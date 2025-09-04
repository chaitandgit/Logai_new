import os
import re
import yaml
import pytz
import numpy as np
import pandas as pd
from dateutil import parser
from elasticsearch import Elasticsearch
from sklearn.preprocessing import OneHotEncoder
from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestDetector, IsolationForestParams
from logai.algorithms.categorical_encoding_algo.label_encoding import LabelEncoding
from autoencoder_detector import AutoEncoderDetector

# --------------------------------------------------------
# 1. Load config
# --------------------------------------------------------
def load_config(path="d:/logai-main/logai/config.yaml"):
    with open(path) as f:
        raw = f.read()
    pattern = re.compile(r"\$\{([^}^{]+)\}")
    return yaml.safe_load(pattern.sub(lambda m: os.environ.get(m.group(1), ""), raw))

cfg = load_config()
es_cfg = cfg["elasticsearch"]

# --------------------------------------------------------
# 2. Connect to Elasticsearch
# --------------------------------------------------------
es = Elasticsearch(
    [es_cfg["host"]],
    basic_auth=(es_cfg["username"], es_cfg["password"]),
    verify_certs=True,
)

resp = es.search(
    index=es_cfg["index"],
    body={"query": es_cfg["query"], "size": 5000}
)

# --------------------------------------------------------
# 3. Flatten ES response
# --------------------------------------------------------
def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

flattened_records = [flatten_dict(hit["_source"]) for hit in resp["hits"]["hits"]]
df = pd.DataFrame(flattened_records)

# --------------------------------------------------------
# 3b. Template cleanup (templates + samples)
# --------------------------------------------------------
def normalize_text(val: str) -> str:
    """Strip junk and normalize."""
    if not isinstance(val, str):
        return ""
    val = val.strip().strip('"')
    val = re.sub(r"^[^\w]+", "", val)     # leading junk
    val = re.sub(r"[^\w.:-]+$", "", val)  # trailing junk
    return val

def is_noisy(val: str) -> bool:
    """Detect garbage strings."""
    if not val or val == "<*>":
        return True
    v = val.lower()
    if len(v) < 5:
        return True
    junk_patterns = [
        r"\baaa\b", r"\bin\b", r"\bodl\b", r"\bder\b",
        r"googleapis\.com", r"\.net$", r"netdev"
    ]
    return any(re.search(p, v) for p in junk_patterns)

for col in ["metadata.template", "metadata.sample_first", "metadata.sample_last"]:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(normalize_text)
        before = len(df)
        df = df[~df[col].apply(is_noisy)]
        after = len(df)
        print(f"🧹 Cleaned {col}: dropped {before - after} noisy rows")

# --------------------------------------------------------
# 3c. Ensure features.tpl_hash exists
# --------------------------------------------------------
if "features.tpl_hash" not in df.columns:
    df["features.tpl_hash"] = df.get("metadata.tpl_hash", "unknown_tpl")

# --------------------------------------------------------
# 4. Robust timestamp parsing
# --------------------------------------------------------
def safe_parse(ts):
    try:
        return parser.isoparse(str(ts)).astimezone(pytz.UTC)
    except Exception:
        return pd.NaT

df["@timestamp"] = df["@timestamp"].apply(safe_parse)
df = df.dropna(subset=["@timestamp"]).sort_values("@timestamp")

# --------------------------------------------------------
# 5. Feature engineering
# --------------------------------------------------------
def extract_code_and_text(row):
    code = row.get("features.tpl_hash") or row.get("metadata.sample_first")
    text = row.get("features.tpl_text") or row.get("metadata.sample_first")
    return code, text

def check_mismatch(row):
    return 0 if row.get("metadata.sample_first") == row.get("metadata.sample_last") else 1

df["extracted_code"], df["extracted_text"] = zip(*df.apply(extract_code_and_text, axis=1))
df["template_mismatch_flag"] = df.apply(check_mismatch, axis=1)

# --------------------------------------------------------
# 6. Burst & sequence anomalies
# --------------------------------------------------------
burst_counts = (
    df.groupby("features.tpl_hash")
      .rolling("5min", on="@timestamp")["metadata.cluster_size"]
      .sum()
      .reset_index()
)
df["burst_count"] = burst_counts["metadata.cluster_size"]

template_stats = df.groupby("features.tpl_hash")["burst_count"].agg(["mean", "std"]).reset_index()
template_stats["threshold"] = template_stats["mean"] + 3 * template_stats["std"]
df = df.merge(template_stats[["features.tpl_hash", "threshold"]], on="features.tpl_hash", how="left")
df["burst_flag"] = (df["burst_count"] > df["threshold"]).astype(int)

df["tpl_id"] = df["features.tpl_hash"].astype("category").cat.codes
df["tpl_prev"] = df["tpl_id"].shift(1)
df["tpl_bigram"] = df["tpl_prev"].astype(str) + "->" + df["tpl_id"].astype(str)

bigram_counts = df["tpl_bigram"].value_counts()
valid_bigrams = set(bigram_counts[bigram_counts > 5].index)
df["sequence_flag"] = (~df["tpl_bigram"].isin(valid_bigrams)).astype(int)

# --------------------------------------------------------
# 7. Schema setup
# --------------------------------------------------------
numerical_features = [
    "features.tpl_len", "features.var_cnt", "features.tpl_complexity",
    "features.hour", "features.msg_len", "features.word_count",
    "metadata.cluster_size", "metadata.tpl_count", "burst_count"
]
boolean_features = ["features.is_dynamic", "features.has_numbers", "burst_flag", "sequence_flag"]
categorical_features = ["metadata.device_id", "metadata.source_file", "extracted_code"]

# Convert features.hour to CST
if "features.hour" in df.columns:
    def convert_hour_to_cst(val):
        try:
            return parser.isoparse(str(val)).astimezone(pytz.timezone("US/Central")).hour
        except Exception:
            return None
    df["features.hour"] = df["features.hour"].apply(convert_hour_to_cst)

# Normalize Unknowns
for col in numerical_features:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
for col in boolean_features:
    if col not in df.columns:
        df[col] = False
    df[col] = df[col].astype(bool)
for col in categorical_features:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(str).replace("Unknown", "")

# --------------------------------------------------------
# 8. Isolation Forest
# --------------------------------------------------------
contamination_value = (
    cfg.get("ml_training", {}).get("autoencoder", {}).get("params", {}).get("contamination")
    or cfg["ml_training"]["isolation_forest"]["params"].get("contamination", 0.05)
)
params = IsolationForestParams(
    n_estimators=cfg["ml_training"]["isolation_forest"]["params"].get("n_estimators", 100),
    contamination=contamination_value,
    random_state=cfg["ml_training"]["isolation_forest"]["params"].get("random_state", 42),
)
detector_iforest = IsolationForestDetector(params)
X_iforest = df[numerical_features + boolean_features + categorical_features].copy()
if categorical_features:
    encoder = LabelEncoding()
    encoded = encoder.fit_transform(X_iforest[categorical_features])
    for i, col in enumerate(categorical_features):
        X_iforest[col] = encoded.iloc[:, i]
detector_iforest.fit(X_iforest)
result_iforest = detector_iforest.predict(X_iforest)
df["anomaly_label_iforest"] = result_iforest["anom_score"]

# --------------------------------------------------------
# 9. AutoEncoder
# --------------------------------------------------------
ae_cfg = cfg["ml_training"].get("autoencoder", {})
ae_params = ae_cfg.get("params", {})
feature_cols_ae = ae_cfg.get("numerical_features", numerical_features)
cat_features_ae = [c for c in categorical_features if c in feature_cols_ae and c in df.columns]
num_features_ae = [c for c in feature_cols_ae if c not in cat_features_ae and c in df.columns]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = ohe.fit_transform(df[cat_features_ae].astype(str)) if cat_features_ae else np.empty((len(df), 0))
num_data = df[num_features_ae].fillna(0).astype(float).values if num_features_ae else np.empty((len(df), 0))
X_ae = np.hstack([num_data, cat_encoded])
detector_ae = AutoEncoderDetector(
    hidden_dim=ae_params.get("hidden_dim", 16),
    latent_dim=ae_params.get("latent_dim", 8),
    lr=ae_params.get("learning_rate", 0.001),
    batch_size=ae_params.get("batch_size", 32),
    num_epochs=ae_params.get("num_epochs", 20),
    contamination=contamination_value,
)
detector_ae.fit(X_ae)
scores_ae, errors_ae = detector_ae.decision_function(X_ae)
labels_ae = detector_ae.predict(X_ae)
df["anomaly_score_autoencoder"] = scores_ae
df["anomaly_label_autoencoder"] = labels_ae

# --------------------------------------------------------
# 10. Explanations
# --------------------------------------------------------
def explain_row(row):
    reasons = []
    if row.get("anomaly_label_autoencoder", 0) == 1:
        recon_err_cols = [c for c in df.columns if c.startswith("recon_err_")]
        if recon_err_cols:
            top_feat = max(recon_err_cols, key=lambda c: row[c])
            reasons.append(f"Autoencoder: unusual {top_feat} = {row[top_feat]}")
    if row.get("burst_flag", 0) == 1:
        reasons.append(f"Burst on {row.get('features.tpl_hash')}")
    if row.get("sequence_flag", 0) == 1:
        reasons.append(f"Unexpected sequence {row.get('tpl_bigram')}")
    if row.get("template_mismatch_flag", 0) == 1:
        reasons.append("Mismatch: sample_first vs sample_last differ")
    return " | ".join(reasons) if reasons else ""

df["explanation_text"] = df.apply(explain_row, axis=1)

# --------------------------------------------------------
# 11. Ensemble
# --------------------------------------------------------
df["anomaly_label_iforest"] = (df["anomaly_label_iforest"] == -1).astype(int)
df["anomaly_label_autoencoder"] = (df["anomaly_label_autoencoder"] == 1).astype(int)
df["anomaly_label_ensemble"] = ((df["anomaly_label_iforest"] + df["anomaly_label_autoencoder"]) >= 1).astype(int)

# --------------------------------------------------------
# 12. Save
# --------------------------------------------------------
df.to_csv("logs_with_anomalies.csv", index=False, escapechar="\\")
print("\n✅ Cleaned results saved to logs_with_anomalies.csv")