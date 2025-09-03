import os
import re
import yaml
import pandas as pd
from elasticsearch import Elasticsearch
from logai.applications.application import LogAI
from logai.data.dataset import LogDataset

# --------------------------------------------------------------------
# 1. Load YAML config (with ${VAR} support)
# --------------------------------------------------------------------
def load_config(path="D:/logai_new/config.yaml"):
    with open(path) as f:
        raw = f.read()

    pattern = re.compile(r'\$\{([^}^{]+)\}')
    def replace_env(match):
        env_var = match.group(1)
        return os.environ.get(env_var, "")

    resolved = pattern.sub(replace_env, raw)
    return yaml.safe_load(resolved)

cfg = load_config()
es_cfg = cfg["elasticsearch"]

# --------------------------------------------------------------------
# 2. Connect to Elasticsearch
# --------------------------------------------------------------------
es = Elasticsearch(
    [es_cfg["host"]],
    basic_auth=(es_cfg["username"], es_cfg["password"]),
    verify_certs=True
)

resp = es.search(
    index=es_cfg["index"],
    body={"query": {"match_all": {}}},
    size=5000
)

# --------------------------------------------------------------------
# 3. Flatten ES records into DataFrame
# --------------------------------------------------------------------
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

flattened_records = []
for hit in resp["hits"]["hits"]:
    flat = flatten_dict(hit.get("_source", {}))
    if "fields" in hit:
        flat.update(flatten_dict(hit["fields"]))
    flattened_records.append(flat)

df = pd.DataFrame(flattened_records)
print("Preview of flattened DataFrame:")
print(df.head())

# --------------------------------------------------------------------
# 4. Ensure required schema (numerical, boolean, categorical)
# --------------------------------------------------------------------
numerical_features = [
    "features.tpl_len", "features.var_cnt", "features.tpl_complexity",
    "features.hour", "features.msg_len", "features.word_count",
    "metadata.cluster_size", "metadata.tpl_count"
]
boolean_features = ["features.is_dynamic", "features.has_numbers"]
categorical_features = ["metadata.device_id", "metadata.source_file"]
metadata_columns = [
    "@timestamp", "metadata.template", "metadata.tpl_hash",
    "metadata.sample_first", "metadata.sample_last"
]

# Add missing numeric columns
for col in numerical_features:
    if col not in df.columns:
        df[col] = float("nan")
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Add missing boolean columns
for col in boolean_features:
    if col not in df.columns:
        df[col] = False
    df[col] = df[col].astype(bool)

# Add missing categorical columns
for col in categorical_features:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(str)

# --------------------------------------------------------------------
# 5. Build LogDataset
# --------------------------------------------------------------------
dataset = LogDataset.from_dataframe(
    dataframe=df,
    feature_names=numerical_features + boolean_features + categorical_features,
    metadata_names=metadata_columns
)

# --------------------------------------------------------------------
# 6. Run Isolation Forest via LogAI
# --------------------------------------------------------------------
app = LogAI(config="D:/logai_new/config.yaml")
result = app.run(dataset)

print("\n=== First few rows with anomaly labels ===")
print(result.head())

print("\n=== Anomaly counts ===")
print(result["anomaly_label"].value_counts())

# --------------------------------------------------------------------
# 7. Save results
# --------------------------------------------------------------------
result.to_csv("logs_with_anomalies.csv", index=False)
print("\n✅ Results saved to logs_with_anomalies.csv")