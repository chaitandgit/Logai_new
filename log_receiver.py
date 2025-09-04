import os
import re
import yaml
import pytz
import numpy as np
import pandas as pd
from dateutil import parser
from elasticsearch import Elasticsearch

# --------------------------------------------------------
# 1. Load config
# --------------------------------------------------------
def load_config(path="d:/logai-main/logai/config.yaml"):
    with open(path) as f:
        raw = f.read()
    pattern = re.compile(r"\$\{([^}^{]+)\}")
    def replace_env(match):
        return os.environ.get(match.group(1), "")
    return yaml.safe_load(pattern.sub(replace_env, raw))

cfg = load_config()
es_cfg = cfg["elasticsearch"]

# --------------------------------------------------------
# 2. Connect to Elasticsearch & run query with _source filter
# --------------------------------------------------------
es = Elasticsearch(
    [es_cfg["host"]],
    basic_auth=(es_cfg["username"], es_cfg["password"]),
    verify_certs=True,
)

resp = es.search(
    index=es_cfg["index"],
    body={
        "_source": [
            "@timestamp",
            "doc_id",
            "message",          # raw log text
            "features.*",
            "metadata.*"
        ],
        "query": es_cfg["query"]
    },
    size=5000
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

flattened_records = []
for hit in resp["hits"]["hits"]:
    src = hit.get("_source", {})
    flattened_records.append(flatten_dict(src))

df = pd.DataFrame(flattened_records)

# --------------------------------------------------------
# 4. Resolve conflicts: prefer features.*, fallback to metadata.*
# --------------------------------------------------------
if "features.tpl_hash" not in df.columns:
    if "metadata.tpl_hash" in df.columns:
        df["features.tpl_hash"] = df["metadata.tpl_hash"]
        print("ℹ️ Using metadata.tpl_hash as features.tpl_hash")
    else:
        df["features.tpl_hash"] = "unknown_tpl"
        print("⚠️ No tpl_hash found, assigning 'unknown_tpl'")

# Drop duplicate metadata columns if features.* already exists
for col in df.columns:
    if col.startswith("metadata.") and col.replace("metadata.", "features.") in df.columns:
        df.drop(columns=[col], inplace=True)

# --------------------------------------------------------
# 5. Parse timestamps robustly
# --------------------------------------------------------
def safe_parse(ts):
    try:
        return parser.isoparse(str(ts)).astimezone(pytz.UTC)
    except Exception:
        return pd.NaT

if "@timestamp" in df.columns:
    df["@timestamp"] = df["@timestamp"].apply(safe_parse)
    bad_ts = df["@timestamp"].isna().sum()
    if bad_ts > 0:
        print(f"⚠️ Dropping {bad_ts} rows with invalid timestamps")
        df = df.dropna(subset=["@timestamp"])
    df = df.sort_values("@timestamp")
else:
    print("⚠️ No @timestamp field found!")

# --------------------------------------------------------
# 6. Final preview
# --------------------------------------------------------
print("\n✅ Final DataFrame ready for ML:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# Optional: save to CSV for inspection
df.to_csv("logs_flattened_clean.csv", index=False, escapechar="\\")
print("\n✅ Cleaned logs saved to logs_flattened_clean.csv")




# --------------------------------------------------------
# 4b. Fix bad templates ("<*>")
# --------------------------------------------------------
if "metadata.template" in df.columns:
    def clean_template(val, fallback):
        if pd.isna(val):
            return fallback
        s = str(val).strip()
        # If template is only <*> tokens (like "<*>" or "<*> <*>")
        if re.fullmatch(r"(<\*>[\s]*)+", s):
            return fallback or "unknown_template"
        return s

    # Use sample_first or tpl_hash as fallback
    df["metadata.template"] = df.apply(
        lambda r: clean_template(r.get("metadata.template"), r.get("metadata.sample_first") or r.get("features.tpl_hash")),
        axis=1
    )