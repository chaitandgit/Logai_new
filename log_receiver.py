import os
import re
import yaml
import pandas as pd
from elasticsearch import Elasticsearch

def load_config(path="config.yaml"):
    with open(path) as f:
        raw = f.read()

    # Replace ${VAR} in YAML with environment values
    pattern = re.compile(r'\$\{([^}^{]+)\}')
    def replace_env(match):
        env_var = match.group(1)
        return os.environ.get(env_var, "")

    resolved = pattern.sub(replace_env, raw)
    return yaml.safe_load(resolved)

cfg = load_config()
es_cfg = cfg["elasticsearch"]

# Connect to ES using environment variables (expanded from YAML)
es = Elasticsearch(
    [es_cfg["host"]],
    basic_auth=(es_cfg["username"], es_cfg["password"]),
    verify_certs=True
)

resp = es.search(
    index=es_cfg["index"],
    body={"query": {"range": {"@timestamp": {"gte": "now-15m"}}}},
    size=500
)

df = pd.DataFrame([hit["_source"] for hit in resp["hits"]["hits"]])
print(df.head())



df = pd.DataFrame([hit["_source"] for hit in resp["hits"]["hits"]])

# --- Ensure correct dtypes for LogAI ---

# Booleans → proper Python bool
if "features" in df.columns:
    if "is_dynamic" in df["features"].iloc[0]:
        df["features.is_dynamic"] = df["features"].apply(lambda x: bool(x.get("is_dynamic", False)))
    if "has_numbers" in df["features"].iloc[0]:
        df["features.has_numbers"] = df["features"].apply(lambda x: bool(x.get("has_numbers", False)))

# Numericals → force numeric dtype
numeric_fields = [
    "features.tpl_len", "features.var_cnt", "features.tpl_complexity",
    "features.hour", "features.msg_len", "features.word_count",
    "metadata.cluster_size", "metadata.tpl_count"
]
for field in numeric_fields:
    df[field] = pd.to_numeric(df[field], errors="coerce")

# Categoricals → keep as strings
categorical_fields = ["metadata.device_id", "metadata.source_file"]
for field in categorical_fields:
    df[field] = df[field].astype(str)

print(df.head())

