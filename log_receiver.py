import os
import re
import yaml
import pandas as pd
from elasticsearch import Elasticsearch
from logai.applications.application import LogAI
from logai.dataloader.opendataloader import OpenSetDataLoader
import sys

# Remove the local 'd:\\logai_new\\logai' path from sys.path
sys.path = [p for p in sys.path if p != 'd:\\logai_new\\logai']

def load_config(path="D:\\logai_new\\config.yaml"):
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
    body={
        "query": {
            "match_all": {}
        },
        "size": 10000  # Adjust size as needed to fetch more records
    }
)

def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Flatten Elasticsearch response
flattened_records = []
for hit in resp["hits"]["hits"]:
    flattened_record = flatten_dict(hit.get("_source", {}))
    if "fields" in hit:
        flattened_record.update(flatten_dict(hit["fields"]))
    flattened_records.append(flattened_record)

df = pd.DataFrame(flattened_records)
print(df.head())

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
# Check for missing numeric fields upfront
missing = [f for f in numeric_fields if f not in df.columns]
if missing:
    print("⚠️ Missing fields in ES response:", missing)
# Handle missing numeric fields gracefully
for field in numeric_fields:
    if field in df.columns:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    else:
        print(f"⚠️ Field '{field}' is missing. Filling with NaN.")
        df[field] = float('nan')

# Categoricals → keep as strings
categorical_fields = ["metadata.device_id", "metadata.source_file"]
for field in categorical_fields:
    df[field] = df[field].astype(str)

print(df.head())

# Ensure all required features exist in the DataFrame
numerical_features = [
    "features.tpl_len", "features.var_cnt", "features.tpl_complexity",
    "features.hour", "features.msg_len", "features.word_count",
    "metadata.cluster_size", "metadata.tpl_count"
]
boolean_features = ["features.is_dynamic", "features.has_numbers"]
categorical_features = ["metadata.device_id", "metadata.source_file"]

# Check for missing features
missing_numerical = [f for f in numerical_features if f not in df.columns]
missing_boolean = [f for f in boolean_features if f not in df.columns]
missing_categorical = [f for f in categorical_features if f not in df.columns]

# Add missing numerical features
for feature in missing_numerical:
    df[feature] = float('nan')

# Add missing boolean features
for feature in missing_boolean:
    df[feature] = False

# Add missing categorical features
for feature in missing_categorical:
    df[feature] = ""

print("Missing numerical features added:", missing_numerical)
print("Missing boolean features added:", missing_boolean)
print("Missing categorical features added:", missing_categorical)

# Log the total hits from Elasticsearch
if "hits" in resp and "total" in resp["hits"]:
    total_hits = resp["hits"]["total"]
    if isinstance(total_hits, dict):
        total_hits = total_hits.get("value", 0)  # Handle ES 7.x+ format
    print(f"Total hits in Elasticsearch: {total_hits}")

# Compare total hits with DataFrame size
if len(df) < total_hits:
    print(f"⚠️ Warning: Retrieved only {len(df)} rows out of {total_hits} total rows.")
else:
    print("✅ All data has been successfully retrieved.")

# Log the number of rows in the DataFrame
print(f"Number of rows retrieved: {len(df)}")

# Save the DataFrame to a CSV file
df.to_csv("flattened_elasticsearch_data_new.csv", index=False, escapechar='\\')
print("Data saved to 'flattened_elasticsearch_data_new.csv'")

# --- Run Isolation Forest using LogAI ---
# 1. Load LogAI config
app = LogAI(config="config.yaml")

# 2. Load flattened CSV into LogDataset
loader = OpenSetDataLoader(
    dataset_name="csv_dataset",
    dataset_path="flattened_elasticsearch_data_new.csv"
)
dataset = loader.load_data()

# 3. Run Isolation Forest (LogAI handles feature extraction per config.yaml)
result = app.run(dataset)

# 4. Inspect results
print("=== First few rows with anomaly labels ===")
print(result.head())

print("\n=== Anomaly counts ===")
print(result["anomaly_label"].value_counts())

# 5. Save enriched results locally
result.to_csv("logs_with_anomalies.csv", index=False)
print("\nSaved results to logs_with_anomalies.csv")
