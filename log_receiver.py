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