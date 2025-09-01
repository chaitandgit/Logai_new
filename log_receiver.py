import yaml, pandas as pd
from elasticsearch import Elasticsearch

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

es = Elasticsearch([cfg["elasticsearch"]["host"]])
resp = es.search(index=cfg["elasticsearch"]["index"],
                 body={"query": cfg["elasticsearch"]["query"]},
                 size=500)

df = pd.DataFrame([hit["_source"] for hit in resp["hits"]["hits"]])
# pass df to LogAI / NLP