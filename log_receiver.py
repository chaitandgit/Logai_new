# On your ML server
echo '
input:
  type: elasticsearch
  config:
    host: "plantxhssrvr11.ops.e2e.labs.att.com:9200"
    index: "log-parser"
    username: "cg722v"
    password: "RGStatus2025"
    
processing:
  - name: anomaly_detection
    type: isolation_forest
    features: ["features.tpl_len", "features.var_cnt", "metadata.cluster_size"]
' > logai_config.yaml

# Install and run
pip install logai
logai --config logai_config.yaml