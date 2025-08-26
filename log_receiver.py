import json
import csv
import hashlib
from datetime import datetime

INPUT_FILE = "Sample_logs.txt"
OUTPUT_FILE = "logai_features.csv"

def parse_timestamp(orig_ts):
    try:
        # Convert ISO timestamp to epoch seconds with fractional part
        dt = datetime.fromisoformat(orig_ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None

def extract_features(entry, record_id):
    logline = entry.get("sample_logs", [""])[0]
    device_id = entry.get("device_id", "")
    template_id = entry.get("template_id", "")
    log_level_text = entry.get("log_level_text", "")
    template_frequency = entry.get("template_frequency", 1)
    orig_timestamp = entry.get("orig_timestamp", "")
    timestamp = parse_timestamp(orig_timestamp)

    return {
        "logline": logline,
        "_id": record_id,
        "is_anomaly": False,
        "device_id": device_id,
        "template_id": template_id,
        "log_level_text": log_level_text,
        "template_frequency": template_frequency,
        "timestamp": timestamp
    }

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=[
            "logline", "_id", "is_anomaly", "device_id",
            "template_id", "log_level_text", "template_frequency", "timestamp"
        ])
        writer.writeheader()
        for i, line in enumerate(infile):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                features = extract_features(entry, record_id=i)
                writer.writerow(features)
            except Exception as e:
                print(f"[WARN] Skipping line {i}: {e}")

if __name__ == "__main__":
    main()