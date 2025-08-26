#!/usr/bin/env python3
"""
Wrapper script to extract features from clustered Drain3 logs (ml_delta.jsonl)
and save them in a CSV format suitable for LogAI input.
"""

import json
import csv
from datetime import datetime

# Input/Output files
INPUT_FILE = "drain_logs/ml_delta.jsonl"
OUTPUT_FILE = "extracted_features.csv"

# Helper: Convert ISO 8601 timestamp to float seconds (optional)
def timestamp_to_float(ts: str) -> float:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0

# Main conversion logic
def convert_jsonl_to_csv(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        # Column headers expected by LogAI
        writer.writerow([
            "logline", "_id", "is_anomaly", "device_id",
            "template_id", "log_level_text", "template_frequency", "timestamp"
        ])

        id_counter = 0
        for line in infile:
            try:
                data = json.loads(line)
                sample_logs = data.get("sample_logs", [])
                for logline in sample_logs:
                    writer.writerow([
                        logline,
                        id_counter,
                        False,
                        data.get("device_id"),
                        data.get("template_id"),
                        data.get("log_level_text"),
                        data.get("template_frequency", 1),
                        timestamp_to_float(data.get("first_seen", ""))
                    ])
                    id_counter += 1
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    convert_jsonl_to_csv(INPUT_FILE, OUTPUT_FILE)
    print(f"Extracted features saved to: {OUTPUT_FILE}")