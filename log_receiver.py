import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

try:
    import pandas as pd
except Exception:
    pd = None


def parse_timestamp(orig_ts):
    if not orig_ts:
        return None
    try:
        dt = datetime.fromisoformat(orig_ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        try:
            return float(orig_ts)
        except Exception:
            return None


def process_jsonl_file(path, agg):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            device = entry.get("device_id") or entry.get("device") or "unknown"
            tmpl = entry.get("template_id") or entry.get("template") or ""
            tf = entry.get("template_frequency")
            try:
                tf = float(tf) if tf is not None else 1.0
            except Exception:
                tf = 1.0
            is_anom = bool(entry.get("is_anomaly") or entry.get("suspicious"))
            ts = parse_timestamp(entry.get("orig_timestamp") or entry.get("timestamp") or entry.get("first_seen"))

            d = agg[device]
            d["total_events"] += 1
            d["sum_template_frequency"] += tf
            d["anomaly_count"] += 1 if is_anom else 0
            if tmpl:
                d["templates_counter"][tmpl] += 1
            if ts:
                if d["earliest_ts"] is None or ts < d["earliest_ts"]:
                    d["earliest_ts"] = ts
                if d["latest_ts"] is None or ts > d["latest_ts"]:
                    d["latest_ts"] = ts


def process_parquet_file(path, agg):
    if pd is None:
        print(f"[WARN] pandas not available, skipping parquet file: {path}")
        return
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[WARN] Failed to read parquet {path}: {e}")
        return

    for _, row in df.iterrows():
        device = row.get("device_id") or row.get("device") or "unknown"
        tmpl = row.get("template_id") or row.get("template") or ""
        tf = row.get("template_frequency")
        try:
            tf = float(tf) if tf is not None else 1.0
        except Exception:
            tf = 1.0
        is_anom = bool(row.get("is_anomaly") or row.get("suspicious"))
        ts = parse_timestamp(row.get("orig_timestamp") or row.get("timestamp") or row.get("first_seen"))

        d = agg[device]
        d["total_events"] += 1
        d["sum_template_frequency"] += tf
        d["anomaly_count"] += 1 if is_anom else 0
        if tmpl:
            d["templates_counter"][tmpl] += 1
        if ts:
            if d["earliest_ts"] is None or ts < d["earliest_ts"]:
                d["earliest_ts"] = ts
            if d["latest_ts"] is None or ts > d["latest_ts"]:
                d["latest_ts"] = ts


def make_empty_device_record():
    return {
        "total_events": 0,
        "sum_template_frequency": 0.0,
        "anomaly_count": 0,
        "templates_counter": Counter(),
        "earliest_ts": None,
        "latest_ts": None,
    }


def write_features_csv(output_path, agg):
    fieldnames = [
        "device_id",
        "total_events",
        "unique_templates",
        "top_template",
        "top_template_count",
        "avg_template_frequency",
        "anomaly_rate",
        "earliest_timestamp",
        "latest_timestamp",
    ]
    with open(output_path, "w", newline='', encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for device, d in sorted(agg.items()):
            total = d["total_events"]
            unique_templates = len(d["templates_counter"]) if d["templates_counter"] else 0
            top_template, top_count = (None, 0)
            if d["templates_counter"]:
                top_template, top_count = d["templates_counter"].most_common(1)[0]
            avg_tf = (d["sum_template_frequency"] / total) if total else 0.0
            anomaly_rate = (d["anomaly_count"] / total) if total else 0.0
            earliest = datetime.fromtimestamp(d["earliest_ts"]).isoformat() if d["earliest_ts"] else ""
            latest = datetime.fromtimestamp(d["latest_ts"]).isoformat() if d["latest_ts"] else ""

            writer.writerow({
                "device_id": device,
                "total_events": total,
                "unique_templates": unique_templates,
                "top_template": top_template,
                "top_template_count": top_count,
                "avg_template_frequency": round(avg_tf, 3),
                "anomaly_rate": round(anomaly_rate, 4),
                "earliest_timestamp": earliest,
                "latest_timestamp": latest,
            })


def write_device_features(output_dir, agg):
    """Write individual device feature files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "device_id",
        "total_events",
        "unique_templates",
        "top_template",
        "top_template_count",
        "avg_template_frequency",
        "anomaly_rate",
        "earliest_timestamp",
        "latest_timestamp",
    ]
    for device, d in agg.items():
        device_file = output_dir / f"{device}_features.csv"
        with open(device_file, "w", newline='', encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            total = d["total_events"]
            unique_templates = len(d["templates_counter"]) if d["templates_counter"] else 0
            top_template, top_count = (None, 0)
            if d["templates_counter"]:
                top_template, top_count = d["templates_counter"].most_common(1)[0]
            avg_tf = (d["sum_template_frequency"] / total) if total else 0.0
            anomaly_rate = (d["anomaly_count"] / total) if total else 0.0
            earliest = datetime.fromtimestamp(d["earliest_ts"]).isoformat() if d["earliest_ts"] else ""
            latest = datetime.fromtimestamp(d["latest_ts"]).isoformat() if d["latest_ts"] else ""

            writer.writerow({
                "device_id": device,
                "total_events": total,
                "unique_templates": unique_templates,
                "top_template": top_template,
                "top_template_count": top_count,
                "avg_template_frequency": round(avg_tf, 3),
                "anomaly_rate": round(anomaly_rate, 4),
                "earliest_timestamp": earliest,
                "latest_timestamp": latest,
            })


def main(uploaded_dir: str = None):
    base_dir = Path(uploaded_dir) if uploaded_dir else Path(__file__).resolve().parents[1] / "uploaded_files"
    if not base_dir.exists():
        print(f"[ERROR] uploaded_files directory not found: {base_dir}")
        return

    agg = defaultdict(make_empty_device_record)

    for p in sorted(base_dir.iterdir()):
        if p.is_dir():
            continue
        name = p.name.lower()
        try:
            if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
                process_jsonl_file(p, agg)
            elif name.endswith(".results.parquet") or name.endswith(".parquet"):
                process_parquet_file(p, agg)
            elif name.endswith(".json"):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        for entry in data:
                            device = entry.get("device_id") or entry.get("device") or "unknown"
                            tmpl = entry.get("template_id") or entry.get("template") or ""
                            tf = entry.get("template_frequency") or 1.0
                            is_anom = bool(entry.get("is_anomaly") or entry.get("suspicious"))
                            ts = parse_timestamp(entry.get("orig_timestamp") or entry.get("timestamp") or entry.get("first_seen"))
                            d = agg[device]
                            d["total_events"] += 1
                            try:
                                d["sum_template_frequency"] += float(tf)
                            except Exception:
                                d["sum_template_frequency"] += 0.0
                            d["anomaly_count"] += 1 if is_anom else 0
                            if tmpl:
                                d["templates_counter"][tmpl] += 1
                            if ts:
                                if d["earliest_ts"] is None or ts < d["earliest_ts"]:
                                    d["earliest_ts"] = ts
                                if d["latest_ts"] is None or ts > d["latest_ts"]:
                                    d["latest_ts"] = ts
                except Exception:
                    continue
            else:
                continue
        except Exception as e:
            print(f"[WARN] Error processing {p}: {e}")

    output_dir = base_dir / "extracted_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_file = output_dir / "combined_features.csv"
    write_features_csv(combined_file, agg)
    print(f"Wrote combined features to: {combined_file}")

    write_device_features(output_dir, agg)
    print(f"Wrote per-device features to: {output_dir}")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
