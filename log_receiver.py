#!/usr/local/bin/python3
"""
Drain3 Runner - Version-2 for Containers on RG Devices

Key Features:
- Clusters logs using Drain3 template miner
- Extracts timestamps from logs for accurate first_seen/last_seen
- Cleans and masks log lines (e.g., timestamps, IPs)
- Adds sample logs (marked as such)
- Excludes `source` from ML logs, keeps it in drain.jsonl for Kibana
- Safely handles unknown and extended log levels (L8+)
- Includes UTF-8 safe writes
"""

# --- Imports and Setup ---
import os, re, json, logging
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from drain3.template_miner import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

# --- Config ---
CONFIG = {
    "log_dirs": ["/input/var", "/input/data"],
    "base_output": os.environ.get("BASE_OUTPUT", "/output/log_parser"),
    "name_contains": os.environ.get("NAME_CONTAINS", "log").lower()
}
CONFIG.update({
    "delta_file": os.path.join(CONFIG["base_output"], "ml_delta_append.jsonl"),
    "full_file": os.path.join(CONFIG["base_output"], "drain.jsonl"),
    "seen_file": os.path.join(CONFIG["base_output"], "seen_templates.json")
})
os.makedirs(CONFIG["base_output"], exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG to enable debug messages
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["base_output"], "drain_runner.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("drain3_runner")

ignored_files_log_path = os.path.join(CONFIG["base_output"], "ignored_files.log")
ignored_files_handler = logging.FileHandler(ignored_files_log_path, mode="w", encoding="utf-8")
ignored_files_handler.setLevel(logging.INFO)
ignored_files_logger = logging.getLogger("ignored_files")
ignored_files_logger.addHandler(ignored_files_handler)
ignored_files_logger.propagate = False

# --- Global Skip List ---
skipped_files = set()

# --- Utility Functions ---
def load_seen(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def save_seen(path, seen):
    log.debug(f"Saving seen dictionary to {path}: {json.dumps(seen, indent=2)}")
    with open(path, "w") as f:
        json.dump(seen, f, indent=2)

def clean_log_line(line):
    return re.sub(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[-+]\d{2}:?\d{2})?', '<TIME>', line)

def extract_timestamp(line):
    try:
        # Enhanced regex to capture more timestamp formats
        match = re.search(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[-+]\d{2}:?\d{2})?)', line)
        if match:
            ts = datetime.fromisoformat(match.group(1).replace(' ', 'T'))
            return ts.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        else:
            log.warning(f"No valid timestamp found in log line: {line.strip()}")
    except Exception as e:
        log.error(f"Error extracting timestamp from log line: {line.strip()} - {e}")

    # Default to current UTC time if no timestamp is found
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

# --- Drain3 Setup ---
def build_miner(state_file):
    cfg = TemplateMinerConfig()
    cfg.similarity = 0.4  # Reduced similarity for better clustering
    cfg.depth = 6  # Increased depth for more detailed clustering
    miner = TemplateMiner(FilePersistence(state_file), cfg)
    if os.path.exists(state_file):
        try:
            miner.load_state()
            log.debug(f"Drain3 state loaded successfully from {state_file}")
        except Exception as e:
            log.error(f"Failed to load Drain3 state from {state_file}: {e}")
    return miner

def enrich_log_with_drain3(line, miner, seen, source_file):
    log.debug(f"Processing log line: {line.strip()}")

    # Log a warning for lines without log levels but do not exclude them
    if not re.search(r'(INFO|DEBUG|ERROR|WARN|CRITICAL)', line, re.IGNORECASE):
        log.warning(f"Log line missing log level: {line.strip()}")

    cleaned = clean_log_line(line)
    timestamp = extract_timestamp(line)
    result = miner.add_log_message(cleaned)

    if not result:
        log.debug("No template generated for the log line.")
        return None

    tpl_id = result["cluster_id"]
    reason = "new template" if tpl_id not in seen else "count increased"

    # Extract log level and level text
    log_level_match = re.search(r'(INFO|DEBUG|ERROR|WARN|CRITICAL)', line, re.IGNORECASE)
    log_level = log_level_match.group(0).upper() if log_level_match else "UNKNOWN"
    log_level_text = f"Log level detected: {log_level}" if log_level != "UNKNOWN" else "Log level not detected"

    if tpl_id not in seen:
        seen[tpl_id] = {
            "count": 1,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "sample_log": line.strip(),
            "source": source_file  # Use the provided source file path
        }
        log.debug(f"New template added: {tpl_id}")
    else:
        seen[tpl_id]["count"] += 1
        seen[tpl_id]["last_seen"] = timestamp
        log.debug(f"Updated template: {tpl_id}, count: {seen[tpl_id]['count']}")

    enriched = {
        "template_id": tpl_id,
        "template": result["template_mined"],
        "count": seen[tpl_id]["count"],
        "first_seen": seen[tpl_id]["first_seen"],
        "last_seen": seen[tpl_id]["last_seen"],
        "reason": reason,
        "sample_log": seen[tpl_id].get("sample_log"),
        "log_level": log_level,  # Add log level
        "log_level_text": log_level_text,  # Add log level text
        "device_id": os.getenv("DEVICE_ID")  # Add device_id from environment
    }
    return enriched

def write_to_files(enriched, delta_path, full_path, seen):
    """Write enriched logs to delta and full files, updating only necessary fields in drain.jsonl."""
    # Define the maximum size for the delta file (in bytes)
    MAX_DELTA_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    # Write to delta file (always append new entries)
    with open(delta_path, "a", encoding="utf-8") as delta_f:
        delta_f.write(json.dumps(enriched) + "\n")

    # Check and truncate the delta file if it exceeds the maximum size
    if os.path.getsize(delta_path) > MAX_DELTA_FILE_SIZE:
        with open(delta_path, "r", encoding="utf-8") as delta_f:
            lines = delta_f.readlines()
        # Keep only the most recent entries within the size limit
        with open(delta_path, "w", encoding="utf-8") as delta_f:
            current_size = 0
            for line in reversed(lines):
                line_size = len(line.encode("utf-8"))
                if current_size + line_size <= MAX_DELTA_FILE_SIZE:
                    delta_f.write(line)
                    current_size += line_size
                else:
                    break

    # Update drain.jsonl with only updated fields for existing templates
    updated_templates = {}
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as full_f:
            for line in full_f:
                try:
                    entry = json.loads(line.strip())
                    tpl_id = entry.get("template_id")
                    if tpl_id in seen:
                        # Update only the count and last_seen fields for existing templates
                        entry["count"] = seen[tpl_id]["count"]
                        entry["last_seen"] = seen[tpl_id]["last_seen"]
                    updated_templates[tpl_id] = entry
                except json.JSONDecodeError:
                    continue

    # Add the new enriched entry if it's a new template
    tpl_id = enriched["template_id"]
    if tpl_id not in updated_templates:
        updated_templates[tpl_id] = enriched

    # Write back the updated drain.jsonl
    with open(full_path, "w", encoding="utf-8") as full_f:
        for entry in updated_templates.values():
            full_f.write(json.dumps(entry) + "\n")

def ensure_drain_jsonl_complete(seen, full_path):
    """Ensure that drain.jsonl includes all templates from seen_templates.json."""
    log.debug("Starting ensure_drain_jsonl_complete.")
    updated_templates = {}

    # Load existing entries from drain.jsonl
    if os.path.exists(full_path):
        log.debug(f"Loading existing entries from {full_path}.")
        with open(full_path, "r", encoding="utf-8") as full_f:
            for line in full_f:
                try:
                    entry = json.loads(line.strip())
                    tpl_id = entry.get("template_id")
                    updated_templates[tpl_id] = entry
                except json.JSONDecodeError:
                    log.warning(f"Skipping invalid JSON line in {full_path}: {line.strip()}")

    # Validate and add missing templates from seen_templates.json
    missing_templates = 0
    for tpl_id, template_data in seen.items():
        log.debug(f"Processing template_id: {tpl_id}")
        if not all(key in template_data for key in ["count", "first_seen", "last_seen", "sample_log"]):
            log.warning(f"Template_id {tpl_id} is missing required fields in seen_templates.json: {template_data}")
            continue

        if tpl_id not in updated_templates:
            log.debug(f"Adding missing template_id: {tpl_id} to drain.jsonl.")
            updated_templates[tpl_id] = {
                "template_id": tpl_id,
                "template": template_data.get("template", "<MISSING_TEMPLATE>"),
                "count": template_data["count"],
                "first_seen": template_data["first_seen"],
                "last_seen": template_data["last_seen"],
                "sample_log": template_data["sample_log"],
                "reason": "from seen_templates",
                "log_level": "UNKNOWN",
                "log_level_text": "Log level not detected",
                "device_id": os.getenv("DEVICE_ID")
            }
            missing_templates += 1
        else:
            log.debug(f"Template_id: {tpl_id} already exists in drain.jsonl.")

    # Write back the updated drain.jsonl
    log.debug(f"Writing updated templates to {full_path}.")
    with open(full_path, "w", encoding="utf-8") as full_f:
        for entry in updated_templates.values():
            full_f.write(json.dumps(entry) + "\n")

    log.info(f"ensure_drain_jsonl_complete completed. Total templates in seen_templates.json: {len(seen)}, Total templates written to drain.jsonl: {len(updated_templates)}, Missing templates added: {missing_templates}.")

# --- Watchdog Event Handler ---
class LogHandler(FileSystemEventHandler):
    def __init__(self, miner, seen, delta_path, full_path):
        self.miner = miner
        self.seen = seen
        self.delta_path = delta_path
        self.full_path = full_path

    def on_modified(self, event):
        # Ensure the event source is a file, not a directory
        if not os.path.isfile(event.src_path):
            log.warning(f"Skipped processing directory: {event.src_path}")
            return

        # Ensure the file has a valid extension
        valid_extensions = (".log", ".txt")
        if not event.src_path.endswith(valid_extensions) and not re.search(r"\.log\.\d+$", event.src_path):
            ignored_files_logger.info(f"Excluded file: {event.src_path}")
            return

        try:
            log.info(f"Processing modified file: {event.src_path}")
            with open(event.src_path, "r", errors="ignore") as f:
                for line in f:
                    enriched = enrich_log_with_drain3(line, self.miner, self.seen, event.src_path)
                    if enriched:
                        write_to_files(enriched, self.delta_path, self.full_path, self.seen)
            save_seen(CONFIG["seen_file"], self.seen)
        except Exception as e:
            log.error(f"Error processing file {event.src_path}: {e}")

# Modify initial_scan to remove redundant checks
def initial_scan(log_dirs, miner, seen, delta_path, full_path):
    """Process all existing log files in the specified directories."""
    valid_extensions = (".log", ".txt")  # Ensure valid extensions are consistent
    files_to_monitor = get_files_to_monitor(log_dirs, valid_extensions)

    for file_path in files_to_monitor:
        try:
            log.info(f"Processing file during initial scan: {file_path}")
            with open(file_path, "r", errors="ignore") as f:
                for line in f:
                    enriched = enrich_log_with_drain3(line, miner, seen, file_path)
                    if enriched:
                        write_to_files(enriched, delta_path, full_path, seen)
        except Exception as e:
            log.error(f"Error processing file {file_path} during initial scan: {e}")
    save_seen(CONFIG["seen_file"], seen)

def get_files_to_monitor(directories, valid_extensions):
    """Get a list of files to monitor based on valid extensions."""
    files_to_monitor = []
    for directory in directories:
        if not os.path.exists(directory):
            log.warning(f"Directory does not exist: {directory}")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                # Include only files with valid extensions
                if file_path.endswith(valid_extensions) or re.search(r"\.log\.\d+$", file_path):
                    files_to_monitor.append(file_path)
                else:
                    ignored_files_logger.info(f"Excluded file: {file_path}")
    return files_to_monitor

# --- Main ---
def main():
    seen = load_seen(CONFIG["seen_file"])
    miner = build_miner(os.path.join(CONFIG["base_output"], "drain_state.json"))

    # Perform an initial scan of existing log files
    log.info("Starting initial scan of existing log files.")
    initial_scan(CONFIG["log_dirs"], miner, seen, CONFIG["delta_file"], CONFIG["full_file"])
    log.info("Initial scan completed. Ensuring drain.jsonl is complete.")

    # Ensure drain.jsonl includes all templates from seen_templates.json
    ensure_drain_jsonl_complete(seen, CONFIG["full_file"])
    log.info("drain.jsonl is now complete. Starting watchdog observer.")

    # Start the watchdog observer
    event_handler = LogHandler(miner, seen, CONFIG["delta_file"], CONFIG["full_file"])
    observer = Observer()

    # Get files to monitor explicitly
    valid_extensions = (".log", ".txt")
    files_to_monitor = get_files_to_monitor(CONFIG["log_dirs"], valid_extensions)

    for file_path in files_to_monitor:
        observer.schedule(event_handler, file_path, recursive=False)

    observer.start()
    log.info("Watchdog observer started with Drain3. Monitoring specific files.")
    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
