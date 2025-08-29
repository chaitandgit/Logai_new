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
import os, re, json, logging, tempfile, time
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, PatternMatchingEventHandler
# Prefer normal import; if running inside a container where `file_filters` wasn't installed,
# fall back to loading a local `file_filters.py` next to this script.
try:
    from file_filters import is_utf8, is_text_file
except ModuleNotFoundError:
    import importlib.util
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _ff_path = os.path.join(_this_dir, "file_filters.py")
    if os.path.exists(_ff_path):
        spec = importlib.util.spec_from_file_location("file_filters", _ff_path)
        _ff = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ff)
        is_utf8 = _ff.is_utf8
        is_text_file = _ff.is_text_file
    else:
        raise
from drain3.template_miner import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
import hashlib

# Config
CONFIG = {
    "log_dirs": ["/input/var", "/input/data"],
    "base_output": os.environ.get("BASE_OUTPUT", "/output/log_parser"),
}
CONFIG.update({
    "excluded_dirs": ["/input/var/memstats", "/input/var/ipcstats"],
    "delta_file": os.path.join(CONFIG["base_output"], "ml_delta_append.jsonl"),
    "full_file": os.path.join(CONFIG["base_output"], "drain.jsonl"),
    # Keep seen templates compact as JSONL for streaming/ingest
    "seen_file": os.path.join(CONFIG["base_output"], "seen_templates.jsonl")
})
os.makedirs(CONFIG["base_output"], exist_ok=True)

LOG_LEVEL = getattr(logging, os.environ.get("DRAIN_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["base_output"], "drain_runner.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("drain3_runner")


def load_seen(path):
    try:
        if not os.path.exists(path):
            return {}
        # Support both JSON and JSONL: detect by peeking at first non-space char
        with open(path, "r", encoding="utf-8") as f:
            first = f.read(1)
            if not first:
                return {}
            f.seek(0)
            if first == "[" or first == "{" :
                try:
                    return json.load(f)
                except Exception:
                    # fall back to JSONL
                    f.seek(0)
            # JSONL: each line is a JSON object with template_id key
            seen = {}
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    tid = obj.get("template_id")
                    if tid:
                        seen[tid] = obj
                except Exception:
                    continue
            return seen
    except Exception:
        log.exception(f"Failed to load seen file: {path}")
        return {}


def save_seen(path, seen):
    try:
        # Write as JSONL: one compact object per line for streaming/append-friendly storage
        temp_dir = os.path.dirname(path) or "."
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=temp_dir)
            for tid, data in seen.items():
                # ensure template_id present
                obj = dict(data)
                obj["template_id"] = tid
                tmp.write(json.dumps(obj, separators=(",", ":")) + "\n")
            tmp.flush()
            tmp_name = tmp.name
            tmp.close()
            os.replace(tmp_name, path)
        finally:
            try:
                if tmp is not None:
                    tmp.close()
            except Exception:
                pass
    except Exception:
        log.exception(f"Failed to save seen file: {path}")


def clean_log_line(line):
    line = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[-+]\d{2}:?\d{2})?", "<TIME>", line)
    line = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", line)
    line = re.sub(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", "<IP>", line)
    # Mask MAC addresses like 01:23:45:67:89:ab or 01-23-45-67-89-ab
    line = re.sub(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b", "<MAC>", line)
    line = re.sub(r"username=[^&\s]+", "username=<REDACTED>", line, flags=re.IGNORECASE)
    line = re.sub(r"password=[^&\s]+", "password=<REDACTED>", line, flags=re.IGNORECASE)
    return line


def extract_timestamp(line):
    try:
        match = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[-+]\d{2}:?\d{2})?)", line)
        if match:
            ts = datetime.fromisoformat(match.group(1).replace(" ", "T"))
            return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def compute_sha1(template):
    return hashlib.sha1(template.encode("utf-8")).hexdigest()


def build_miner(state_file):
    cfg = TemplateMinerConfig()
    cfg.similarity = 0.4
    cfg.depth = 6
    miner = TemplateMiner(FilePersistence(state_file), cfg)
    if os.path.exists(state_file):
        try:
            miner.load_state()
            log.debug(f"Drain3 state loaded successfully from {state_file}")
        except Exception as e:
            log.error(f"Failed to load Drain3 state from {state_file}: {e}")
    return miner


def enrich_log_with_drain3(line, miner, seen, source_file):
    log.debug(f"Processing log line: {line.strip()}.")
    # Accept legacy/embedded levels like L4 as well as standard levels
    log_level_regex = re.compile(r"(INFO|DEBUG|ERROR|WARN|CRITICAL|L\d+)", re.IGNORECASE)
    if not log_level_regex.search(line):
        log.warning(f"Log line missing log level: {line.strip()}.")
    cleaned = clean_log_line(line)
    timestamp = extract_timestamp(line)
    result = miner.add_log_message(cleaned)
    if not result:
        log.debug("No template generated for the log line.")
        return None
    template = result["template_mined"]
    tpl_id = compute_sha1(template)
    reason = "new template" if tpl_id not in seen else "count increased"
    log_level_match = log_level_regex.search(line)
    raw_level = log_level_match.group(0).upper() if log_level_match else None
    # try to capture a short text following the level up to the next ':' (e.g. "L4 logproxy:")
    level_text = None
    if raw_level:
        mlt = re.search(r"\b(?:INFO|DEBUG|ERROR|WARN|CRITICAL|L\d+)\b\s+([^:]+):", line, re.IGNORECASE)
        if mlt:
            level_text = mlt.group(1).strip()

    # Normalize L# into a coarse severity mapping (adjust thresholds as needed)
    log_level = "UNKNOWN"
    if raw_level:
        if raw_level.startswith("L") and raw_level[1:].isdigit():
            lvl_num = int(raw_level[1:])
            if lvl_num <= 3:
                log_level = "DEBUG"
            elif lvl_num == 4:
                log_level = "INFO"
            elif lvl_num == 5:
                log_level = "WARNING"
            else:
                log_level = "ERROR"
        else:
            log_level = raw_level
    suspicious = timestamp < "2010-01-01T00:00:00Z"
    # Lightweight extraction of network tuple fields for analytics
    src = None
    dst = None
    sport = None
    dport = None
    m = re.search(r"\bSRC=([^\s]+)", line, re.IGNORECASE)
    if m:
        src = m.group(1)
    m = re.search(r"\bDST=([^\s]+)", line, re.IGNORECASE)
    if m:
        dst = m.group(1)
    m = re.search(r"\bSPT=(\d+)", line, re.IGNORECASE)
    if m:
        sport = m.group(1)
    m = re.search(r"\bDPT=(\d+)", line, re.IGNORECASE)
    if m:
        dport = m.group(1)
    if tpl_id not in seen:
        seen[tpl_id] = {
            "count": 1,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "sample_log": line.strip(),
            "source": source_file,
            "template": template,
            "suspicious": suspicious,
            "last_src": src,
            "last_dst": dst,
            "last_sport": sport,
            "last_dport": dport,
            "last_log_level": log_level,
            "last_log_level_text": level_text,
        }
        log.debug(f"New template added: {tpl_id}.")
    else:
        seen[tpl_id]["count"] += 1
        seen[tpl_id]["last_seen"] = timestamp
        seen[tpl_id]["suspicious"] = suspicious
        seen[tpl_id]["last_src"] = src
        seen[tpl_id]["last_dst"] = dst
        seen[tpl_id]["last_sport"] = sport
        seen[tpl_id]["last_dport"] = dport
        seen[tpl_id]["last_log_level"] = log_level
        seen[tpl_id]["last_log_level_text"] = level_text
        log.debug(f"Updated template: {tpl_id}, count: {seen[tpl_id]['count']}.")
    enriched = {
        "template_id": tpl_id,
        "template": template,
        "count": seen[tpl_id]["count"],
        "first_seen": seen[tpl_id]["first_seen"],
        "last_seen": seen[tpl_id]["last_seen"],
        "reason": reason,
        "sample_log": seen[tpl_id].get("sample_log"),
        "log_level": log_level,
        "log_level_text": level_text,
    "src": src,
    "dst": dst,
    "sport": sport,
    "dport": dport,
        "device_id": os.getenv("DEVICE_ID"),
        "suspicious": suspicious,
    }
    return enriched


def write_to_files(enriched, delta_path, full_path, seen):
    MAX_DELTA_FILE_SIZE = 10 * 1024 * 1024
    # Append new delta entry
    with open(delta_path, "a", encoding="utf-8") as delta_f:
        delta_f.write(json.dumps(enriched) + "\n")

    # If delta file exceeded max, keep the most recent entries up to MAX_DELTA_FILE_SIZE
    if os.path.getsize(delta_path) > MAX_DELTA_FILE_SIZE:
        with open(delta_path, "r", encoding="utf-8") as delta_f:
            lines = delta_f.readlines()
        # Accumulate lines from the end until we reach MAX_DELTA_FILE_SIZE, preserving order
        accumulated = []
        current_size = 0
        for line in reversed(lines):
            line_size = len(line.encode("utf-8"))
            if current_size + line_size <= MAX_DELTA_FILE_SIZE:
                accumulated.append(line)
                current_size += line_size
            else:
                break
        final_lines = list(reversed(accumulated))
        with open(delta_path, "w", encoding="utf-8") as delta_f:
            for line in final_lines:
                delta_f.write(line)
    updated_templates = {}
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as full_f:
            for line in full_f:
                try:
                    entry = json.loads(line.strip())
                    tpl_id = entry.get("template_id")
                    if tpl_id in seen:
                        entry["count"] = seen[tpl_id]["count"]
                        entry["last_seen"] = seen[tpl_id]["last_seen"]
                    updated_templates[tpl_id] = entry
                except json.JSONDecodeError:
                    log.warning(f"Skipping invalid JSON line in {full_path}")
    tpl_id = enriched["template_id"]
    if tpl_id not in updated_templates:
        updated_templates[tpl_id] = enriched
    # Write atomically to avoid partial files
    temp_dir = os.path.dirname(full_path) or "."
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=temp_dir)
        for entry in updated_templates.values():
            tmp.write(json.dumps(entry) + "\n")
        tmp.flush()
        tmp_name = tmp.name
        tmp.close()
        os.replace(tmp_name, full_path)
    finally:
        try:
            if tmp is not None:
                tmp.close()
        except Exception:
            pass


class LogHandler(FileSystemEventHandler):
    valid_extensions = (".log", ".txt")
    excluded_extensions = (".mem", ".stat", ".bin", ".tmp")
    excluded_directories = tuple(CONFIG.get("excluded_dirs", []))

    def __init__(self, miner, seen, delta_path, full_path):
        self.miner = miner
        self.seen = seen
        self.delta_path = delta_path
        self.full_path = full_path

    def on_created(self, event):
        if event.is_directory:
            return
        if self.is_in_excluded_directories(event.src_path):
            log.debug(f"Ignoring event from excluded directory: {event.src_path}")
            return
        if is_text_file(event.src_path, self.valid_extensions, self.excluded_extensions):
            self.process_file(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self.is_in_excluded_directories(event.src_path):
            log.debug(f"Ignoring event from excluded directory: {event.src_path}")
            return
        if is_text_file(event.src_path, self.valid_extensions, self.excluded_extensions):
            self.process_file(event.src_path)

    def is_in_excluded_directories(self, file_path):
        normalized = os.path.normpath(file_path)
        return any(normalized.startswith(os.path.normpath(ed)) for ed in self.excluded_directories)

    def process_file(self, file_path):
        try:
            if not is_utf8(file_path):
                log.debug(f"Skipping non-UTF8 file: {file_path}")
                return
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    enriched = enrich_log_with_drain3(line, self.miner, self.seen, file_path)
                    if enriched:
                        write_to_files(enriched, self.delta_path, self.full_path, self.seen)
            save_seen(CONFIG["seen_file"], self.seen)
        except Exception:
            log.exception(f"Failed while processing file: {file_path}")


class FilteredEventHandler(PatternMatchingEventHandler):
    def __init__(self, delegate_handler, patterns, ignore_patterns=None):
        super().__init__(patterns=patterns, ignore_patterns=ignore_patterns, ignore_directories=True, case_sensitive=False)
        self.delegate = delegate_handler

    def on_created(self, event):
        if event.is_directory:
            return
        if is_utf8(event.src_path):
            self.delegate.process_file(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if is_utf8(event.src_path):
            self.delegate.process_file(event.src_path)


def get_files_to_monitor(directories, valid_extensions, excluded_extensions):
    files_to_monitor = []
    excluded_directories = tuple(CONFIG.get("excluded_dirs", []))
    for directory in directories:
        if not os.path.exists(directory):
            continue
        for root, _, files in os.walk(directory):
            if any(os.path.normpath(root).startswith(os.path.normpath(ed)) for ed in excluded_directories):
                continue
            for fname in files:
                path = os.path.join(root, fname)
                if path.endswith(valid_extensions) and not path.endswith(excluded_extensions):
                    if is_utf8(path):
                        files_to_monitor.append(path)
    return files_to_monitor


def get_watch_roots(root_dirs, excluded_dirs, valid_exts, excluded_exts):
    watch_roots = []
    for root in root_dirs:
        if not os.path.exists(root):
            continue
        for current_root, dirs, files in os.walk(root):
            norm = os.path.normpath(current_root)
            if any(norm.startswith(os.path.normpath(ed)) for ed in excluded_dirs):
                dirs[:] = []
                continue
            has_valid = False
            for fname in files:
                if (fname.endswith(valid_exts) or re.search(r"\.log\.\d+$", fname)) and not fname.endswith(excluded_exts):
                    has_valid = True
                    break
            if has_valid:
                watch_roots.append(current_root)
    return sorted(set(watch_roots))


def initial_scan(log_dirs, miner, seen, delta_path, full_path):
    files = get_files_to_monitor(log_dirs, LogHandler.valid_extensions, LogHandler.excluded_extensions)
    for p in files:
        try:
            log.info(f"Processing file during initial scan: {p}")
            if not is_utf8(p):
                log.debug(f"Skipping non-UTF8 file during initial scan: {p}")
                continue
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    enriched = enrich_log_with_drain3(line, miner, seen, p)
                    if enriched:
                        write_to_files(enriched, delta_path, full_path, seen)
        except Exception as e:
            log.exception(f"Error processing file {p} during initial scan: {e}")
    save_seen(CONFIG["seen_file"], seen)


def ensure_drain_jsonl_complete(seen, full_path):
    """Ensure the drain.jsonl file contains entries for all templates in `seen`.

    This is intentionally robust: it skips invalid JSON lines and preserves counts/last_seen
    from the in-memory `seen` mapping when available.
    """
    log.debug("Starting ensure_drain_jsonl_complete.")
    updated_templates = {}
    if os.path.exists(full_path):
        log.debug(f"Loading existing entries from {full_path}.")
        with open(full_path, "r", encoding="utf-8") as full_f:
            for line in full_f:
                try:
                    entry = json.loads(line.strip())
                    tpl_id = entry.get("template_id")
                    if tpl_id:
                        updated_templates[tpl_id] = entry
                except json.JSONDecodeError:
                    log.warning(f"Skipping invalid JSON line in {full_path}: {line.strip()}")

    missing_templates = 0
    for tpl_id, template_data in seen.items():
        log.debug(f"Processing template_id: {tpl_id}")
        if not all(key in template_data for key in ["count", "first_seen", "last_seen", "sample_log"]):
            log.warning(f"Template_id {tpl_id} is missing required fields in seen_templates.json: {template_data}")
            continue
        if tpl_id not in updated_templates:
            log.debug(f"Adding missing template_id: {tpl_id} to drain.jsonl.")
            # Prefer level information from the in-memory `seen` mapping when available.
            inferred_level = template_data.get("last_log_level", "UNKNOWN")
            inferred_level_text = template_data.get("last_log_level_text", "Log level not detected")
            updated_templates[tpl_id] = {
                "template_id": tpl_id,
                "template": template_data.get("template", "<MISSING_TEMPLATE>"),
                "count": template_data["count"],
                "first_seen": template_data["first_seen"],
                "last_seen": template_data["last_seen"],
                "sample_log": template_data["sample_log"],
                "reason": "from seen_templates",
                "log_level": inferred_level,
                "log_level_text": inferred_level_text,
                "device_id": os.getenv("DEVICE_ID")
            }
            missing_templates += 1
        else:
            log.debug(f"Template_id: {tpl_id} already exists in drain.jsonl.")

    log.debug(f"Writing updated templates to {full_path}.")
    with open(full_path, "w", encoding="utf-8") as full_f:
        for entry in updated_templates.values():
            full_f.write(json.dumps(entry) + "\n")

    log.info(f"ensure_drain_jsonl_complete completed. Total templates in seen_templates.json: {len(seen)}, Total templates written to drain.jsonl: {len(updated_templates)}, Missing templates added: {missing_templates}.")


def backfill_delta_levels(delta_path, seen):
    """Backfill missing/UNKNOWN log level fields in the delta file using the `seen` mapping.

    For each JSON line in `delta_path`, if log_level is missing or UNKNOWN, try to find a
    matching template in `seen` (by exact template string) and fill last_log_level/last_log_level_text.
    The file is rewritten atomically if any changes are made.
    Returns number of lines updated.
    """
    if not os.path.exists(delta_path):
        return 0
    changed = 0
    out_lines = []
    try:
        with open(delta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except Exception:
                    out_lines.append(line)
                    continue
                needs = (not entry.get("log_level")) or entry.get("log_level") == "UNKNOWN"
                if needs:
                    tpl = entry.get("template")
                    if tpl:
                        match = None
                        for tid, data in seen.items():
                            if data.get("template") == tpl:
                                match = data
                                break
                        if match:
                            entry["log_level"] = match.get("last_log_level", "UNKNOWN")
                            entry["log_level_text"] = match.get("last_log_level_text", "Log level not detected")
                            changed += 1
                out_lines.append(json.dumps(entry) + "\n")
    except Exception:
        log.exception(f"Failed to read delta file for backfill: {delta_path}")
        return 0

    if changed:
        try:
            temp_dir = os.path.dirname(delta_path) or "."
            tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=temp_dir)
            tmp.writelines(out_lines)
            tmp.flush()
            tmp_name = tmp.name
            tmp.close()
            os.replace(tmp_name, delta_path)
            log.info(f"Backfilled {changed} entries in {delta_path} using seen mapping.")
        finally:
            try:
                if tmp is not None:
                    tmp.close()
            except Exception:
                pass

    return changed


def main():
    # Load seen templates (expects compact JSONL at CONFIG['seen_file']).
    seen = load_seen(CONFIG["seen_file"])
    miner = build_miner(os.path.join(CONFIG["base_output"], "drain_state.json"))
    log.info("Starting initial scan of existing log files.")
    initial_scan(CONFIG["log_dirs"], miner, seen, CONFIG["delta_file"], CONFIG["full_file"])
    log.info("Initial scan completed. Ensuring drain.jsonl is complete.")
    ensure_drain_jsonl_complete(seen, CONFIG["full_file"]) if os.path.exists(CONFIG["full_file"]) else None
    log.info("drain.jsonl is now complete. Starting watchdog observer.")

    observer = Observer()
    watch_roots = get_watch_roots(CONFIG["log_dirs"], tuple(CONFIG.get("excluded_dirs", [])), LogHandler.valid_extensions, LogHandler.excluded_extensions)
    if watch_roots:
        patterns = ["*.log", "*.txt", "*.log.*"]
        for wr in watch_roots:
            filtered = FilteredEventHandler(LogHandler(miner, seen, CONFIG["delta_file"], CONFIG["full_file"]), patterns)
            observer.schedule(filtered, wr, recursive=False)
            log.info(f"Scheduled watch on: {wr}")
    else:
        for d in CONFIG["log_dirs"]:
            log.warning(f"Directory does not exist or no watchable subdirs found: {d}")

    observer.start()
    log.info("Watchdog observer started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, stopping observer...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
