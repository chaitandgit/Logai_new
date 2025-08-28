# Logai_new


1.	Monitor /input/var and /input/data directories for .log and .txt files using watchdog
	2.	Use Drain3 to cluster log lines into templates, persisting state with FilePersistence
	3.	Extract timestamps (from ISO 8601 or common formats) and track first_seen and last_seen
	4.	Track log-level (INFO, DEBUG, ERROR, WARN, CRITICAL) and also allow unknown levels as "UNKNOWN"
	5.	Mask dynamic content in logs like timestamps and IP addresses
	6.	Write enriched log templates to:
	•	ml_delta_append.jsonl (only new/updated templates)
	•	drain.jsonl (full templates with count and last_seen updates)
	7.	Maintain a seen_templates.json dictionary to deduplicate and persist known clusters
	8.	Ensure UTF-8 safe writes and handle corrupted JSON lines gracefully
	9.	Skip writing to disk (i.e., dry-run mode, just print enriched JSON to console for now)
	10.	Include logging to console and to drain_runner.log
	11.	Skip files not ending in .log or .txt, and track them in ignored_files.log
	12.	Be well-commented and organized with clear functions for: initial_scan, write_to_files, extract_timestamp, build_miner, etc.
	13.	Use os.environ.get to allow configurable paths like BASE_OUTPUT, DEVICE_ID, and NAME_CONTAINS
	14.	Ensure drain.jsonl is complete with any missing templates from seen_templates.json
	15.	Log everything with debug/info/warning levels and clearly explain what each function does