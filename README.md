# Logai_new


Please enhance this existing Python script used to cluster log files using Drain3 on embedded/containerized devices.

✅ The following features are already implemented, so keep them as-is:
	•	Uses watchdog to monitor .log and .txt files in /input/var and /input/data
	•	Extracts timestamps from log lines in ISO and other formats
	•	Uses Drain3 for clustering
	•	Tracks first_seen, last_seen, count, sample_log, and template_id
	•	Writes full clusters to drain.jsonl and delta updates to ml_delta_append.jsonl
	•	Maintains seen_templates.json for persistence
	•	Skips invalid extensions, tracks skipped files in ignored_files.log
	•	Logs to console and drain_runner.log, supports BASE_OUTPUT, DEVICE_ID, and NAME_CONTAINS from env vars
	•	Handles malformed JSON, unknown log levels, and UTF-8 encoding

🚧 Now add or fix the following missing pieces:
	1.	Mask IP addresses in each log line (both IPv4 and IPv6) using regex before clustering.
	2.	Add a DRY_RUN toggle (boolean or env var). When True, the script should skip writing to disk and just print the enriched JSON to stdout.
	3.	Fix the watchdog observer to recursively monitor all subdirectories inside /input/var and /input/data, not just specific files.

🧩 Bonus if possible:
	•	Ensure comments and structure are clean and readable.
	•	Validate that each function logs what it’s doing with DEBUG or INFO level messages.
