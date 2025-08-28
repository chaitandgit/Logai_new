# drain3_test
"""
Write a Python function that post-processes parsed log entries from a JSONL file (e.g., ml_delta_append.jsonl).

Goals:
1. For any log with a timestamp before 1990 (e.g., 1969/1970), override the timestamp to a fixed value like "1970-01-01T00:00:00Z"
2. Add a new field "suspicious": true for such logs, else "suspicious": false
3. Ensure the output remains in valid JSONL format (one JSON object per line)
4. Save the modified logs to a new output file, e.g., ml_delta_tagged.jsonl
5. Preserve all original fields except the modified timestamp and added suspicious tag

Make sure this logic can be reused as a function, with input/output file paths passed as arguments.
"""

Enhance this log clustering script to implement SHA1-based template tracking:
	1.	After clustering a log line using Drain3 or similar, compute the template_id as a SHA1 hash of the cleaned template string (remove variable parts like timestamps/IPs if not already done).
	2.	Maintain a persistent file seen_templates.json that stores for each template_id:
	•	count: number of times this template was seen
	•	first_seen: ISO UTC timestamp of the first occurrence
	•	last_seen: ISO UTC timestamp of the most recent occurrence
	3.	For each new log cluster:
	•	If template_id is not in seen_templates.json, add it and write the cluster to ml_delta_append.jsonl and ml_delta_full.jsonl
	•	If already seen, only update count and last_seen, but do not re-write to ML output
	4.	Save the updated seen_templates.json file after processing all logs.
	5.	Use UTF-8-safe JSON read/write, and ensure the script can resume from previous state.

(Optional: Include a field "suspicious": true in the output if the log timestamp is before 2010 or looks abnormal.)

