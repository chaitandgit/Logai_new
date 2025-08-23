
import os
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
API_SECRET = os.getenv("API_SECRET", "changeme123")  # fallback if not set

@app.post("/ingest")
async def ingest(request: Request):
    # Secret check
    auth = request.headers.get("X-API-KEY")
    if auth != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    device_id = request.headers.get("X-Device-ID", "unknown-device")
    data = await request.body()

    os.makedirs("received_logs", exist_ok=True)
    file_path = f"received_logs/{device_id}.jsonl"

    with open(file_path, "ab") as f:
        f.write(data + b"\n")

    return {"status": "ok", "device": device_id}
