import json
import os
import time
import requests

FASTAPI_URL = "http://127.0.0.1:8000"


def run_scenario(scenario_folder="scenario_01"):
    print(f"--- STARTING EVALUATION FOR {scenario_folder} ---")

    # Clear old inference log file before replaying to keep outputs clean
    if os.path.exists("inferred_events.jsonl"):
        os.remove("inferred_events.jsonl")

    # 1. Load and process history seed into memory
    history_file = f"{scenario_folder}/history_seed.jsonl"
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    # Directly feed history seed to ingestion stream
                    requests.post(f"{FASTAPI_URL}/events/stream", json=event)
        print(f"✅ History seed from {history_file} loaded into memory.")
    except FileNotFoundError:
        print(f"⚠️️ Could not find {history_file}")

    # Clear log file again after seeding so history logs don't contaminate the evaluation run
    if os.path.exists("inferred_events.jsonl"):
        os.remove("inferred_events.jsonl")

    # 2. Stream live events line-by-line
    stream_file = f"{scenario_folder}/live_stream.jsonl"
    print(f"🚀 Replaying live event stream from {stream_file}...")

    try:
        with open(stream_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    res = requests.post(
                        f"{FASTAPI_URL}/events/stream", json=event
                    )
                    print(
                        f"Processed Event [{event.get('customer_id')}]: Status {res.status_code}"
                    )
                    time.sleep(0.1)  # Simulate live stream pacing
    except FileNotFoundError:
        print(f"❌ Error: Stream file {stream_file} not found.")
        return

    print(
        f"\n✅ Finished replaying {scenario_folder}. Check inferred_events.jsonl for outputs.\n"
    )


if __name__ == "__main__":
    # Test on scenario 1
    run_scenario("scenario_01")