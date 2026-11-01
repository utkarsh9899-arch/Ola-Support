"""Tasks 11 + 12 demo against a REAL uvicorn server on localhost.

Task 11: POST /ask, POST /add-document (Pydantic models), WebSocket /ws/chat; a client disconnects
         mid-conversation and the server keeps serving other clients.
Task 12: every request -> one JSON-Lines log entry with trace id + timing; the raw phone number used
         in a request never reaches the log file.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import httpx  # noqa: E402
import uvicorn  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from api.main import app  # noqa: E402
from demos._transcript import transcript  # noqa: E402
from ola_agent.logging_utils import LOG_FILE, read_log, reset_log  # noqa: E402
from ola_agent.rag.index import delete_document  # noqa: E402

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
WS = f"ws://127.0.0.1:{PORT}/ws/chat"
RAW_PHONE = "+91 98765 43210"  # fabricated
NEW_DOC = {
    "doc_id": "13_lost_item_policy",
    "title": "Lost Item Recovery Policy",
    "text": ("Riders who leave an item in a cab can report it from the trip details screen within 24 hours. "
             "Support opens a masked call bridge so the rider and driver can arrange the return without sharing "
             "phone numbers. A return fee of INR 150 is charged when the driver delivers the item to the rider."),
}


def start_server() -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return server


def show(label, resp):
    print(f"{label} -> HTTP {resp.status_code}  X-Trace-ID={resp.headers.get('x-trace-id')}")
    return resp.json()


if __name__ == "__main__":
    reset_log()
    with transcript("task11_12_api"):
        server = start_server()
        print(f"uvicorn serving api.main:app on {BASE} (MOCK_LLM={config.MOCK_LLM})\n")
        expected_entries = 0
        with httpx.Client(base_url=BASE, timeout=120) as http:
            print("=" * 100 + "\nTASK 11 - HTTP endpoints (Pydantic request/response models)\n" + "=" * 100)
            print(show("GET  /health", http.get("/health")))
            expected_entries += 1

            body = {"query": f"Hi, my phone is {RAW_PHONE}. What is the status of ticket OLA-T-0044?"}
            data = show("POST /ask", http.post("/ask", json=body))
            expected_entries += 1
            print(f"  request : {body}")
            print(f"  response: {json.dumps(data['response'])[:400]}...\n")

            data = show("POST /add-document", http.post("/add-document", json=NEW_DOC))
            expected_entries += 1
            print(f"  {data}")
            data = show("POST /ask (uses the new document)", http.post("/ask", json={"query": "I left my bag in the cab. How do I get it back?"}))
            expected_entries += 1
            print(f"  answer: {data['response']['answer']}\n")

            bad = http.post("/ask", json={"query": ""})
            expected_entries += 1
            print(f"POST /ask with empty query -> HTTP {bad.status_code} (Pydantic validation): {bad.json()['detail'][0]['msg']}\n")

            print("=" * 100 + "\nTASK 11 - WebSocket /ws/chat, client disconnects mid-conversation\n" + "=" * 100)
            with connect(f"{WS}?session_id=ws-client-A") as ws:
                ws.send("What is the status of ticket OLA-T-0016?")
                print(f"client A turn 1 <- {json.loads(ws.recv())['response']['answer'][:110]}...")
                ws.send("Should it be escalated?")
                print("client A sent turn 2 and DISCONNECTED before reading the reply")
            # turn 1 + turn 2 (the server still processes it after the client left) + the disconnect event
            expected_entries += 3
            time.sleep(3)  # let the server finish turn 2 and notice the closed socket

            print(f"server still alive? GET /health -> {http.get('/health').json()}")
            expected_entries += 1
            with connect(f"{WS}?session_id=ws-client-B") as ws:
                for i, msg in enumerate(["What is the status of ticket OLA-T-0005?", "Should it be escalated?"], 1):
                    ws.send(msg)
                    print(f"client B turn {i} -> {msg}\n                <- {json.loads(ws.recv())['response']['answer'][:110]}...")
            expected_entries += 3  # two messages + normal close
            time.sleep(1)

        removed = delete_document(NEW_DOC["doc_id"])  # keep the KB identical for later tasks
        print(f"\n(cleanup: removed demo document chunks {removed})")
        server.should_exit = True
        time.sleep(1)

        print("\n" + "=" * 100 + f"\nTASK 12 - structured JSON-Lines log: {LOG_FILE.relative_to(config.ROOT)}\n" + "=" * 100)
        entries = read_log()
        for e in entries:
            print(json.dumps(e))
        raw = LOG_FILE.read_text(encoding="utf-8")
        print(f"\nlog lines: {len(entries)}  (requests made: {expected_entries})")
        print(f"every line is valid JSON with a trace.id and event.duration_ms: "
              f"{all('trace.id' in e and 'event.duration_ms' in e for e in entries)}")
        print(f"trace ids unique: {len({e['trace.id'] for e in entries}) == len(entries)}")
        events = [e["event.action"] for e in entries]
        print(f"events: {', '.join(f'{x} x{events.count(x)}' for x in dict.fromkeys(events))}")
        disconnects = [e for e in entries if e["event.action"] == "ws_disconnect"]
        print(f"ws disconnects logged (session, turns completed, exception caught): "
              f"{[(d['session.id'], d['ws.turns_completed'], d['ws.disconnect']) for d in disconnects]}")
        print(f"one log entry per request: {len(entries) == expected_entries}")
        digits = RAW_PHONE.replace(" ", "").replace("+", "")
        print(f"raw phone {RAW_PHONE!r} in log file: {RAW_PHONE in raw}; digits {digits[2:]!r} in log file: {digits[2:] in raw}")
        print(f"[PHONE_REDACTED] in log file: {'[PHONE_REDACTED]' in raw}  -> PII leak check: "
              f"{'PASS' if RAW_PHONE not in raw and digits[2:] not in raw else 'FAIL'}")
        sample = config.LOG_DIR / "requests.sample.jsonl"
        sample.write_text(raw, encoding="utf-8")
        print(f"(copied to {sample.relative_to(config.ROOT)} for the repository)")
