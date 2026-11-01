import pytest
from fastapi.testclient import TestClient

from api.main import app
from ola_agent import logging_utils
from ola_agent.rag.index import delete_document


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    log_file = tmp_path_factory.mktemp("logs") / "requests.jsonl"
    original = logging_utils.LOG_FILE
    logging_utils.reset_log(original)
    logging_utils.LOG_FILE = log_file
    logging_utils.get_request_logger.__defaults__ = (log_file,)
    with TestClient(app) as c:
        yield c
    logging_utils.reset_log(log_file)
    logging_utils.LOG_FILE = original
    logging_utils.get_request_logger.__defaults__ = (original,)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "mock_llm": True}


def test_ask_returns_validated_response_and_trace_id(client):
    r = client.post("/ask", json={"query": "What is the status of ticket OLA-T-0044?"})
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] == r.headers["x-trace-id"]
    assert body["response"]["record_id"] == "OLA-T-0044"


def test_ask_rejects_empty_query(client):
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_add_document_validates_and_upserts(client):
    assert client.post("/add-document", json={"doc_id": "Bad Id!", "title": "x", "text": "y"}).status_code == 422
    r = client.post("/add-document", json={"doc_id": "99_test_doc", "title": "Test Policy",
                                           "text": "This is a test sentence. This is another test sentence."})
    assert r.status_code == 200 and r.json()["chunks_upserted"]["ola_kb_sentence"] == 2
    delete_document("99_test_doc")


def test_websocket_disconnect_does_not_kill_server(client):
    with client.websocket_connect("/ws/chat?session_id=t") as ws:
        ws.send_text("What is the status of ticket OLA-T-0005?")
        assert ws.receive_json()["response"]["record_id"] == "OLA-T-0005"
    assert client.get("/health").status_code == 200


def test_every_request_logged_with_trace_id_and_no_raw_phone(client):
    logging_utils.reset_log(logging_utils.LOG_FILE)
    client.post("/ask", json={"query": "my number is 9876543210, status of OLA-T-0001?"})
    entries = logging_utils.read_log(logging_utils.LOG_FILE)
    assert len(entries) == 1 and entries[0]["trace.id"] and "event.duration_ms" in entries[0]
    assert "9876543210" not in logging_utils.LOG_FILE.read_text(encoding="utf-8")
