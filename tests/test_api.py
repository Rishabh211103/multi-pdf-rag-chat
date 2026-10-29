import json

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_document_service, get_rag_service, get_registry
from app.main import app


@pytest.fixture
def client(document_service, make_rag):
    app.dependency_overrides[get_document_service] = lambda: document_service
    app.dependency_overrides[get_registry] = lambda: document_service.registry
    app.dependency_overrides[get_rag_service] = lambda: make_rag(["Answer from the docs [1]."])
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload(client, *files):
    return client.post(
        "/documents",
        files=[("files", (name, data, "application/pdf")) for name, data in files],
    )


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_upload_list_and_delete(client, make_pdf):
    pdf = make_pdf(["Quarterly revenue grew 12 percent."])

    first = _upload(client, ("report.pdf", pdf), ("bad.pdf", b"not a pdf")).json()
    assert [r["status"] for r in first] == ["indexed", "failed"]
    assert "not a valid PDF" in first[1]["error"]

    again = _upload(client, ("report-copy.pdf", pdf)).json()
    assert again[0]["status"] == "duplicate"

    docs = client.get("/documents").json()
    assert [d["filename"] for d in docs] == ["report.pdf"]

    doc_id = docs[0]["doc_id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get("/documents").json() == []


def test_chat(client, make_pdf):
    _upload(client, ("report.pdf", make_pdf(["Quarterly revenue grew 12 percent."])))

    body = client.post("/chat", json={"question": "How much did revenue grow?"}).json()

    assert body["answer"] == "Answer from the docs [1]."
    assert body["sources"][0]["filename"] == "report.pdf"


def test_chat_stream(client, make_pdf):
    _upload(client, ("report.pdf", make_pdf(["Quarterly revenue grew 12 percent."])))

    with client.stream("POST", "/chat/stream", json={"question": "Revenue growth?"}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    assert "".join(e["data"] for e in events if e["type"] == "token") == "Answer from the docs [1]."


def test_chat_validates_input(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422
