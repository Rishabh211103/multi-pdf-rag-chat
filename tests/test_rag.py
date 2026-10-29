from app.rag import NO_CONTEXT_ANSWER, normalize_citations, retrieve
from app.schemas import ChatTurn


def _index_two_docs(document_service, make_pdf):
    a, _ = document_service.ingest(make_pdf(["Alpha document about cats.\nCats purr."]), "a.pdf")
    b, _ = document_service.ingest(make_pdf(["Beta document about dogs.\nDogs bark."]), "b.pdf")
    return a, b


def test_retrieve_respects_document_filter(document_service, store, make_pdf):
    _, b = _index_two_docs(document_service, make_pdf)

    docs = retrieve(store, "animals", k=5, fetch_k=10, use_mmr=True, doc_ids=[b.doc_id])

    assert docs
    assert {d.metadata["doc_id"] for d in docs} == {b.doc_id}


def test_answer_returns_citable_sources(document_service, make_pdf, make_rag):
    _index_two_docs(document_service, make_pdf)
    rag = make_rag(["Cats purr [1]."])

    response = rag.answer("What do cats do?", history=[])

    assert response.answer == "Cats purr [1]."
    assert response.standalone_question == "What do cats do?"  # no history -> no rewrite
    assert [s.index for s in response.sources] == list(range(1, len(response.sources) + 1))
    assert all(s.filename in {"a.pdf", "b.pdf"} for s in response.sources)


def test_follow_up_question_is_condensed(document_service, make_pdf, make_rag):
    _index_two_docs(document_service, make_pdf)
    rag = make_rag(["What sound do dogs make?", "Dogs bark [1]."])
    history = [
        ChatTurn(role="user", content="Tell me about dogs"),
        ChatTurn(role="assistant", content="Dogs are animals."),
    ]

    response = rag.answer("What sound do they make?", history=history)

    assert response.standalone_question == "What sound do dogs make?"
    assert response.answer == "Dogs bark [1]."


def test_normalize_citations():
    assert normalize_citations("Revenue grew 12%【1】【2】.") == "Revenue grew 12%[1][2]."


def test_no_documents_gives_fallback_answer(make_rag):
    rag = make_rag(["should not be called"])

    response = rag.answer("Anything?", history=[])

    assert response.answer == NO_CONTEXT_ANSWER
    assert response.sources == []


def test_stream_event_order(document_service, make_pdf, make_rag):
    _index_two_docs(document_service, make_pdf)
    rag = make_rag(["Cats purr."])

    events = list(rag.stream("What do cats do?", history=[]))

    assert events[0]["type"] == "sources"
    assert events[-1] == {"type": "done"}
    tokens = "".join(e["data"] for e in events if e["type"] == "token")
    assert tokens == "Cats purr."
