"""Conversational RAG: rewrite question -> retrieve -> answer with citations."""

from collections.abc import Iterator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import Settings
from app.schemas import ChatResponse, ChatTurn, Source

NO_CONTEXT_ANSWER = "I couldn't find anything relevant in the selected documents."

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a follow-up question, rewrite the follow-up "
            "as a standalone question that can be understood without the conversation. "
            "Do NOT answer it. Return only the rewritten question.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about the user's PDF documents.\n"
            "Answer using ONLY the numbered context passages below. Cite the passages you use "
            "inline like [1] or [2][3]. If the context does not contain the answer, say that you "
            "couldn't find it in the documents. Do not make up information.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def retrieve(
    store: Chroma,
    query: str,
    k: int,
    fetch_k: int,
    use_mmr: bool,
    doc_ids: list[str] | None = None,
) -> list[Document]:
    search_kwargs: dict = {"k": k}
    if doc_ids:
        search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}
    if use_mmr:
        # MMR trades a little relevance for diversity, which helps when
        # several PDFs contain near-duplicate passages.
        search_kwargs["fetch_k"] = fetch_k
        retriever = store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        retriever = store.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def to_sources(docs: list[Document], snippet_chars: int = 300) -> list[Source]:
    return [
        Source(
            index=i,
            doc_id=d.metadata["doc_id"],
            filename=d.metadata["source"],
            page=d.metadata["page"],
            snippet=d.page_content[:snippet_chars],
        )
        for i, d in enumerate(docs, start=1)
    ]


def normalize_citations(text: str) -> str:
    # Some models (e.g. gpt-oss) cite with full-width brackets: 【1】 -> [1]
    return text.replace("【", "[").replace("】", "]")


def to_messages(history: list[ChatTurn]) -> list[BaseMessage]:
    return [
        HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content)
        for turn in history
    ]


class RAGService:
    def __init__(self, llm: BaseChatModel, store: Chroma, settings: Settings):
        self.store = store
        self.settings = settings
        self._condense = CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer = ANSWER_PROMPT | llm | StrOutputParser()

    def _prepare(
        self, question: str, history: list[ChatTurn], doc_ids: list[str] | None
    ) -> tuple[dict, list[Document], str]:
        messages = to_messages(history[-2 * self.settings.max_history_turns :])

        standalone = question
        if messages:
            standalone = self._condense.invoke({"history": messages, "question": question}).strip()
            standalone = standalone or question

        docs = retrieve(
            self.store,
            standalone,
            k=self.settings.retrieval_k,
            fetch_k=self.settings.retrieval_fetch_k,
            use_mmr=self.settings.use_mmr,
            doc_ids=doc_ids,
        )
        inputs = {"context": format_context(docs), "history": messages, "question": question}
        return inputs, docs, standalone

    def answer(
        self, question: str, history: list[ChatTurn], doc_ids: list[str] | None = None
    ) -> ChatResponse:
        inputs, docs, standalone = self._prepare(question, history, doc_ids)
        answer = normalize_citations(self._answer.invoke(inputs)) if docs else NO_CONTEXT_ANSWER
        return ChatResponse(answer=answer, sources=to_sources(docs), standalone_question=standalone)

    def stream(
        self, question: str, history: list[ChatTurn], doc_ids: list[str] | None = None
    ) -> Iterator[dict]:
        """Yield events: one `sources`, many `token`, then `done`."""
        inputs, docs, _ = self._prepare(question, history, doc_ids)
        yield {"type": "sources", "data": [s.model_dump() for s in to_sources(docs)]}

        if not docs:
            yield {"type": "token", "data": NO_CONTEXT_ANSWER}
        else:
            for token in self._answer.stream(inputs):
                yield {"type": "token", "data": normalize_citations(token)}

        yield {"type": "done"}
