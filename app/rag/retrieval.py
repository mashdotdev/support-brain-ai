from pydantic import BaseModel, Field
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.core.config import get_settings, Settings
from app.rag.ingest import get_vector_store

set_tracing_disabled(True)
settings: Settings = get_settings()

open_router_client = AsyncOpenAI(
    api_key=settings.open_router_api_key,
    base_url="https://openrouter.ai/api/v1",
)

open_router_model = OpenAIChatCompletionsModel(
    model="nvidia/nemotron-3-super-120b-a12b:free", openai_client=open_router_client
)


# ── Schemas ──────────────────────────────────────────────────────────────────


class GradeDocument(BaseModel):
    score: str = Field(description="'yes' if relevant to the question, 'no' if not")
    reason: str = Field(description="One sentence explanation")


# ── Agents ───────────────────────────────────────────────────────────────────

grader_agent = Agent(
    name="DocumentGrader",
    instructions=(
        "You are grading whether a document is relevant to a user's support question. "
        "Answer 'yes' if the document is relevant, 'no' if not. Give a one sentence reason."
    ),
    model=open_router_model,
    output_type=GradeDocument,
)

rag_agent = Agent(
    name="SupportRAGAgent",
    instructions=(
        "You are a helpful customer support assistant. Answer the question using ONLY "
        "the provided documentation. If the answer isn't in the docs, say so clearly. "
        "Always mention which page/section your answer comes from."
    ),
    model=open_router_model,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.4


async def grade_documents(question: str, docs: list) -> list:
    relevant = []
    for doc in docs:
        result = await Runner.run(
            starting_agent=grader_agent,
            input=f"Question: {question}\n\nDocument: {doc.page_content}",
        )
        grade: GradeDocument = result.final_output
        if grade.score == "yes":
            relevant.append(doc)
    return relevant


def format_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Main query function ──────────────────────────────────────────────────────


async def query(
    question: str,
    category: str | None = None,
    k: int = 5,
) -> dict:
    client = QdrantClient(url=settings.qdrant_url)
    store = get_vector_store(client)

    search_filter = None
    if category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )

    retriever = store.as_retriever(search_kwargs={"k": k, "filter": search_filter})
    docs = retriever.invoke(question)

    relevant_docs = await grade_documents(question, docs)
    confidence_ratio = len(relevant_docs) / len(docs) if docs else 0
    confident = confidence_ratio >= CONFIDENCE_THRESHOLD

    if not confident or not relevant_docs:
        return {
            "answer": "I don't have enough information to answer this confidently.",
            "citations": [],
            "confident": False,
        }

    context = format_docs(relevant_docs)
    result = await Runner.run(
        starting_agent=rag_agent,
        input=f"Documentation:\n{context}\n\nQuestion: {question}",
    )

    citations = list({doc.metadata.get("source", "") for doc in relevant_docs})

    return {
        "answer": result.final_output,
        "citations": citations,
        "confident": True,
    }
