# SupportBrain — Build Specification

> **Tagline:** "Your docs, answered instantly. Escalates when it's not sure."
>
> **Stack:** Python 3.13 · uv · Gemini · Qdrant · FastAPI · Streamlit
>
> **Demo dataset:** Supabase public docs (https://supabase.com/docs) — free, no auth

---

## What You're Building

A customer support chatbot that ingests any product's documentation and answers
user questions in plain English. When confidence is low (CRAG grades docs as poor),
it escalates instead of hallucinating. Every answer cites the exact doc page it used.

### The 60-Second Demo Story
1. Paste a sitemap URL (e.g. Supabase docs)
2. Click "Ingest" — watch chunks appear
3. Ask "How do I set up row-level security in Supabase?"
4. Get a streamed answer with a citation link
5. Ask something obscure — watch it say "I'm not confident, here's the support link"

### Why Clients Pay
Every SaaS company has a support burden. If you can handle 80% of tier-1 tickets
automatically and escalate the rest, that's real money saved. Charge $500-2K/month.

---

## RAG Techniques Used

| Lesson | Technique | Where |
|--------|-----------|-------|
| L03 | Basic RAG chain | Core query pipeline |
| L04 | Hybrid search | Dense + BM25 for exact term matching |
| L05 | Metadata filtering | Filter by doc section/category |
| L07 | CRAG | Confidence grading + escalation |
| L09 | FastAPI | Production API |

No agents needed for v1. Keep it simple.

---

## Project Structure

```
supportbrain/
├── .env
├── .python-version         # 3.13
├── pyproject.toml
├── docker-compose.yml
│
├── src/
│   ├── __init__.py
│   ├── config.py           # pydantic-settings
│   ├── ingest.py           # Load docs → chunk → store in Qdrant
│   ├── retrieval.py        # Hybrid search + CRAG grading
│   └── api.py              # FastAPI app
│
└── demo/
    └── app.py              # Streamlit UI
```

Flat structure — no nested subfolders. This is a first project, keep it simple.

---

## Dependencies

### `.env`
```
GEMINI_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
```

### `pyproject.toml`
```toml
[project]
name = "supportbrain"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "langchain>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "langchain-qdrant>=0.2.0",
    "langchain-text-splitters>=0.3.0",
    "langchain-community>=0.3.0",
    "qdrant-client>=1.12.0",
    "fastembed>=0.4.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "pydantic-settings>=2.0.0",
    "streamlit>=1.40.0",
    "python-dotenv>=1.0.0",
    "beautifulsoup4>=4.12.0",
    "httpx>=0.27.0",
]
```

### `docker-compose.yml`
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

---

## `src/config.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    gemini_api_key: str = Field(default="")
    qdrant_url: str     = Field(default="http://localhost:6333")
    collection: str     = Field(default="supportbrain_docs")
    embed_dim: int      = Field(default=768)

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## `src/ingest.py`

### What it does
Loads documentation pages from a sitemap or list of URLs, splits them into chunks,
and stores them in Qdrant with metadata.

### Key functions

```python
from langchain_community.document_loaders import SitemapLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, PayloadSchemaType
)
import hashlib

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient) -> None:
    """Create the hybrid collection + payload indexes. Safe to call multiple times."""
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection in existing:
        return

    client.create_collection(
        settings.collection,
        vectors_config={
            "dense": VectorParams(size=settings.embed_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    # Index these fields so filtering is fast (from Lesson 05)
    client.create_payload_index(settings.collection, "source",   PayloadSchemaType.KEYWORD)
    client.create_payload_index(settings.collection, "category", PayloadSchemaType.KEYWORD)


def get_vector_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.gemini_api_key,
        ),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


def compute_chunk_id(url: str, chunk_index: int) -> str:
    """Deterministic ID — re-ingesting same URL never creates duplicates."""
    return hashlib.sha256(f"{url}:{chunk_index}".encode()).hexdigest()[:16]


def load_from_sitemap(sitemap_url: str) -> list:
    """Load all pages listed in a sitemap XML. Returns LangChain Documents."""
    loader = SitemapLoader(web_path=sitemap_url)
    loader.requests_per_second = 2      # be polite
    return loader.load()


def load_from_urls(urls: list[str]) -> list:
    """Load specific URLs. Use this if there's no sitemap."""
    loader = WebBaseLoader(urls)
    return loader.load()


def ingest(sitemap_url: str = None, urls: list[str] = None) -> int:
    """
    Main ingestion entry point.
    Pass either a sitemap_url OR a list of urls.
    Returns the number of chunks stored.
    """
    # 1. Load raw documents
    if sitemap_url:
        docs = load_from_sitemap(sitemap_url)
    elif urls:
        docs = load_from_urls(urls)
    else:
        raise ValueError("Provide sitemap_url or urls")

    # 2. Attach a 'category' from the URL path
    #    e.g. https://supabase.com/docs/guides/auth → category = "guides"
    for doc in docs:
        url = doc.metadata.get("source", "")
        parts = url.rstrip("/").split("/")
        doc.metadata["category"] = parts[-2] if len(parts) >= 2 else "general"

    # 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)

    # 4. Add deterministic chunk IDs to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]    = compute_chunk_id(chunk.metadata.get("source", ""), i)
        chunk.metadata["chunk_index"] = i

    # 5. Upsert into Qdrant
    client = get_client()
    ensure_collection(client)
    store  = get_vector_store(client)
    store.add_documents(chunks)

    return len(chunks)
```

### Testing Day 1
```python
# Run this directly to test ingestion
if __name__ == "__main__":
    count = ingest(sitemap_url="https://supabase.com/docs/sitemap.xml")
    print(f"Ingested {count} chunks")
    # Then open http://localhost:6333/dashboard to see them
```

---

## `src/retrieval.py`

### What it does
Retrieves relevant chunks for a question and uses CRAG to grade confidence.
Returns the answer, citations, and a confidence flag.

### Key functions

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=settings.gemini_api_key,
)


# ── CRAG: Grade document relevance ──────────────────────────────────────────

class GradeDocument(BaseModel):
    score:  str = Field(description="'yes' if relevant to the question, 'no' if not")
    reason: str = Field(description="One sentence explanation")


grader_llm = llm.with_structured_output(GradeDocument)

GRADE_PROMPT = ChatPromptTemplate.from_template("""
You are grading whether a document is relevant to a user's support question.

Question: {question}
Document: {document}

Is this document relevant? Answer 'yes' or 'no'.
""")

def grade_documents(question: str, docs: list) -> list:
    """Return only the docs that are relevant to the question."""
    relevant = []
    for doc in docs:
        result = (GRADE_PROMPT | grader_llm).invoke({
            "question": question,
            "document": doc.page_content,
        })
        if result.score == "yes":
            relevant.append(doc)
    return relevant


# ── RAG: Generate answer from docs ──────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant. Answer the question using ONLY
the provided documentation. If the answer isn't in the docs, say so clearly.
Always mention which page/section your answer comes from.

Documentation:
{context}

Question: {question}

Answer:""")

def format_docs(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Main query function ──────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.4   # if fewer than 40% of docs are relevant → escalate

def query(
    question: str,
    category: str = None,   # optional: restrict to a doc category
    k: int = 5,
) -> dict:
    """
    Full CRAG query pipeline.
    Returns:
        {
            "answer":    str,
            "citations": list[str],   # source URLs used
            "confident": bool,        # False = escalate to human
        }
    """
    client = get_client()
    store  = get_vector_store(client)

    # Build optional filter (from Lesson 05)
    search_filter = None
    if category:
        search_filter = Filter(must=[
            FieldCondition(key="category", match=MatchValue(value=category))
        ])

    # 1. Retrieve
    retriever = store.as_retriever(
        search_kwargs={"k": k, "filter": search_filter}
    )
    docs = retriever.invoke(question)

    # 2. CRAG: grade relevance
    relevant_docs = grade_documents(question, docs)
    confidence_ratio = len(relevant_docs) / len(docs) if docs else 0
    confident = confidence_ratio >= CONFIDENCE_THRESHOLD

    # 3. If too few relevant docs → return low-confidence signal
    if not confident or not relevant_docs:
        return {
            "answer":    "I don't have enough information to answer this confidently.",
            "citations": [],
            "confident": False,
        }

    # 4. Generate answer from relevant docs only
    context = format_docs(relevant_docs)
    answer  = (RAG_PROMPT | llm | StrOutputParser()).invoke({
        "context":  context,
        "question": question,
    })

    citations = list({doc.metadata.get("source", "") for doc in relevant_docs})

    return {
        "answer":    answer,
        "citations": citations,
        "confident": True,
    }
```

### Testing Day 2
```python
if __name__ == "__main__":
    result = query("How do I enable row-level security in Supabase?")
    print("Answer:", result["answer"])
    print("Confident:", result["confident"])
    print("Citations:", result["citations"])
```

---

## `src/api.py`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Status + chunk count |
| `POST` | `/ingest` | Ingest a sitemap or URL list |
| `POST` | `/query`  | Ask a question, get answer + citations |

### Full implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.ingest import ingest, get_client, ensure_collection
from src.retrieval import query as rag_query


# ── Schemas ──────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    sitemap_url: str | None = None
    urls:        list[str] | None = None

class IngestResponse(BaseModel):
    chunks_added: int

class QueryRequest(BaseModel):
    question: str
    category: str | None = None

class QueryResponse(BaseModel):
    answer:    str
    citations: list[str]
    confident: bool


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = get_client()
    ensure_collection(client)
    yield

app = FastAPI(title="SupportBrain", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    client = get_client()
    try:
        info = client.get_collection("supportbrain_docs")
        return {"status": "ok", "chunks": info.points_count}
    except Exception:
        return {"status": "ok", "chunks": 0}

@app.post("/ingest", response_model=IngestResponse)
def ingest_docs(req: IngestRequest):
    if not req.sitemap_url and not req.urls:
        raise HTTPException(400, "Provide sitemap_url or urls")
    count = ingest(sitemap_url=req.sitemap_url, urls=req.urls)
    return IngestResponse(chunks_added=count)

@app.post("/query", response_model=QueryResponse)
def ask(req: QueryRequest):
    result = rag_query(question=req.question, category=req.category)
    return QueryResponse(**result)
```

### Testing Day 4
```bash
# Start API
uv run uvicorn src.api:app --reload

# Health check
curl http://localhost:8000/health

# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"sitemap_url": "https://supabase.com/docs/sitemap.xml"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I set up authentication in Supabase?"}'
```

---

## `demo/app.py` — Streamlit UI

### What it looks like
- **Sidebar:** Ingest panel (sitemap URL input + button)
- **Main area:** Chat interface with streaming responses
- **Each answer:** Shows a "Sources" expander with clickable links
- **Low confidence:** Shows a yellow warning + "Contact support" link instead of answer

### Implementation

```python
import streamlit as st
import httpx

API_URL = "http://localhost:8000"

EXAMPLE_QUESTIONS = [
    "How do I set up row-level security?",
    "How do I create a new table in Supabase?",
    "How does Supabase authentication work?",
    "How do I connect Supabase to a Next.js app?",
    "What are Supabase Edge Functions?",
]

st.set_page_config(page_title="SupportBrain", page_icon="🧠", layout="wide")
st.title("🧠 SupportBrain")
st.caption("Docs-powered support — answers from your documentation, not guesswork.")


# ── Sidebar: Ingestion ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("Ingest Docs")
    sitemap_url = st.text_input(
        "Sitemap URL",
        value="https://supabase.com/docs/sitemap.xml",
        placeholder="https://yourproduct.com/sitemap.xml",
    )
    if st.button("Ingest", type="primary"):
        with st.spinner("Fetching and indexing docs..."):
            resp = httpx.post(
                f"{API_URL}/ingest",
                json={"sitemap_url": sitemap_url},
                timeout=300,    # ingestion takes time
            )
        if resp.status_code == 200:
            count = resp.json()["chunks_added"]
            st.success(f"Ingested {count} chunks")
        else:
            st.error(f"Error: {resp.text}")

    st.divider()

    # Health check
    try:
        health = httpx.get(f"{API_URL}/health", timeout=3).json()
        st.metric("Chunks indexed", health.get("chunks", 0))
    except Exception:
        st.warning("API not reachable — start the server first")

    st.divider()
    st.subheader("Example questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q


# ── Main area: Chat ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for url in msg["citations"]:
                    st.markdown(f"- [{url}]({url})")

# Prefill from example button
prefill = st.session_state.pop("prefill", None)

# Input
question = st.chat_input("Ask anything about the docs...") or prefill

if question:
    # Show user message
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            resp = httpx.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=60,
            )

        if resp.status_code != 200:
            st.error("API error — check the server logs")
        else:
            data = resp.json()

            if not data["confident"]:
                # Low confidence → escalation message
                st.warning(
                    "I couldn't find a confident answer in the docs. "
                    "Please [contact support](mailto:support@example.com) for help."
                )
                answer = "I don't have enough information to answer this confidently."
                citations = []
            else:
                st.markdown(data["answer"])
                if data["citations"]:
                    with st.expander("Sources"):
                        for url in data["citations"]:
                            st.markdown(f"- [{url}]({url})")
                answer    = data["answer"]
                citations = data["citations"]

            st.session_state.messages.append({
                "role":      "assistant",
                "content":   answer,
                "citations": citations,
            })
```

### Testing Day 5
```bash
uv run streamlit run demo/app.py
# Open http://localhost:8501
# Ingest Supabase docs → ask a question → verify cited answer
```

---

## Build Order

**Day 1 — Ingestion**
1. `docker compose up -d` — start Qdrant
2. Write `src/config.py`
3. Write `src/ingest.py`
4. Test: run `python src/ingest.py` — should print chunk count
5. Open `http://localhost:6333/dashboard` — verify chunks with metadata

**Day 2 — Retrieval**
1. Write `src/retrieval.py`
2. Test: run `python src/retrieval.py` — verify answer + citations print
3. Try a question you know is NOT in the docs — verify `confident: False`

**Day 3 — Polish retrieval**
1. Tune `CONFIDENCE_THRESHOLD` — try 0.3 and 0.5, see which feels right
2. Tune `CHUNK_SIZE` — if answers feel cut off, increase to 800
3. Try filtering by category — ask "auth questions only"

**Day 4 — FastAPI**
1. Write `src/api.py`
2. Run server + test all 3 endpoints with curl
3. Fix any import or startup errors

**Day 5 — Streamlit UI**
1. Write `demo/app.py`
2. Run the full demo: ingest → chat → see citations
3. Click an example question button — verify prefill works

---

## Verification Checklist

```
[ ] docker compose up -d                     # Qdrant at localhost:6333
[ ] uv run python src/ingest.py              # Prints chunk count
[ ] localhost:6333/dashboard shows chunks    # With source/category metadata
[ ] uv run python src/retrieval.py           # Prints answer + confident: True
[ ] Ask unknown question → confident: False  # CRAG escalation works
[ ] uv run uvicorn src.api:app --reload      # API starts without errors
[ ] curl localhost:8000/health               # {"status":"ok","chunks":N}
[ ] POST /ingest works                       # Returns chunks_added
[ ] POST /query works                        # Returns answer + citations
[ ] uv run streamlit run demo/app.py         # UI loads
[ ] Full demo flow works end-to-end          # Ingest → question → cited answer
```

---

## Common Issues and Fixes

**SitemapLoader is slow**
Normal. Supabase docs have hundreds of pages. Let it run. Add a print statement
inside `load_from_sitemap` to see progress.

**`confident: False` for everything**
Your chunks might be too small. Increase `CHUNK_SIZE` to 800 or 1000.
Or your grader prompt is too strict — adjust the GRADE_PROMPT wording.

**Embedding API rate limit errors**
Add `time.sleep(1)` between batches in `store.add_documents()`.
Or split chunks into batches of 50 and embed one batch at a time.

**Qdrant filter not working**
Make sure you called `create_payload_index` for the field you're filtering on.
Without an index, Qdrant still works but ignores filters silently in some versions.

**Streamlit chat resets on each interaction**
Make sure ALL state goes through `st.session_state`. Never use local variables
for anything that needs to persist across interactions.

---

## What to Show on LinkedIn

**Demo video script (60 seconds):**
1. Open Streamlit — show empty chat (0 chunks indexed)
2. Paste `https://supabase.com/docs/sitemap.xml` → click Ingest
3. Watch chunk counter go up
4. Ask: "How do I enable row-level security?"
5. Show streamed answer + click the source link → it opens the exact Supabase docs page
6. Ask something obscure → show the yellow escalation warning

**Caption:**
> Built a customer support chatbot that reads your docs and knows when it doesn't know.
> Every answer cites the exact page. Low-confidence answers escalate instead of hallucinating.
> Stack: Gemini + Qdrant hybrid search + CRAG self-correction + FastAPI + Streamlit.
