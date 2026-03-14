import hashlib
from app.core.config import get_settings, Settings

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.document_loaders import SitemapLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Distance,
    PayloadSchemaType,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

settings: Settings = get_settings()


def create_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name in existing:
        return

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config={
            "dense": VectorParams(
                size=settings.embedding_dimension, distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
        },
    )

    client.create_payload_index(
        collection_name=settings.qdrant_collection_name,
        field_name="source",
        field_type=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_collection_name,
        field_name="category",
        field_type=PayloadSchemaType.KEYWORD,
    )


def get_vector_store(client: QdrantClient) -> QdrantVectorStore:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
        embedding=GoogleGenerativeAIEmbeddings(
            api_key=settings.gemini_api_key, model=settings.embedding_model
        ),
        vector_name="dense",
        sparse_vector_name="sparse",
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
    )

    return vector_store


def create_chunk_id(url: str, chunk_index: int) -> str:
    # Create a unique identifier for the chunk using a hash of the URL and chunk index
    unique_string = f"{url}_{chunk_index}"
    return hashlib.sha256(unique_string.encode()).hexdigest()


def load_from_sitemap(sitemap_url: str) -> list[Document]:
    loader = SitemapLoader(web_path=sitemap_url)
    loader.requests_per_second = 2
    return loader.load()


def load_from_urls(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(web_paths=urls)
    return loader.load()


def inngest(sitemap_url: str | None, urls: list[str] | None):
    if sitemap_url:
        docs = load_from_sitemap(sitemap_url)
    elif urls:
        docs = load_from_urls(urls)
    else:
        raise ValueError("Sitemap url or URL not provided")

    for doc in docs:
        url = doc.metadata.get("source", "")
        parts = url.rstrip("/").split("/")
        doc.metadata["category"] = parts[-2] if len(parts) >= 2 else "general"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
        len_func=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = create_chunk_id(
            url=chunk.metadata.get("source", ""), chunk_index=i
        )
        chunk.metadata["chunk_index"] = i

    client = QdrantClient(url=settings.qdrant_url)
    create_collection(client)
    store = get_vector_store(client)
    store.add_documents(chunks)

    return len(chunks)
