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
