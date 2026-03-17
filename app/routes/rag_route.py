from app.rag.ingest import ingest
from app.schema.rag_schema import IngestSchema
from app.models.user import User
from app.core.dependency import get_current_user

from fastapi import APIRouter, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address


limiter = Limiter(get_remote_address)
rag_router = APIRouter(prefix="/rag", tags=["rag"])


@rag_router.post(path="/ingest", description="Ingest a document url for the vector db")
@limiter.limit("5/minute")
def ingest_document(
    request: Request,
    ingest_data: IngestSchema,
    current_user: User = Depends(get_current_user),
):
    return ingest(sitemap_url=ingest_data.sitemap_url, urls=ingest_data.urls)
