from typing import Optional
from pydantic import BaseModel


class IngestSchema(BaseModel):
    sitemap_url: Optional[str] = None
    urls: Optional[list[str]] = None
