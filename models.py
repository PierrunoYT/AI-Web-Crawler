from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PageSummary(BaseModel):
    title: str = Field(..., description="Title of the page.")
    summary: str = Field(..., description="Summary of the page.")
    brief_summary: str = Field(..., description="Brief summary of the page.")
    keywords: list = Field(..., description="Keywords assigned to the page.")

class CrawlResult(BaseModel):
    url: str
    html: str
    success: bool
    cleaned_html: Optional[str] = None
    media: Dict[str, List[Dict]] = {}
    links: Dict[str, List[Dict]] = {}
    screenshot: Optional[str] = None
    markdown: Optional[str] = None
    extracted_content: Optional[str] = None
    metadata: Optional[dict] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    responser_headers: Optional[dict] = None
    status_code: Optional[int] = None
