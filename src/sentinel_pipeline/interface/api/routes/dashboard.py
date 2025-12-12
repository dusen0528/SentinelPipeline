from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter()


@router.get("/admin/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard(_: Request) -> HTMLResponse:
    base_dir = Path(__file__).resolve().parent.parent / "static" / "dashboard"
    html_path = base_dir / "index.html"
    return FileResponse(html_path)

