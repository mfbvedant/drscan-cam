"""
DRScan Cam — FastAPI Application Entry Point
AI-Powered Medical Image Analysis Platform
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from backend.config import APP_NAME, APP_VERSION, APP_DESCRIPTION
from backend.routers import health, predict

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for hackathon (lock down in production)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(predict.router)

# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")

    # Check if assets directory exists before mounting
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
async def serve_frontend():
    """Serve the main dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": f"{APP_NAME} API is running. Visit /api/docs for API documentation."}


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    favicon_path = FRONTEND_DIR / "assets" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    # Return empty response if no favicon
    from fastapi.responses import Response
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print(f"[+] {APP_NAME} v{APP_VERSION} starting...")
    print(f"    Inference mode: {__import__('backend.config', fromlist=['INFERENCE_MODE']).INFERENCE_MODE}")
    print(f"    Frontend: {'Found' if FRONTEND_DIR.exists() else 'Not found'}")
    print(f"    API docs: /api/docs")
    print(f"    Ready to analyze medical images!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print(f"[-] {APP_NAME} shutting down...")
