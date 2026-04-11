"""
DRScan AI — FastAPI Application Entry Point
AI-Powered Medical Image Analysis & Diabetic Retinopathy Screening Platform
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
RETINAL_ASSETS = Path(__file__).parent.parent / "hack bluebit" / "frontend" / "assets"

if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")

    # Check if assets directory exists before mounting
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Mount retinal sample images from hack bluebit if available
if RETINAL_ASSETS.exists():
    app.mount("/retinal-assets", StaticFiles(directory=RETINAL_ASSETS), name="retinal-assets")


@app.get("/")
async def serve_landing():
    """Serve the landing page."""
    landing_path = FRONTEND_DIR / "landing.html"
    if landing_path.exists():
        return FileResponse(landing_path)
    # Fallback to scan dashboard if no landing page
    return await serve_scanner()


@app.get("/scan")
async def serve_scanner():
    """Serve the scan/analysis dashboard."""
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
    print(f"    Frontend: {'Found' if FRONTEND_DIR.exists() else 'Not found'}")
    print(f"    Landing page: {'Found' if (FRONTEND_DIR / 'landing.html').exists() else 'Not found'}")
    print(f"    Retinal assets: {'Found' if RETINAL_ASSETS.exists() else 'Not found'}")
    print(f"    API docs: /api/docs")

    # Initialize inference engine (auto-loads BiomedCLIP + retinal model if available)
    from backend.services.inference import engine
    engine.initialize()

    print(f"    Inference mode: {engine.mode}")
    print(f"    Ready to analyze medical images!")
    print(f"    Routes: / (landing) | /scan (dashboard) | /api/predict | /api/health")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print(f"[-] {APP_NAME} shutting down...")
