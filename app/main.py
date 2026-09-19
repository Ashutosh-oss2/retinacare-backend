import os
os.environ.setdefault("KERAS_BACKEND", "torch")
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.utils.logging import logger
from app.services.model_registry import registry
from app.api import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Pre-loads all 3 AI models into memory once at startup."""
    logger.info("RetinaCare backend startup: Initializing AI models...")
    weights_dir = os.getenv("MODEL_WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "..", "weights"))
    try:
        registry.initialize(weights_dir=weights_dir)
        logger.info("AI models initialized successfully.")
    except Exception as e:
        logger.critical(f"Fatal error during model initialization: {e}", exc_info=True)
    yield
    logger.info("RetinaCare backend shutdown: Releasing resources...")


app = FastAPI(
    title="RetinaCare Triple AI Retinal Diagnostic Suite Backend",
    description=(
        "Production-grade, modular FastAPI backend providing AI-assisted Diabetic Retinopathy screening, "
        "Retinal Vessel segmentation, and Multi-Lesion segmentation."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================
# CORS CONFIGURATION (Environment-Driven, No Unrestricted *)
# ============================================================

DEFAULT_ORIGINS = [
    "https://retina-care-frontend-theta.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

def get_allowed_origins() -> list[str]:
    """Parse comma-separated origins from environment with whitespace/quote/slash sanitization."""
    origins = list(DEFAULT_ORIGINS)
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if raw.strip():
        for item in raw.split(","):
            cleaned = item.strip().strip("'\"").rstrip("/")
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
    return origins

ALLOWED_ORIGINS = get_allowed_origins()
raw_regex = os.getenv("ALLOW_ORIGIN_REGEX", r"https://.*\.vercel\.app").strip().strip("'\"")
ALLOW_ORIGIN_REGEX = raw_regex if raw_regex else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ============================================================
# EXCEPTION HANDLERS
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "validation_error", "detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "message": "An unexpected error occurred during processing."}
    )

# ============================================================
# ROUTER REGISTRATION
# ============================================================

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
