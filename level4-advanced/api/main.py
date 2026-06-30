"""
FastAPI web service for Profanity Filter.

Auto-generated interactive documentation at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
import time
from contextlib import asynccontextmanager

from profanity_filter import ProfanityDetector

# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class PredictRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(..., description="Text to check for toxicity", min_length=1, max_length=10000)
    model: Optional[Literal["traditional-ml", "modernbert-binary", "modernbert-multiclass", "toxic-bert"]] = Field(
        None,
        description="Specific model to use (optional, defaults to auto/hybrid)"
    )
    mode: Literal["single", "hybrid", "auto"] = Field(
        "auto",
        description="Detection mode (auto uses hybrid)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a test message",
                "model": "modernbert-multiclass",
                "mode": "auto"
            }
        }


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction."""
    texts: List[str] = Field(..., description="List of texts to check", min_length=1, max_length=100)
    model: Optional[str] = Field(None, description="Specific model to use")
    mode: Literal["single", "hybrid", "auto"] = Field("auto", description="Detection mode")

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 texts per batch")
        for text in v:
            if not text or len(text) > 10000:
                raise ValueError("Each text must be 1-10000 characters")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "Nice game!", "Good morning"],
                "mode": "hybrid"
            }
        }


class ToxicityResponse(BaseModel):
    """Response model for single prediction."""
    text: str = Field(..., description="The input text that was analyzed")
    is_toxic: bool = Field(..., description="Whether the text is toxic")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    toxicity_type: Optional[str] = Field(None, description="Type of toxicity (for multi-class models)")
    latency_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="Model used for prediction")
    two_stage: bool = Field(False, description="Whether hybrid two-stage filtering was used")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a test message",
                "is_toxic": False,
                "confidence": 0.95,
                "toxicity_type": "Clean",
                "latency_ms": 15.5,
                "model_name": "modernbert-multiclass",
                "two_stage": True
            }
        }


class BatchToxicityResponse(BaseModel):
    """Response model for batch prediction."""
    results: List[ToxicityResponse]
    total_count: int
    toxic_count: int
    clean_count: int
    average_latency_ms: float
    total_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: List[str]
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    description: str
    f1_gametox: float
    latency_ms: str
    best_for: str


# ============================================================================
# Application Lifecycle
# ============================================================================

# Global detector instance (loaded on startup)
detector: Optional[ProfanityDetector] = None
app_start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global detector, app_start_time

    # Startup
    print("=" * 60)
    print("Starting Profanity Filter API...")
    print("=" * 60)
    app_start_time = time.time()

    # Initialize detector (hybrid mode by default)
    print("\nInitializing detector in hybrid mode...")
    detector = ProfanityDetector(mode="auto")

    # Pre-load models by making a test prediction
    print("Pre-loading models...")
    test_result = detector.predict("test")
    print(f"✓ Models loaded and ready!")
    print(f"✓ Test prediction: {test_result.model_name}")

    print("\n" + "=" * 60)
    print("API ready! Visit http://localhost:8000/docs")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\nShutting down Profanity Filter API...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Profanity Filter API",
    description="""
    **Production-ready profanity filter for gaming chat.**

    Combines three state-of-the-art approaches:
    - Traditional ML: Ultra-fast screening (0.008ms)
    - ModernBERT Multi-Class: Best in-domain accuracy (F1=0.85)
    - Toxic-BERT: Best cross-domain generalization (F1=0.67)

    ## Features
    - ✅ Hybrid two-stage filtering for optimal speed/accuracy
    - ✅ Multi-class detection (Profanity, Insult, Hate Speech)
    - ✅ Batch processing support
    - ✅ Auto-generated interactive documentation
    - ✅ Production-ready with health checks

    ## Quick Start
    1. Visit `/docs` (this page) for interactive testing
    2. Click "Try it out" on any endpoint
    3. Enter your text and click "Execute"
    4. See results immediately!

    ## Models
    - `traditional-ml`: F1=0.68, 0.008ms latency
    - `modernbert-binary`: F1=0.78, 14ms latency
    - `modernbert-multiclass`: F1=0.85, 16ms latency ⭐
    - `toxic-bert`: F1=0.67 (external), 10ms latency
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Profanity Filter API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time

    # Check which models are loaded
    models_loaded = []
    if detector and detector._models:
        models_loaded = list(detector._models.keys())

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=models_loaded,
        uptime_seconds=uptime
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Info"])
async def list_models():
    """List available models and their characteristics."""
    models = [
        ModelInfo(
            name="traditional-ml",
            description="Traditional ML (Level 3) - Fastest",
            f1_gametox=0.68,
            latency_ms="0.008",
            best_for="High-throughput screening"
        ),
        ModelInfo(
            name="modernbert-binary",
            description="ModernBERT Binary (Level 4)",
            f1_gametox=0.78,
            latency_ms="14",
            best_for="Binary classification"
        ),
        ModelInfo(
            name="modernbert-multiclass",
            description="ModernBERT Multi-Class (Level 4) - Best in-domain",
            f1_gametox=0.85,
            latency_ms="16",
            best_for="Gaming chat with toxicity types"
        ),
        ModelInfo(
            name="toxic-bert",
            description="Toxic-BERT (pre-trained) - Best generalization",
            f1_gametox=0.63,
            latency_ms="10",
            best_for="General-purpose toxicity detection"
        ),
    ]
    return models


@app.post("/predict", response_model=ToxicityResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Check a single message for toxicity.

    **Modes:**
    - `auto` (default): Uses hybrid two-stage filtering
    - `hybrid`: Explicitly use two-stage filtering
    - `single`: Use single model only

    **Models:**
    - Leave empty for auto-selection (hybrid mode)
    - Specify model for single-model mode

    **Returns:**
    - Toxicity classification with confidence score
    - For multi-class models: specific toxicity type
    - Latency information
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create detector with requested configuration
        det = ProfanityDetector(
            model=request.model,
            mode=request.mode
        )

        # Make prediction
        result = det.predict(request.text)

        # Convert to response model
        return ToxicityResponse(
            text=result.text,
            is_toxic=result.is_toxic,
            confidence=result.confidence,
            toxicity_type=result.toxicity_type,
            latency_ms=result.latency_ms,
            model_name=result.model_name,
            two_stage=result.two_stage
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch", response_model=BatchToxicityResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Check multiple messages for toxicity (batch processing).

    **Limits:**
    - Maximum 100 texts per batch
    - Each text: 1-10,000 characters

    **Returns:**
    - Individual results for each text
    - Summary statistics (toxic count, average latency)
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Create detector with requested configuration
        det = ProfanityDetector(
            model=request.model,
            mode=request.mode
        )

        # Make batch prediction
        results = det.predict_batch(request.texts)

        total_time_ms = (time.time() - start_time) * 1000

        # Convert to response models
        response_results = [
            ToxicityResponse(
                text=r.text,
                is_toxic=r.is_toxic,
                confidence=r.confidence,
                toxicity_type=r.toxicity_type,
                latency_ms=r.latency_ms,
                model_name=r.model_name,
                two_stage=r.two_stage
            )
            for r in results
        ]

        # Calculate statistics
        toxic_count = sum(1 for r in results if r.is_toxic)
        clean_count = len(results) - toxic_count
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        return BatchToxicityResponse(
            results=response_results,
            total_count=len(results),
            toxic_count=toxic_count,
            clean_count=clean_count,
            average_latency_ms=avg_latency,
            total_time_ms=total_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return {
        "error": "Not found",
        "message": "Endpoint not found. Visit /docs for API documentation.",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
