"""
FastAPI Async Backend for EOU Detection UI
"""

import os
import asyncio
import logging
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from deepgram import DeepgramClient, PrerecordedOptions, LiveOptions, LiveTranscriptionEvents

from model_loader import engine

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("eou_app")

# Load environment variables
load_dotenv()

# Initialize Deepgram
try:
    deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
    logger.info("Deepgram Initialized")
except Exception as e:
    logger.error(f"Failed to initialize Deepgram: {e}")
    deepgram = None

# ============================================================
# Configuration
# ============================================================

# ⬇️ CHANGE THIS to your model directory
MODEL_DIR = os.environ.get(
    "EOU_MODEL_DIR",
    "./model"  # default: local folder with model files
)

# ============================================================
# App Lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    logger.info("🚀 Starting EOU Detection Server...")

    try:
        info = await engine.load_model(MODEL_DIR)
        logger.info(f"✅ Model loaded: {info}")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.warning("Server running without model. Load via /api/load endpoint.")

    yield

    logger.info("👋 Shutting down...")

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="EOU Detector",
    description="End-of-Utterance Detection with DeBERTa",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ============================================================
# Pydantic Models
# ============================================================

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Utterance text")

class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., max_length=50)

class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)

class LoadModelRequest(BaseModel):
    model_dir: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    text: str
    is_complete: bool
    confidence: float
    complete_probability: float
    incomplete_probability: float
    threshold: float
    inference_time_ms: float
    features: dict

# ============================================================
# Routes — Pages
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve main UI"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_loaded": engine.is_loaded,
    })

# ============================================================
# Routes — API
# ============================================================

@app.get("/api/status")
async def get_status():
    """Get model status"""
    return engine.get_status()


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict EOU for a single utterance"""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = await engine.predict(req.text)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """Transcribe audio with Deepgram and predict EOU"""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not deepgram:
        raise HTTPException(status_code=500, detail="Deepgram is not configured")

    try:
        # Read file into memory
        audio_data = await file.read()
        
        # Call Deepgram
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            punctuate=True,
        )
        
        payload = {"buffer": audio_data}
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        
        # Extract text from v3 response structure
        transcript = ""
        if response and response.results and response.results.channels:
            transcript = response.results.channels[0].alternatives[0].transcript
            
        if not transcript.strip():
            return JSONResponse(status_code=200, content={
                "text": "",
                "prediction": None,
                "message": "No speech detected"
            })
            
        # Run EOU Prediction
        result = await engine.predict(transcript)
        
        return {
            "text": transcript,
            "prediction": result
        }

    except Exception as e:
        logger.error(f"Audio prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    """Predict EOU for multiple utterances"""
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        texts = [item.text for item in req.items]
        results = await engine.predict_batch(texts)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/threshold")
async def update_threshold(req: ThresholdUpdate):
    """Update classification threshold"""
    result = await engine.update_threshold(req.threshold)
    return result


@app.post("/api/load")
async def load_model(req: LoadModelRequest):
    """Load/reload model from directory"""
    try:
        info = await engine.load_model(req.model_dir)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# WebSocket — Real-time Predictions
# ============================================================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        logger.info(f"WebSocket disconnected. Total: {len(self.active)}")

    async def broadcast(self, message: dict):
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                pass

ws_manager = ConnectionManager()


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Real-time WebSocket predictions (as you type)"""
    await ws_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "").strip()
            request_id = data.get("request_id", "")

            if not text:
                await websocket.send_json({
                    "request_id": request_id,
                    "error": "Empty text",
                })
                continue

            if not engine.is_loaded:
                await websocket.send_json({
                    "request_id": request_id,
                    "error": "Model not loaded",
                })
                continue

            try:
                result = await engine.predict(text)
                result["request_id"] = request_id
                await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({
                    "request_id": request_id,
                    "error": str(e),
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# ============================================================
# WebSocket — Deepgram Audio Streaming
# ============================================================

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """Real-time audio streaming to Deepgram + EOU Prediction"""
    await ws_manager.connect(websocket)

    if not engine.is_loaded:
        await websocket.send_json({"error": "Model not loaded"})
        ws_manager.disconnect(websocket)
        return

    if not deepgram:
        await websocket.send_json({"error": "Deepgram is not configured"})
        ws_manager.disconnect(websocket)
        return

    try:
        # Create a Deepgram Async Live Connection
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        async def on_message(self, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            is_final = result.is_final
            
            if transcript.strip() and is_final:
                # Run EOU Prediction for final result
                prediction = await engine.predict(transcript)
                await websocket.send_json({
                    "type": "final",
                    "text": transcript,
                    "prediction": prediction
                })
            elif transcript.strip() and not is_final:
                # Interim result
                await websocket.send_json({
                    "type": "interim",
                    "text": transcript
                })

        async def on_error(self, error, **kwargs):
            logger.error(f"Deepgram WS Error: {error}")
            await websocket.send_json({"error": str(error)})

        # Register handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            sample_rate=16000,
            interim_results=True,
            endpointing=300,
        )

        # Connect
        if await dg_connection.start(options) is False:
            logger.error("Failed to start Deepgram connection")
            await websocket.send_json({"error": "Failed to connect to ASR service"})
            return

        logger.info("Deepgram Live Connection Started")

        while True:
            data = await websocket.receive_bytes()
            await dg_connection.send(data)

    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
        if 'dg_connection' in locals():
            await dg_connection.finish()
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket Audio error: {e}")
        if 'dg_connection' in locals():
            await dg_connection.finish()
        ws_manager.disconnect(websocket)

# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded,
    }


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info",
    )