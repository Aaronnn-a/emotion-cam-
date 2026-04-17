from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from emotion import EmotionDetector, decode_data_url, decode_image_bytes


class AnalyzeDataUrlRequest(BaseModel):
    image_data_url: str


def create_app() -> FastAPI:
    app = FastAPI(title="Real-Time Emotion Detection API", version="0.1.0")

    allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").strip()
    allow_origins = [o.strip() for o in allowed.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    detector: Optional[EmotionDetector] = None

    @app.on_event("startup")
    def _startup() -> None:
        nonlocal detector
        detector = EmotionDetector()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/api/analyze")
    async def analyze_image(file: UploadFile = File(...)) -> dict:
        if detector is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail=f"Unsupported content-type: {file.content_type}")

        image_bytes = await file.read()
        try:
            frame = decode_image_bytes(image_bytes)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e))

        result = detector.analyze_bgr(frame)
        return {
            "emotion": result.emotion,
            "confidence": result.confidence,
            "face_box": result.face_box,
            "probabilities": result.probabilities,
        }

    @app.post("/api/analyze-data-url")
    async def analyze_data_url(payload: AnalyzeDataUrlRequest) -> dict:
        if detector is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            image_bytes = decode_data_url(payload.image_data_url)
            frame = decode_image_bytes(image_bytes)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(e))

        result = detector.analyze_bgr(frame)
        return {
            "emotion": result.emotion,
            "confidence": result.confidence,
            "face_box": result.face_box,
            "probabilities": result.probabilities,
        }

    return app


app = create_app()

