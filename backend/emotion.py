import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np


MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "emotion-ferplus-8.onnx"

MODEL_EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def ensure_model_exists() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        urlretrieve(MODEL_URL, MODEL_PATH)


def _softmax(logits: np.ndarray) -> Optional[np.ndarray]:
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    denom = np.sum(exps)
    if denom == 0 or np.isnan(denom):
        return None
    return exps / denom


def _predict_emotion(face_gray: np.ndarray, emotion_net: cv2.dnn.Net) -> Tuple[str, float, Optional[np.ndarray]]:
    face_preprocessed = cv2.equalizeHist(face_gray)
    face_resized = cv2.resize(face_preprocessed, (64, 64))
    blob = cv2.dnn.blobFromImage(
        image=face_resized,
        scalefactor=1.0,
        size=(64, 64),
        mean=(0,),
        swapRB=False,
        crop=False,
        ddepth=cv2.CV_32F,
    )
    emotion_net.setInput(blob)
    output = emotion_net.forward().flatten()
    probabilities = _softmax(output)
    if probabilities is None or np.any(np.isnan(probabilities)):
        return "neutral", 0.0, None
    best_idx = int(np.argmax(probabilities))
    emotion_key = MODEL_EMOTIONS[best_idx]
    confidence = float(probabilities[best_idx])
    return emotion_key, confidence, probabilities


def _select_expression(probabilities: Optional[np.ndarray]) -> Tuple[str, float]:
    if probabilities is None:
        return "neutral", 0.0

    sorted_indices = np.argsort(probabilities)[::-1]
    best_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])
    best_label = MODEL_EMOTIONS[best_idx]
    best_conf = float(probabilities[best_idx])
    second_label = MODEL_EMOTIONS[second_idx]
    second_conf = float(probabilities[second_idx])

    if best_label == "neutral":
        small_gap = (best_conf - second_conf) < 0.10
        expressive_signal = second_conf >= 0.22
        if small_gap and expressive_signal:
            return second_label, second_conf

    return best_label, best_conf


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes")
    return frame


def decode_data_url(data_url: str) -> bytes:
    # supports: data:image/jpeg;base64,...
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    header, b64 = data_url.split(",", 1)
    if "base64" not in header:
        raise ValueError("Data URL is not base64-encoded")
    return base64.b64decode(b64)


@dataclass(frozen=True)
class EmotionResult:
    emotion: str
    confidence: float
    face_box: Optional[Tuple[int, int, int, int]]  # x,y,w,h
    probabilities: Optional[Dict[str, float]]


class EmotionDetector:
    def __init__(self) -> None:
        ensure_model_exists()
        self._emotion_net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._face_cascade.empty():
            raise RuntimeError("OpenCV face cascade could not be loaded")

    def analyze_bgr(self, frame_bgr: np.ndarray) -> EmotionResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return EmotionResult(emotion="no_face", confidence=0.0, face_box=None, probabilities=None)

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        face_gray = gray[y : y + h, x : x + w]
        _, _, probs = _predict_emotion(face_gray, self._emotion_net)
        emotion, conf = _select_expression(probs)
        probs_dict = None
        if probs is not None:
            probs_dict = {MODEL_EMOTIONS[i]: float(probs[i]) for i in range(len(MODEL_EMOTIONS))}
        return EmotionResult(emotion=emotion, confidence=conf, face_box=(int(x), int(y), int(w), int(h)), probabilities=probs_dict)

