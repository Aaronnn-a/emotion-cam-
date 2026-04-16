import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve


WINDOW_NAME = "Gesichtsemotionen-Erkennung"
FACE_COLOR = (0, 255, 0)
EMOTION_COLORS = {
    "neutral": (0, 255, 0),      # gruen
    "happiness": (0, 255, 255),  # gelb
    "surprise": (0, 165, 255),   # orange
    "sadness": (255, 0, 0),      # blau
    "anger": (0, 0, 255),        # rot
    "disgust": (128, 0, 128),    # lila
    "fear": (255, 0, 255),       # magenta
    "contempt": (200, 200, 200), # grau
}
PERFORMANCE_MODES = [
    {"name": "Qualitaet", "infer_every_n_frames": 1},
    {"name": "Ausbalanciert", "infer_every_n_frames": 2},
    {"name": "Performance", "infer_every_n_frames": 3},
]
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
MODEL_PATH = Path("models") / "emotion-ferplus-8.onnx"
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
EMOTION_LABELS_DE = {
    "neutral": "Neutral",
    "happiness": "Gluecklich",
    "surprise": "Ueberrascht",
    "sadness": "Traurig",
    "anger": "Wuetend",
    "disgust": "Ekel",
    "fear": "Angst",
    "contempt": "Verachtung",
}


def draw_label(frame, text, x, y, color=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    label_color = FACE_COLOR if color is None else color
    cv2.putText(frame, text, (x, y), font, scale, label_color, thickness, cv2.LINE_AA)


def get_emotion_color(emotion_key):
    return EMOTION_COLORS.get(emotion_key, FACE_COLOR)


def ensure_model_exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("Lade Emotion-Modell herunter (einmalig, ca. 35 MB)...")
        urlretrieve(MODEL_URL, MODEL_PATH)


def softmax(logits):
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    denominator = np.sum(exps)
    if denominator == 0 or np.isnan(denominator):
        return None
    return exps / denominator


def predict_emotion(face_gray, emotion_net):
    # FER+ works more robustly with histogram-equalized grayscale
    # and raw pixel range (0..255) instead of manual normalization.
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
    probabilities = softmax(output)
    if probabilities is None or np.any(np.isnan(probabilities)):
        return "neutral", 0.0, None
    best_idx = int(np.argmax(probabilities))
    emotion_key = MODEL_EMOTIONS[best_idx]
    confidence = float(probabilities[best_idx])
    return emotion_key, confidence, probabilities


def select_expression(probabilities):
    if probabilities is None:
        return "neutral", 0.0

    sorted_indices = np.argsort(probabilities)[::-1]
    best_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1])
    best_label = MODEL_EMOTIONS[best_idx]
    best_conf = float(probabilities[best_idx])
    second_label = MODEL_EMOTIONS[second_idx]
    second_conf = float(probabilities[second_idx])

    # FER+ often sticks to neutral. If neutral is only slightly above the
    # next emotion, prefer the second one to better reflect expressions.
    if best_label == "neutral":
        small_gap = (best_conf - second_conf) < 0.10
        expressive_signal = second_conf >= 0.22
        if small_gap and expressive_signal:
            return second_label, second_conf

    return best_label, best_conf


def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    ensure_model_exists()
    emotion_net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    recent_predictions = deque(maxlen=5)
    display_emotion_key = "neutral"
    display_confidence = 0.0
    performance_mode_idx = 0
    frame_counter = 0
    fps_ema = 0.0
    last_tick = time.perf_counter()

    if not cap.isOpened():
        print("Kamera konnte nicht geoeffnet werden.")
        return
    if face_cascade.empty():
        print("OpenCV Gesichts-Cascade konnte nicht geladen werden.")
        return

    print("Starte Emotionserkennung. Beenden mit 'q'.")

    while True:
        frame_counter += 1
        ret, frame = cap.read()
        if not ret:
            print("Frame konnte nicht gelesen werden.")
            break
        now = time.perf_counter()
        dt = max(now - last_tick, 1e-6)
        instant_fps = 1.0 / dt
        fps_ema = instant_fps if fps_ema == 0.0 else (0.90 * fps_ema + 0.10 * instant_fps)
        last_tick = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            draw_label(frame, "Kein Gesicht erkannt", 10, 30)

        if len(faces) > 0:
            # Use the largest face to avoid unstable switching.
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face_gray = gray[y : y + h, x : x + w]
            infer_every_n = PERFORMANCE_MODES[performance_mode_idx]["infer_every_n_frames"]
            should_infer = (frame_counter % infer_every_n) == 0

            if should_infer:
                _, _, probabilities = predict_emotion(face_gray, emotion_net)
                emotion_key, confidence = select_expression(probabilities)
                recent_predictions.append((emotion_key, confidence))

            # Majority vote over recent frames keeps it stable but responsive.
            if recent_predictions:
                votes = {}
                for pred_key, _ in recent_predictions:
                    votes[pred_key] = votes.get(pred_key, 0) + 1
                stable_emotion_key = max(votes, key=votes.get)

                stable_confidences = [
                    conf for pred_key, conf in recent_predictions if pred_key == stable_emotion_key
                ]
                stable_confidence = sum(stable_confidences) / max(len(stable_confidences), 1)

                # Fast label reaction with light hysteresis to avoid jitter.
                if stable_emotion_key != display_emotion_key:
                    confidence_gap = stable_confidence - display_confidence
                    if confidence_gap > -0.03:
                        display_emotion_key = stable_emotion_key

                # Smooth percentage separately so numbers "chill" more.
                alpha = 0.18 if display_emotion_key == stable_emotion_key else 0.30
                display_confidence = (1 - alpha) * display_confidence + alpha * stable_confidence

            current_color = get_emotion_color(display_emotion_key)
            cv2.rectangle(frame, (x, y), (x + w, y + h), current_color, 2)
            emotion_label_de = EMOTION_LABELS_DE.get(display_emotion_key, "Unbekannt")
            draw_label(
                frame,
                f"Gefuehl: {emotion_label_de} ({display_confidence:.0%})",
                x,
                max(y - 10, 30),
                color=current_color,
            )
        mode_name = PERFORMANCE_MODES[performance_mode_idx]["name"]
        draw_label(frame, f"FPS: {fps_ema:.1f}", 10, 60, color=(255, 255, 255))
        draw_label(frame, f"Modus: {mode_name} (Taste m)", 10, 90, color=(255, 255, 255))

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("m"):
            performance_mode_idx = (performance_mode_idx + 1) % len(PERFORMANCE_MODES)
            recent_predictions.clear()
            print(f"Performance-Modus: {PERFORMANCE_MODES[performance_mode_idx]['name']}")
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
