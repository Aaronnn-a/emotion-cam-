## Real-Time Emotion Detection (Web-based)

Flow: **Kamera (Browser) → FastAPI → React UI**

### Live Deployment (Vercel + Railway)

- **Frontend**: Vercel → `emotion.aaronashraf.com`
- **Backend**: Railway → z.B. `https://emotion-api-production.up.railway.app`

#### Backend auf Railway

1. Neues Railway Project erstellen → “Deploy from GitHub” (dieses Repo)
2. Railway erkennt `backend/railway.toml` + `backend/Dockerfile`
3. Environment Variable setzen:
   - `ALLOWED_ORIGINS=https://emotion.aaronashraf.com,https://aaronashraf.com`
4. Deploy abwarten, dann prüfen:
   - `GET /health`

#### Frontend auf Vercel

1. Neues Vercel Project → Repo auswählen
2. **Root Directory**: `frontend`
3. Environment Variable setzen:
   - `VITE_API_BASE=<deine Railway-URL>` (z.B. `https://emotion-api-production.up.railway.app`)
4. Domain verbinden: `emotion.aaronashraf.com`

### 1) Backend starten (FastAPI)

Im Projekt-Root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Test:

- `GET` `http://127.0.0.1:8000/health`

### 2) Frontend starten (React + Vite)

In einem zweiten Terminal:

```bash
cd frontend
npm install
npm run dev
```

Dann im Browser öffnen: `http://localhost:5173`

### API

- `POST /api/analyze` (multipart form-data)
  - field: `file` (image/jpeg, image/png, image/webp)
  - response: `{ emotion, confidence, face_box, probabilities }`

# Gesichtsausdruck-Erkennung (Python + OpenCV)

Ein kleines Projekt zur Live-Erkennung von Gesichts-Emotionen ueber die Webcam.
Es nutzt OpenCV + ONNX (FER+) und erkennt u. a.:
`Gluecklich`, `Traurig`, `Wuetend`, `Ueberrascht`, `Neutral`.

## Voraussetzungen

- Python 3.14 (oder andere aktuelle Versionen)
- Webcam

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start

```bash
python main.py
```

- Mit `q` beenden.
- Keine TensorFlow- oder DeepFace-Abhaengigkeiten.
- Beim ersten Start wird ein ONNX-Modell (ca. 35 MB) automatisch heruntergeladen.

## Hinweis

Falls die Kamera nicht erkannt wird, pruefe ob sie von einer anderen App belegt ist.

## Hinweis zur Genauigkeit

Die Emotionserkennung ist eine Schaetzung und kann je nach Licht, Kamerawinkel
und Gesichtsausdruck schwanken.
