```javascriptreact
import React, { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_API_BASE = import.meta.env?.VITE_API_BASE || "http://127.0.0.1:8000";

function formatPercent(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "-";
  return Math.round(x * 100) + "%";
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function blobFromVideoFrame(videoEl, quality = 0.7) {
  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
  if (!blob) throw new Error("could not capture frame");
  return blob;
}

async function analyzeFrame({ apiBase, blob, signal }) {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  const res = await fetch(`${apiBase}/api/analyze`, { method: "POST", body: form, signal });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${txt || res.statusText}`);
  }
  return res.json();
}

export default function App() {
  const videoRef = useRef(null);
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [isRunning, setIsRunning] = useState(false);
  const [fpsTarget, setFpsTarget] = useState(5);
  const [jpegQuality, setJpegQuality] = useState(0.7);
  const [last, setLast] = useState(null);
  const [status, setStatus] = useState("Bereit.");
  const [error, setError] = useState("");

  const intervalMs = useMemo(() => Math.max(80, Math.round(1000 / Math.max(1, fpsTarget))), [fpsTarget]);

  useEffect(() => {
    let stream;
    (async () => {
      try {
        setStatus("Kamera wird angefragt …");
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setStatus("Kamera bereit.");
      } catch (e) {
        setError(String(e?.message || e));
        setStatus("Kamera-Fehler.");
      }
    })();
    return () => {
      if (stream) {
        for (const track of stream.getTracks()) track.stop();
      }
    };
  }, []);

  useEffect(() => {
    if (!isRunning) return;
    if (!videoRef.current) return;
    if (!videoRef.current.videoWidth) return;

    const controller = new AbortController();
    let cancelled = false;

    (async () => {
      setError("");
      setStatus("Analyse läuft …");

      while (!cancelled) {
        const t0 = performance.now();
        try {
          const blob = await blobFromVideoFrame(videoRef.current, jpegQuality);
          const data = await analyzeFrame({ apiBase, blob, signal: controller.signal });
          setLast({ data, at: Date.now() });
          setStatus("Analyse läuft …");
        } catch (e) {
          if (!String(e?.name).includes("Abort")) {
            setError(String(e?.message || e));
            setStatus("API-Fehler.");
          }
        }

        const dt = performance.now() - t0;
        await sleep(Math.max(0, intervalMs - dt));
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
      setStatus("Gestoppt.");
    };
  }, [apiBase, intervalMs, isRunning, jpegQuality]);

  const emotion = last?.data?.emotion ?? "-";
  const confidence = last?.data?.confidence;
  const faceBox = last?.data?.face_box;

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">Real-Time Emotion Detection</div>
          <div className="subtitle">Browser-Kamera → FastAPI → Live UI</div>
        </div>
        <div className="controls">
          <button className={isRunning ? "btn btnStop" : "btn"} onClick={() => setIsRunning((v) => !v)}>
            {isRunning ? "Stop" : "Start"}
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="card">
          <div className="cardTitle">Kamera</div>
          <video ref={videoRef} className="video" playsInline muted />
          <div className="hint">Tipp: gutes Licht + Gesicht nah an die Kamera.</div>
        </section>

        <section className="card">
          <div className="cardTitle">Ergebnis</div>
          <div className="resultRow">
            <div className="metric">
              <div className="metricLabel">Emotion</div>
              <div className="metricValue">{emotion}</div>
            </div>
            <div className="metric">
              <div className="metricLabel">Confidence</div>
              <div className="metricValue">{formatPercent(confidence)}</div>
            </div>
          </div>

          <div className="small">
            <div><span className="k">Status</span> {status}</div>
            {faceBox ? (
              <div><span className="k">Face Box</span> x={faceBox[0]} y={faceBox[1]} w={faceBox[2]} h={faceBox[3]}</div>
            ) : (
              <div><span className="k">Face Box</span> -</div>
            )}
            <div><span className="k">API</span> {apiBase}</div>
          </div>

          {error ? <div className="error">{error}</div> : null}

          <div className="divider" />

          <div className="form">
            <label className="field">
              <div className="label">API Base URL</div>
              <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://127.0.0.1:8000" />
            </label>
            <label className="field">
              <div className="label">FPS Target</div>
              <input type="range" min="1" max="15" value={fpsTarget} onChange={(e) => setFpsTarget(Number(e.target.value))} />
              <div className="value">{fpsTarget} fps</div>
            </label>
            <label className="field">
              <div className="label">JPEG Qualität</div>
              <input type="range" min="0.3" max="0.95" step="0.05" value={jpegQuality} onChange={(e) => setJpegQuality(Number(e.target.value))} />
              <div className="value">{jpegQuality.toFixed(2)}</div>
            </label>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span className="small">Hinweis: Alles läuft lokal; Kamera bleibt im Browser, nur Frames gehen an dein Backend.</span>
      </footer>
    </div>
  );
}


```