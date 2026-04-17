import React, { useEffect, useMemo, useRef, useState } from "react";

var DEFAULT_API_BASE = (import.meta.env && import.meta.env.VITE_API_BASE) || "http://127.0.0.1:8000";

function formatPercent(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "-";
  return Math.round(x * 100) + "%";
}

function sleep(ms) {
  return new Promise(function(r) { return setTimeout(r, ms); });
}

async function blobFromVideoFrame(videoEl, quality) {
  var q = quality || 0.7;
  var canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  var ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  var blob = await new Promise(function(resolve) { canvas.toBlob(resolve, "image/jpeg", q); });
  if (!blob) throw new Error("could not capture frame");
  return blob;
}

async function analyzeFrame(opts) {
  var blob = opts.blob;
  var signal = opts.signal;
  var form = new FormData();
  form.append("file", blob, "frame.jpg");
  var url = DEFAULT_API_BASE + "/api/analyze";
  var res = await fetch(url, { method: "POST", body: form, signal: signal });
  if (!res.ok) {
    var txt = await res.text().catch(function() { return ""; });
    throw new Error("API " + res.status + ": " + (txt || res.statusText));
  }
  return res.json();
}

export default function App() {
  var videoRef = useRef(null);
  var canvasRef = useRef(null);
  var [isRunning, setIsRunning] = useState(false);
  var [fpsTarget, setFpsTarget] = useState(5);
  var [jpegQuality, setJpegQuality] = useState(0.7);
  var [last, setLast] = useState(null);
  var [status, setStatus] = useState("Bereit.");
  var [error, setError] = useState("");

  var intervalMs = useMemo(function() {
    return Math.max(80, Math.round(1000 / Math.max(1, fpsTarget)));
  }, [fpsTarget]);

  useEffect(function() {
    var stream;
    (async function() {
      try {
        setStatus("Kamera wird angefragt ...");
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setStatus("Kamera bereit.");
      } catch (e) {
        setError(String(e && e.message ? e.message : e));
        setStatus("Kamera-Fehler.");
      }
    })();
    return function() {
      if (stream) stream.getTracks().forEach(function(t) { t.stop(); });
    };
  }, []);

  useEffect(function() {
    var canvas = canvasRef.current;
    var video = videoRef.current;
    if (!canvas || !video) return;
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var faceBox = last && last.data && last.data.face_box;
    if (!faceBox) return;
    var scaleX = canvas.width / (video.videoWidth || canvas.width);
    var scaleY = canvas.height / (video.videoHeight || canvas.height);
    var x = faceBox[0], y = faceBox[1], w = faceBox[2], h = faceBox[3];
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 3;
    ctx.shadowColor = "#00ff88";
    ctx.shadowBlur = 10;
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
    var emotion = (last && last.data && last.data.emotion) || "";
    var conf = (last && last.data && last.data.confidence)
      ? Math.round(last.data.confidence * 100) + "%"
      : "";
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#00ff88";
    ctx.font = "bold 15px monospace";
    ctx.fillText(emotion + " " + conf, x * scaleX + 4, Math.max(16, y * scaleY - 6));
  }, [last]);

  useEffect(function() {
    if (!isRunning) return;
    if (!videoRef.current || !videoRef.current.videoWidth) return;
    var controller = new AbortController();
    var cancelled = false;
    (async function() {
      setError("");
      setStatus("Analyse laeuft ...");
      while (!cancelled) {
        var t0 = performance.now();
        try {
          var blob = await blobFromVideoFrame(videoRef.current, jpegQuality);
          var data = await analyzeFrame({ blob: blob, signal: controller.signal });
          setLast({ data: data, at: Date.now() });
        } catch (e) {
          var name = e && e.name ? String(e.name) : "";
          if (!name.includes("Abort")) {
            setError(String(e && e.message ? e.message : e));
            setStatus("API-Fehler.");
          }
        }
        var dt = performance.now() - t0;
        await sleep(Math.max(0, intervalMs - dt));
      }
    })();
    return function() {
      cancelled = true;
      controller.abort();
      setStatus("Gestoppt.");
      var canvas = canvasRef.current;
      if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [intervalMs, isRunning, jpegQuality]);

  var emotion = (last && last.data && last.data.emotion) ? last.data.emotion : "-";
  var confidence = last && last.data && last.data.confidence;

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">Real-Time Emotion Detection</div>
          <div className="subtitle">Browser-Kamera / FastAPI / Live UI</div>
        </div>
        <button className={isRunning ? "btn btnStop" : "btn"} onClick={function() { setIsRunning(function(v) { return !v; }); }}>
          {isRunning ? "Stop" : "Start"}
        </button>
      </header>

      <main className="grid">
        <section className="card">
          <div className="cardTitle">Kamera</div>
          <div style={{ position: "relative", display: "block", width: "100%" }}>
            <video ref={videoRef} className="video" playsInline muted style={{ display: "block", width: "100%" }} />
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
            />
          </div>
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
          </div>
          {error ? <div className="error">{error}</div> : null}
          <div className="divider" />
          <div className="form">
            <label className="field">
              <div className="label">FPS Target</div>
              <input type="range" min="1" max="15" value={fpsTarget} onChange={function(e) { setFpsTarget(Number(e.target.value)); }} />
              <div className="value">{fpsTarget} fps</div>
            </label>
            <label className="field">
              <div className="label">JPEG Qualitaet</div>
              <input type="range" min="0.3" max="0.95" step="0.05" value={jpegQuality} onChange={function(e) { setJpegQuality(Number(e.target.value)); }} />
              <div className="value">{jpegQuality.toFixed(2)}</div>
            </label>
          </div>
        </section>
      </main>

      <footer className="footer">
        <span className="small">Hinweis: Kamera bleibt im Browser, nur Frames gehen an dein Backend.</span>
      </footer>
    </div>
  );
}