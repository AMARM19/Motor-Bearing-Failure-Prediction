"""Flask API: bearing fault diagnosis + per-motor early-warning health index.

POST /predict    form-data: file (.wav/.csv/.npz), optional rpm      -> fault type (12 kHz+ recordings)
POST /baseline   form-data: healthy (file), test (file)              -> health index of test vs healthy
GET  /samples, /samples/<name>                                        -> bundled CWRU demo clips
GET  /health, /model-info
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bearing import anomaly as A  # noqa: E402
from bearing import signals as S  # noqa: E402
from bearing.catalog import ADVICE, CLASSES, DEFECT_ORDERS  # noqa: E402

MODEL = joblib.load(ROOT / "models" / "fault_classifier.joblib")
META = json.loads((ROOT / "models" / "model_metadata.json").read_text())
SAMPLES = ROOT / "data" / "samples"
MAX_SECONDS = 60
MIN_FS_CLASSIFIER = 10_000  # the classifier needs the 2-5 kHz band

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def _read(field: str):
    f = request.files.get(field)
    if f is None or not f.filename:
        raise ValueError(f"missing file field '{field}'")
    x, fs, desc = S.load_recording(f.read(), f.filename)
    x = x[: int(MAX_SECONDS * fs)]
    if not np.all(np.isfinite(x)):
        raise ValueError("recording contains NaN/inf values")
    return x, fs, desc, f.filename


def _rpm(default: float | None) -> float:
    v = request.form.get("rpm") or request.args.get("rpm")
    rpm = float(v) if v else (default or META["default_rpm"])
    if not 300 <= rpm <= 6000:
        raise ValueError("rpm must be between 300 and 6000")
    return rpm


@app.errorhandler(ValueError)
def bad_request(e):
    return jsonify(error=str(e)), 400


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="file too large (max 25 MB)"), 413


@app.get("/health")
def health():
    return jsonify(status="ok", model=META["model"])


@app.get("/model-info")
def model_info():
    return jsonify({k: META[k] for k in ("model", "classes", "trained_on", "validation", "baseline_detector", "bearing")})


@app.get("/samples")
def samples():
    out = []
    for p in sorted(SAMPLES.glob("*.npz")):
        z = np.load(p)
        out.append(dict(name=p.stem, source=str(z["source"]), seconds=round(len(z["signal"]) / float(z["fs"]), 1)))
    return jsonify(out)


@app.get("/samples/<name>")
def sample_file(name: str):
    p = SAMPLES / f"{Path(name).stem}.npz"
    if not p.exists():
        return jsonify(error="unknown sample"), 404
    return send_file(p, as_attachment=True, download_name=p.name)


@app.post("/predict")
def predict():
    x, fs, desc, fname = _read("file")
    default_rpm = None
    if fname.lower().endswith(".npz"):
        request.files["file"].stream.seek(0)
        z = np.load(request.files["file"].stream)
        default_rpm = float(z["rpm"]) if "rpm" in z else None
    rpm = _rpm(default_rpm)
    if fs < MIN_FS_CLASSIFIER:
        raise ValueError(f"sample rate {fs:.0f} Hz is too low for fault-type diagnosis (needs >= {MIN_FS_CLASSIFIER} Hz). "
                         "Phone accelerometer data: use /baseline (early warning) instead.")
    x12 = S.resample(x, fs, S.CLASSIFIER_FS)
    w = S.windows(x12, S.CLASSIFIER_WINDOW, step=S.CLASSIFIER_WINDOW // 2)
    proba = MODEL.predict_proba(S.classifier_features(w, S.CLASSIFIER_FS, rpm))
    mean = proba.mean(axis=0)
    votes = np.bincount(proba.argmax(axis=1), minlength=len(CLASSES)) / len(proba)
    label = CLASSES[int(mean.argmax())]
    warn = None
    if abs(fs - S.CLASSIFIER_FS) > 1 or "audio" in desc:
        warn = ("This recording differs from the training data (CWRU 12 kHz accelerometer on an SKF 6205 bearing). "
                "Treat the fault type as indicative only; use the baseline check for early warning.")
    return jsonify(
        prediction=label, confidence=round(float(mean.max()), 3),
        probabilities={c: round(float(p), 3) for c, p in zip(CLASSES, mean)},
        window_votes={c: round(float(v), 3) for c, v in zip(CLASSES, votes)},
        windows=len(w), seconds=round(len(x) / fs, 2), sample_rate_hz=round(fs, 1), rpm=rpm, input=desc,
        defect_frequencies_hz={k: round(o * rpm / 60, 1) for k, o in DEFECT_ORDERS.items()},
        advice=ADVICE[label], warning=warn,
    )


@app.post("/baseline")
def baseline():
    xh, fsh, dh, _ = _read("healthy")
    xt, fst, dt, _ = _read("test")
    if abs(fsh - fst) / fsh > 0.05:
        raise ValueError(f"healthy ({fsh:.0f} Hz) and test ({fst:.0f} Hz) recordings must use the same sample rate")
    win = S.baseline_window(fsh)
    names = S.feature_names()
    fh = A.select(S.extract_features(S.windows(xh, win)), names)
    ft = A.select(S.extract_features(S.windows(xt, win)), names)
    det = A.BaselineDetector().fit(fh)
    hi = det.health_index(ft)
    share = float((hi > 1).mean())
    status = "ALARM" if share >= 0.5 else "WATCH" if share >= 0.1 else "OK"
    msg = {"OK": "Vibration matches the healthy baseline.",
           "WATCH": "Some windows drift from the baseline. Record again; if it persists, schedule an inspection.",
           "ALARM": "Vibration is clearly different from the healthy baseline. Inspect the bearing (and check for "
                    "looseness, imbalance or misalignment) at the next stop."}[status]
    return jsonify(status=status, message=msg, alarm_share=round(share, 3), median_health_index=round(float(np.median(hi)), 3),
                   health_index=[round(float(v), 3) for v in hi], window_seconds=round(win / fsh, 3),
                   healthy_windows=len(fh), test_windows=len(ft), sample_rate_hz=round(fsh, 1), inputs=[dh, dt])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
