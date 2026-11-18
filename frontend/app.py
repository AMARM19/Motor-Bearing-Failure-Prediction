"""Streamlit UI for the bearing fault API."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
for cand in (ROOT / "src", ROOT.parent / "src"):
    if cand.exists():
        sys.path.insert(0, str(cand))
from bearing import signals as S  # noqa: E402

API = os.environ.get("BACKEND_URL", "http://localhost:5000").rstrip("/")
st.set_page_config(page_title="Bearing Fault Early Warning", page_icon="⚙️", layout="wide")


@st.cache_data(ttl=600)
def api_get(path: str):
    r = requests.get(f"{API}{path}", timeout=90)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def sample_bytes(name: str) -> bytes:
    r = requests.get(f"{API}/samples/{name}", timeout=90)
    r.raise_for_status()
    return r.content


def post(path: str, files: dict, data: dict | None = None):
    r = requests.post(f"{API}{path}", files=files, data=data or {}, timeout=120)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text[:300]}
    return r.ok, body


def plots(data: bytes, name: str, defect: dict | None = None):
    x, fs, _ = S.load_recording(data, name)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    n = min(len(x), int(0.2 * fs))
    axes[0].plot(np.arange(n) / fs * 1000, x[:n], lw=0.7, color="#264653")
    axes[0].set(title="Waveform (first 0.2 s)", xlabel="ms")
    f, p = S.spectrum(x, fs)
    axes[1].semilogy(f, p, lw=0.7, color="#264653"); axes[1].set(title="Spectrum", xlabel="Hz")
    if fs >= 10_000:
        from scipy import signal as sps
        xf = sps.sosfiltfilt(sps.butter(4, (2000, min(5000, fs / 2 * 0.95)), btype="band", fs=fs, output="sos"), x)
        f, p = S.envelope_spectrum(xf, fs, n=16384)
        axes[2].plot(f, 10 * np.log10(p + 1e-20), lw=0.7, color="#e76f51"); axes[2].set_xlim(0, 400)
        for (k, v), c in zip((defect or {}).items(), ["#e76f51", "#264653", "#b08900", "#999"]):
            axes[2].axvline(v, ls="--", lw=1, color=c, label=k)
        if defect:
            axes[2].legend(fontsize=7)
        axes[2].set(title="Envelope spectrum (2-5 kHz band)", xlabel="Hz", ylabel="dB")
    else:
        axes[2].axis("off"); axes[2].text(0.05, 0.5, f"Sample rate {fs:.0f} Hz:\nenvelope analysis needs ≥10 kHz", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)


st.title("⚙️ Motor Bearing Fault Early Warning")
st.caption("Vibration-based bearing fault detection. Trained on the CWRU bearing dataset (12 kHz accelerometer). "
           "Free hosting: the first request after idle can take ~1 minute while the API wakes up.")

try:
    sample_list = api_get("/samples")
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach the API at {API}. It may be waking up; refresh in a minute. ({e.__class__.__name__})")
    st.stop()
names = [s["name"] for s in sample_list]

tab1, tab2, tab3 = st.tabs(["🔍 Diagnose a recording", "📈 Early warning vs healthy baseline", "ℹ️ About the model"])

with tab1:
    st.markdown("Pick a demo clip or upload your own recording. Fault-type diagnosis needs a **≥10 kHz** recording "
                "(accelerometer or audio). Phone accelerometer data (100-500 Hz) goes in the early-warning tab.")
    c1, c2 = st.columns([1, 1])
    with c1:
        src = st.radio("Source", ["Demo clip", "Upload"], horizontal=True)
        if src == "Demo clip":
            pick = st.selectbox("Demo clip (CWRU)", names, index=names.index("outer_race_007_1hp") if "outer_race_007_1hp" in names else 0)
            data, fname = sample_bytes(pick), f"{pick}.npz"
        else:
            up = st.file_uploader("Recording (.wav, .csv, .npz)", type=["wav", "csv", "npz", "txt"])
            data, fname = (up.getvalue(), up.name) if up else (None, None)
    with c2:
        rpm = st.number_input("Shaft speed (rpm), leave 0 to use the file's value / 1797", min_value=0, max_value=6000, value=0, step=10)
    if data and st.button("Diagnose", type="primary"):
        with st.spinner("Analysing..."):
            ok, res = post("/predict", {"file": (fname, data)}, {"rpm": rpm} if rpm else None)
        if not ok:
            st.error(res.get("error"))
        else:
            color = "green" if res["prediction"] == "Normal" else "red"
            st.markdown(f"### :{color}[{res['prediction']}]  ·  confidence {res['confidence']:.0%}")
            st.write(res["advice"])
            if res.get("warning"):
                st.warning(res["warning"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Windows analysed", res["windows"]); m2.metric("Length", f"{res['seconds']} s")
            m3.metric("Sample rate", f"{res['sample_rate_hz']:.0f} Hz"); m4.metric("Shaft speed", f"{res['rpm']:.0f} rpm")
            st.bar_chart(res["probabilities"], horizontal=True)
            plots(data, fname, res["defect_frequencies_hz"])

with tab2:
    st.markdown("""
**How a mill would use it:** record the motor once when it is known to be healthy (e.g. after an overhaul), then record
again every week from the same spot. The detector learns *this* motor's healthy vibration and reports a **health index**
(above 1 = abnormal window). No fault labels needed, so it works for any motor, pump or fan, including phone recordings.

Recording with a phone: lay it flat on the motor/fan housing, open **phyphox → Acceleration (without g)**, record 30-60 s,
export as CSV. Keep the same position and speed for both recordings.
""")
    c1, c2 = st.columns(2)
    with c1:
        hsrc = st.radio("Healthy baseline", ["Demo: CWRU healthy (1 hp)", "Upload"], key="h")
        if hsrc.startswith("Demo"):
            hd, hn = sample_bytes("healthy_1hp"), "healthy_1hp.npz"
        else:
            u = st.file_uploader("Healthy recording", type=["wav", "csv", "npz", "txt"], key="hu")
            hd, hn = (u.getvalue(), u.name) if u else (None, None)
    with c2:
        tsrc = st.radio("Recording to check", ["Demo clip", "Upload"], key="t")
        if tsrc == "Demo clip":
            tp = st.selectbox("Demo clip", names, index=names.index("inner_race_007_1hp") if "inner_race_007_1hp" in names else 0, key="tp")
            td, tn = sample_bytes(tp), f"{tp}.npz"
        else:
            u = st.file_uploader("New recording", type=["wav", "csv", "npz", "txt"], key="tu")
            td, tn = (u.getvalue(), u.name) if u else (None, None)
    if hd and td and st.button("Check against baseline", type="primary"):
        with st.spinner("Fitting baseline and scoring..."):
            ok, res = post("/baseline", {"healthy": (hn, hd), "test": (tn, td)})
        if not ok:
            st.error(res.get("error"))
        else:
            icon = {"OK": "🟢", "WATCH": "🟠", "ALARM": "🔴"}[res["status"]]
            st.markdown(f"### {icon} {res['status']}")
            st.write(res["message"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Abnormal windows", f"{res['alarm_share']:.0%}")
            m2.metric("Median health index", res["median_health_index"])
            m3.metric("Windows (healthy / test)", f"{res['healthy_windows']} / {res['test_windows']}")
            hi = np.array(res["health_index"])
            fig, ax = plt.subplots(figsize=(12, 3))
            t = np.arange(len(hi)) * res["window_seconds"]
            ax.plot(t, hi, ".-", ms=3, lw=0.6, color="#264653"); ax.axhline(1, ls="--", color="#e76f51", label="alarm threshold")
            ax.set_yscale("log"); ax.set(xlabel="seconds", ylabel="health index"); ax.legend()
            plt.tight_layout(); st.pyplot(fig)
            st.caption("Status: ALARM if ≥50% of windows exceed the threshold, WATCH if ≥10%. Note: the demo healthy_0hp clip also alarms against the 1 hp baseline. In CWRU the 0 hp healthy run differs from the others, so a real baseline should cover every load the motor runs at.")

with tab3:
    info = api_get("/model-info")
    v = info["validation"]
    st.markdown(f"""
**Data:** {info['trained_on']}. Bearing: {info['bearing']}.

**Two models**
1. **Fault-type classifier** ({info['model']}): 0.5 s windows, time-domain indicators (RMS, kurtosis, crest factor...),
   spectral band energies, and envelope-spectrum peaks at the bearing defect frequencies (BPFI/BPFO/BSF).
2. **Baseline detector** (early warning): Mahalanobis distance to the motor's own healthy recording, on scale-free features.

**Honest validation** (splits by recording, never by window):
* Train on 3 motor loads, test on the 4th: **{v['leave_one_load_out_acc']:.1%}** accuracy.
* Train on 2 fault sizes, test on the 3rd: **{v['leave_one_size_out_acc']:.1%}**. Unseen fault sizes are much harder;
  the fault *type* should be read as a hint, the baseline alarm as the main signal.
* Baseline detector with a healthy baseline across loads 0-3: every fault window of every size flagged; 0-2% false alarms on held-out healthy data.

**Limits:** CWRU faults are artificially seeded on one test rig. Phone accelerometers (100-500 Hz) cannot see the 2-5 kHz
band used for diagnosis, so phone data is checked with the baseline detector only.
""")
    st.json(info["baseline_detector"], expanded=False)
