"""Load vibration recordings and turn them into fixed-length feature windows.

Used by the notebook, the API and the tests, so training and serving share one code path.
"""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats
from scipy.io import loadmat, wavfile

WINDOW = 2048  # samples per window (~0.17 s at 12 kHz)


# ----------------------------------------------------------------------------- loading
def load_cwru_mat(path: str | Path) -> tuple[np.ndarray, float | None]:
    """Drive-end signal and rpm from a CWRU .mat file (some files hold extra series)."""
    path = Path(path)
    m = loadmat(path)
    fid = f"X{int(path.stem):03d}"
    de_keys = [k for k in m if k.endswith("_DE_time")]
    key = f"{fid}_DE_time" if f"{fid}_DE_time" in m else de_keys[0]
    rpm_key = f"{fid}RPM"
    rpm = float(np.ravel(m[rpm_key])[0]) if rpm_key in m else None
    return np.ravel(m[key]).astype(np.float64), rpm


def load_recording(data: bytes, filename: str) -> tuple[np.ndarray, float, str]:
    """Load an uploaded recording. Returns (signal, sample rate Hz, description).

    Supported:
      * .wav  - phone microphone / any audio (mono mix)
      * .csv  - phone accelerometer export (e.g. phyphox: a time column + x/y/z columns),
                or a single-column signal with the sample rate in a 'fs=12000' header line
      * .npz  - arrays 'signal' and 'fs' (the bundled demo samples)
    """
    name = filename.lower()
    if name.endswith(".wav"):
        fs, x = wavfile.read(io.BytesIO(data))
        x = x.astype(np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x, float(fs), "audio (microphone)"
    if name.endswith(".npz"):
        z = np.load(io.BytesIO(data), allow_pickle=False)
        return z["signal"].astype(np.float64), float(z["fs"]), str(z["source"]) if "source" in z else "array"
    if name.endswith(".csv") or name.endswith(".txt"):
        return _load_csv(data)
    raise ValueError("Unsupported file type. Upload .wav, .csv or .npz")


def _load_csv(data: bytes) -> tuple[np.ndarray, float, str]:
    text = data.decode("utf-8", errors="replace")
    first = text.splitlines()[0] if text else ""
    if first.lower().startswith("fs="):
        fs = float(first.split("=", 1)[1].strip().rstrip(","))
        x = pd.read_csv(io.StringIO(text), skiprows=1, header=None).iloc[:, 0].to_numpy(np.float64)
        return x, fs, "single-channel CSV"
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    if df.shape[1] < 2:
        raise ValueError("CSV needs a time column plus at least one signal column, or a 'fs=<rate>' first line")
    tcol = next((c for c in df.columns if "time" in str(c).lower()), df.columns[0])
    t = df[tcol].to_numpy(np.float64)
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not read a sample rate from the time column")
    fs = 1.0 / dt
    sig_cols = [c for c in df.columns if c != tcol and "abs" not in str(c).lower()]
    axes = df[sig_cols].to_numpy(np.float64)
    axes = axes - axes.mean(axis=0)
    # use the axis with the most vibration energy (the one pointing along the shaft load)
    best = int(np.argmax(axes.std(axis=0)))
    t_uniform = np.arange(t[0], t[-1], dt)  # phone sensors jitter; put samples on a uniform grid
    x = np.interp(t_uniform, t, axes[:, best])
    return x, fs, f"accelerometer CSV, axis '{sig_cols[best]}'"


def resample(x: np.ndarray, fs: float, target: float) -> np.ndarray:
    if abs(fs - target) < 1e-6:
        return x
    a, b = int(round(target)), int(round(fs))
    g = gcd(a, b)
    return sps.resample_poly(x, a // g, b // g)


def windows(x: np.ndarray, size: int = WINDOW, step: int | None = None, max_windows: int | None = None) -> np.ndarray:
    step = step or size
    n = 1 + (len(x) - size) // step if len(x) >= size else 0
    if max_windows:
        n = min(n, max_windows)
    if n <= 0:
        raise ValueError(f"Recording too short: need at least {size} samples")
    idx = np.arange(size)[None, :] + step * np.arange(n)[:, None]
    return x[idx]


# ----------------------------------------------------------------------------- features
TIME_FEATURES = ["rms", "std", "peak", "p2p", "kurtosis", "skewness", "crest", "shape", "impulse", "margin"]
N_BANDS = 16


def _time_features(w: np.ndarray) -> np.ndarray:
    w = w - w.mean(axis=1, keepdims=True)
    absw = np.abs(w)
    rms = np.sqrt((w ** 2).mean(axis=1))
    std = w.std(axis=1)
    peak = absw.max(axis=1)
    p2p = w.max(axis=1) - w.min(axis=1)
    kurt = stats.kurtosis(w, axis=1, fisher=False)
    skew = stats.skew(w, axis=1)
    mean_abs = absw.mean(axis=1)
    sqrt_amp = (np.sqrt(absw).mean(axis=1)) ** 2
    eps = 1e-12
    return np.column_stack([rms, std, peak, p2p, kurt, skew, peak / (rms + eps), rms / (mean_abs + eps),
                            peak / (mean_abs + eps), peak / (sqrt_amp + eps)])


def _band_fractions(spec: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """Share of spectral energy in n equal-width bands (0..Nyquist). Scale-free."""
    bands = np.array_split(spec, n_bands, axis=1)
    e = np.column_stack([b.sum(axis=1) for b in bands])
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def feature_names() -> list[str]:
    return (TIME_FEATURES + ["spectral_centroid", "spectral_kurtosis", "env_kurtosis"]
            + [f"band_{i:02d}" for i in range(N_BANDS)] + [f"env_band_{i:02d}" for i in range(N_BANDS)])


def extract_features(w: np.ndarray) -> np.ndarray:
    """Features for each window (rows of w). Frequencies are relative to Nyquist, so the same
    code works for any sample rate; the fault classifier is only trained at 12 kHz."""
    w = w - w.mean(axis=1, keepdims=True)
    tf = _time_features(w)
    hann = np.hanning(w.shape[1])
    spec = np.abs(np.fft.rfft(w * hann, axis=1)) ** 2
    freqs = np.linspace(0, 1, spec.shape[1])
    centroid = (spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-12)
    spec_kurt = stats.kurtosis(spec, axis=1, fisher=False)
    # envelope analysis: bearing impacts show up as periodic bursts in the Hilbert envelope
    env = np.abs(sps.hilbert(w, axis=1))
    env_kurt = stats.kurtosis(env, axis=1, fisher=False)
    env = env - env.mean(axis=1, keepdims=True)
    env_spec = np.abs(np.fft.rfft(env * hann, axis=1)) ** 2
    # envelope spectrum: low-frequency part holds the defect frequencies (first quarter of Nyquist)
    env_spec = env_spec[:, : env_spec.shape[1] // 4]
    return np.column_stack([tf, centroid, spec_kurt, env_kurt, _band_fractions(spec), _band_fractions(env_spec)])


def spectrum(x: np.ndarray, fs: float, n: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    seg = x[: min(len(x), 8 * n)]
    f, p = sps.welch(seg - seg.mean(), fs=fs, nperseg=min(n, len(seg)))
    return f, p


def envelope_spectrum(x: np.ndarray, fs: float, n: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    seg = x[: min(len(x), 8 * n)]
    env = np.abs(sps.hilbert(seg - seg.mean()))
    f, p = sps.welch(env - env.mean(), fs=fs, nperseg=min(n, len(seg)))
    return f, p


# ----------------------------------------------------------------------------- physics features
CLASSIFIER_FS = 12_000
CLASSIFIER_WINDOW = 6000  # 0.5 s: enough frequency resolution (2 Hz) to see the defect lines
_ORDERS = [("bpfi", 5.4152), ("bpfo", 3.5848), ("bsf", 4.7135)]  # SKF 6205 drive-end bearing


def defect_features(w: np.ndarray, fs: float, rpm: float) -> np.ndarray:
    """Envelope-spectrum SNR (dB) at the bearing defect frequencies (1st+2nd harmonic averaged),
    plus each one relative to the strongest. Band-pass 2-5 kHz first: bearing impacts excite
    structural resonances there, the envelope recovers how often they repeat."""
    sos = sps.butter(4, (2000, 5000), btype="band", fs=fs, output="sos")
    xf = sps.sosfiltfilt(sos, w, axis=1)
    env = np.abs(sps.hilbert(xf, axis=1))
    env -= env.mean(axis=1, keepdims=True)
    n = w.shape[1]
    p = np.abs(np.fft.rfft(env * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    base = np.median(p[:, (f > 20) & (f < 600)], axis=1) + 1e-30
    fr = rpm / 60
    snr = []
    for _, o in _ORDERS:
        s = 0
        for h in (1, 2):
            m = (f > o * fr * h * 0.98 - 1) & (f < o * fr * h * 1.02 + 1)
            s = s + 10 * np.log10(p[:, m].max(axis=1) / base)
        snr.append(s / 2)
    snr = np.column_stack(snr)
    return np.column_stack([snr, snr - snr.max(axis=1, keepdims=True)])


def classifier_feature_names() -> list[str]:
    return feature_names() + [f"snr_{k}" for k, _ in _ORDERS] + [f"rel_{k}" for k, _ in _ORDERS]


def classifier_features(w: np.ndarray, fs: float, rpm: float) -> np.ndarray:
    return np.column_stack([extract_features(w), defect_features(w, fs, rpm)])


def baseline_window(fs: float) -> int:
    """Window for the baseline detector: ~0.1-0.6 s whatever the sample rate (phone accelerometers run at 100-500 Hz)."""
    return 2048 if fs >= 8000 else 256 if fs >= 300 else 128
