import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app import app  # noqa: E402

SAMPLES = ROOT / "data" / "samples"


@pytest.fixture()
def client():
    return app.test_client()


def upload(name):
    return (io.BytesIO((SAMPLES / f"{name}.npz").read_bytes()), f"{name}.npz")


def test_health(client):
    assert client.get("/health").json["status"] == "ok"


def test_samples_listed(client):
    names = {s["name"] for s in client.get("/samples").json}
    assert {"healthy_1hp", "inner_race_007_1hp", "outer_race_007_1hp", "ball_007_1hp"} <= names


@pytest.mark.parametrize("name,expected", [("healthy_1hp", "Normal"), ("inner_race_007_1hp", "Inner race"),
                                           ("outer_race_007_1hp", "Outer race"), ("ball_007_1hp", "Ball")])
def test_predict_samples(client, name, expected):
    r = client.post("/predict", data={"file": upload(name)}, content_type="multipart/form-data")
    assert r.status_code == 200, r.json
    assert r.json["prediction"] == expected  # samples were in training data: this is a smoke test, not an accuracy test


def _npz_part(name, a, b):
    z = np.load(SAMPLES / f"{name}.npz")
    fs = float(z["fs"]); buf = io.BytesIO()
    np.savez(buf, signal=z["signal"][int(a * fs):int(b * fs)], fs=fs)
    buf.seek(0)
    return buf, f"{name}_{a}_{b}.npz"


def test_baseline_ok_on_healthy_alarm_on_fault(client):
    # baseline = first 6 s of a healthy recording; unseen last 4 s should pass, a 0.007" fault should alarm
    ok = client.post("/baseline", data={"healthy": _npz_part("healthy_1hp", 0, 6), "test": _npz_part("healthy_1hp", 6, 10)}, content_type="multipart/form-data")
    bad = client.post("/baseline", data={"healthy": _npz_part("healthy_1hp", 0, 6), "test": upload("inner_race_007_1hp")}, content_type="multipart/form-data")
    assert ok.json["status"] == "OK", ok.json["alarm_share"]
    assert bad.json["status"] == "ALARM"


def test_phone_csv_baseline_and_low_rate_rejected_for_predict(client):
    rng = np.random.default_rng(0)
    t = np.arange(0, 30, 1 / 400)

    def csv(sig):
        rows = "\n".join(f"{a:.5f},{b:.5f},{c:.5f},{d:.5f},{abs(b):.5f}" for a, b, c, d in zip(t, sig, 0.1 * sig, rng.normal(0, .01, len(t)), ))
        return io.BytesIO(("Time (s),Acceleration x (m/s^2),Acceleration y (m/s^2),Acceleration z (m/s^2),Absolute acceleration (m/s^2)\n" + rows).encode())

    healthy = np.sin(2 * np.pi * 25 * t) + rng.normal(0, 0.2, len(t))
    faulty = healthy + 3 * (np.sin(2 * np.pi * 25 * t) > 0.995) * rng.normal(0, 3, len(t))  # periodic impacts
    r = client.post("/baseline", data={"healthy": (csv(healthy), "h.csv"), "test": (csv(faulty), "f.csv")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.json
    assert r.json["status"] in ("WATCH", "ALARM")
    r = client.post("/predict", data={"file": (csv(healthy), "h.csv")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "too low" in r.json["error"]


def test_bad_input(client):
    r = client.post("/predict", data={"file": (io.BytesIO(b"hello"), "x.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 400
