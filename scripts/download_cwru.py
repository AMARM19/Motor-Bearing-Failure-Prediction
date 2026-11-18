"""Download the 40 CWRU files used here (12 kHz drive end, ~134 MB) into data/raw/cwru/."""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bearing.catalog import BASE_URL, FILES  # noqa: E402

out = ROOT / "data" / "raw" / "cwru"
out.mkdir(parents=True, exist_ok=True)
for fid, *_ in FILES:
    dest = out / f"{fid}.mat"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        continue
    print("downloading", dest.name)
    urllib.request.urlretrieve(BASE_URL.format(id=fid), dest)
print("done:", len(list(out.glob("*.mat"))), "files in", out)
