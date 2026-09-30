"""Summarise the members of the DeepSense V2V archives without extracting them.

Writes results/members_<zipstem>.json: counts and bytes per (top-level folder, extension),
and the full list of small non-media members (csv, txt, json, mat, npy under 50 MB).
"""
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

DATA = Path(r"N:\Datasets\DeepSense 6G")
OUT = Path(__file__).resolve().parents[1] / "results"


def summarise(zpath: Path) -> dict:
    groups = defaultdict(lambda: [0, 0, 0])  # count, compressed, uncompressed
    small = []
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
        for i in infos:
            if i.is_dir():
                continue
            parts = i.filename.split("/")
            folder = "/".join(parts[:-1])
            ext = Path(parts[-1]).suffix.lower()
            g = groups[f"{folder} | {ext}"]
            g[0] += 1
            g[1] += i.compress_size
            g[2] += i.file_size
            if ext in {".csv", ".json", ".xlsx", ".md", ".pdf"}:
                small.append([i.filename, i.file_size])
    return {
        "zip": zpath.name,
        "members": len(infos),
        "groups": {k: v for k, v in sorted(groups.items())},
        "index_files": small,
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pattern = sys.argv[1] if len(sys.argv) > 1 else "Scenario3[6-9]*.zip"
    for zp in sorted(DATA.glob(pattern)):
        s = summarise(zp)
        (OUT / f"members_{zp.stem.replace(' ', '_')}.json").write_text(json.dumps(s, indent=1))
        print(f"\n== {s['zip']}: {s['members']} members", flush=True)
        for k, (n, c, u) in s["groups"].items():
            print(f"  {n:>8}  {u / 1e9:8.2f} GB  {k}", flush=True)
        for name, size in s["index_files"]:
            print(f"  INDEX {size:>12}  {name}", flush=True)
