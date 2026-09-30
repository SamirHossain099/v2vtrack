"""Unpack DeepSense V2V inner archives from the extracted Drive parts into extracted/scenarioNN/.

The Drive zips (already unpacked by hand to "extracted/ScenarioNN Dataset - Part K/") hold split inner
archives: scenarioNN_<modality>.zip.001, .002, ... A split zip is one zip cut into pieces, so the parts
are read through a file object that spans them; nothing is concatenated on disk.

Usage: python unpack_inner.py 36 37 38 39 --modalities gps pwr        (light: ~1 GB total)
       python unpack_inner.py 36 --modalities radar rgb90 lidar        (heavy, one scenario at a time)
Index files (csv, mat, p) are always copied.
"""
import argparse
import io
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(r"N:\Datasets\DeepSense 6G\extracted")


class SplitFile(io.RawIOBase):
    """Read-only, seekable view over consecutive part files."""

    def __init__(self, parts):
        self.parts = [open(p, "rb") for p in parts]
        self.sizes = [p.stat().st_size for p in parts]
        self.starts = [sum(self.sizes[:i]) for i in range(len(parts))]
        self.total = sum(self.sizes)
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.total + off}[whence]
        return self.pos

    def readinto(self, b):
        n, view = 0, memoryview(b)
        while n < len(b) and self.pos < self.total:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            f = self.parts[i]
            f.seek(self.pos - self.starts[i])
            got = f.readinto(view[n:n + min(len(b) - n, self.sizes[i] - (self.pos - self.starts[i]))])
            if not got:
                break
            n += got
            self.pos += got
        return n

    def close(self):
        for f in self.parts:
            f.close()
        super().close()


def parts_of(scen: int):
    files = {}
    for d in ROOT.glob(f"Scenario{scen} Dataset - Part *"):
        for f in d.iterdir():
            files[f.name] = f
    return files


def unpack(scen: int, modalities, force=False):
    dest = ROOT / f"scenario{scen}"
    dest.mkdir(exist_ok=True)
    files = parts_of(scen)
    for name in (f"scenario{scen}.csv", f"scenario{scen}.mat", f"scenario{scen}.p"):
        if name in files and not (dest / name).exists():
            shutil.copy2(files[name], dest / name)
            print(f"  copied {name}")
    for mod in modalities:
        parts = sorted(f for n, f in files.items() if n.startswith(f"scenario{scen}_{mod}.zip."))
        if not parts:
            print(f"  {mod}: no parts found")
            continue
        marker = dest / f".{mod}.done"
        if marker.exists() and not force:
            print(f"  {mod}: already unpacked")
            continue
        t0 = time.time()
        with SplitFile(parts) as sf, zipfile.ZipFile(io.BufferedReader(sf, 1 << 24)) as z:
            names = z.namelist()
            print(f"  {mod}: {len(parts)} parts, {sum(p.stat().st_size for p in parts) / 1e9:.1f} GB, {len(names)} members", flush=True)
            z.extractall(dest)
        marker.write_text(f"{len(names)} members from {[p.name for p in parts]}\n")
        print(f"  {mod}: done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", type=int, nargs="+")
    ap.add_argument("--modalities", nargs="+", default=["gps", "pwr"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for s in a.scenarios:
        print(f"== scenario {s}", flush=True)
        unpack(s, a.modalities, a.force)
