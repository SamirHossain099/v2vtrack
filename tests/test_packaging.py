"""Packaging guard: the staged repository must hold no working notes, no dataset-derived arrays and no manuscript.

Checks the directory, not only git (RESEARCH.md standing rule 16): a gitignored file still sits in a hand-made
zip. Proven to fail on a planted CLAUDE.md before it was trusted.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {"CLAUDE.md", "PLAN.md", "FINDINGS.md", "CORRECTIONS.md", "PAPER-DRAFT.md", "RESULTS.md",
                   "MANUSCRIPT.md", "MANUSCRIPT.docx", "MANUSCRIPT-IEEE.docx", "MANUSCRIPT-IEEE.pdf", "zotero_keys.json",
                   "references.bib", "references.ris", "references-missing.ris"}
FORBIDDEN_DIRS = {"external", "data", "checkpoints", "submission", "public-repo"}
FORBIDDEN_SUFFIX = (".npz", ".npy", ".pt", ".mat")
FORBIDDEN_PREFIX = ("make_docx", "make_refs", "sync_zotero", "radar_profiles_", "members_")


def walk():
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", ".pytest_cache")]
        for f in files:
            yield Path(d) / f


def test_no_working_notes_or_manuscript():
    bad = [p for p in walk() if p.name in FORBIDDEN_NAMES or p.name.startswith(FORBIDDEN_PREFIX)]
    assert not bad, bad


def test_no_private_directories():
    bad = [p for p in walk() if set(p.relative_to(ROOT).parts[:-1]) & FORBIDDEN_DIRS]
    assert not bad, bad


def test_no_dataset_derived_arrays():
    bad = [p for p in walk() if p.suffix in FORBIDDEN_SUFFIX]
    assert not bad, bad


def test_no_local_paths_in_results():
    bad = []
    for p in (ROOT / "results").glob("*.json"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "N:\\\\" in txt or "N:/" in txt or "C:\\\\Users" in txt:
            bad.append(p.name)
    assert not bad, bad


def test_no_em_or_en_dashes_in_released_text():
    """RESEARCH.md standing rule 1, extended 2026-09-30: nothing that leaves the machine carries an em dash."""
    bad = []
    for p in walk():
        if p.suffix in (".md", ".py", ".cff", ".txt", ".yml", ".toml"):
            s = p.read_text(encoding="utf-8", errors="replace")
            for ch in ("\u2014", "\u2013"):
                if ch in s and f"\\u{ord(ch):04x}" not in s:
                    bad.append((str(p.relative_to(ROOT)), hex(ord(ch))))
    assert not bad, bad
