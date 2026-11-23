"""Erzeugt die Pipeline-Masken aller markierten Frames für Evaluation.ipynb, für alle drei Verfahren.

Aufruf (im Projektordner):
    uv run python scripts/make_eval_masks.py                 # alle Verfahren, nur fehlende Masken
    uv run python scripts/make_eval_masks.py --force         # alle Masken neu erzeugen
    uv run python scripts/make_eval_masks.py --only yunet    # nur ein Verfahren (sam3, yunet, scrfd)

Markierte Frames = alle data/interim/PAT_1/gt/VID_76_f<Frame>.json (aus X-AnyLabeling).
Ergebnis: data/interim/PAT_1/eval/masks/VID_76_f<Frame>_<sam3det|yunet|scrfd>.npz

Der Pipeline-Code wird NICHT kopiert, sondern direkt aus den Notebooks gelesen und ausgeführt
(Setup-, Funktions- und Masken-Zellen). Ändert sich ein Notebook, gilt automatisch die neue Version.
Achtung: Es zählt der Stand auf der Festplatte. Ungespeicherte Änderungen im Editor werden nicht gesehen.

Jedes Verfahren läuft in einem eigenen Python-Prozess (getrennte Variablen, GPU-Speicher wird danach frei).
Dauer: SAM 3 ca. 40 s pro Frame (GPU), YuNet ca. 5 s pro Frame, SCRFD wenige Sekunden insgesamt
(nutzt die gespeicherten Rohfunde des ganzen Videos, SCRFD-10G.ipynb Abschnitt 17).
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"
GT_DIR = PROJECT_ROOT / "data" / "interim" / "PAT_1" / "gt"
MASK_DIR = PROJECT_ROOT / "data" / "interim" / "PAT_1" / "eval" / "masks"
VIDEO_STEM = "VID_76"

# Pro Verfahren: Notebook, Kürzel im Masken-Dateinamen, Zellen (erkannt am Anfang bzw. Inhalt der Zelle)
METHODS = {
    "sam3": {
        "notebook": "SAM3.ipynb", "key": "sam3det",
        "cells": [("startswith", "from pathlib import Path"),   # 0. Setup
                  ("contains", "SPLIT_X = 961"),                # 1. Ansichten, Hilfsfunktionen
                  ("contains", "def run_video_detect"),         # 1b. Einzelbild-Modell + Pipeline
                  ("contains", "GT_FRAMES = sorted(")],         # 2. Masken für die Evaluation
    },
    "yunet": {
        "notebook": "YuNet.ipynb", "key": "yunet",
        "cells": [("startswith", "import csv"),                 # Schritt 1
                  ("startswith", "SCORE_THRESHOLD"),            # Schritt 2
                  ("startswith", "def pixelate_region"),        # Schritt 4
                  ("contains", "def anonymize_video_views"),    # Schritt 7
                  ("contains", "GT_FRAMES = sorted(")],         # Schritt 8, Masken
    },
    "scrfd": {
        "notebook": "SCRFD-10G.ipynb", "key": "scrfd",
        "cells": [("contains", "MODEL_SCRFD = "),               # 1. Setup
                  ("startswith", "class SCRFD"),                # 3.
                  ("startswith", "def iou"),                    # 4.
                  ("startswith", "def near_track"),             # 5.
                  ("startswith", "def bridge_gaps"),            # 6.
                  ("startswith", "def smooth_tracks"),          # 7.
                  ("contains", "def detect_video_views"),       # 16.
                  ("contains", "GT_FRAMES = sorted(")],         # 18. Masken
    },
}


def gt_frames():
    return sorted(int(p.stem.split("_f")[-1]) for p in GT_DIR.glob(f"{VIDEO_STEM}_f*.json"))


def collect_code(notebook, rules, frames):
    # Holt die benötigten Code-Zellen aus dem Notebook und beschränkt die Masken-Zelle auf `frames`
    nb = json.loads((NB_DIR / notebook).read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    parts = []
    for how, text in rules:
        hits = [s for s in code_cells if (s.startswith(text) if how == "startswith" else text in s)]
        if len(hits) != 1:
            raise RuntimeError(f"{notebook}: Zelle '{text}' {len(hits)}-mal gefunden (erwartet 1)")
        parts.append(hits[0])
    # In der Masken-Zelle nur die gewünschten Frames bearbeiten (Zeile direkt nach "GT_FRAMES = sorted(...)")
    lines = parts[-1].splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("GT_FRAMES = sorted("))
    lines.insert(k + 1, f"GT_FRAMES = [f for f in GT_FRAMES if f in {sorted(frames)!r}]")
    parts[-1] = "\n".join(lines)
    return "\n\n".join(parts)


def main():
    sys.stdout.reconfigure(line_buffering=True)   # eigene Ausgaben sofort zeigen (vor denen der Unterprozesse)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(METHODS), default=list(METHODS))
    ap.add_argument("--force", action="store_true", help="vorhandene Masken neu erzeugen")
    args = ap.parse_args()

    frames = gt_frames()
    print("Markierte Frames:", frames)
    if not frames:
        print("Keine markierten Frames in", GT_DIR)
        return

    summary = {}
    for name in args.only:
        m = METHODS[name]
        todo = [f for f in frames
                if args.force or not (MASK_DIR / f"{VIDEO_STEM}_f{f:05d}_{m['key']}.npz").exists()]
        print(f"\n===== {name}: {len(todo)} von {len(frames)} Frames zu erzeugen =====")
        if not todo:
            summary[name] = "alle vorhanden"
            continue
        code = collect_code(m["notebook"], m["cells"], todo)
        with tempfile.NamedTemporaryFile("w", suffix=f"_{name}.py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
        t0 = time.time()
        # cwd = notebooks/, weil die Notebooks PROJECT_ROOT = Path.cwd().parent verwenden
        res = subprocess.run([sys.executable, "-u", tmp.name], cwd=NB_DIR)
        Path(tmp.name).unlink(missing_ok=True)
        summary[name] = f"{'OK' if res.returncode == 0 else 'FEHLER'} ({time.time() - t0:.0f} s)"

    print("\n===== Zusammenfassung =====")
    for name, s in summary.items():
        print(f"{name:6s}: {s}")
    for f in frames:
        have = [m["key"] for m in METHODS.values() if (MASK_DIR / f"{VIDEO_STEM}_f{f:05d}_{m['key']}.npz").exists()]
        print(f"Frame {f}: Masken für {have}")


if __name__ == "__main__":
    main()
