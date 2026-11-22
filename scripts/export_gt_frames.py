"""Einzelne Frames aus einem Video als PNG speichern (für die Ground-Truth-Markierung).

Aufruf (im Projektordner):
    uv run python scripts/export_gt_frames.py --frames 2500
    uv run python scripts/export_gt_frames.py --video data/raw/PAT_1/VID_76.AVI --frames 2500 6500 8150

Die Bilder landen in data/interim/<Patient>/gt/ (durch .gitignore geschützt, nie committen).
Frames werden der Reihe nach gelesen statt per Sprung (CAP_PROP_POS_FRAMES), weil Springen
bei AVI-Dateien manchmal einen falschen Frame liefert. So stimmt die Frame-Nummer sicher
mit den SAM-3-Läufen überein.
"""
import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw/PAT_1/VID_76.AVI")
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    args = ap.parse_args()

    video = (PROJECT_ROOT / args.video).resolve()
    out_dir = PROJECT_ROOT / "data" / "interim" / video.parent.name / "gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = sorted(set(args.frames))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video}")

    nr = 0
    for target in wanted:
        while nr < target:          # bis zum Ziel-Frame vorspulen (nur lesen, nicht dekodieren)
            if not cap.grab():
                raise RuntimeError(f"Video endet vor Frame {target}")
            nr += 1
        ok, frame = cap.read()      # Ziel-Frame wirklich dekodieren
        if not ok:
            raise RuntimeError(f"Frame {target} konnte nicht gelesen werden")
        nr += 1
        path = out_dir / f"{video.stem}_f{target:05d}.png"
        cv2.imwrite(str(path), frame)
        print(f"Frame {target}: {frame.shape[1]} x {frame.shape[0]} -> {path.relative_to(PROJECT_ROOT)}")

    cap.release()


if __name__ == "__main__":
    main()
