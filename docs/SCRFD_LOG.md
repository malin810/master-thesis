# SCRFD-10G – Log: Gesichtserkennung und Verpixelung

Ziel: Gesichter in Videos der Epilepsiestation mit dem Gesichtsdetektor SCRFD-10G (InsightFace) finden und verpixeln, als Vergleich zu SAM 3 ([`SAM3_LOG.md`](SAM3_LOG.md)) und YuNet ([`YUNET_LOG.md`](YUNET_LOG.md)).

Notebook: [`notebooks/SCRFD-10G.ipynb`](../notebooks/SCRFD-10G.ipynb). Ergebnisse liegen lokal in `data/interim/PAT_1/*.pkl` und `data/processed/PAT_1/*.mp4` (nicht in Git). Ausgewertet wurde nur über Zahlen aus den `.pkl`-Dateien.

---

## 1. Umgebung

| Was | Stand |
|---|---|
| Rechner | Laptop, NVIDIA RTX 3070 (SCRFD läuft aber über `onnxruntime` auf der **CPU**) |
| Python | 3.12.12 über `uv` |
| onnxruntime | 1.31.1 (nur CPU) |
| Modell | `models/buffalo_l/buffalo_l/det_10g.onnx` (16,9 MB), aus `buffalo_l.zip` der InsightFace-Releases (`github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`), heruntergeladen am 23.11.2026, per `.gitignore` ausgeschlossen |
| SHA-256 | `buffalo_l.zip`: `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`; `det_10g.onnx`: `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| Lizenz | InsightFace-Modelle: nur nicht-kommerzielle Forschung |
| Eingabegröße | variabel (`[1, 3, ?, ?]`). Das Modell speichert Beispielgrößen, daher die harmlose Warnung „Expected shape … does not match“ (ausgeblendet mit `ort.set_default_logger_severity(3)`) |

Testvideo und Bildaufbau wie in `SAM3_LOG.md`, Abschnitt 2: `VID_76.AVI`, 1920 × 1080, 25 fps, 8759 Frames. Zwei Ansichten: **F (Farbe)** x 0–960, volle Höhe; **SW (Schwarz-Weiß)** x 991–1909, y 10–526.

---

## 2. Versuche (chronologisch)

### 2.1 v1: ganzes Bild (Abschnitte 9–11, frühere Arbeit)

Auf dem früheren Rechner. Ablauf in zwei Durchgängen:
1. Rohfunde für alle Frames berechnen und speichern (ganzes Bild, Eingabe 1280 × 736, ab Score 0,2), damals ca. 0,9 s pro Frame (≈ 2,2 h für VID_76)
2. Offline-Kette: Tracking (`track_with_ids`: neue Spur ab Score 0,5, Weiterführen ab 0,2, Halten bis 8 Frames, Geschwindigkeit mit Dämpfung) → Lücken füllen (`bridge_gaps`, linear bis 40 Frames) → Glätten (`smooth_tracks`, k = 3) → Doppelte entfernen (`dedupe`) → verpixeln mit 50 % Rand, Blockgröße 22 px

Eingabegröße 1280 × 736 statt 1920 × 1088 gewählt (erste 500 Frames: kein Gewinn bei 2,5-facher Zeit, siehe Ende des Notebooks).

**Bekannte Problemstelle:** Frames 8100–8250, dort 140 von 150 Frames ohne Box. Darum wurde zusätzlich ein Kopfdetektor (YOLO11s, Abschnitte 12–15) getestet; der half in Farbe, aber nicht in S/W. **Für den Vergleich hier wird YOLO11 nicht verwendet.**

### 2.2 v2: getrennte Ansichten (Abschnitte 16–18, 23.11.2026)

| | v1 | v2 |
|---|---|---|
| Suche | ganzes Bild, 1280 × 736 (Maßstab 0,67) | jede Ansicht einzeln: **Farbe 640 × 736** (Maßstab 0,67), **S/W 928 × 544** (Maßstab 1,0, volle Auflösung) |
| Tracking, Lücken füllen, Glätten | gemeinsam | **pro Ansicht** |
| Rand der Boxen | am Bildrand abgeschnitten | am Rand der Ansicht abgeschnitten |
| außerhalb der Ansichten | sichtbar | schwarz |
| Ausgabe | Video | `_pixel.mp4`, Kontrollvideo `_masken.mp4`, `.pkl` mit Boxen, Masken für `Evaluation.ipynb` |

Alle Schwellen und die Kette wie in v1.

**Tempo auf diesem Rechner** (Frame 2500): ganzes Bild 1280 × 736: 0,15 s; Farbe 640 × 736: 0,08 s + S/W 928 × 544: 0,08 s. Getrennte Ansichten kosten also kaum mehr Zeit.

#### Test Frames 2400–3000 (Lauf `VID_76_scrfdviews_2400_3000_2026-11-23_16-06-00`)

| | Farbe | S/W |
|---|---|---|
| Frames ohne Verpixelung | **0** | **0** |
| ohne sicheren Fund (Score ≥ 0,5) | 76 (längste Folge 11) | 89 (längste 28) |
| sichere Funde pro Frame | 0: 76, 1: 416, 2: 108 | 0: 89, 1: 511 |
| Höhe sicherer Funde min / median / max | 60 / 91 / 161 px | 90 / 113 / 134 px |
| verpixelte Fläche der Ansicht, Median / Max | 4,6 / 10,1 % | 9,6 / 12,5 % |

Erkennung 5,9 Frames/s, Videos schreiben 60 s. Auf dieser Stelle schließt die Kette alle Lücken. Den Teilkopf am rechten Rand der Farbansicht (Frame 2500, Ground Truth) findet SCRFD nicht.

#### Ganzes Video (Lauf `VID_76_scrfdviews_0_8759_2026-11-23_16-31-36`)

Rohfunde: `data/interim/PAT_1/VID_76_scrfdviews_raw_0_8759.pkl`. Erkennung 1412 s für 8500 Frames (**6,0 Frames/s**, gesamt ca. 24 min), Videos schreiben 851 s.

| | Farbe | S/W |
|---|---|---|
| **Frames ohne Verpixelung** | **879** (10 %, längste Lücke **267** = 10,7 s) | **2987** (34 %, längste Lücke **840** = 33,6 s) |
| ohne sicheren Fund (Score ≥ 0,5) | 2376 (27 %, längste 281) | 6429 (73 %, längste 856) |
| sichere Funde pro Frame | 0: 2376, 1: 5404, 2: 953, 3: 26 | 0: 6429, 1: 2325, 2: 5 |
| Höhe sicherer Funde min / median / max | 35 / 111 / 201 px | 53 / 116 / 157 px |
| verpixelte Fläche der Ansicht, Median / Max | 4,0 / 18,3 % | 9,7 / 27,0 % |

Lücken ab 1 s (Frames):
- **Farbe:** 3477–3511, 3533–3577, 3977–4014, 5576–5621, 6164–6244 (04:06), 6466–6539 (04:18), 6742–6801 (04:29), 7789–7902, 7927–8045 (05:11–05:21), **8101–8367 (05:24–05:35)**
- **S/W:** 3112–3153, 3195–3292, 3573–3618, 4392–4454, 5158–5257, 5273–5401, 5572–5754, 5862–5997, **6018–6697 (04:00–04:27)**, 6738–6806, 7080–7242, 7261–7483, 7500–7618, **7772–8611 (05:11–05:44)**, 8644–8721

---

## 3. Vergleich ganzes Video (alle mit getrennten Ansichten)

| Frames ohne Verpixelung | Farbe | S/W | längste Lücke Farbe / S/W | Tempo |
|---|---|---|---|---|
| SAM 3 (`run_video_detect`) | **0** | **0** | 0 / 0 | 1,25 Frames/s (GPU) |
| YuNet v3 (clip/full) | 43 (0,5 %) | 2938 (34 %) | 0,8 s / 16 s | 7,2 Frames/s (CPU) |
| SCRFD-10G v2 | 879 (10 %) | 2987 (34 %) | 10,7 s / 33,6 s | 6,0 Frames/s Erkennung (CPU) + Kette offline |

---

## 4. Erkenntnisse

- **S/W:** SCRFD versagt ähnlich wie YuNet (34 % ungeschützt). Tracking und Lücken füllen (bis 40 Frames) helfen nicht, wenn das Gesicht über Sekunden bis Minuten nicht gefunden wird. Die längste Lücke (05:11–05:44) liegt an derselben Stelle wie bei YuNet.
- **Farbe:** schlechter als YuNet (10 % statt 0,5 %). Vermutlicher Grund: Eine Spur beginnt erst ab Score 0,5 (YuNet: alles ab 0,3). Die längste Farblücke ist die bekannte Problemstelle 8100–8250.
- **Keine riesigen Fehlboxen** in S/W (anders als YuNet im Ausschnitt): Boxen 53–157 px hoch, verpixelte Fläche moderat.
- Die Teststelle 2400–3000 ist für SCRFD leicht; die Probleme liegen in Minute 3–6 (Bewegung, gedrehter Kopf, Personen im Raum).
- **Fazit für den Vergleich:** Reine Gesichtsdetektoren (YuNet, SCRFD) reichen für dieses Video nicht aus, vor allem nicht in S/W. SAM 3 mit dem Prompt „head“ ist das einzige der drei Verfahren ohne Lücken.

---

## 5. Offene Punkte

- [ ] Kontrollvideo an den Lücken sichten (v. a. Farbe 05:24–05:35, S/W 04:00–04:27 und 05:11–05:44): abgewandt, Profil, verdeckt?
- [ ] Optional: Startschwelle in Farbe auf 0,3–0,4 senken und vergleichen (mehr Treffer, aber mehr Fehlalarme)
- [ ] Masken für die Evaluation erzeugen (Abschnitt 18) und mit `Evaluation.ipynb` auswerten
- [ ] Ground-Truth-Frames gezielt aus den Lücken wählen
