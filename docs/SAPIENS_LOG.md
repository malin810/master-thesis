# Sapiens – Log: Oberflächennormalen (nach Ponnambalam et al.)

Ziel: Das Video der Epilepsiestation (`VID_76.AVI`) so bearbeiten wie im Paper und als Ergebnis ein Video aus **Oberflächennormalen** erzeugen. Später, bei der Anfallserkennung, soll es mit den verpixelten SAM-3-Videos verglichen werden (siehe [`SAM3_LOG.md`](SAM3_LOG.md)).

Notebook: [`notebooks/Sapiens-Normalen.ipynb`](../notebooks/Sapiens-Normalen.ipynb). Ergebnisse liegen lokal in `data/interim/PAT_1/` und `data/processed/PAT_1/` (nicht in Git).

---

## 1. Quelle: das Paper

K. G. Ponnambalam, T. Ilyas, S. Sivathamboo, Z. Ge, P. Kwan, L. Kuhlmann, D. Mehta: *Privacy-Centric Seizure Detection Using Surface Normals, Pose and Segmentation Masks.* AI 2025 (38th Australasian Joint Conference on AI), LNAI 16371, S. 322–334, Springer 2026.
DOI: [10.1007/978-981-95-4972-6_25](https://doi.org/10.1007/978-981-95-4972-6_25) · Eintrag: [research.monash.edu](https://research.monash.edu/en/publications/privacy-centric-seizure-detection-using-surface-normals-pose-ands/)

Was das Paper zur Vorverarbeitung angibt (Abschnitte 3 und 4.1):

| Schritt | Paper |
|---|---|
| Eingangsvideo | 1920 × 1080, 30 fps (VSViG-Datensatz) |
| Bildrate | gleichmäßig auf **5 fps** reduziert |
| Personen finden | **YOLOv8-m**, Schwelle τ = 0,25 |
| Boxen filtern | Boxen kleiner als **50 % der Bildhöhe** verwerfen |
| Ausschnitt | Box mit **50 px Rand** („patient-centric ROI“) |
| Größe für Sapiens | auf **1024 × 768** skaliert |
| Normalen | **Sapiens-2B** (ViT-Huge), rohe Normalen pro Pixel (nicht maskiert) |
| Tempo (NVIDIA A40) | 1,50 ± 0,09 s pro Frame |
| danach | Clips à 32 Frames (6,4 s), Schritt 16, SlowFast-R50 (nicht Teil dieses Notebooks) |

Ergebnis im Paper (patientenunabhängig, LOPO): rohe Normalen 89,4 % Genauigkeit, F1 0,899. Pose nur 82,8 %. Bei Verdeckung oder mehreren Personen hielten die Normalen F1 0,833, Pose fiel auf 0.

---

## 2. Modell (18.11.2026)

**Sapiens** von Meta: Khirodkar et al., *Sapiens: Foundation for Human Vision Models*, ECCV 2024.
Paper: [arXiv 2408.12569](https://arxiv.org/abs/2408.12569) · Code: [github.com/facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) · Alle Modelle: [huggingface.co/facebook/sapiens](https://huggingface.co/facebook/sapiens)

| Was | Stand |
|---|---|
| Modellkarte | [facebook/sapiens-normal-2b-torchscript](https://huggingface.co/facebook/sapiens-normal-2b-torchscript) |
| Datei | `sapiens_2b_normal_render_people_epoch_70_torchscript.pt2` (8,7 GB, float32) |
| Eckdaten laut Modellkarte | ViT, 2,163 Mrd. Parameter, 48 Schichten, 32 Heads, Patch 16 × 16, Eingabe **1024 × 768 (H × B)**, vortrainiert auf 300 Mio. Menschenbildern |
| Lizenz | **CC BY-NC 4.0** (nur nicht-kommerziell, Namensnennung). Darf nicht ins Repo: `models/sapiens/` steht in der `.gitignore`. |
| Ablage | `models/sapiens/` |
| Herunterladen | `uv run python -c "from huggingface_hub import hf_hub_download as d; d('facebook/sapiens-normal-2b-torchscript', 'sapiens_2b_normal_render_people_epoch_70_torchscript.pt2', local_dir='models/sapiens')"` (kein Login nötig) |

**Verworfen: bfloat16-Variante** [facebook/sapiens-normal-2b-bfloat16](https://huggingface.co/facebook/sapiens-normal-2b-bfloat16) (`..._bfloat16.pt2`, 4,4 GB). Sie wäre kleiner gewesen, lässt sich mit torch 2.14.1 aber nicht laden:
`RuntimeError: Serialized version ['5', '1'] does not match our current schema version (8, 20).`
Sie wurde mit einer alten PyTorch-Version per `torch.export` gespeichert. Die Datei kann gelöscht werden.

Plan: TorchScript-Variante mit `torch.jit.load(..., map_location="cpu")` laden, dann auf bfloat16 umwandeln (~4,4 GB GPU) und auf die GPU schieben. In float32 (8,7 GB) passt sie nicht auf die RTX 3070 (8 GB). RAM: 15,9 GB gesamt.

### Vor- und Nachverarbeitung (wie in der offiziellen Demo)

Quelle: [`lite/demo/vis_normal.py`](https://github.com/facebookresearch/sapiens/blob/main/lite/demo/vis_normal.py) im Sapiens-Repo.

- Bild auf 1024 (Höhe) × 768 (Breite) skalieren
- normieren mit Mittelwert `[123.5, 116.5, 103.5]` und Standardabweichung `[58.5, 57.0, 57.5]` (pro Kanal)
- Ausgabe (3 Kanäle = x, y, z) bilinear auf die Originalgröße zurückskalieren, dann jeden Pixelvektor auf Länge 1 normieren (+1e-5 gegen Division durch 0)
- Darstellung: Werte von [-1, 1] auf [0, 255], Kanalreihenfolge umdrehen (für OpenCV/BGR)

---

## 3. Abweichungen vom Paper (geplant)

| Punkt | Paper | hier | Grund |
|---|---|---|---|
| Bildrate Eingang | 30 fps → 5 fps | 25 fps → 5 fps (jeder 5. Frame) | VID_76 hat 25 fps |
| Zwei Kameras | – | Farb- und S/W-Ansicht getrennt (wie in `SAM3.ipynb`), jede Ansicht ein eigener Durchlauf durch Sapiens | VID_76 enthält ein eingeblendetes S/W-Fenster |
| Personendetektor + Ausschnitt | YOLOv8-m, Boxen < 50 % Höhe weg, 50 px Rand | **weggelassen**, ganze Ansicht | Die Kameras sind bereits auf das Patientenbett gerichtet (Entscheidung 18.11.2026). Vorteil: fester Ausschnitt, das Normalenvideo „zoomt“ nicht von Frame zu Frame, und es gibt keinen Ausfall, wenn unter der Decke niemand erkannt wird. |
| Skalieren auf 1024 × 768 | direkt skaliert | erst oben/unten **schwarz auffüllen** auf 3 : 4 (hochkant), dann skalieren, danach den Rand wieder abschneiden | Beide Ansichten sind breiter als 3 : 4 (Farbe 961 × 1080, S/W 919 × 517). Direktes Skalieren würde vor allem die S/W-Ansicht stark stauchen. |
| Modellformat | nicht angegeben | TorchScript, in bfloat16 umgewandelt | bfloat16-Datei nicht ladbar, float32 zu groß für 8 GB |

---

## 4. Tests

| Datum | Test | Ergebnis |
|---|---|---|
| 18.11.2026 | bfloat16-Variante laden (torch 2.14.1) | Fehler: altes Speicherformat, siehe oben |
| 18.11.2026 | TorchScript-Variante laden (`torch.jit.load`, CPU), in bfloat16 umwandeln, auf GPU; Zufallsbild 1024 × 768, 4 Durchläufe | läuft. Laden 6,2 s. GPU: 4,40 GB nach dem Laden, **7,28 GB Spitze** (von 8 GB). Pro Bild 3,0 s beim ersten Mal, danach **2,2 s**. Ausgabe 3 × 512 × 384 (halbe Auflösung), keine NaN. |
| 18.11.2026 | Notebook-Funktionen mit Zufallsbildern in Größe der Ansichten (1080 × 961, 517 × 919), ohne Patientendaten | Ausgabegröße stimmt mit der Ansicht überein, Normalen haben Länge 1. Bewegungsmaß zuerst mit `acos`: 0,37° für zwei identische Karten (Rundungsfehler), deshalb auf `atan2` umgestellt: jetzt 0° (gleich) bzw. 180° (entgegengesetzt). |

**Folgerungen:**
- SAM 3 (~1,7 GB) passt nicht gleichzeitig auf die GPU. Für das Normalen-Notebook den SAM-3-Kernel vorher beenden.
- Geschätzte Dauer für VID_76: 8759 Frames / 5 = 1752 Frames × 2 Ansichten × 2,2 s ≈ **2,1 h**.
- Ob die Umwandlung auf bfloat16 die Werte verfälscht, ist noch nicht geprüft. Ein Vergleich mit float32 ginge nur auf der CPU (8,7 GB RAM).
