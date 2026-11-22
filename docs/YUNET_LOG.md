# YuNet – Log: Gesichtserkennung und Verpixelung

Ziel: Gesichter in Videos der Epilepsiestation mit dem leichten Gesichtsdetektor YuNet finden und verpixeln, als schneller Vergleich zu SAM 3 (siehe [`SAM3_LOG.md`](SAM3_LOG.md)).

Notebook: [`notebooks/YuNet.ipynb`](../notebooks/YuNet.ipynb). Ergebnisse liegen lokal in `data/interim/PAT_1/*.pkl` und `data/processed/PAT_1/*.mp4` (nicht in Git). Ausgewertet wurde nur über Zahlen aus den `.pkl`-Dateien plus eigene Sichtprüfung der Kontrollvideos.

---

## 1. Umgebung

| Was | Stand |
|---|---|
| Rechner | Laptop, NVIDIA RTX 3070 (YuNet läuft aber auf der **CPU**) |
| Python | 3.12.12 über `uv` |
| OpenCV | 5.0.0 (`opencv-python-headless`), YuNet über `cv2.FaceDetectorYN`. Die Warnung „Targets are not supported by the new graph engine“ ist harmlos. |
| Modell | `models/face_detection_yunet_2023mar.onnx` (232 KB) |

Testvideo und Bildaufbau wie in `SAM3_LOG.md`, Abschnitt 2: `VID_76.AVI`, 1920 × 1080, 25 fps, 8759 Frames. Zwei Ansichten: **F (Farbe)** x 0–960, volle Höhe; **SW (Schwarz-Weiß)** x 991–1909, y 10–526.

---

## 2. Versuche (chronologisch)

### 2.1 v1: ganzes Bild (`anonymize_video`, Schritt 5/6)

Erster Lauf (vor dem 20.11.2026), nur die ersten 500 Frames:

| Einstellung | Wert |
|---|---|
| Score-Schwelle | 0,3 |
| Rand | 50 % links/rechts/unten, 80 % oben |
| hold | 15 Frames (0,6 s) |
| trail | 5 Frames |
| Verpixelung | 8 Blöcke in der Breite der Box (`N_BLOCKS`) |

Ergebnis: 487 von 500 Frames mit Erkennung, 13 gehalten, 0 ohne Schutz.

**Einschränkung:** „Erkennung“ hieß hier: *irgendein* Gesicht *irgendwo* im Bild. Ob Farb- **und** S/W-Ansicht geschützt waren, war nicht messbar. Ein Treffer in Farbe hat außerdem die gehaltene Box in S/W gelöscht (hold war gemeinsam fürs ganze Bild).

### 2.2 v2: getrennte Ansichten (`anonymize_video_views`, Schritt 7/8, 22.11.2026)

Änderungen gegenüber v1:

| | v1 | v2 |
|---|---|---|
| Suche | ganzes Bild | jede Ansicht einzeln, eigener Detektor in der Größe der Ansicht |
| hold / trail | gemeinsam | pro Ansicht |
| zu große Boxen | verpixelt | verworfen, wenn breiter oder höher als 40 % der Ansicht (`MAX_FACE_FRAC = 0.4`) |
| außerhalb der Ansichten | sichtbar | schwarz |
| Ausgabe | 1 Video + CSV | `_pixel.mp4`, Kontrollvideo `_masken.mp4`, `.pkl` mit allen Boxen |

Score-Schwelle, Rand, hold, trail und Verpixelung wie in v1.

**Grund für den Größenfilter:** In der S/W-Ansicht meldet YuNet im Ausschnitt oft ein riesiges „Gesicht“ (Score 0,30–0,45, ca. 250–400 × 300–450 px). Mit trail und 50/80 % Rand wurde dadurch fast die ganze S/W-Ansicht verpixelt (Frame 2500: Überverpixelung 24-fache Kopffläche, siehe `Evaluation.ipynb`).

#### Test Frames 2400–3000 (Lauf `VID_76_yunetviews_2400_3000_2026-11-22_12-53-42`)

| | Farbe | S/W |
|---|---|---|
| Frames ohne Verpixelung | **0** | **30** (2684–2713 am Stück) |
| ohne frische Erkennung | 10 (längste Folge 5) | 89 (längste 50, Frames 2664–2713) |
| gehalten / nur trail | 10 / 0 | 54 / 5 |
| Frames mit verworfener Box | 4 | 40 |
| Gesichter pro Frame | 0: 10, 1: 159, 2: 247, 3: 116, 4: 61, 5: 6, 6: 1 | 0: 89, 1: 511 |
| Score min / median / max | 0,30 / 0,55 / 0,90 | 0,30 / 0,72 / 0,89 |

Tempo 7,2 Frames/s (mit zwei Videos schreiben). SAM 3 (`run_video_detect`) auf derselben Stelle: 0 / 0 Frames ohne Verpixelung, S/W-Kopf in allen 600 Frames frisch erkannt.

- Die Lücke 2684–2713 liegt **nach** der letzten verworfenen S/W-Box (Frame 2554), ist also ein echter Aussetzer von YuNet.
- In Farbe meldet YuNet meist 2–4 Gesichter, also mehr als Personen im Bild. Häufungen am linken Rand (x 0–100, y 400–600) und oben (x 600–900, y 0–100). Noch zu prüfen: echte Personen oder Fehlerkennungen?

#### Ganzes Video (Lauf `VID_76_yunetviews_0_8759_2026-11-22_13-03-00`)

8759 Frames in 1197 s (20 min), **7,3 Frames/s** (SAM 3: 1,25 Frames/s).

| | Farbe | S/W | SAM 3 (zum Vergleich, `SAM3_LOG.md` 3.5) |
|---|---|---|---|
| **Frames ohne Verpixelung** | **43** (0,5 %, längste Lücke 20) | **3396** (39 %, längste Lücke **397** = 16 s) | 0 / 0 |
| ohne frische Erkennung | 1207 (längste 40) | 6047 (69 %, längste 417) | 0 / 29 |
| gehalten / nur trail | 1141 / 23 | 2296 / 355 | |
| Frames mit verworfener Box | 149 | 608 | |
| Gesichter pro Frame | 0: 1207, 1: 2970, 2: 3244, 3: 960, 4: 295, 5: 69, 6: 14 | 0: 6047, 1: 2664, 2: 48 | |
| Score min / median / max | 0,30 / 0,53 / 0,93 | 0,30 / 0,71 / 0,94 | |

S/W-Frames ohne Verpixelung pro Minute: 66, 36, **931**, 491, **1003**, **869**. YuNet bricht in S/W ab ca. 02:12 ein.

Längste Lücken:
- **S/W:** 3311–3441 (02:12–02:17), 4082–4245 (02:43–02:49), 4333–4563 (02:53–03:02), 6163–6293 (04:06–04:11), 6406–6657 (04:16–04:26), 7251–7422 (04:50–04:56), **7842–8238 (05:13–05:29, 16 s)**, 8262–8393 (05:30–05:35)
- **Farbe:** 6782–6801 (04:31–04:32, 20 Frames), 7282–7298 (04:51, 17 Frames), 6208–6212, 3328. Die Stelle 04:31 ist dieselbe, an der auch SAM 3 in S/W kurz keine frische Maske hatte (Frames 6755–6792).

**Größenfilter über das ganze Video:**
- S/W: angenommene Boxen 38–206 px hoch (Median 122, 95 %: 182), verworfene 207–667 px (Median 342). Kein Abstand mehr zwischen beiden Gruppen wie im Test.
- Farbe: angenommene 27–431 px hoch (Median 97, 99 %: 316), verworfene 353–663 px.
- In 205 der 3396 ungeschützten S/W-Frames gab es eine verworfene Box. Höchstens diese gehen auf den Filter zurück, die übrigen > 3000 sind echte Aussetzer.
- **Sichtprüfung (22.11.2026):** Unter den roten (verworfenen) Boxen stecken **manchmal Gesichter oder Gesichtsteile**, aber **nicht in der Mitte** der Box. Die Boxen sind meist riesig. YuNet „spürt“ dort also ein Gesicht, lokalisiert es aber schlecht.

---

## 3. Erkenntnisse

- **Farbe:** YuNet ist brauchbar (0,5 % ungeschützt, Lücken ≤ 0,8 s), erzeugt aber viele Zusatzerkennungen (bis 6 pro Frame).
- **S/W:** YuNet ist **ungeeignet**: 39 % der Frames ungeschützt, Lücken bis 16 s. Genau diese Ansicht löst SAM 3 vollständig.
- Der Größenfilter verhindert massive Überverpixelung, verwirft aber auch schlecht lokalisierte echte Treffer. Einfaches Verwerfen ist daher für den Datenschutz nicht die beste Lösung.
- YuNet ist ca. 6-mal schneller als SAM 3 und braucht keine GPU (wichtig für den Klinik-PC).

---

## 4. Offene Punkte

- [ ] Umgang mit zu großen Boxen verbessern, statt sie zu verwerfen. Möglichkeiten:
  - auf die letzte gute Gesichtsbox (erweitert) zuschneiden, sonst ganz verpixeln (wie „clip“/„full“ bei SAM 3, `SAM3_LOG.md` 3.2 v3)
  - ganz verpixeln (sicher, aber viel Überverpixelung)
  - Filter pro Ansicht unterschiedlich einstellen
- [ ] Kontrollvideo an den S/W-Lücken sichten (v. a. 05:13–05:29): Warum findet YuNet nichts (abgewandt, Profil, dunkel)?
- [ ] Zusatzerkennungen in Farbe prüfen (linker Rand, oberer Rand): echte Personen oder Fehlerkennungen?
- [ ] Ground-Truth-Frames gezielt aus den S/W-Lücken wählen (`Evaluation.ipynb`)
- [ ] Masken für die Evaluation erzeugen (Schritt 8, letzte Zelle)
