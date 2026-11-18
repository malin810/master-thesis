# SAM 3 – Log: Kopferkennung und Verpixelung

Ziel: Köpfe in Videos der Epilepsiestation zuverlässig finden und verpixeln (später ggf. Face Swapping). Vorherige Ansätze (YuNet, SCRFD-10G/34G, YOLO11-Kopfdetektor) hatten Aussetzer, vor allem in der Schwarz-Weiß-Ansicht. Getestet wird [SAM 3](https://huggingface.co/facebook/sam3) (Segmentierung per Text-Prompt).

Notebook: [`notebooks/SAM3.ipynb`](../notebooks/SAM3.ipynb). Alle Zahlen stammen aus den dortigen Läufen. Ergebnisse liegen lokal in `data/interim/PAT_1/*.pkl` und `data/processed/PAT_1/*.mp4` (nicht in Git).

---

## 1. Umgebung (15.11.2026)

| Was | Stand |
|---|---|
| Rechner | Laptop, NVIDIA RTX 3070 (8 GB), Treiber 591.86 (CUDA 13.1), Windows 11 |
| Python | 3.12.12 über `uv` (`.python-version`) |
| PyTorch | 2.14.1+cu130 (GPU). PyPI liefert unter Windows nur die CPU-Version, daher eigene Paketquelle in `pyproject.toml` (`[[tool.uv.index]] pytorch-cu130`, `explicit = true`). `cu128` gab es nur bis torch 2.11, `cu132` (torch 2.15) bräuchte einen neueren Treiber. Deshalb Mindestversionen auf `torch>=2.14.1`, `torchvision>=0.29.1` gesenkt. |
| SAM-3-Pakete | eigene Abhängigkeitsgruppe `sam3`: `transformers` (5.23.0), `accelerate`, `huggingface_hub`, `pillow`. Installieren mit **`uv sync --group sam3`** (ein einfaches `uv sync` entfernt die Gruppe wieder). Das Klinik-Paket (`scripts/build_klinik_paket.ps1`) bleibt dadurch unverändert. |
| Modell | `facebook/sam3` (Hugging Face, zugangsbeschränkt, Lizenz von Meta), heruntergeladen nach `models/sam3/` (6,9 GB), per `.gitignore` ausgeschlossen. `uv run hf auth login`, dann `uv run hf download facebook/sam3 --local-dir models/sam3`. |
| `kernels` | Versuchsweise installiert, wieder entfernt: Das zugehörige Modul (`kernels-community/cv-utils`) gibt es nur für Linux. |

Commits: `5345fb7` (PyTorch cu130), `8b5a598` (Gruppe sam3), `ffebc30` (`models/sam3` ignorieren).

---

## 2. Testvideo und Bildaufbau

`data/raw/PAT_1/VID_76.AVI`: 1920 × 1080, 25 fps, 8759 Frames (5,8 min). Problemstelle der früheren Detektoren: Frames 8100–8250.

Das Video zeigt **zwei Kameras**. Gemessen über Farbsättigung und Helligkeit von 60 Frames (rein numerisch, ohne Bilder anzusehen):

```
x=0                        961  981                    1919
 ┌──────────────────────────┬──┬───────────────────────┐ y=0
 │                          │▓▓│   S/W-Ansicht         │
 │      Farbansicht         │▓▓│   (Fenster oben rechts)│
 │                          └──┴───────────────────────┤ y≈527
 │      Farbansicht (rechts unten nichts Relevantes)   │
 └─────────────────────────────────────────────────────┘ y=1079
                               ▓▓ = schwarzer Balken
```

Verwendete Ausschnitte (ab Lauf v2):
- **F (Farbe):** x 0–960, volle Höhe (961 × 1080)
- **SW (Schwarz-Weiß):** x 991–1909, y 10–526 (919 × 517), links/oben/rechts 10 px eingerückt
- **Alles andere wird im fertigen Video schwarz.**

---

## 3. Versuche (chronologisch)

### 3.1 Ganzes Bild, SAM 3 mit Tracking (`Sam3VideoModel`)

| Versuch | Ergebnis | Erkenntnis |
|---|---|---|
| Einzelbild, Frame 250, „face“ vs. „head“ | „head“: 2 Köpfe (Farbe 0,87, S/W 0,86). „face“: 4 Treffer, davon Duplikate | „head“ robuster als „face“ |
| `VID_76_A.mp4` (500 Frames), „face“ | 1646 s, kurzer Aussetzer am S/W-Gesicht | – |
| `VID_76_A.mp4`, „head“ | 665 s, alles verpixelt | „head“ als Prompt gewählt |
| Speicher | GPU-Speicher wächst ~10–13 MB pro Frame; ab ~6,5 GB lagert Windows aus, ein Frame dauert dann ~12 s statt ~0,6 s | **`prune_session`**: Tracker liest höchstens ~16 Frames zurück (`num_maskmem = 7`, `max_object_pointers_in_encoder = 16`, `hotstart_delay = 15`, `max_cond_frame_num = 4`), ältere Zwischenergebnisse werden gelöscht → Speicher konstant |
| VID_76, 8000–8300 | 0 von 300 Frames ohne Treffer, 1,37 Frames/s | Problemstelle von SCRFD/YOLO11 vollständig verpixelt |
| VID_76, ganzes Video | bis Frame ~1350 stabil (0,59 s/Frame), danach bis zu 11 Objekte (große Masken über ~¾ des Bildes, ganze Person, Hand), bis 10 s/Frame; bei Frame 4345 abgebrochen, 24 IDs | Das eingeblendete S/W-Fenster wird als Teil des Raums behandelt → Ansichten trennen |

### 3.2 Getrennte Ansichten, SAM 3 mit Tracking (`run_video_split`), Frames 2400–3000

Jede Ansicht hat eine eigene SAM-3-Sitzung.

| Lauf | Änderung | Farbe ohne Kopf | S/W ohne Kopf | S/W normale Masken | Tempo |
|---|---|---|---|---|---|
| v1 | F = ganzes Bild mit geschwärztem Fenster; zu große Masken (> 60 % der Ansicht) verworfen | 71 (längste Lücke 4) | 162 (längste 15) | 540 | 0,91 Frames/s |
| v2 | F = nur linke Hälfte, SW eingerückt, Rest schwarz | 8 (3) | 342 (31) | 292 | 0,86 |
| v3 | zu große Masken nicht verwerfen: auf letzte gute Kopfbox derselben ID zuschneiden („clip“), sonst ganz verpixeln („full“) | **0** | **0** (clip 511, full 208 Frames) | 292 | 0,88 |
| v4 | + Ersatz für `kernels` (NMS, Löcher füllen, Sprenkel entfernen) | **0** | **0** (clip 494, full 207 Frames) | 362 | 0,86 |

Erkenntnisse:
- Die zu großen Masken gehören zu **denselben IDs wie die Köpfe**: Die Kopfmaske „läuft aus“ über Kissen, Decke oder Körper.
- Farbansicht ab v2 gut, **S/W-Ansicht** schlecht: in v4 nur 362 von 1314 Masken normal (~28 %).
- Der Score bleibt beim Tracking auf dem Wert der ersten Erkennung stehen und sagt nichts über die aktuelle Maske aus.
- NMS-Ersatz: +24 % normale S/W-Masken, aber das Hauptproblem bleibt.

### 3.3 Stichprobe ohne Tracking (Anhang A3)

Jeder 10. Frame von 2400–3000, nur S/W-Ansicht, **Einzelbild-Modell** (`Sam3Model`):

| | ohne Tracking | mit Tracking (v4) |
|---|---|---|
| Frames mit normaler Kopfmaske | **60 von 60** (Score 0,82–0,92) | 32 von 60 |
| Frames mit zu großer Maske | **0** | 28 |

→ Die **Erkennung** von SAM 3 ist gut. Das **Tracking** trägt eine einmal ausgelaufene Maske weiter.

### 3.4 Erkennung pro Frame (`run_video_detect`), Frames 2400–3000 (17.11.2026)

Einzelbild-Modell in jedem Frame und jeder Ansicht, eigene einfache Regeln statt SAM-3-Tracking:
- Score ≥ 0,5 (`DET_THR`)
- gleiche ID, wenn sich die Box mit einem Kopf aus dem Frame davor um ≥ 30 % überlappt (`MATCH_IOU`)
- kurz nicht erkannter Kopf: letzte Maske bis 1 s weiter verpixeln („hold“, `HOLD = 25` Frames)
- Masken um 8 px vergrößern (`DILATE`)
- zu große Masken ganz verpixeln („full“)

| | Tracking (v4) | **pro Frame (v5)** |
|---|---|---|
| Frames ohne Verpixelung (F / SW) | 0 / 0 | **0 / 0** |
| S/W: Frames ohne frische Kopfmaske | 293 (längste Folge 27) | **0** |
| S/W: ausgelaufene Masken | 952 | **0** |
| Farbe: Frames ohne frische Kopfmaske | 2 | **0** |
| Notlösungen | S/W: clip 494, full 207 | F: hold 85, SW: hold 198 |
| Tempo | 0,86 Frames/s | **1,29 Frames/s** |
| GPU reserviert | 3,2 GB | 4,0 GB (zwei Modelle geladen) |

- Patientenkopf in S/W (ID 0) in **allen 600 Frames** frisch erkannt, durchgehend dieselbe ID.
- „hold“ betrifft nur kurze Nebenerkennungen (1–3 Frames), z. B. eine Pflegekraft, die mit einem kleinen Teil des Kopfes oberhalb des Patienten erscheint und sich schnell bewegt. Visuell unproblematisch (etwas mehr verpixelt).
- Sichtprüfung des Kontrollvideos: die gelben „hold“-Flächen stören nicht.

**Entscheidung:** Erkennung pro Frame (`run_video_detect`) ist das Standardverfahren. `run_video_split` bleibt als Vergleich im Notebook.

### 3.5 Ganzes VID_76 mit `run_video_detect` (17.11.2026)

Lauf `VID_76_sam3det_head_0_8759_2026-11-17_17-17-32` mit den Einstellungen aus 3.4. Ausgewertet nur über Zahlen aus der pkl-Datei.

| | Farbe | S/W |
|---|---|---|
| Frames ohne Verpixelung | **0** | **0** |
| Frames ohne frische Kopfmaske | **0** | 29 (längste Folge 4) |
| hold (Objekt-Frames) | 3295 | 2529 |
| full (zu große Maske) | 0 | 5 |
| frische Masken pro Frame | 1: 3213, 2: 4192, 3: 1312, 4: 41, 5: 1 | 0: 29, 1: 4998, 2: 3335, 3: 385, 4: 12 |
| IDs gesamt | 114 | 97 |
| Score normale Masken min / median / max | 0,50 / 0,91 / 0,96 | 0,50 / 0,88 / 0,96 |

- **Tempo:** 8759 Frames in 7026 s (1 h 57 min), **1,25 Frames/s**. GPU-Speicher über den ganzen Lauf konstant (3,51 GB belegt, 4,03 GB reserviert), also kein Speicherleck.
- **S/W ohne frische Maske:** einzelne Frames bei 03:14, 03:45 und 03:58, dazu zwei Häufungen um **04:19–04:20** (Frames 6494–6514) und **04:30–04:31** (Frames 6755–6792), jeweils höchstens 4 Frames am Stück. Durch „hold“ trotzdem verpixelt, allerdings an der zuletzt bekannten Stelle. → Im Kontrollvideo prüfen, ob der Kopf dort von der hold-Maske abgedeckt bleibt.
- **full (S/W):** 5 Masken bei 03:26, 03:45 und 04:02, alle am rechten Rand der S/W-Ansicht (x ≈ 1600–1911), hoch und schmal, Score 0,53–0,72. Wahrscheinlich eine Person am Bildrand. Die ganze Box wird verpixelt, das bedeutet höchstens zu viel Verpixelung.
- **Mehr Personen ab ca. Frame 5000 (03:20):** In Farbe hat ab dann etwa jeder dritte Frame ≥ 3 Kopfmasken (vorher fast nie). → Im Video prüfen, ob dort wirklich mehr Personen im Raum sind.
- **Kurzlebige IDs:** Farbe 40, S/W 58 IDs mit ≤ 5 frischen Frames. Ihre Scores sind deutlich niedriger (Median ≈ 0,6 statt ≈ 0,9). Vermutlich Teilansichten von Köpfen, schnell bewegte Personen oder Fehlerkennungen (z. B. Hand). Für die Anonymisierung unkritisch, weil sie nur zusätzliche Verpixelung erzeugen.
- **ID-Wechsel:** Langlebige Köpfe behalten ihre ID über Hunderte bis Tausende Frames, z. B. S/W ID 0 in Frames 0–3248 oder ID 72 in Frames 6781–8758. Danach bekommt derselbe Kopf manchmal eine neue ID, etwa wenn sich die Box zwischen zwei Frames zu wenig überlappt. Für das Verpixeln egal, für Face Swapping ggf. relevant.
- **Einschränkung:** „0 Frames ohne Verpixelung“ heißt nur, dass in jeder Ansicht mindestens eine Maske aktiv war. Ob **jeder** Kopf abgedeckt ist, zeigt erst die Sichtprüfung bzw. eine Ground-Truth-Bewertung.
- **Sichtprüfung (18.11.2026):** Im ganzen Video sind **alle Köpfe verpixelt**. Die Auffälligkeiten am rechten S/W-Rand (full-Masken, Lücken) stammen von einer Pflegekraft, die kurz am Rand auftaucht. Ihr Kopf wird dabei mehrmals kurz erkannt. Nachdem sie gegangen ist, hält „hold“ die Maske noch bis zu 1 s an der leeren Stelle. Das ist zu viel Verpixelung, aber kein Datenschutzproblem, und wird bewusst in Kauf genommen.

---

## 4. Technische Details und Abweichungen

- **Ersatz für `kernels`** (Notebook, Abschnitt 0): SAM 3 (Video) lädt für NMS und Maskenbereinigung ein Linux-Modul. Unter Windows wurden diese Schritte stillschweigend übersprungen (`transformers/models/sam3_video/modeling_sam3_video.py`, `nms_masks`, `fill_holes_in_mask_scores`). Der Ersatz bildet beide Funktionen mit PyTorch/OpenCV nach und wird zur Laufzeit in die globale Variable `cv_utils_kernel` geschrieben (Monkey Patching). Getestet gegen die SAM-3-Funktionen. **Wirkt nur beim Video-Modell (Tracking).** Das Einzelbild-Modell (`run_video_detect`) nutzt `kernels` nicht und macht in `post_process_instance_segmentation` keine NMS, nur einen Score-Schwellwert.
- **Score beim Tracking:** Sicherheit der ersten Erkennung, ändert sich danach nicht.
- **Reproduzierbarkeit:** Gleiche Eingabe ergab exakt gleiche Rohdaten (v3 und v4 vor der Änderung identisch).
- **Ausgabedateien:** Name enthält Verfahren, Prompt, Frame-Bereich, Datum und Uhrzeit, z. B. `VID_76_sam3det_head_2400_3000_2026-11-17_16-53-21_pixel.mp4` (`sam3det` = pro Frame, `sam3split` = Tracking). Speichern auch bei Abbruch (`try/finally`).
- **Kontrollvideo-Farben:** ID-Farbe = normale Kopfmaske, gelb = hold, orange = clip, magenta = full, rot dünn = ursprüngliche zu große Maske, weiß = Grenzen der Ansichten, abgedunkelt = nicht ausgewertet (im fertigen Video schwarz).
- **Datenschutz:** Patientenvideos werden lokal verarbeitet. Bilder wurden nur lokal angesehen. Notebook-Ausgaben vor jedem Commit löschen.

---

## 5. Offene Punkte

- [x] Ganzes VID_76 mit `run_video_detect` (1 h 57 min, Ergebnisse in 3.5)
- [x] Kontrollvideo des ganzen Laufs sichten: alle Köpfe verpixelt, am Rand nur kurzes Nachhalten nach Abgang der Pflegekraft (siehe 3.5)
- [ ] Optional: `DET_THR` auf 0,6 anheben und vergleichen. Das würde viele kurzlebige Erkennungen entfernen, könnte aber echte Teilköpfe verlieren.
- [ ] Weitere Videos/Situationen testen (Nacht, Anfall, schnelle Bewegung, mehrere Personen)
- [ ] Optional: einfache NMS in `run_video_detect` gegen doppelte Erkennungen desselben Kopfes
- [ ] Bewertung mit Ground Truth (annotierte Stichprobe), z. B. Recall pro Frame
- [ ] Face Swapping (z. B. GHOST): SAM-3-Kopfbox als Suchbereich für Gesichtsdetektor/Landmarks, Verpixeln als Rückfall
- [ ] Klinik-PC: kein GPU, offline → SAM 3 dort nicht sinnvoll; Verarbeitung auf GPU-Rechner klären
- [ ] `notebooks/test_pipeline.ipynb` enthält ein gespeichertes Bild aus `data/raw/test.mp4` (Inhalt unklar) → prüfen, ggf. aus Notebook und Git-Historie entfernen
- [ ] `notebooks/SAM3.ipynb` ohne Ausgaben committen
