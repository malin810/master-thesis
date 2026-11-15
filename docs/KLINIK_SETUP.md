# Klinik-PC einrichten

Der Klinik-PC kann Pakete nicht von `pypi.org` laden (Proxy blockiert). Deshalb kommen die Python-Pakete aus einem **Paketordner** (`KlinikPC_Paket`), den du auf dem Laptop baust. Python selbst, VS Code und den Code holst du auf dem Klinik-PC direkt (python.org, GitHub, code.visualstudio.com), das geht dort.

## Auf dem Laptop

### 1. Aenderungen auf GitHub bringen
Das Notebook und die Skripte muessen auf GitHub sein (im Notebook ist `torch` optional gemacht, das braucht der Klinik-PC):
```
cd C:\Users\malin\Masterthesis\master-thesis
git status
git add .gitignore notebooks/SCRFD-10G.ipynb scripts docs/KLINIK_SETUP.md
git commit -m "Klinik-Paket: Skripte, torch optional"
git push
```

### 2. Paket bauen (mit Internet, ca. 10-15 Minuten)
```
powershell -ExecutionPolicy Bypass -File scripts\build_klinik_paket.ps1
```
Ergebnis: `C:\Users\malin\Masterthesis\KlinikPC_Paket` (neben dem Projekt, nicht im Repo), ungefaehr 190 MB:
`uv.exe`, `uv-cache.zip` (alle Pakete aus `uv.lock`, ohne torch), `models\det_10g.onnx`, `repo.zip` (Code als Reserve), `setup_klinik.ps1`, `SHA256SUMS.txt`.
Optionen: `-MitPython` (Python gleich mitpacken), `-MitTorch`.

Das Paket enthaelt **keine Patientendaten**. Das Modell darf nicht ins GitHub-Repo (InsightFace: nur Forschung).

### 3. Paket zum Klinik-PC bringen
Ordner `KlinikPC_Paket` ins Intranet oder auf USB legen.

## Auf dem Klinik-PC

### 4. Python 3.12 installieren
python.org -> Downloads -> **Python 3.12.x** (Windows installer, 64-bit). **Nicht 3.13 oder neuer**, die Pakete sind fuer 3.12 gebaut.
Im Installer "Add python.exe to PATH" ankreuzen und "Install for current user only" waehlen (kein Admin noetig).
Pruefen in einem neuen Fenster: `python --version` zeigt 3.12.x.

### 5. Code holen
```
mkdir C:\Klinik
cd C:\Klinik
git clone https://github.com/malin810/master-thesis.git
```
(Kein Git? Auf GitHub "Code -> Download ZIP" und nach `C:\Klinik\master-thesis` entpacken. Das Repo ist privat, dafuer ist eine Anmeldung noetig.)

### 6. Paket kopieren und einrichten
Ordner `KlinikPC_Paket` auf die **lokale Platte** kopieren (z. B. `C:\Temp\KlinikPC_Paket`), dann in einem normalen PowerShell-Fenster (kein Admin):
```
cd C:\Temp\KlinikPC_Paket
Unblock-File .\setup_klinik.ps1
powershell -ExecutionPolicy Bypass -File .\setup_klinik.ps1
```
Das Skript prueft die Pruefsummen, legt das Modell ab, baut `.venv` mit `uv sync --offline` (ohne Internet) und fuehrt einen Selbsttest aus. Am Ende steht `ALLES OK`.
Anderer Zielordner: `-Ziel D:\Klinik` (dann muss auch der Code dort liegen).

### 7. Notebook benutzen
1. **VS Code** installieren, Erweiterungen "Python" und "Jupyter" installieren. Der Extension-Marketplace kann blockiert sein; dann bei der IT nachfragen oder die `.vsix`-Dateien separat besorgen.
2. Ordner `C:\Klinik\master-thesis` oeffnen, Notebook `notebooks\SCRFD-10G.ipynb` oeffnen, Kernel `C:\Klinik\master-thesis\.venv\Scripts\python.exe` waehlen.
3. Video nach `C:\Klinik\master-thesis\data\raw\<Ordner>\` legen. Es bleibt lokal.
4. In der Setup-Zelle `INPUT_VIDEO` und `FACES_FILE` (neuer Name!) anpassen.
5. **Nur Abschnitte 1-11 ausfuehren** (ohne Kopfdetektor). Erst mit `max_frames=500` in der Erkennungszelle testen und die Zeit messen.

## Danach
- Vor dem Mitnehmen eines Notebooks alle Ausgaben loeschen.
- Verpixelte Videos und `.pkl`-Dateien bleiben auf dem Klinik-PC (auch sie sind aus Patientendaten abgeleitet).
- Ob Programme aus diesen Ordnern gestartet werden duerfen (Virenscanner), entscheidet die Klinik-IT. Wird `uv.exe` oder `python.exe` blockiert, bitte die IT fragen.
