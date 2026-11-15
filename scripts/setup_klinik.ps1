# setup_klinik.ps1
# Laufzeit: auf dem KLINIK-PC, OHNE Internet. Liegt im Paketordner (KlinikPC_Paket).
#
# Voraussetzungen:
#   * Python 3.12 ist installiert (NICHT 3.13 oder neuer - die Pakete im Paket sind fuer 3.12 gebaut),
#     ausser das Paket enthaelt python.zip (dann wird dieses Python benutzt).
#   * Den Projektordner holst du von GitHub nach <Ziel>\master-thesis (git clone oder ZIP).
#     Fehlt er und das Paket enthaelt repo.zip, wird der Code daraus entpackt.
#
# Das Skript:  prueft Dateien -> entpackt den Paket-Cache -> legt das Modell ab -> baut .venv (offline) -> Selbsttest.
#
# Aufruf (PowerShell, normales Fenster, kein Admin noetig):
#   powershell -ExecutionPolicy Bypass -File .\setup_klinik.ps1
#   powershell -ExecutionPolicy Bypass -File .\setup_klinik.ps1 -Ziel D:\Klinik
# Falls Windows das Skript blockiert (aus dem Intranet kopiert):  Unblock-File .\setup_klinik.ps1

param(
    [string]$Ziel = "C:\Klinik"
)

$ErrorActionPreference = "Stop"
$Paket = $PSScriptRoot

function Run($exe, $argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "Fehler bei: $exe $($argList -join ' ') (Code $LASTEXITCODE)" }
}

function Extract-Zip($zip, $dest) {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        Run "tar.exe" @("-x", "-f", $zip, "-C", $dest)
    } else {
        Expand-Archive -Path $zip -DestinationPath $dest -Force
    }
}

Write-Host "Paket: $Paket"
Write-Host "Ziel : $Ziel"
if ($Ziel.StartsWith("\\")) { throw "Ziel darf kein Netzlaufwerk sein (zu langsam). Bitte lokalen Pfad waehlen, z. B. C:\Klinik" }

# 1. Pruefsummen kontrollieren
$sumFile = Join-Path $Paket "SHA256SUMS.txt"
if (-not (Test-Path $sumFile)) { throw "SHA256SUMS.txt fehlt im Paketordner" }
$sums = @{}
foreach ($line in Get-Content $sumFile) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split "\s+", 2
    $expected = $parts[0]; $name = $parts[1].Trim()
    $sums[$name] = $expected
    $file = Join-Path $Paket $name
    if (-not (Test-Path $file)) { throw "Datei fehlt im Paket: $name" }
    $actual = (Get-FileHash -Algorithm SHA256 $file).Hash
    if ($actual -ne $expected) { throw "Pruefsumme stimmt nicht: $name (Datei beim Kopieren beschaedigt? Bitte neu kopieren)" }
}
Write-Host "1/5 Pruefsummen ok"

# 2. Entpacken / bereitstellen
New-Item -ItemType Directory -Force -Path $Ziel | Out-Null

function Ensure-Extracted($zipName, $destName) {
    # entpackt neu, wenn das Zip neu ist oder sich seit dem letzten Mal geaendert hat (Paket-Update)
    $dest = Join-Path $Ziel $destName
    $marker = Join-Path $Ziel ($destName + ".sha256")
    $want = $sums[$zipName]
    if ((Test-Path $dest) -and (Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $want)) {
        Write-Host "  $destName ist aktuell"
        return
    }
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Extract-Zip (Join-Path $Paket $zipName) $dest
    Set-Content -Path $marker -Value $want -Encoding ASCII
    Write-Host "  $destName entpackt"
}

$mitPython = Test-Path (Join-Path $Paket "python.zip")
if ($mitPython) { Ensure-Extracted "python.zip" "python" }
Ensure-Extracted "uv-cache.zip" "uv-cache"
Copy-Item (Join-Path $Paket "uv.exe") (Join-Path $Ziel "uv.exe") -Force
$uv = Join-Path $Ziel "uv.exe"

$projekt = Join-Path $Ziel "master-thesis"
if (Test-Path (Join-Path $projekt "pyproject.toml")) {
    Write-Host "Projektordner gefunden (wird NICHT ueberschrieben, Daten bleiben erhalten): $projekt"
} elseif (Test-Path (Join-Path $Paket "repo.zip")) {
    Extract-Zip (Join-Path $Paket "repo.zip") $projekt
    Write-Host "Code aus repo.zip entpackt: $projekt"
} else {
    throw "Projektordner fehlt: bitte zuerst  git clone https://github.com/malin810/master-thesis.git  nach $projekt  (oder als ZIP von GitHub entpacken)"
}
$modellZiel = Join-Path $projekt "models\buffalo_l\buffalo_l"
New-Item -ItemType Directory -Force -Path $modellZiel | Out-Null
Copy-Item (Join-Path $Paket "models\det_10g.onnx") (Join-Path $modellZiel "det_10g.onnx") -Force

# Passt uv.lock im Projekt zu dem, woraus das Paket gebaut wurde?
$lockSumFile = Join-Path $Paket "UV_LOCK_SHA256.txt"
if (Test-Path $lockSumFile) {
    $lockText = (Get-Content (Join-Path $projekt "uv.lock") -Raw) -replace "`r`n", "`n"
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $lockHash = ([System.BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($lockText)))).Replace("-", "")
    if ($lockHash -ne (Get-Content $lockSumFile -Raw).Trim()) {
        Write-Host "WARNUNG: uv.lock im Projekt ist ein anderer Stand als beim Bauen des Pakets. Es koennen Pakete fehlen." -ForegroundColor Yellow
    }
}
Write-Host "2/5 Bereitgestellt (Pakete, Code, Modell)"

# 3. Umgebung bauen - nur aus dem Paket, ohne Internet
$env:UV_CACHE_DIR = Join-Path $Ziel "uv-cache"
$env:UV_PYTHON_DOWNLOADS = "never"
$env:UV_LINK_MODE = "copy"
if ($mitPython) {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Ziel "python"
    $env:UV_PYTHON_PREFERENCE = "only-managed"
} else {
    $env:UV_PYTHON_PREFERENCE = "only-system"
    $ErrorActionPreference = "Continue"
    $pyFound = & $uv python find 3.12 2>$null
    $found = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"
    if (-not $found) {
        throw "Python 3.12 nicht gefunden. Bitte Python 3.12 (nicht 3.13) von python.org installieren (nur fuer aktuellen Benutzer, 'Add python.exe to PATH') und ein NEUES PowerShell-Fenster oeffnen."
    }
    Write-Host "Python gefunden: $pyFound"
}
$syncArgs = @((Get-Content (Join-Path $Paket "SYNC_ARGS.txt") -Raw).Trim() -split "\s+") + @("--offline")
Push-Location $projekt
try { Run $uv $syncArgs } finally { Pop-Location }
Write-Host "3/5 Umgebung gebaut (.venv)"

# 4. Selbsttest
$py = Join-Path $projekt ".venv\Scripts\python.exe"
$test = @"
import sys, onnxruntime as ort, cv2, numpy as np
print('Python      :', sys.version.split()[0], '|', sys.executable)
print('onnxruntime :', ort.__version__)
print('OpenCV      :', cv2.__version__)
print('numpy       :', np.__version__)
s = ort.InferenceSession(r'$modellZiel\det_10g.onnx', providers=['CPUExecutionProvider'])
print('Modell      : geladen,', len(s.get_outputs()), 'Ausgaenge (erwartet 9)')
print('ALLES OK')
"@
$testFile = Join-Path $Ziel "_selbsttest.py"
Set-Content -Path $testFile -Value $test -Encoding ASCII
Run $py @($testFile)
Remove-Item $testFile -Force
Write-Host "4/5 Selbsttest bestanden"

Write-Host ""
Write-Host "5/5 FERTIG. Naechste Schritte:"
Write-Host "  Projektordner : $projekt"
Write-Host "  Python        : $py   (in VS Code als Kernel/Interpreter auswaehlen)"
Write-Host "  Notebook      : $projekt\notebooks\SCRFD-10G.ipynb  (Abschnitte 1-11; in der Setup-Zelle INPUT_VIDEO setzen)"
Write-Host "  Video ablegen : $projekt\data\raw\<Ordner>\<Video>   (bleibt lokal, nicht ins Intranet)"
