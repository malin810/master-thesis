# build_klinik_paket.ps1
# Laufzeit: auf dem LAPTOP, mit Internet.
# Baut einen Ordner, der alles enthaelt, was der Klinik-PC nicht selbst herunterladen kann:
#   uv.exe, alle Pakete aus uv.lock (uv-Cache), Modell, Code (repo.zip), Setup-Skript; optional Python (-MitPython).
# Auf dem Klinik-PC wird dann setup_klinik.ps1 aus dem Paketordner gestartet (ohne Internet).
#
# Aufruf (Projektordner = master-thesis):
#   powershell -ExecutionPolicy Bypass -File scripts\build_klinik_paket.ps1
# Optionen:
#   -Paket <Ordner>   Zielordner (Standard: ..\KlinikPC_Paket neben dem Projekt)
#   -MitTorch         torch/torchvision mit ins Paket (Standard: ohne, die Gesichts-Kette braucht sie nicht)
#   -MitPython        Python (Standalone) mit ins Paket (Standard: nein, Python 3.12 wird auf dem Klinik-PC von Hand installiert)

param(
    [string]$Paket = "",
    [switch]$MitTorch,
    [switch]$MitPython
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Paket) { $Paket = Join-Path (Split-Path -Parent $Repo) "KlinikPC_Paket" }

function Run($exe, $argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "Fehler bei: $exe $($argList -join ' ') (Code $LASTEXITCODE)" }
}

function Make-Zip($zipPath, $quelle) {
    # bsdtar (tar.exe in Windows 10/11) ist schnell und kommt mit vielen kleinen Dateien klar
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        Run "tar.exe" @("-a", "-c", "-f", $zipPath, "-C", $quelle, ".")
    } else {
        Compress-Archive -Path (Join-Path $quelle "*") -DestinationPath $zipPath -CompressionLevel Optimal
    }
}

Write-Host "Paketordner: $Paket"
$tmpPython = Join-Path $Paket "_tmp_python"
$tmpCache  = Join-Path $Paket "_tmp_cache"
$tmpBuild  = Join-Path $Paket "_tmp_build"
foreach ($d in @($tmpPython, $tmpCache, $tmpBuild)) { if (Test-Path $d) { Remove-Item $d -Recurse -Force } }
New-Item -ItemType Directory -Force -Path $Paket | Out-Null
foreach ($f in @("python.zip", "uv-cache.zip", "repo.zip", "uv.exe", "SYNC_ARGS.txt", "UV_LOCK_SHA256.txt", "SHA256SUMS.txt", "setup_klinik.ps1", "ANLEITUNG.md")) {
    $p = Join-Path $Paket $f; if (Test-Path $p) { Remove-Item $p -Force }
}

# 1. uv.exe
$uvExe = (Get-Command uv -ErrorAction Stop).Source
Copy-Item $uvExe (Join-Path $Paket "uv.exe")
Write-Host "1/7 uv.exe kopiert: $uvExe"

# 2. Python-Standalone von uv herunterladen (Internet noetig)
$env:UV_PYTHON_INSTALL_DIR = $tmpPython
$env:UV_PYTHON_PREFERENCE = "only-managed"
Run "uv" @("python", "install", "3.12")
# nur den echten Ordner mitnehmen, NICHT die Verknuepfung cpython-3.12-... (absoluter Pfad, waere kaputt)
$pyDir = Get-ChildItem $tmpPython -Directory | Where-Object { $_.Name -like "cpython-3.12.*" } | Select-Object -First 1
if (-not $pyDir) { throw "Python-Ordner nicht gefunden in $tmpPython" }
$pyStage = Join-Path $Paket "_tmp_pystage"
if (Test-Path $pyStage) { Remove-Item $pyStage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $pyStage | Out-Null
Copy-Item $pyDir.FullName (Join-Path $pyStage $pyDir.Name) -Recurse
if ($MitPython) {
    Make-Zip (Join-Path $Paket "python.zip") $pyStage
    Write-Host "2/7 Python: $($pyDir.Name) -> python.zip"
} else {
    Write-Host "2/7 Python wird NICHT ins Paket gepackt (nur zum Bauen benutzt). Auf dem Klinik-PC Python 3.12 selbst installieren."
}
Remove-Item $pyStage -Recurse -Force

# 3. Pakete in einen eigenen uv-Cache laden (genau die aus uv.lock)
New-Item -ItemType Directory -Force -Path $tmpBuild | Out-Null
foreach ($f in @("pyproject.toml", "uv.lock", ".python-version")) { Copy-Item (Join-Path $Repo $f) $tmpBuild }
$syncArgs = @("sync", "--frozen", "--no-install-project", "--python", "3.12")
if (-not $MitTorch) { $syncArgs += @("--no-install-package", "torch", "--no-install-package", "torchvision") }
$env:UV_CACHE_DIR = $tmpCache
$env:UV_LINK_MODE = "copy"
Push-Location $tmpBuild
try { Run "uv" $syncArgs } finally { Pop-Location }
# dieselben Argumente (plus --offline) nutzt das Setup auf dem Klinik-PC
Set-Content -Path (Join-Path $Paket "SYNC_ARGS.txt") -Value ($syncArgs -join " ") -Encoding ASCII
Make-Zip (Join-Path $Paket "uv-cache.zip") $tmpCache
Write-Host "3/7 Pakete geladen und in uv-cache.zip gepackt (mit torch: $($MitTorch.IsPresent))"

# uv.lock-Pruefsumme (CRLF/LF egal), fuer den Abgleich auf dem Klinik-PC
$lockText = (Get-Content (Join-Path $Repo "uv.lock") -Raw) -replace "`r`n", "`n"
$sha = [System.Security.Cryptography.SHA256]::Create()
$lockHash = ([System.BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($lockText)))).Replace("-", "")
Set-Content -Path (Join-Path $Paket "UV_LOCK_SHA256.txt") -Value $lockHash -Encoding ASCII

# 4. Modell
$model = Join-Path $Repo "models\buffalo_l\buffalo_l\det_10g.onnx"
if (-not (Test-Path $model)) { throw "Modell fehlt: $model" }
New-Item -ItemType Directory -Force -Path (Join-Path $Paket "models") | Out-Null
Copy-Item $model (Join-Path $Paket "models\det_10g.onnx")
Write-Host "4/7 Modell kopiert"

# 5. Code: alle von Git verfolgten und neuen, nicht ignorierten Dateien (inkl. noch nicht committeter Aenderungen)
$list = Join-Path $Paket "_tmp_repo_files.txt"
Push-Location $Repo
try {
    # Nur das Noetige fuer den Klinik-PC (keine PDFs/PowerPoints, keine anderen Notebooks)
    $alle = & git -c core.quotepath=off ls-files --cached --others --exclude-standard
    $nimm = @("pyproject.toml", "uv.lock", ".python-version", ".gitignore", "README.md",
              "notebooks/SCRFD-10G.ipynb", "docs/KLINIK_SETUP.md", "models/models_go_here.txt")
    $dateien = $alle | Where-Object {
        ($nimm -contains $_) -or $_.StartsWith("src/") -or $_.StartsWith("scripts/") -or ($_ -like "data/*README.md")
    }
    $dateien | Set-Content -Path $list -Encoding ASCII
} finally { Pop-Location }
Write-Host ("    Code-Archiv: {0} Dateien" -f @($dateien).Count)
Run "tar.exe" @("-a", "-c", "-f", (Join-Path $Paket "repo.zip"), "-C", $Repo, "-T", $list)
Remove-Item $list -Force
Write-Host "5/7 Code gepackt (repo.zip)"

# 6. Setup-Skript und Anleitung
Copy-Item (Join-Path $PSScriptRoot "setup_klinik.ps1") (Join-Path $Paket "setup_klinik.ps1")
$doc = Join-Path $Repo "docs\KLINIK_SETUP.md"
if (Test-Path $doc) { Copy-Item $doc (Join-Path $Paket "ANLEITUNG.md") }

# 7. Pruefsummen und Aufraeumen
foreach ($d in @($tmpPython, $tmpCache, $tmpBuild)) { if (Test-Path $d) { Remove-Item $d -Recurse -Force } }
$sumFile = Join-Path $Paket "SHA256SUMS.txt"
$lines = @()
foreach ($f in @("uv.exe", "python.zip", "uv-cache.zip", "repo.zip", "SYNC_ARGS.txt", "UV_LOCK_SHA256.txt", "models\det_10g.onnx", "setup_klinik.ps1")) {
    if (-not (Test-Path (Join-Path $Paket $f))) { continue }
    $h = (Get-FileHash -Algorithm SHA256 (Join-Path $Paket $f)).Hash
    $lines += "$h  $f"
}
Set-Content -Path $sumFile -Value $lines -Encoding ASCII
Write-Host "6/7 Pruefsummen: SHA256SUMS.txt"

$sizes = Get-ChildItem $Paket -Recurse -File | Measure-Object -Property Length -Sum
Write-Host ""
Write-Host ("7/7 FERTIG. Paket: {0} Dateien, {1:N0} MB" -f $sizes.Count, ($sizes.Sum / 1MB))
Get-ChildItem $Paket -Recurse -File | ForEach-Object { "{0,10:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Substring($Paket.Length + 1) }
