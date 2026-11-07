# One-time setup: creates the Python environment. Voices and auto-tag use free online services,
# so there are no voice models to download.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:PIP_CACHE_DIR = "$root\.cache\pip"

if (-not (Test-Path "$root\.venv")) { py -3.12 -m venv "$root\.venv" }
$py = "$root\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r "$root\requirements.txt"

Write-Host "Setup complete. Start the app with run.bat, then paste your free Gemini key in Settings."
