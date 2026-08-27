$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    python scripts/build_spy_dca_dashboard.py --refresh
}
finally {
    Pop-Location
}
