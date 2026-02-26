param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [switch]$RunSetup
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$BackendDir = Join-Path $RepoRoot "webapp\backend"
$FrontendDir = Join-Path $RepoRoot "webapp\frontend"
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
$SetupScript = Join-Path $RepoRoot "scripts\setup_handover.ps1"
$ApiBaseUrl = "http://localhost:$BackendPort"

if ($RunSetup -and (Test-Path $SetupScript)) {
    & $SetupScript
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Python venv not found at $VenvPython" -ForegroundColor Red
    Write-Host "Run: .\scripts\setup_handover.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $BackendDir)) {
    Write-Host "Backend directory not found at $BackendDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $FrontendDir)) {
    Write-Host "Frontend directory not found at $FrontendDir" -ForegroundColor Red
    exit 1
}

Write-Host "Starting FastAPI Backend..." -ForegroundColor Green
$backendCmd = "Set-Location `"$BackendDir`"; & `"$VenvPython`" -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port $BackendPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Write-Host "Starting Next.js Frontend..." -ForegroundColor Cyan
$frontendCmd = "Set-Location `"$FrontendDir`"; `$env:NEXT_PUBLIC_API_BASE_URL = '$ApiBaseUrl'; npm run dev -- --port=$FrontendPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host "Platform Startup Initiated." -ForegroundColor Yellow
Write-Host "Backend: $ApiBaseUrl/docs"
Write-Host "Frontend: http://localhost:$FrontendPort"
