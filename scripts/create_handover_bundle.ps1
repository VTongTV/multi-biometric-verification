param(
    [ValidateSet("full", "lean")]
    [string]$Profile = "full",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "handover"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$bundleName = "verify_handover_{0}_{1}.zip" -f $Profile, $timestamp
$bundlePath = Join-Path $OutputDir $bundleName
$stageDir = Join-Path $env:TEMP ("verify_handover_stage_" + [Guid]::NewGuid().ToString("N"))

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

Write-Host "Profile: $Profile" -ForegroundColor Green
Write-Host "Repository: $RepoRoot" -ForegroundColor Green
Write-Host "Bundle: $bundlePath" -ForegroundColor Green

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

$itemsToCopy = @(
    "data",
    "scripts",
    "src",
    "tests",
    "webapp",
    "weights",
    "requirements.txt",
    "start_platform.ps1",
    "train_models.ps1",
    "HANDOVER.md",
    "DIRECTORY_MAP.md",
    "implementation_plan.md",
    ".gitignore"
)

if ($Profile -eq "full") {
    $itemsToCopy += "venv"
}

Write-Step "Copying repository files to staging"
foreach ($item in $itemsToCopy) {
    $source = Join-Path $RepoRoot $item
    if (-not (Test-Path $source)) {
        continue
    }
    Copy-Item -Path $source -Destination $stageDir -Recurse -Force
}

$removePaths = @(
    ".git",
    ".pytest_cache",
    "logs",
    "outputs",
    "paper",
    "webapp\frontend\.next",
    "webapp\backend\src\api\__pycache__",
    "webapp\backend\src\api\core\__pycache__",
    "webapp\backend\src\api\endpoints\__pycache__"
)

if ($Profile -eq "lean") {
    $removePaths += @(
        "venv",
        "webapp\frontend\node_modules"
    )
}

Write-Step "Removing non-essential files from staging"
foreach ($rel in $removePaths) {
    $target = Join-Path $stageDir $rel
    if (Test-Path $target) {
        Remove-Item -Path $target -Recurse -Force
    }
}

Write-Step "Creating ZIP bundle"
if (Test-Path $bundlePath) {
    Remove-Item -Path $bundlePath -Force
}
Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $bundlePath -CompressionLevel Optimal

Remove-Item -Path $stageDir -Recurse -Force

$bundleSizeGB = [math]::Round(((Get-Item $bundlePath).Length / 1GB), 2)
Write-Host "`nBundle created: $bundlePath ($bundleSizeGB GB)" -ForegroundColor Green
