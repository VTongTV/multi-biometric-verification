param(
    [switch]$ForceRebuildVenv,
    [switch]$ForceNpmInstall,
    [switch]$SkipPython,
    [switch]$SkipNode
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsFile = Join-Path $RepoRoot "requirements.txt"
$FrontendDir = Join-Path $RepoRoot "webapp\frontend"
$EnvLocalFile = Join-Path $FrontendDir ".env.local"
$DefaultApiBase = "http://localhost:8000"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Resolve-PythonExecutable {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($version in @("-3.12", "-3", "")) {
            try {
                if ($version) {
                    $candidate = (& py $version -c "import sys; print(sys.executable)" 2>$null).Trim()
                } else {
                    $candidate = (& py -c "import sys; print(sys.executable)" 2>$null).Trim()
                }
                if ($candidate) {
                    return $candidate
                }
            } catch {
                # Try the next candidate.
            }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    return $null
}

function Test-VenvPython {
    param([string]$PythonPath)
    if (-not (Test-Path $PythonPath)) {
        return $false
    }

    try {
        & $PythonPath -c "import sys; print(sys.version)" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

Set-Location $RepoRoot
Write-Host "Repository: $RepoRoot" -ForegroundColor Green

if (-not $SkipPython) {
    $venvIsHealthy = Test-VenvPython -PythonPath $VenvPython
    $needsRebuild = $ForceRebuildVenv -or -not $venvIsHealthy

    if ($needsRebuild) {
        Write-Step "Preparing Python virtual environment"
        $basePython = Resolve-PythonExecutable
        if (-not $basePython) {
            throw "Python is not installed. Install Python 3.12+ and re-run this script."
        }

        if (Test-Path $VenvDir) {
            $backupName = "venv_backup_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
            $backupPath = Join-Path $RepoRoot $backupName
            Write-Host "Existing venv appears invalid. Moving it to $backupName" -ForegroundColor Yellow
            Move-Item -Path $VenvDir -Destination $backupPath
        }

        & $basePython -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment."
        }

        & $VenvPython -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to upgrade pip in venv."
        }

        if (Test-Path $RequirementsFile) {
            Write-Host "Installing Python dependencies from requirements.txt..." -ForegroundColor Yellow
            & $VenvPython -m pip install -r $RequirementsFile
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install requirements.txt packages."
            }
        }
    } else {
        Write-Step "Python virtual environment is already usable"
    }

    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip check reported dependency issues. Re-run with -ForceRebuildVenv if needed." -ForegroundColor Yellow
    } else {
        Write-Host "Python dependencies look consistent." -ForegroundColor Green
    }
}

if (-not $SkipNode) {
    if (-not (Test-Path $FrontendDir)) {
        throw "Frontend directory not found: $FrontendDir"
    }

    Write-Step "Checking Node.js frontend dependencies"
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) {
        Write-Host "npm not found. Install Node.js LTS to run the frontend." -ForegroundColor Yellow
    } else {
        $nodeModulesDir = Join-Path $FrontendDir "node_modules"
        $mustInstall = $ForceNpmInstall -or -not (Test-Path $nodeModulesDir)

        Push-Location $FrontendDir
        try {
            if ($mustInstall) {
                Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
                npm install
                if ($LASTEXITCODE -ne 0) {
                    throw "npm install failed."
                }
            } else {
                Write-Host "node_modules already present. Skipping npm install." -ForegroundColor Green
            }
        } finally {
            Pop-Location
        }
    }
}

Write-Step "Configuring frontend API base URL"
$apiLine = "NEXT_PUBLIC_API_BASE_URL=$DefaultApiBase"
if (-not (Test-Path $EnvLocalFile)) {
    Set-Content -Path $EnvLocalFile -Value $apiLine
    Write-Host "Created $EnvLocalFile" -ForegroundColor Green
} else {
    $content = Get-Content $EnvLocalFile
    $existingIndex = -1
    for ($i = 0; $i -lt $content.Count; $i++) {
        if ($content[$i] -match '^NEXT_PUBLIC_API_BASE_URL=') {
            $existingIndex = $i
            break
        }
    }

    if ($existingIndex -ge 0) {
        $content[$existingIndex] = $apiLine
        Set-Content -Path $EnvLocalFile -Value $content
    } else {
        Add-Content -Path $EnvLocalFile -Value $apiLine
    }
    Write-Host "Updated $EnvLocalFile" -ForegroundColor Green
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Start everything with:" -ForegroundColor Yellow
Write-Host "  .\start_platform.ps1" -ForegroundColor Yellow
