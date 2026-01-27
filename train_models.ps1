# Biometric Model Training Script
# Live Dashboard and Extensive Logging

param(
    [int]$Option = 0,
    [int]$Epochs = 60,
    [int]$BatchSize = 32,
    [float]$SampleFraction = 1.0
)

$ErrorActionPreference = "Stop"
# Threading config
$env:OMP_NUM_THREADS = 4
$env:MKL_NUM_THREADS = 4
$env:OPENBLAS_NUM_THREADS = 4
$env:VECLIB_MAXIMUM_THREADS = 4
$env:NUMEXPR_NUM_THREADS = 4
$Script:LogDir = Join-Path $PSScriptRoot "logs"
$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

# Ensure Log Directory Exists
if (-not (Test-Path $Script:LogDir)) {
    New-Item -ItemType Directory -Path $Script:LogDir | Out-Null
}

function Show-Dashboard {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$LogFile,
        [string]$Title,
        [string]$ModelName
    )

    $startTime = Get-Date
    Write-Host "[DEBUG] Entering Dashboard Loop..." -ForegroundColor DarkGray
    
    # Initialize best metrics tracking
    $script:bestAcc = 0.0
    $script:minLoss = 9999.0
    
    # Initialize ETA session tracking
    $script:sessionStartRecorded = $false
    $script:sessionStartEpoch = 0
    $script:sessionStartBatch = 0
    $script:sessionStartTime = 0

    while (-not $Process.HasExited) {
        # 1. Get Hardware Metrics
        try {
            $cpuObj = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue
            $cpu = $cpuObj.CounterSamples.CookedValue
        }
        catch {
            $cpu = 0
        }

        # RAM
        try {
            $memCounter = New-Object System.Diagnostics.PerformanceCounter("Memory", "Available MBytes")
            $freeMB = $memCounter.NextValue()
            $ramDisplay = "Free: $([math]::Round($freeMB/1024, 2)) GB"
        }
        catch {
            $ramDisplay = "N/A"
        }

        # GPU (NVIDIA-SMI) - Only query every 5 seconds
        if ($null -eq $script:gpuCounter) { $script:gpuCounter = 0 }
        
        if ($script:gpuCounter % 5 -eq 0) {
            try {
                $gpuResults = & nvidia-smi "--query-gpu=utilization.gpu,memory.used,memory.total" "--format=csv,noheader,nounits"
                $gpuData = $gpuResults | Where-Object { $_ -match "\d" -and $_ -match "," } | Select-Object -First 1
                if ($gpuData) {
                    $gpuParts = $gpuData -split ','
                    if ($gpuParts.Count -ge 3) {
                        $gpuUtil = $gpuParts[0].Trim()
                        $gpuMemUsed = $gpuParts[1].Trim()
                        $gpuMemTotal = $gpuParts[2].Trim()
                        $script:lastGpuStr = "$gpuUtil% ($gpuMemUsed / $gpuMemTotal MiB)"
                    }
                    else {
                        $script:lastGpuStr = "Format Error"
                    }
                }
                else {
                    $script:lastGpuStr = "No Output"
                }
            }
            catch {
                $script:lastGpuStr = "Not Detected"
            }
        }
        $script:gpuCounter++
        $gpuStr = $script:lastGpuStr

        # 2. Parse progress from recent logs for ETA
        $currentEpoch = 0
        $totalEpochs = $Epochs
        $currentBatch = 0
        $totalBatches = 1
        $currentLoss = 0.0
        
        if (Test-Path $LogFile) {
            try {
                $lastLines = Get-Content $LogFile -Tail 5 -ErrorAction SilentlyContinue
                foreach ($line in $lastLines) {
                    if ($line -match "Epoch\s+(\d+)/(\d+)\s+Batch\s+(\d+)/(\d+)\s+Loss:\s+([\d.]+)") {
                        $currentEpoch = [int]$matches[1]
                        $totalEpochs = [int]$matches[2]
                        $currentBatch = [int]$matches[3]
                        $totalBatches = [int]$matches[4]
                        $currentLoss = [float]$matches[5]
                    }
                }
            }
            catch { }
        }
        
        # Calculate ETA using Delta logic
        $etaStr = "Calculating..."
        
        if ($currentEpoch -gt 0 -and $totalEpochs -gt 0) {
            if (-not $script:sessionStartRecorded) {
                $script:sessionStartEpoch = $currentEpoch
                $script:sessionStartBatch = $currentBatch
                $script:sessionStartTime = Get-Date
                $script:sessionStartRecorded = $true
            }
            
            $calcProgress = {
                param($e, $b, $tb)
                if ($tb -eq 0) { return ($e - 1) }
                return ($e - 1) + ($b / $tb)
            }
            
            $currentProgressVal = & $calcProgress $currentEpoch $currentBatch $totalBatches
            $startProgressVal = & $calcProgress $script:sessionStartEpoch $script:sessionStartBatch $totalBatches
            
            $progressDelta = $currentProgressVal - $startProgressVal
            $timeDelta = (Get-Date) - $script:sessionStartTime
            
            if ($progressDelta -gt 0.001 -and $timeDelta.TotalSeconds -gt 15) {
                $epochsPerSecond = $progressDelta / $timeDelta.TotalSeconds
                $remainingEpochs = $totalEpochs - $currentProgressVal
                
                if ($remainingEpochs -gt 0) {
                    $secondsLeft = $remainingEpochs / $epochsPerSecond
                    $eta = [TimeSpan]::FromSeconds($secondsLeft)
                    $completionTime = (Get-Date).AddSeconds($secondsLeft)
                    $etaStr = "$($eta.ToString('hh\:mm\:ss')) (@ $($completionTime.ToString('HH:mm')))"
                }
                else {
                    $etaStr = "Finishing..."
                }
            }
        }

        # 3. Draw Dashboard
        $duration = (Get-Date) - $startTime
        
        Clear-Host
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "  Biometric Training Dashboard - $ModelName" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  [Status]" -ForegroundColor White
        Write-Host "  State: " -NoNewline; Write-Host "RUNNING" -ForegroundColor Green
        Write-Host "  Time Elapsed: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
        Write-Host "  ETA:          " -NoNewline; Write-Host "$etaStr" -ForegroundColor Magenta
        
        # Track best/min values
        if ($currentLoss -gt 0 -and $currentLoss -lt $script:minLoss) {
            $script:minLoss = $currentLoss
        }
        
        # Parse recent logs for Validation Accuracy
        foreach ($line in $recentLogs) {
            if ($line -match "Val Acc:\s+([\d.]+)%") {
                $acc = [float]$matches[1]
                if ($acc -gt $script:bestAcc) { $script:bestAcc = $acc }
            }
            if ($line -match "Loss:\s+([\d.]+)") {
                $l = [float]$matches[1]
                if ($l -gt 0 -and $l -lt $script:minLoss) { $script:minLoss = $l }
            }
        }

        Write-Host "  Progress:     Epoch $currentEpoch/$totalEpochs | Batch $currentBatch/$totalBatches | Loss: $("{0:F4}" -f $currentLoss)" -ForegroundColor Cyan
        Write-Host "  Stats:        Best Acc: $("{0:F2}" -f $script:bestAcc)% | Min Loss: $("{0:F4}" -f $script:minLoss)" -ForegroundColor Cyan
        Write-Host "  Log File:     $LogFile" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  CPU Usage: $([math]::Round($cpu, 1))%" -ForegroundColor Green
        if ($ramDisplay) {
            Write-Host "  RAM Usage: $ramDisplay" -ForegroundColor Green
        }
        Write-Host "  GPU Usage: $gpuStr" -ForegroundColor Magenta
        Write-Host ""
        Write-Host "  [Recent Logs]" -ForegroundColor White
        Write-Host "  ------------------------------------------------------------" -ForegroundColor DarkGray
        
        # Custom robust file reading
        $recentLogs = @()
        if (Test-Path $LogFile) {
            try {
                $fs = [System.IO.File]::Open($LogFile, 'Open', 'Read', 'ReadWrite')
                $sr = New-Object System.IO.StreamReader($fs)
                
                if ($fs.Length -gt 10000) {
                    $null = $fs.Seek(-10000, [System.IO.SeekOrigin]::End)
                }
                
                $content = $sr.ReadToEnd()
                $lines = $content -split "`r`n|`n"
                $recentLogs = $lines | Select-Object -Last 30 | Where-Object { $_ -ne "" }
                
                $sr.Close()
                $fs.Close()
            }
            catch {
                $recentLogs = @("Error reading log: $_")
            }
        }

        if ($recentLogs) {
            $recentLogs | ForEach-Object { 
                if ($_ -match "Error" -or $_ -match "Exception") {
                    Write-Host "  $_" -ForegroundColor Red
                }
                elseif ($_ -match "Epoch" -or $_ -match "Accuracy") {
                    Write-Host "  $_" -ForegroundColor Cyan
                }
                else {
                    Write-Host "  $_" -ForegroundColor Gray 
                }
            }
        }
        Write-Host "  ------------------------------------------------------------" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Press Ctrl+C to stop script (Process ID: $($Process.Id))" -ForegroundColor DarkGray

        if ($Process.HasExited) { break }
    }

    # Final summary after exit
    $exitCode = $Process.ExitCode
    Clear-Host
    $endColor = "Red"
    if ($exitCode -eq 0) { $endColor = "Green" }
    
    Clear-Host
    Write-Host "============================================================" -ForegroundColor $endColor
    Write-Host "  Training Finished - $ModelName" -ForegroundColor $endColor
    Write-Host "============================================================" -ForegroundColor $endColor
    Write-Host "  Exit Code: $exitCode"
    Write-Host "  Total Time: $(((Get-Date) - $startTime).ToString('hh\:mm\:ss'))"
    Write-Host "  Log File: $LogFile"
    Write-Host ""
    if ($exitCode -ne 0) {
        Write-Host "  [ERROR] Process failed. Checking last logs:" -ForegroundColor Red
        Get-Content $LogFile -Tail 20 | Write-Host -ForegroundColor Red
    }
    Write-Host ""
}

function Run-TrainingStep {
    param(
        [string]$ScriptPath,
        [string]$ModelName,
        [string]$ExtraArgs = ""
    )
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $Script:LogDir "train_${ModelName}_${timestamp}.log"
    
    # Simplify argument passing: Use only specific extra args
    # This ensures python config files are respected for epochs/batch_size
    $FinalArgs = "$ExtraArgs"
    
    Write-Host "Starting $ModelName training..." -ForegroundColor Yellow
    if ($FinalArgs) {
        Write-Host "Arguments: $FinalArgs" -ForegroundColor DarkGray
    } else {
        Write-Host "Arguments: (Using defaults from config.py)" -ForegroundColor DarkGray
    }
    
    $pythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (-not (Test-Path $pythonPath)) {
        Write-Host "Error: Python not found at $pythonPath" -ForegroundColor Red
        return
    }

    $pythonArgs = "-u -m $ScriptPath $FinalArgs"
    $cmdArgs = "/c echo [DASHBOARD] Starting training... > ""$logFile"" && ""$pythonPath"" $pythonArgs >> ""$logFile"" 2>&1"
    
    $proc = Start-Process -FilePath "cmd" -ArgumentList $cmdArgs -PassThru -NoNewWindow
    
    try {
        Show-Dashboard -Process $proc -LogFile $logFile -Title "Training" -ModelName $ModelName
    }
    finally {
        if (-not $proc.HasExited) {
            Write-Host "Stopping process $($proc.Id)..." -ForegroundColor DarkGray
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

function Run-Ensemble {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $Script:LogDir "build_ensemble_${timestamp}.log"
    
    Write-Host "Building Multi-Modal Ensemble..." -ForegroundColor Yellow
    
    $pythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (-not (Test-Path $pythonPath)) {
        Write-Host "Error: Python not found at $pythonPath" -ForegroundColor Red
        return
    }

    $pythonArgs = "-u -m src.models.ensemble.build_ensemble"
    $cmdArgs = "/c echo [ENSEMBLE] Building ensemble... > ""$logFile"" && ""$pythonPath"" $pythonArgs >> ""$logFile"" 2>&1"
    
    $proc = Start-Process -FilePath "cmd" -ArgumentList $cmdArgs -PassThru -NoNewWindow
    
    try {
        Show-Dashboard -Process $proc -LogFile $logFile -Title "Ensemble" -ModelName "Multi-Modal Ensemble"
    }
    finally {
        if (-not $proc.HasExited) {
            Write-Host "Stopping process $($proc.Id)..." -ForegroundColor DarkGray
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

function Activate-Venv {
    if (Test-Path "venv\Scripts\Activate.ps1") {
        $env:Path = "$PSScriptRoot\venv\Scripts;$env:Path"
        if (Get-Command python -ErrorAction SilentlyContinue) {
            $pyPath = (Get-Command python).Source
            Write-Host "[OK] Using Python: $pyPath" -ForegroundColor Green
        }
        else {
            Write-Host "[ERROR] Python not found in venv" -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "[ERROR] Virtual environment not found! Run setup first." -ForegroundColor Red
        exit 1
    }
}

function Show-Menu {
    Clear-Host
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Biometric Training Dashboard" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  [1] Train Face Model" -ForegroundColor Green
    Write-Host "  [2] Train Fingerprint Model" -ForegroundColor Green
    Write-Host "  [3] Train Iris NIR Model (CASIA/MMU)" -ForegroundColor Green
    Write-Host "  [4] Train Iris Visible Model (UBIRIS/SBVPI)" -ForegroundColor Green
    Write-Host "  [5] Train Iris Pipeline (NIR + Visible)" -ForegroundColor Green
    Write-Host "  [6] Custom Training Sequence" -ForegroundColor Cyan
    Write-Host "  [7] Build Ensemble (combine trained models)" -ForegroundColor Magenta
    Write-Host "  [8] Quit" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Configuration:" -ForegroundColor DarkGray
    Write-Host "    Epochs: $Epochs" -ForegroundColor DarkGray
    Write-Host "    Batch Size: $BatchSize" -ForegroundColor DarkGray
    Write-Host "    Sample Fraction: $SampleFraction" -ForegroundColor DarkGray
    Write-Host "    Logs: $Script:LogDir" -ForegroundColor DarkGray
    Write-Host ""
}

# Main execution
Set-Location $PSScriptRoot
Activate-Venv

# Interactive Loop
do {
    if ($Option -gt 0) {
        $choice = "$Option"
        $Option = 0
    }
    else {
        Show-Menu
        $choice = Read-Host "Select an option (1-8)"
    }

    switch ($choice) {
        "1" { 
            Run-TrainingStep -ScriptPath "src.models.face.train" -ModelName "Face"
            Read-Host "Press Enter to continue"
        }
        "2" { 
            Run-TrainingStep -ScriptPath "src.models.fingerprint.train" -ModelName "Fingerprint"
            Read-Host "Press Enter to continue"
        }
        "3" { 
            Run-TrainingStep -ScriptPath "src.models.iris.train" -ModelName "Iris-NIR" -ExtraArgs "--domain nir_casia"
            Read-Host "Press Enter to continue"
        }
        "4" { 
            Run-TrainingStep -ScriptPath "src.models.iris.train" -ModelName "Iris-Visible" -ExtraArgs "--domain visible"
            Read-Host "Press Enter to continue"
        }
        "5" { 
            Run-TrainingStep -ScriptPath "src.models.iris.train_pipeline" -ModelName "Iris-Pipeline"
            Read-Host "Press Enter to continue"
        }
        "6" {
            Write-Host ""
            Write-Host "Enter training sequence using numbers:" -ForegroundColor Cyan
            Write-Host "  1 = Face, 2 = Fingerprint, 3 = Iris NIR, 4 = Iris Visible, 5 = Iris Pipeline" -ForegroundColor Gray
            Write-Host "  Example: '1 2 3 4' trains Face, Fingerprint, Iris NIR, then Iris Visible" -ForegroundColor Gray
            Write-Host ""
            $seq = Read-Host "Enter sequence"
            $ids = $seq -split "[\s,]+"
            foreach ($id in $ids) {
                switch ($id) {
                    "1" { Run-TrainingStep -ScriptPath "src.models.face.train" -ModelName "Face" }
                    "2" { Run-TrainingStep -ScriptPath "src.models.fingerprint.train" -ModelName "Fingerprint" }
                    "3" { Run-TrainingStep -ScriptPath "src.models.iris.train" -ModelName "Iris-NIR" -ExtraArgs "--domain nir_casia" }
                    "4" { Run-TrainingStep -ScriptPath "src.models.iris.train" -ModelName "Iris-Visible" -ExtraArgs "--domain visible" }
                    "5" { Run-TrainingStep -ScriptPath "src.models.iris.train_pipeline" -ModelName "Iris-Pipeline" }
                    default { Write-Host "Skipping invalid ID: $id" -ForegroundColor Red }
                }
            }
            Read-Host "Press Enter to continue"
        }
        "7" {
            Run-Ensemble
            Read-Host "Press Enter to continue"
        }
        "8" { break }
        "Q" { break }
        "q" { break }
        default { 
            if ($choice -notin "Q", "q", "", "8") {
                Write-Host "Invalid choice. Please select 1-8." -ForegroundColor Red
                Start-Sleep -Seconds 1 
            }
        }
    }
} while ($choice -notin "Q", "q", "8")
