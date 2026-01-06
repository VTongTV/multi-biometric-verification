"""
GPU Monitoring Script for Training Diagnosis.

Continuously logs GPU metrics to detect thermal/power issues before crash.
Run this ALONGSIDE training: python scripts/monitor_gpu.py

Output: logs/gpu_monitor_<timestamp>.csv
"""

import subprocess
import time
import csv
from datetime import datetime
from pathlib import Path

# Configuration
POLL_INTERVAL = 1.0  # seconds between polls
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# nvidia-smi query fields
QUERY_FIELDS = [
    "timestamp",
    "gpu_name",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "temperature.memory",  # May not be available on all GPUs
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.memory",
    "fan.speed",  # May show N/A on laptops
]

def get_gpu_metrics():
    """Query nvidia-smi for GPU metrics."""
    query = ",".join(QUERY_FIELDS)
    cmd = f'nvidia-smi --query-gpu={query} --format=csv,noheader,nounits'
    
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(", ")
            return dict(zip(QUERY_FIELDS, values))
        else:
            return {"error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timeout"}
    except Exception as e:
        return {"error": str(e)}


def monitor():
    """Main monitoring loop."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"gpu_monitor_{timestamp}.csv"
    
    print(f"=" * 60)
    print(f"GPU MONITORING STARTED")
    print(f"=" * 60)
    print(f"Log file: {log_file}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Press Ctrl+C to stop")
    print(f"=" * 60)
    
    # Track for alerts
    max_temp = 0
    max_power = 0
    start_time = time.time()
    
    with open(log_file, 'w', newline='') as f:
        writer = None
        
        try:
            while True:
                metrics = get_gpu_metrics()
                
                if "error" in metrics:
                    print(f"[ERROR] {metrics['error']}")
                    time.sleep(POLL_INTERVAL)
                    continue
                
                # Add elapsed time
                elapsed = time.time() - start_time
                metrics["elapsed_seconds"] = f"{elapsed:.1f}"
                
                # Initialize CSV writer with headers on first successful read
                if writer is None:
                    fieldnames = list(metrics.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                
                writer.writerow(metrics)
                f.flush()  # Ensure data is written immediately
                
                # Parse and check for warnings
                try:
                    temp = float(metrics.get("temperature.gpu", 0))
                    power_str = metrics.get("power.draw", "0")
                    power = float(power_str) if power_str != "[N/A]" else 0
                    
                    max_temp = max(max_temp, temp)
                    max_power = max(max_power, power)
                    
                    # Console output
                    elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                    util = metrics.get("utilization.gpu", "?")
                    mem = metrics.get("memory.used", "?")
                    pstate = metrics.get("pstate", "?")
                    
                    status = f"[{elapsed_str}] Temp: {temp}°C | Power: {power_str}W | Util: {util}% | Mem: {mem}MB | PState: {pstate}"
                    
                    # Thermal warning
                    if temp >= 85:
                        status += " ⚠️ THERMAL WARNING!"
                    elif temp >= 80:
                        status += " 🔥 HIGH TEMP"
                    
                    print(status)
                    
                except (ValueError, TypeError):
                    print(f"[{elapsed:.1f}s] {metrics}")
                
                time.sleep(POLL_INTERVAL)
                
        except KeyboardInterrupt:
            print(f"\n" + "=" * 60)
            print(f"MONITORING STOPPED")
            print(f"=" * 60)
            print(f"Duration: {elapsed:.1f} seconds")
            print(f"Max Temperature: {max_temp}°C")
            print(f"Max Power Draw: {max_power}W")
            print(f"Log saved to: {log_file}")
            print(f"=" * 60)


if __name__ == "__main__":
    monitor()
