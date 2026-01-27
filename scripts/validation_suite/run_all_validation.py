
import os
import subprocess
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts/validation_suite"

def run_script(script_name):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return
    
    logger.info(f"Running {script_name}...")
    try:
        # Run using the same python interpreter
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        if result.returncode == 0:
            logger.info(f"{script_name} completed successfully.")
        else:
            logger.error(f"{script_name} failed with code {result.returncode}")
    except subprocess.CalledProcessError as e:
        logger.error(f"{script_name} failed: {e}")

def main():
    logger.info("STARTING FULL VALIDATION SUITE (Identity Split)")
    
    scripts = [
        "validate_face.py",
        "validate_fingerprint.py",
        "validate_iris.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    logger.info("VALIDATION SUITE COMPLETE")
    logger.info(f"Check results in {PROJECT_ROOT / 'outputs/validation_suite'}")

if __name__ == "__main__":
    main()
