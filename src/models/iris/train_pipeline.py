"""
Unified Iris Training Pipeline.

Orchestrates the entire flow:
1. Train NIR Domain Model
2. Train Visible Domain Model
3. Run Ensemble Evaluation (+ evaluation plots)
"""
import sys
from pathlib import Path
from loguru import logger
import argparse

# Project root setup
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.iris.train import train_iris
from src.models.iris.config import TrainingConfig
from src.models.iris.domain_config import DOMAINS, get_domain_config_overrides
from src.utils.logging_utils import setup_logging

def run_pipeline(quick_test=False, start_stage=1):
    """Run the 2-stage pipeline.
    
    Args:
        quick_test: Run in quick test mode
        start_stage: Stage to start from (1=NIR, 2=Visible)
    """
    logger.info(f"STARTING UNIFIED IRIS PIPELINE (from stage {start_stage})")
    
    import subprocess

    # Preprocess helper
    def run_preprocessing(domain):
        logger.info(f"Running Incremental Preprocessing for {domain}...")
        try:
            # We assume we are in project root context or src context. 
            # Best to invoke via python -m or script path.
            # train_pipeline is likely run as -m src.models.iris.train_pipeline from root.
            # So scripts/preprocess_with_onnx.py is relative to CWD.
            cmd = [sys.executable, '-m', 'src.models.iris.preprocess', '--domain', domain]
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            logger.error(f"Preprocessing failed for {domain}: {e}")
            logger.warning("Continuing with training (might overlap with on-the-fly preprocessing if enabled)...")

    # Helper to train a domain
    def run_domain_training(domain_name):
        domain_info = DOMAINS[domain_name]
        config_kwargs = {
            'output_dir': f'outputs/iris_domain_{domain_name}',
            'allowed_datasets': set(domain_info['datasets']),
        }
        
        # APPLY DOMAIN-SPECIFIC OVERRIDES (margin, scale, subcenter, etc.)
        domain_overrides = get_domain_config_overrides(domain_name)
        logger.info(f"Applying {len(domain_overrides)} domain overrides for '{domain_name}':")
        for key, value in domain_overrides.items():
            config_kwargs[key] = value
            logger.info(f"  {key} = {value}")
        
        if quick_test:
            config_kwargs['quick_test'] = True
            config_kwargs['epochs'] = 3
        config = TrainingConfig(**config_kwargs)
        train_iris(config)
    
    # --- STAGE 1: NIR DOMAIN ---
    if start_stage <= 1:
        logger.info(">>> STAGE 1: NIR Domain (Preprocessing & Training)")
        run_preprocessing('nir_casia')
        
        # Train NIR
        run_domain_training('nir_casia')
    else:
        logger.info(">>> SKIPPING STAGE 1: NIR Domain (--start-stage > 1)")
    
    # --- STAGE 2: VISIBLE DOMAIN ---
    if start_stage <= 2:
        logger.info(">>> STAGE 2: Visible Domain (Preprocessing & Training)")
        run_preprocessing('visible')
        
        run_domain_training('visible')
    else:
        logger.info(">>> SKIPPING STAGE 2: Visible Domain (--start-stage > 2)")
    
    logger.info("PIPELINE COMPLETE - SUCCESS")
    logger.info("=" * 60)
    # sys.exit(0)  # Return to main for proper exit

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--start-stage', type=int, default=1, choices=[1, 2],
                        help='Stage to start from: 1=NIR, 2=Visible')
    args = parser.parse_args()
    
    run_pipeline(quick_test=args.quick_test, start_stage=args.start_stage)
    
    # Explicit success exit for PowerShell wrapper
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE - SUCCESS")
    logger.info("=" * 60)
    sys.exit(0)
