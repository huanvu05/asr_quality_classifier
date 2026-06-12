"""
train.py — Main pipeline runner.

Runs all 5 phases in sequence or executes a specific phase.
Usage:
  python train.py --all
  python train.py --phase 2
"""

import argparse
import sys
from pathlib import Path

# Add src to python path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import config, logger
from src.data_loader import sync_data_from_azure

# Lazy load experiments to handle imports cleanly
def get_experiment_runner(phase: int):
    if phase == 1:
        from experiments.phase1_eda import run_eda
        return run_eda
    elif phase == 2:
        from experiments.phase2_baseline import run_baseline
        return run_baseline
    elif phase == 3:
        from experiments.phase3_deep_audio import run_deep_audio
        return run_deep_audio
    elif phase == 4:
        from experiments.phase4_crossmodal import run_crossmodal
        return run_crossmodal
    elif phase == 5:
        from experiments.phase5_ensemble import run_ensemble
        return run_ensemble
    else:
        raise ValueError(f"Unknown phase: {phase}")

def parse_args():
    parser = argparse.ArgumentParser(description="ASR Quality Classifier Pipeline Runner")
    parser.add_argument(
        "--phase", 
        type=int, 
        choices=[1, 2, 3, 4, 5],
        help="Run a specific phase only (1: EDA, 2: Baseline, 3: Deep Audio, 4: Cross-modal, 5: Ensemble)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 phases in sequence"
    )
    parser.add_argument(
        "--sync-azure",
        action="store_true",
        help="Download dataset from Azure Blob Storage before starting"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Optional Azure sync
    if args.sync_azure:
        logger.info("Synchronizing data from Azure Blob Storage...")
        sync_data_from_azure(config)
        
    if args.all:
        logger.info("Running complete 5-phase pipeline...")
        # Phase 1: EDA
        logger.info("=== [PHASE 1] Exploratory Data Analysis ===")
        get_experiment_runner(1)()
        
        # Phase 2: Baseline
        logger.info("=== [PHASE 2] Tabular LightGBM Baseline ===")
        get_experiment_runner(2)()
        
        # Phase 3: Deep Audio
        logger.info("=== [PHASE 3] Deep Audio Branch ===")
        get_experiment_runner(3)()
        
        # Phase 4: Cross-modal
        logger.info("=== [PHASE 4] Cross-Modal Fusion ===")
        get_experiment_runner(4)()
        
        # Phase 5: Ensemble
        logger.info("=== [PHASE 5] Model Ensemble & Ablation ===")
        get_experiment_runner(5)()
        
        logger.info("Pipeline executed successfully. Outputs saved in outputs/ directory.")
        
    elif args.phase:
        logger.info(f"Running Phase {args.phase}...")
        get_experiment_runner(args.phase)()
    else:
        print("Please specify either --all or --phase [1-5].")
        print("Use --help for options.")
        sys.exit(1)

if __name__ == "__main__":
    main()
