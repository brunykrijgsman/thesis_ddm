"""
Directed DDM Fitting Script for Factorial Data

This script fits the directed drift-diffusion model (DDM) to factorial experimental data.

Examples:
    Fit to standard factorial data:
    > uv run scripts/directed_ddm_fit_factorial.py --prefix ddmdata_

    Fit to cross-validated factorial data:
    > uv run scripts/directed_ddm_fit_factorial.py --prefix cross_directed_ddm_data_

"""

# =====================================================================================
# Import modules
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from directed_model.simulation import fit_directed_ddm

SEED = 2025

DIRECTED_MODEL_DIR = PROJECT_ROOT / 'directed_model'
DATA_DIR = DIRECTED_MODEL_DIR / 'data_new_sigma_z'

# =====================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Fit directed DDM to factorial data')
parser.add_argument('--prefix', type=str, default='ddmdata_', 
                    help='Glob prefix for data files (default: ddmdata_). Use cross_directed_ddm_data_ for cross-validated data.')
args = parser.parse_args()

# Get all mat files using the specified prefix
mat_files = sorted(DATA_DIR.glob(f"{args.prefix}*.mat"))

if not mat_files:
    print(f"No .mat files found with prefix '{args.prefix}'!")
    sys.exit()

print(f"Found {len(mat_files)} .mat files to process with prefix '{args.prefix}'")

# Loop through all generated datasets
for f in mat_files:
    print(f"\n--- Fitting model to {f} ---")
    
    base = Path(f).stem
    out_dir = DIRECTED_MODEL_DIR / 'results_new_sigma_z' / base
    
    # Check if results already exist
    if out_dir.exists():
        print(f"Results already exist for {base}. Skipping...")
        continue
    
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Use the utility function to fit the model
    fit = fit_directed_ddm(
        mat_file_path=f,
        chains=4,
        parallel_chains=4,
        iter_sampling=1000,
        iter_warmup=500,
        seed=SEED,
        show_console=True
    )

    # Save output
    out_dir.mkdir(parents=True, exist_ok=True)
    fit.save_csvfiles(dir=out_dir)
    print(f"Done. Results saved to {out_dir}")
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {end_time - start_time}")