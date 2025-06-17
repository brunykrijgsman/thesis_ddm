# =====================================================================================
# Import modules
import sys
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from directed_model.simulation import fit_directed_ddm

SEED = 2025

# Project directories
DIRECTED_MODEL_DIR = PROJECT_ROOT / 'directed_model'
DATA_DIR = DIRECTED_MODEL_DIR / 'data'

def process_mat_file(mat_file):
    """Process a single mat file and fit the model."""
    print(f"\n--- Fitting model to {mat_file} ---")

    # Check if results already exist
    base = Path(mat_file).stem
    out_dir = DIRECTED_MODEL_DIR / 'results' / base

    if out_dir.exists():
        print(f"Results already exist for {base}. Skipping...")
        return

    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Use the utility function to fit the model
        fit = fit_directed_ddm(
            mat_file_path=mat_file,
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

        end_time = datetime.now()
        print(f"Done. Results saved to {out_dir}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {end_time - start_time}")

        return True

    except Exception as e:
        print(f"Error processing {mat_file}: {str(e)}")
        return False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fit directed DDM to data files in parallel')
parser.add_argument('--prefix', type=str, default='ddmdata_', 
                    help='Glob prefix for data files (default: ddmdata_)')
args = parser.parse_args()

# Get all mat files using the specified prefix
mat_files = sorted(DATA_DIR.glob(f"{args.prefix}*.mat"))

if not mat_files:
    print(f"No .mat files found with prefix '{args.prefix}'!")
    sys.exit()

print(f"Found {len(mat_files)} .mat files to process with prefix '{args.prefix}'")

# Process files in parallel using up to 6 processes
with ProcessPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(process_mat_file, mat_files))

# Summary
successful = sum(1 for r in results if r is True)
print(f"\nProcessing complete! Successfully processed {successful} out of {len(mat_files)} files")
