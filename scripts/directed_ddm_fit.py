# =====================================================================================
# Import modules
from pathlib import Path
import sys
import argparse

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from directed_model.simulation import fit_directed_ddm

SEED = 2025

# =====================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Fit directed DDM to a specific model/dataset')
parser.add_argument('--model', type=str, default='directed_ddm_base', 
                    help='Model name (default: directed_ddm_base)')
args = parser.parse_args()

model_name = args.model
filename = f'{model_name}.mat'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIRECTED_MODEL_DIR = PROJECT_ROOT / 'directed_model'
DATA_DIR = DIRECTED_MODEL_DIR / 'data'
filepath = DATA_DIR / filename

# Check if the file exists
if not filepath.exists():
    print(f"Error: File {filepath} does not exist!")
    sys.exit(1)

print(f"Fitting model: {model_name}")
print(f"Data file: {filepath}")

# Use the utility function to fit the model
fit = fit_directed_ddm(
    mat_file_path=filepath,
    chains=4,
    parallel_chains=4,
    iter_sampling=1000,
    iter_warmup=500,
    seed=SEED,
    show_console=True
)

print('Done!')

# =====================================================================================
# Save results
RESULTS_DIR = DIRECTED_MODEL_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
fit.save_csvfiles(RESULTS_DIR / model_name)