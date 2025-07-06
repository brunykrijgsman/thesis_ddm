"""
Posterior Predictive Checks for directed DDM results for factorial data

Usage:
> uv run scripts/directed_ddm_analyze_factorial_ppc.py --prefix ddmdata_
> uv run scripts/directed_ddm_analyze_factorial_ppc.py --prefix cross_directed_ddm_data_
"""

# =====================================================================================
# Import modules
from pathlib import Path
import sys
import argparse

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import from_csv
import scipy.io as sio
from directed_model.analysis_ppc import posterior_predictive_check, generate_predicted_data

# =====================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Posterior Predictive Checks for directed DDM results for factorial data')
parser.add_argument('--prefix', type=str, default='ddmdata_', 
                    help='Glob prefix for data files (default: ddmdata_)')
args = parser.parse_args()

# =====================================================================================
# Set up paths
DIRECTED_MODEL_DIR = PROJECT_ROOT / "directed_model"
DATA_DIR = DIRECTED_MODEL_DIR / "data"
RESULTS_DIR = DIRECTED_MODEL_DIR / "results"
FIGURES_ROOT = DIRECTED_MODEL_DIR / "figures"
FIGURES_ROOT.mkdir(exist_ok=True)

# =====================================================================================
# Load baseline model results from 'directed_ddm_base'
model_name = 'directed_ddm_base'
baseline_results_dir = RESULTS_DIR / model_name
baseline_data_file = DATA_DIR / f"{model_name}.mat"

# Check if baseline data file exists
if not baseline_data_file.exists():
    print(f"Error: Baseline data file {baseline_data_file} does not exist!")
    sys.exit(1)

# Check if baseline results directory exists
if not baseline_results_dir.exists():
    print(f"Error: Baseline results directory {baseline_results_dir} does not exist!")
    sys.exit(1)

print(f"Loading baseline model: {model_name}")

# Load baseline data
baseline_genparam = sio.loadmat(baseline_data_file)
baseline_participants = np.squeeze(baseline_genparam["participant"]).astype(int)
n_participants = baseline_participants.max()

# Load baseline CmdStanMCMC results from CSV
baseline_csv_files = sorted(baseline_results_dir.glob("*.csv"))
if not baseline_csv_files:
    print(f"No CSV files found in {baseline_results_dir}, exiting.")
    sys.exit()

baseline_fit = from_csv([str(p) for p in baseline_csv_files])
baseline_df = baseline_fit.draws_pd()

# =====================================================================================
# Load factorial conditions data
print(f"\nLoading factorial conditions data with prefix '{args.prefix}'...")

# Get all .mat files using the specified prefix
factorial_mat_files = sorted(DATA_DIR.glob(f"{args.prefix}*.mat"))

if not factorial_mat_files:
    print(f"No .mat files found with prefix '{args.prefix}'!")
    sys.exit()

print(f"Found {len(factorial_mat_files)} factorial condition files")

# Load data for each condition
conditions = []
for mat_path in factorial_mat_files:
    base_name = mat_path.stem
    condition_name = base_name.replace(f"{args.prefix}", "")
    if base_name.startswith("cross_"):
        condition_name = f"cross_{condition_name}"
    
    # Load condition data
    condition_genparam = sio.loadmat(mat_path)
    condition_y = np.squeeze(condition_genparam["y"])
    condition_z = np.squeeze(condition_genparam["z"])
    condition_participants = np.squeeze(condition_genparam["participant"]).astype(int)
    
    # Add to conditions list
    conditions.append({
        'condition_y': condition_y,
        'condition_z': condition_z,
        'condition_participants': condition_participants,
        'condition_name': condition_name
    })

# =====================================================================================
# Posterior Predictive Checks
# Loop through each condition individually to create separate plots
for condition in conditions:
    condition_name = condition['condition_name']
    print(f"\nGenerating PPC for condition: {condition_name}")
    
    # Create condition-specific directory (same as in directed_ddm_analyze_factorial.py)
    fig_dir = FIGURES_ROOT / condition_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Call posterior_predictive_check for this single condition
    fig = posterior_predictive_check(fit=baseline_fit, df=baseline_df,
                                     participants=baseline_participants,
                                     true_y=None, true_z=None,
                                     nparts=n_participants,
                                     conditions_data=[condition])  # Pass single condition as list
    
    # Save individual plot in the condition-specific directory
    fig.savefig(fig_dir / "posterior_predictive_check.png", dpi=300)
    plt.close(fig)
    print(f"Saved plot for condition: {condition_name} in {fig_dir}")

print(f"\nCompleted posterior predictive checks for {len(conditions)} conditions")
