"""
Integrative DDM Analysis with Conditions

This script performs analysis of the integrative drift-diffusion model (DDM) 
including recovery plots, posterior predictive checks, and simulation-based calibration.

Analyze results for factorial data
> uv run scripts/integrative_ddm_analyze_results_factorial.py --prefix integrative_ddm_data_

Analyze results for cross-validated factorial data
> uv run scripts/integrative_ddm_analyze_results_factorial.py --prefix cross_integrative_ddm_data_

"""

# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

# =====================================================================================
# Import modules
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import keras
import bayesflow as bf
from bayesflow.simulators import make_simulator

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import modules for integrative DDM analysis
from integrative_model.simulation import prior, likelihood
from integrative_model.analysis import calibration_histogram
from shared.plots import recovery_plot, compute_recovery_metrics

# Load checkpoint
CHECKPOINT = "checkpoint_integrative_ddm_seed_12_variable_trials.keras"

# =====================================================================================
# Setup paths and directories
INTEGRATIVE_MODEL_DIR = PROJECT_ROOT / "integrative_model"
CHECKPOINTS_DIR = INTEGRATIVE_MODEL_DIR / "checkpoints"
FIGURES_ROOT = INTEGRATIVE_MODEL_DIR / "figures_new_sigma_new_conditions"
DATA_DIR = INTEGRATIVE_MODEL_DIR / "data_new_sigma_new_conditions"
CHECKPOINT_PATH = CHECKPOINTS_DIR / CHECKPOINT

# =====================================================================================
# Setup simulator and approximator
def meta():
    return dict(n_obs=100)

# Make simulator
print("Making simulator...")
simulator = make_simulator([prior, likelihood], meta_fn=meta)

# Load approximator
approximator = keras.saving.load_model(CHECKPOINT_PATH)

# =====================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze integrative DDM results for factorial data')
parser.add_argument('--prefix', type=str, default='integrative_ddm_data_', 
                    help='Glob prefix for data files (default: integrative_ddm_data_)')
args = parser.parse_args()

# Get all MATLAB files in data directory using the specified prefix
matlab_files = sorted(DATA_DIR.glob(f"{args.prefix}*.mat"))

if not matlab_files:
    print(f"No .mat files found with prefix '{args.prefix}'!")
    sys.exit()

print(f"Found {len(matlab_files)} .mat files to process with prefix '{args.prefix}'")

# =====================================================================================
# Define variable names for analysis
parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]

# =====================================================================================
# Initialize combined gamma datasets
combined_gamma_estimates = {}  # estimates[condition_name] = np.ndarray
combined_gamma_targets = {}    # targets[condition_name] = np.ndarray

# Simulate validation data once (unseen during training)
val_sims = simulator.sample(50)
print(f"Validation simulation shapes: {val_sims['alpha'].shape}, {val_sims['choicert'].shape}")

# =====================================================================================
# Loop through all MATLAB files
for matlab_file in matlab_files:
    # Extract condition name from filename
    base_name = matlab_file.stem
    condition_name = base_name.replace(f"{args.prefix}", "")
    if base_name.startswith("cross_"):
        condition_name = f"cross_{condition_name}"

    # Create output directory for the figures
    figdir = FIGURES_ROOT / condition_name
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Analyzing condition: {condition_name}")
    print(f"File: {matlab_file.name}")
    print(f"{'='*80}")

    # =====================================================================================
    # Load and prepare data for current condition
    data = sio.loadmat(matlab_file)
    nparts = data["nparts"][0][0]
    ntrials = data["ntrials"][0][0]
    print(f"Number of participants: {nparts}, Number of trials: {ntrials}")

    # Reshape data for current condition
    reshaped_data = {
        "n_obs": np.empty((nparts, 1)),
        "alpha": np.empty((nparts, 1)),
        "tau": np.empty((nparts, 1)),
        "beta": np.empty((nparts, 1)),
        "mu_delta": np.empty((nparts, 1)),
        "eta_delta": np.empty((nparts, 1)),
        "gamma": np.empty((nparts, 1)),
        "sigma": np.empty((nparts, 1)),
        "choicert": np.empty((nparts, ntrials)),
        "z": np.empty((nparts, ntrials)),
    }

    for npart in range(nparts):
        data_start = npart * ntrials
        data_end = (npart + 1) * ntrials

        reshaped_data["n_obs"][npart] = ntrials
        reshaped_data["alpha"][npart] = data["alpha"][0][npart]
        reshaped_data["tau"][npart] = data["tau"][0][npart]
        reshaped_data["beta"][npart] = data["beta"][0][npart]
        reshaped_data["mu_delta"][npart] = data["mu_delta"][0][npart]
        reshaped_data["eta_delta"][npart] = data["eta_delta"][0][npart]
        reshaped_data["gamma"][npart] = data["gamma"][0][npart]
        reshaped_data["sigma"][npart] = data["sigma"][0][npart]
        reshaped_data["choicert"][npart] = data["choicert"][0][data_start:data_end]
        reshaped_data["z"][npart] = data["z"][0][data_start:data_end]

    # Generate posterior draws for current condition
    post_draws = approximator.sample(conditions=reshaped_data, num_samples=ntrials)

    # =====================================================================================
    # Store gamma data for combined analysis
    combined_gamma_estimates[condition_name] = post_draws["gamma"]
    combined_gamma_targets[condition_name] = reshaped_data["gamma"]

    # =====================================================================================
    # Recovery analysis for current condition
    print(f"Performing recovery analysis for {condition_name}...")

    split_estimates = post_draws.copy()
    split_targets = reshaped_data.copy()

    # Generate recovery plot
    f = recovery_plot(split_estimates, split_targets)
    if f is not None:
        recovery_filename = f"recovery_plot_{condition_name}.png"
        f.savefig(figdir / recovery_filename)
        print(f"Saved recovery plot: {recovery_filename}")
    
    # =====================================================================================
    # Compute recovery metrics
    print(f"\nComputing recovery metrics for {condition_name}...")

    split_parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]
    val_sims_params = {k: v for k, v in split_targets.items() if k in split_parameter_names}
    compute_recovery_metrics(split_estimates, val_sims_params)

    # =====================================================================================
    print(f"Completed analysis for {condition_name}")

# =====================================================================================
# Create condition display title mapping for combined plots
condition_display_titles = {
    'SNR_no_noise_COUP_low_DIST_gaussian': r'Gaussian, No Noise, Low Coupling',
    'cross_SNR_no_noise_COUP_low_DIST_gaussian': r'Gaussian, No Noise, Low Coupling',
    'SNR_no_noise_COUP_low_DIST_laplace': r'Laplace, No Noise, Low Coupling',
    'cross_SNR_no_noise_COUP_low_DIST_laplace': r'Laplace, No Noise, Low Coupling',
    'SNR_no_noise_COUP_low_DIST_uniform': r'Uniform, No Noise, Low Coupling',
    'cross_SNR_no_noise_COUP_low_DIST_uniform': r'Uniform, No Noise, Low Coupling',

    'SNR_low_COUP_low_DIST_gaussian': r'Gaussian, Low SNR, Low Coupling',
    'cross_SNR_low_COUP_low_DIST_gaussian': r'Gaussian, Low SNR, Low Coupling',
    'SNR_low_COUP_low_DIST_laplace': r'Laplace, Low SNR, Low Coupling',
    'cross_SNR_low_COUP_low_DIST_laplace': r'Laplace, Low SNR, Low Coupling',
    'SNR_low_COUP_low_DIST_uniform': r'Uniform, Low SNR, Low Coupling',
    'cross_SNR_low_COUP_low_DIST_uniform': r'Uniform, Low SNR, Low Coupling',

    'SNR_high_COUP_low_DIST_gaussian': r'Gaussian, High SNR, Low Coupling',
    'cross_SNR_high_COUP_low_DIST_gaussian': r'Gaussian, High SNR, Low Coupling',
    'SNR_high_COUP_low_DIST_laplace': r'Laplace, High SNR, Low Coupling',
    'cross_SNR_high_COUP_low_DIST_laplace': r'Laplace, High SNR, Low Coupling',
    'SNR_high_COUP_low_DIST_uniform': r'Uniform, High SNR, Low Coupling',
    'cross_SNR_high_COUP_low_DIST_uniform': r'Uniform, High SNR, Low Coupling',

    'SNR_no_noise_COUP_high_DIST_gaussian': r'Gaussian, No Noise, High Coupling',
    'cross_SNR_no_noise_COUP_high_DIST_gaussian': r'Gaussian, No Noise, High Coupling',
    'SNR_no_noise_COUP_high_DIST_laplace': r'Laplace, No Noise, High Coupling',
    'cross_SNR_no_noise_COUP_high_DIST_laplace': r'Laplace, No Noise, High Coupling',
    'SNR_no_noise_COUP_high_DIST_uniform': r'Uniform, No Noise, High Coupling',
    'cross_SNR_no_noise_COUP_high_DIST_uniform': r'Uniform, No Noise, High Coupling',

    'SNR_low_COUP_high_DIST_gaussian': r'Gaussian, Low SNR, High Coupling',
    'cross_SNR_low_COUP_high_DIST_gaussian': r'Gaussian, Low SNR, High Coupling',
    'SNR_low_COUP_high_DIST_laplace': r'Laplace, Low SNR, High Coupling',
    'cross_SNR_low_COUP_high_DIST_laplace': r'Laplace, Low SNR, High Coupling',
    'SNR_low_COUP_high_DIST_uniform': r'Uniform, Low SNR, High Coupling',  
    'cross_SNR_low_COUP_high_DIST_uniform': r'Uniform, Low SNR, High Coupling',  

    'SNR_high_COUP_high_DIST_gaussian': r'Gaussian, High SNR, High Coupling',
    'cross_SNR_high_COUP_high_DIST_gaussian': r'Gaussian, High SNR, High Coupling',
    'SNR_high_COUP_high_DIST_laplace': r'Laplace, High SNR, High Coupling',
    'cross_SNR_high_COUP_high_DIST_laplace': r'Laplace, High SNR, High Coupling',
    'SNR_high_COUP_high_DIST_uniform': r'Uniform, High SNR, High Coupling',
    'cross_SNR_high_COUP_high_DIST_uniform': r'Uniform, High SNR, High Coupling',
}

print(f"\n===  Recovery Plot for all gammas (split into two figures)===")

# Separate conditions by coupling level
low_coupling_conditions = [k for k in combined_gamma_estimates.keys() if "_COUP_low_" in k]
high_coupling_conditions = [k for k in combined_gamma_estimates.keys() if "_COUP_high_" in k]

coupling_groups = {
    "Low Coupling": low_coupling_conditions,
    "High Coupling": high_coupling_conditions
}

# Define SNR order for sorting
snr_levels = ["low", "high", "no_noise"]

def group_conditions_by_snr_level(conditions):
    """Sort conditions by SNR level (no_noise, low, high)"""
    grouped = {snr: [] for snr in snr_levels}
    for cond in conditions:
        for snr in snr_levels:
            if snr in cond:
                grouped[snr].append(cond)
    return grouped

# Plot one figure per coupling level
for coupling_level, conditions in coupling_groups.items():
    if not conditions:  # skip empty groups
        print(f"No conditions for {coupling_level}, skipping...")
        continue
    
    # Group by SNR
    grouped_conditions = group_conditions_by_snr_level(conditions)
    # Flatten in desired order
    conditions_ordered = grouped_conditions["low"] + grouped_conditions["high"] + grouped_conditions["no_noise"]
    
    subset_estimates = {k: combined_gamma_estimates[k] for k in conditions_ordered}
    subset_targets = {k: combined_gamma_targets[k] for k in conditions_ordered}
    subset_titles = {k: condition_display_titles[k] for k in conditions_ordered}
    
    print(f"Plotting recovery plot for {coupling_level} conditions: {list(subset_estimates.keys())}")

    # Recovery plot on specific axis
    fig_gamma = recovery_plot(subset_estimates, subset_targets, 
                             parameter_display_titles=subset_titles, fig_height=9, fig_width=15, is_all_coupling=True)
    fig_gamma.savefig(FIGURES_ROOT / f"recovery_plot_combined_gamma_{coupling_level}_{args.prefix}.png", dpi=300)
    plt.close(fig_gamma)

print(f"\n{'='*80}")
print("Analysis complete for all conditions!")
print(f"{'='*80}") 