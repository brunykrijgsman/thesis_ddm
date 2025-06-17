"""
Integrative DDM Analysis with Conditions

This script performs analysis of the integrative drift-diffusion model (DDM) 
including recovery plots, posterior predictive checks, and simulation-based calibration.

Analyze results for factorial data
> uv run scripts/integrative_ddm_analyze_results_factorial.py --prefix integrative_ddm_

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

from integrative_model.simulation import prior, likelihood
from integrative_model.analysis import calibration_histogram
from shared.plots import recovery_plot, compute_recovery_metrics

CHECKPOINT = "checkpoint_integrative_ddm_seed_12_150e.keras"

# =====================================================================================
# Setup paths and directories
INTEGRATIVE_MODEL_DIR = PROJECT_ROOT / "integrative_model"
CHECKPOINTS_DIR = INTEGRATIVE_MODEL_DIR / "checkpoints"
FIGURES_ROOT = INTEGRATIVE_MODEL_DIR / "figures"
DATA_DIR = INTEGRATIVE_MODEL_DIR / "data"
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

# Simulate validation data once (unseen during training)
val_sims = simulator.sample(50)
print(f"Validation simulation shapes: {val_sims['alpha'].shape}, {val_sims['choicert'].shape}")

# =====================================================================================
# Loop through all MATLAB files
for matlab_file in matlab_files:
    # Extract condition name from filename
    condition_name = matlab_file.stem.replace(f"{args.prefix}", "")

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
    # Recovery analysis for current condition
    print(f"Performing recovery analysis for {condition_name}...")

    split_estimates = post_draws.copy()
    split_targets = reshaped_data.copy()
    
    # Handle gamma parameter for high coupling conditions (only for recovery plots)
    high_coupling = "COUP_high" in condition_name
    if high_coupling:
        print(f"Splitting gamma plots for {condition_name}...")
        
        gamma_samples = split_estimates['gamma'].squeeze()  # Shape: (n_participants, n_draws)
        gamma_true = split_targets['gamma'].squeeze()  # Flatten to match shape

        neg_idx = np.where(gamma_true < 0)[0]
        pos_idx = np.where(gamma_true >= 0)[0]

        if len(neg_idx) > 0:
            split_estimates['gamma_negative'] = gamma_samples[neg_idx, :]
            split_targets['gamma_negative'] = gamma_true[neg_idx]

        if len(pos_idx) > 0:
            split_estimates['gamma_positive'] = gamma_samples[pos_idx, :]
            split_targets['gamma_positive'] = gamma_true[pos_idx]

        del split_estimates['gamma']
        del split_targets['gamma']
    
    # Generate recovery plot
    f = recovery_plot(split_estimates, split_targets)
    if f is not None:
        recovery_filename = f"recovery_plot_{condition_name}.png"
        f.savefig(figdir / recovery_filename)
        print(f"Saved recovery plot: {recovery_filename}")

    # =====================================================================================
    # Simulation-Based Calibration for current condition
    print(f"Performing calibration analysis for {condition_name}...")

    # Filter reshaped_data to only include parameter keys
    val_sims_params = {k: v for k, v in reshaped_data.items() if k in parameter_names}

    # Calibration ECDF plot
    print(f"post_draws: {post_draws.keys()}")
    print(f"reshaped_data: {reshaped_data.keys()}")
    print(f"val_sims_params: {val_sims_params.keys()}")
    print(f"parameter_names: {parameter_names}")

    ecdf = bf.diagnostics.plots.calibration_ecdf(
        estimates=post_draws,
        targets=val_sims_params,
        variable_names=parameter_names,
        difference=True,
        rank_type="distance",
    )
    ecdf_filename = f"calibration_ecdf_{condition_name}.png"
    ecdf.savefig(figdir / ecdf_filename)
    print(f"Saved calibration ECDF: {ecdf_filename}")
    plt.close(ecdf)  # Close figure to free memory

    # Calibration histogram
    sbc = calibration_histogram(
        estimates=post_draws,
        targets=val_sims_params,
        variable_keys=parameter_names,
        num_bins=10,
        binomial_interval=0.99,
        label_fontsize=16,
        title_fontsize=18,
    )
    sbc_filename = f"calibration_histogram_{condition_name}.png"
    sbc.savefig(figdir / sbc_filename)
    print(f"Saved calibration histogram: {sbc_filename}")
    plt.close(sbc)  # Close figure to free memory   

    # =====================================================================================
    # Compute recovery metrics
    print(f"\nComputing recovery metrics for {condition_name}...")

    if high_coupling:
        split_parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma_negative", "gamma_positive", "sigma"]
    else:
        split_parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]

    val_sims_params = {k: v for k, v in split_targets.items() if k in split_parameter_names}
    compute_recovery_metrics(split_estimates, val_sims_params)

    # =====================================================================================
    print(f"Completed analysis for {condition_name}")

print(f"\n{'='*80}")
print("Analysis complete for all conditions!")
print(f"{'='*80}") 