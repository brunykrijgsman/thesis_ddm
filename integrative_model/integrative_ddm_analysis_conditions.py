"""
Integrative DDM Analysis with Conditions

This script performs analysis of the integrative drift-diffusion model (DDM) 
including recovery plots, posterior predictive checks, and simulation-based calibration.
"""

# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

# =====================================================================================
# Import modules
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
import numpy as np
import sys
import glob
from bayesflow.approximators import ContinuousApproximator

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to path for imports
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import integrative_ddm_sim as ddm
import keras
import bayesflow as bf
from bayesflow.networks import SetTransformer, CouplingFlow
from bayesflow.adapters import Adapter
from bayesflow.simulators import make_simulator

from plots import calibration_histogram
from shared.plots import recovery, compute_recovery_metrics

# =====================================================================================
# Setup paths and directories
# Load checkpoint - relative to current file
CHECKPOINT_PATH = os.path.join(current_dir, 'checkpoints', 'jax_simple_integrative_ddm_checkpoint_seed12_uniform_new_sigma_beta.keras')

# Create save directory - relative to current file
save_dir = os.path.join(current_dir, 'Figures')
os.makedirs(save_dir, exist_ok=True)

# =====================================================================================
# Setup simulator and approximator
def meta():
    return dict(n_obs=100)

# Make simulator
print("Making simulator...")
simulator = make_simulator([ddm.prior, ddm.likelihood], meta_fn=meta)

adapter = (
    Adapter()
    .broadcast("n_obs", to="choicert")    
    .as_set(["choicert", "z"])
    .standardize(exclude=["n_obs"])
    .convert_dtype("float64", "float32")
    .concatenate(["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"], into="inference_variables")
    .concatenate(["choicert", "z"], into="summary_variables")
    .rename("n_obs", "inference_conditions")
)

# Load approximator
approximator = keras.saving.load_model(CHECKPOINT_PATH)

# =====================================================================================
# Get all MATLAB files in data directory
data_dir = os.path.join(current_dir, 'data')
matlab_files = glob.glob(os.path.join(data_dir, '*.mat'))
matlab_files.sort()  # Sort for consistent ordering

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
    filename = os.path.basename(matlab_file)
    condition_name = filename.replace('integrative_ddm_data_', '').replace('.mat', '')
    
    print(f"\n{'='*80}")
    print(f"Analyzing condition: {condition_name}")
    print(f"File: {filename}")
    print(f"{'='*80}")
    
    # =====================================================================================
    # Load and prepare data for current condition
    data = sio.loadmat(matlab_file)
    nparts = data['nparts'][0][0]
    ntrials = data['ntrials'][0][0]
    print(f"Number of participants: {nparts}, Number of trials: {ntrials}")
    
    # Reshape data for current condition
    reshaped_data = {
        'n_obs': np.empty((nparts, 1)),
        'alpha': np.empty((nparts, 1)),
        'tau': np.empty((nparts, 1)), 
        'beta': np.empty((nparts, 1)),
        'mu_delta': np.empty((nparts, 1)),
        'eta_delta': np.empty((nparts, 1)),
        'gamma': np.empty((nparts, 1)),
        'sigma': np.empty((nparts, 1)),
        'choicert': np.empty((nparts, ntrials)),
        'z': np.empty((nparts, ntrials))
    }

    for npart in range(nparts):
        data_start = npart * ntrials
        data_end = (npart + 1) * ntrials

        reshaped_data['n_obs'][npart] = ntrials
        reshaped_data['alpha'][npart] = data['alpha'][0][npart]
        reshaped_data['tau'][npart] = data['tau'][0][npart]
        reshaped_data['beta'][npart] = data['beta'][0][npart]
        reshaped_data['mu_delta'][npart] = data['mu_delta'][0][npart]
        reshaped_data['eta_delta'][npart] = data['eta_delta'][0][npart]
        reshaped_data['gamma'][npart] = data['gamma'][0][npart]
        reshaped_data['sigma'][npart] = data['sigma'][0][npart]
        reshaped_data['choicert'][npart] = data['choicert'][0][data_start:data_end]
        reshaped_data['z'][npart] = data['z'][0][data_start:data_end]

    print(f"reshaped_data: {reshaped_data['gamma'].shape}")

    # Generate posterior draws for current condition
    post_draws = approximator.sample(conditions=reshaped_data, num_samples=ntrials)
    
    # Reshape any 3D arrays in post_draws to 2D by squeezing the last dimension
    for param in post_draws:
        post_draws[param] = np.squeeze(post_draws[param], axis=-1)

    # =====================================================================================
    # Recovery analysis for current condition
    print(f"Performing recovery analysis for {condition_name}...")
    
    # Handle gamma parameter for high coupling conditions
    if "COUP_high" in condition_name:
        print(f"Splitting gamma plots for {condition_name}...")
        
        gamma_samples = post_draws['gamma']  # Shape: (n_draws, n_participants)
        gamma_true = reshaped_data['gamma'].flatten()  # Flatten to match shape
        
        neg_idx = np.where(gamma_true < 0)[0]
        pos_idx = np.where(gamma_true >= 0)[0]
        
        if len(neg_idx) > 0:
            post_draws['gamma_negative'] = gamma_samples[:, neg_idx].T
            reshaped_data['gamma_negative'] = gamma_true[neg_idx]
            
        if len(pos_idx) > 0:
            post_draws['gamma_positive'] = gamma_samples[:, pos_idx].T
            reshaped_data['gamma_positive'] = gamma_true[pos_idx]
            
        # Remove original gamma to avoid triple-plotting
        del post_draws['gamma']
        del reshaped_data['gamma']
    
    # Recovery plot
    f = recovery(post_draws, reshaped_data)
    if f is not None:
        recovery_filename = f'recovery_plot_{condition_name}.png'
        f.savefig(os.path.join(save_dir, recovery_filename))
        print(f"Saved recovery plot: {recovery_filename}")

    # =====================================================================================
    # Simulation-Based Calibration for current condition
    print(f"Performing calibration analysis for {condition_name}...")
    
    # Filter reshaped_data to only include parameter keys
    val_sims_params = {k: v for k, v in reshaped_data.items() if k in parameter_names}

    # Calibration ECDF plot
    try:
        ecdf = bf.diagnostics.plots.calibration_ecdf(
            estimates=post_draws, 
            targets=reshaped_data,
            variable_names=parameter_names,
            difference=True,
            rank_type="distance"
        )
        ecdf_filename = f'calibration_ecdf_{condition_name}.png'
        ecdf.savefig(os.path.join(save_dir, ecdf_filename))
        print(f"Saved calibration ECDF: {ecdf_filename}")
        plt.close(ecdf)  # Close figure to free memory
    except Exception as e:
        print(f"Error generating calibration ECDF for {condition_name}: {e}")

    # Calibration histogram
    try:
        sbc = calibration_histogram(
            estimates=post_draws, 
            targets=val_sims_params,
            variable_keys=parameter_names,
            num_bins=10,
            binomial_interval=0.99,
            label_fontsize=16,
            title_fontsize=18
        )
        sbc_filename = f'calibration_histogram_{condition_name}.png'
        sbc.savefig(os.path.join(save_dir, sbc_filename))
        print(f"Saved calibration histogram: {sbc_filename}")
        plt.close(sbc)  # Close figure to free memory
    except Exception as e:
        print(f"Error generating calibration histogram for {condition_name}: {e}")

    # =====================================================================================
    # Compute recovery metrics
    print(f"\nComputing recovery metrics for {condition_name}...")
    val_sims_params = {k: v for k, v in reshaped_data.items() if k in parameter_names}
    compute_recovery_metrics(post_draws, val_sims_params)
    
    # =====================================================================================
    print(f"Completed analysis for {condition_name}")

print(f"\n{'='*80}")
print("Analysis complete for all conditions!")
print(f"All plots saved to: {save_dir}")
print(f"{'='*80}") 