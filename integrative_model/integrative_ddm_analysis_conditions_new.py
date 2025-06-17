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

# Define variable names for analysis
parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]

# Simulate validation data once (unseen during training)
val_sims = simulator.sample(50)

# =====================================================================================
# Loop through all MATLAB files
for matlab_file in matlab_files:

    # Extract condition name from filename
    filename = os.path.basename(matlab_file)
    condition_name = filename.replace('integrative_ddm_data_', '').replace('.mat', '')
    
    # Create condition-specific subfolder
    condition_save_dir = os.path.join(save_dir, condition_name)
    os.makedirs(condition_save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Analyzing condition: {condition_name}")
    print(f"File: {filename}")
    print(f"Saving plots to: {condition_save_dir}")
    print(f"{'='*80}")

    # Load and prepare data for current condition
    data = sio.loadmat(matlab_file)
    nparts = data['nparts'][0][0]
    ntrials = data['ntrials'][0][0]

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

    # Fill reshaped data
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

    # Generate posterior draws
    post_draws = approximator.sample(conditions=reshaped_data, num_samples=ntrials)

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
    f = recovery(split_estimates, split_targets)
    f.savefig(os.path.join(condition_save_dir, 'recovery_plot.png'))
    plt.close(f)
    
    # =====================================================================================
    # Generate calibration plots
    parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]
    
    # Calibration ECDF plot
    ecdf = bf.diagnostics.plots.calibration_ecdf(
        estimates=post_draws, 
        targets=reshaped_data,
        variable_names=parameter_names,
        difference=True,
        rank_type="distance"
    )
    ecdf.savefig(os.path.join(condition_save_dir, 'calibration_ecdf.png'))
    plt.close(ecdf)
    
    # Calibration histogram
    sbc = calibration_histogram(
        estimates=post_draws, 
        targets={k: v for k, v in reshaped_data.items() if k in parameter_names},
        variable_keys=parameter_names,
        num_bins=10,
        binomial_interval=0.99,
        label_fontsize=16,
        title_fontsize=18
    )
    sbc.savefig(os.path.join(condition_save_dir, 'calibration_histogram.png'))
    plt.close(sbc)

    # =====================================================================================
    # Compute recovery metrics
    if high_coupling:
        parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma_negative", "gamma_positive", "sigma"]
    else:
        parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]

    val_sims_params = {k: v for k, v in split_targets.items() if k in parameter_names}
    compute_recovery_metrics(split_estimates, val_sims_params)

# =====================================================================================
# Main function
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("Starting integrative DDM analysis...")
    print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"All plots saved to: {save_dir}")
    print(f"{'='*80}")