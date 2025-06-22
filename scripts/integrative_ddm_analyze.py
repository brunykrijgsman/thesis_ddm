"""
Integrative DDM Analysis for Default Conditions

This script performs analysis of the integrative drift-diffusion model (DDM) 
for default conditions including recovery plots, posterior predictive checks, 
and simulation-based calibration.

Usage:
> uv run python scripts/integrative_ddm_analyze.py
"""

# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

# =====================================================================================
# Import modules
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import keras
import bayesflow as bf
from bayesflow.simulators import make_simulator

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from integrative_model.simulation import prior, likelihood
from integrative_model.analysis import calibration_histogram
from shared.plots import recovery_plot, compute_recovery_metrics

CHECKPOINT = "checkpoint_integrative_ddm_seed_12_new_sigma.keras"

# =====================================================================================
# Setup paths and directories
INTEGRATIVE_MODEL_DIR = PROJECT_ROOT / "integrative_model"
CHECKPOINTS_DIR = INTEGRATIVE_MODEL_DIR / "checkpoints"
FIGURES_ROOT = INTEGRATIVE_MODEL_DIR / "figures"
CHECKPOINT_PATH = CHECKPOINTS_DIR / CHECKPOINT

# Create figures directory for default analysis
figdir = FIGURES_ROOT / "default_analysis"
figdir.mkdir(parents=True, exist_ok=True)

# =====================================================================================
# Setup simulator and approximator
def meta():
    return dict(n_obs=100)

# Make simulator
print("Making simulator...")
simulator = make_simulator([prior, likelihood], meta_fn=meta)

# Load approximator
print("Loading approximator...")
approximator = keras.saving.load_model(CHECKPOINT_PATH)

# =====================================================================================
# Define variable names for analysis
parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]

# =====================================================================================
# Generate samples for analysis
print("Generating samples for analysis...")

# Recovery plot
# Set the number of posterior draws 
num_samples = 2000
# Simulate validation data (unseen during training)
val_sims = simulator.sample(100)
print(f"Validation simulation shapes: {val_sims['alpha'].shape}, {val_sims['choicert'].shape}")

# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)
print(f"Posterior draws shapes: {post_draws['alpha'].shape}")

# =====================================================================================
# Recovery analysis
print("Performing recovery analysis...")

# Generate recovery plot
f = recovery_plot(post_draws, val_sims)
if f is not None:
    recovery_filename = "recovery_plot_new_sigma.png"
    f.savefig(figdir / recovery_filename)
    print(f"Saved recovery plot: {recovery_filename}")

# =====================================================================================
# Simulation-Based Calibration
print("Performing calibration analysis...")
# Set the number of posterior draws 
num_samples = 100
# Simulate validation data (unseen during training)
val_sims = simulator.sample(2000)
# Posterior draws
post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

# Filter val_sims to only include parameter keys
val_sims_params = {k: v for k, v in val_sims.items() if k in parameter_names}

print(f"post_draws keys: {post_draws.keys()}")
print(f"val_sims_params keys: {val_sims_params.keys()}")
print(f"parameter_names: {parameter_names}")

# Calibration ECDF plot
ecdf = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws,
    targets=val_sims_params,
    variable_names=parameter_names,
    difference=True,
    rank_type="distance",
)
ecdf_filename = "calibration_ecdf_new_sigma.png"
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
sbc_filename = "calibration_histogram_new_sigma.png"
sbc.savefig(figdir / sbc_filename)
print(f"Saved calibration histogram: {sbc_filename}")
plt.close(sbc)  # Close figure to free memory   

# =====================================================================================
# Compute recovery metrics
print("\nComputing recovery metrics...")

compute_recovery_metrics(post_draws, val_sims_params)

# =====================================================================================
# Posterior Predictive Checks using validation data
# print("\nPerforming posterior predictive checks...")

# # Use validation data from recovery analysis
# observed_choicert = val_sims['choicert']
# observed_z = val_sims['z']

# print(f"Observed data shapes: choicert {observed_choicert.shape}, z {observed_z.shape}")
# print(f"Using validation data with {observed_choicert.shape[0]} datasets")

# # Generate predictions using posterior samples
# print("Generating posterior predictions...")

# def generate_predictions_from_posterior(posterior_samples, n_datasets, n_obs_per_dataset=100):
#     """Generate predictions using posterior samples"""
#     n_posterior_samples = posterior_samples['alpha'].shape[1]  # Number of posterior draws
    
#     # Initialize arrays for predictions
#     pred_choicert = []
#     pred_z = []
    
#     for dataset_idx in range(n_datasets):
#         # Collect predictions from all posterior samples for this dataset
#         dataset_choicert = []
#         dataset_z = []
        
#         for posterior_idx in range(n_posterior_samples):
#             # Extract parameter values for this dataset and posterior sample
#             params = {
#                 'alpha': posterior_samples['alpha'][dataset_idx, posterior_idx],
#                 'tau': posterior_samples['tau'][dataset_idx, posterior_idx],
#                 'beta': posterior_samples['beta'][dataset_idx, posterior_idx],
#                 'mu_delta': posterior_samples['mu_delta'][dataset_idx, posterior_idx],
#                 'eta_delta': posterior_samples['eta_delta'][dataset_idx, posterior_idx],
#                 'gamma': posterior_samples['gamma'][dataset_idx, posterior_idx],
#                 'sigma': posterior_samples['sigma'][dataset_idx, posterior_idx]
#             }
            
#             # Simulate data using these parameters
#             sim_data = likelihood(**params, n_obs=n_obs_per_dataset)
#             dataset_choicert.extend(sim_data['choicert'])
#             dataset_z.extend(sim_data['z'])
        
#         pred_choicert.extend(dataset_choicert)
#         pred_z.extend(dataset_z)
    
#     return np.array(pred_choicert), np.array(pred_z)

# # Generate predictions using posterior samples from val_sims
# predicted_choicert, predicted_z = generate_predictions_from_posterior(
#     post_draws, observed_choicert.shape[0]
# )

# # Flatten observed arrays for plotting
# observed_choicert_flat = observed_choicert.flatten()
# observed_z_flat = observed_z.flatten()

# # Create posterior predictive check plot (1x2 subplots)
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))

# # Set font sizes
# title_fontsize = 18
# label_fontsize = 14
# legend_fontsize = 12
# tick_fontsize = 12

# # PPC for choicert
# sns.histplot(observed_choicert_flat, label='Observed', stat='density', 
#              color='orange', ax=axes[0], alpha=0.7, bins=30)
# sns.kdeplot(predicted_choicert, label='Predicted', color='blue', ax=axes[0])
# axes[0].set_title("Choice/RT Posterior Predictive Check", fontsize=title_fontsize, fontweight='bold')
# axes[0].set_xlabel("Signed RT", fontsize=label_fontsize)
# axes[0].set_ylabel("Density", fontsize=label_fontsize)
# axes[0].legend(fontsize=legend_fontsize)
# axes[0].tick_params(labelsize=tick_fontsize)
# axes[0].grid(False)

# # PPC for z
# sns.histplot(observed_z_flat, label='Observed', stat='density', 
#              color='orange', ax=axes[1], alpha=0.7, bins=30)
# sns.kdeplot(predicted_z, label='Predicted', color='blue', ax=axes[1])
# axes[1].set_title("P300 Posterior Predictive Check", fontsize=title_fontsize, fontweight='bold')
# axes[1].set_xlabel("P300 Response (z)", fontsize=label_fontsize)
# axes[1].set_ylabel("Density", fontsize=label_fontsize)
# axes[1].legend(fontsize=legend_fontsize)
# axes[1].tick_params(labelsize=tick_fontsize)
# axes[1].grid(False)

# plt.tight_layout()
# ppc_filename = "posterior_predictive_checks.png"
# fig.savefig(figdir / ppc_filename, dpi=300)
# print(f"Saved posterior predictive checks: {ppc_filename}")
# plt.close(fig)

# # Summary statistics
# print(f"\nPosterior Predictive Check Results:")
# print(f"{'='*50}")
# print(f"Choice/RT metrics:")
# print(f"  Mean - Observed: {np.mean(observed_choicert_flat):.3f}, Predicted: {np.mean(predicted_choicert):.3f}")
# print(f"  Std  - Observed: {np.std(observed_choicert_flat):.3f}, Predicted: {np.std(predicted_choicert):.3f}")

# print(f"\nP300 (z) metrics:")
# print(f"  Mean - Observed: {np.mean(observed_z_flat):.3f}, Predicted: {np.mean(predicted_z):.3f}")
# print(f"  Std  - Observed: {np.std(observed_z_flat):.3f}, Predicted: {np.std(predicted_z):.3f}")

# =====================================================================================
print(f"\n{'='*80}")
print("Analysis complete for default conditions!")
print(f"Results saved in: {figdir}")
print(f"{'='*80}") 