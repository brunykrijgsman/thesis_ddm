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
import keras
import bayesflow as bf
from bayesflow.simulators import make_simulator

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from integrative_model.simulation import prior, likelihood
from integrative_model.analysis import calibration_histogram
from shared.plots import recovery_plot, compute_recovery_metrics

CHECKPOINT = "checkpoint_integrative_ddm_seed_12_200e.keras"

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
    recovery_filename = "recovery_plot_default.png"
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
ecdf_filename = "calibration_ecdf_default.png"
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
sbc_filename = "calibration_histogram_default.png"
sbc.savefig(figdir / sbc_filename)
print(f"Saved calibration histogram: {sbc_filename}")
plt.close(sbc)  # Close figure to free memory   

# =====================================================================================
# Compute recovery metrics
print("\nComputing recovery metrics...")

compute_recovery_metrics(post_draws, val_sims_params)

# =====================================================================================
print(f"\n{'='*80}")
print("Analysis complete for default conditions!")
print(f"Results saved in: {figdir}")
print(f"{'='*80}") 