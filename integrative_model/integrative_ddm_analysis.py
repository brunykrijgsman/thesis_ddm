# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

# =====================================================================================
# Import modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
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

from integrative_model.plots import simulated_data_check, calibration_histogram
from shared.plots import recovery

# Load checkpoint - relative to current file
CHECKPOINT_PATH = os.path.join(current_dir, 'checkpoints', 'jax_simple_integrative_ddm_checkpoint_seed12_mixed_new_sigma_beta.keras')

# Create save directory - relative to current file
save_dir = os.path.join(current_dir, 'Figures')
os.makedirs(save_dir, exist_ok=True)

# Define meta function
def meta():
    return dict(n_obs=100)

# Make simulator
print("Making simulator...")
simulator = make_simulator([ddm.prior, ddm.likelihood], meta_fn=meta)

# Sample simulator draws
sim_draws = simulator.sample(100)
print("Simulator draws keys:", sim_draws.keys())
simulated_data_check(sim_draws)

summary_network = bf.networks.SetTransformer(summary_dim=8)
inference_network = bf.networks.CouplingFlow()

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

# Set the number of posterior draws 
num_samples = 100

# Simulate validation data (unseen during training)
val_sims = simulator.sample(2000)

# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

# Post_draws is a dictionary of draws with one element per named parameters
post_draws.keys() 

# =====================================================================================
# Recovery plot
# Set the number of posterior draws 
num_samples = 2000
# Simulate validation data (unseen during training)
val_sims = simulator.sample(100)
# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

# Plot and save recovery plot
f = recovery(post_draws, val_sims)
f.savefig(os.path.join(save_dir, 'recovery_plot_basic_integrative_model_sigma_mixed.png'))

# =====================================================================================
# Posterior predictive check
# You need to define how to simulate new data given posterior samples
# Check your simulator's interface
# ppc_fig = plot_posterior_predictive_check(
#     make_simulator([ddm.prior, ddm.likelihood], meta_fn=meta), post_draws, val_sims,
#     stat_fn=np.mean,  # or np.std, skew, etc.
#     observed_key="choicert",
# )
# ppc_fig.savefig(os.path.join(save_dir, "posterior_predictive_check.png"))

# =====================================================================================
# Simulation-based calibration
# Define variable names explicitly
parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]

# Set the number of posterior draws 
num_samples = 100

# Simulate validation data (unseen during training)
val_sims = simulator.sample(2000)

# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

# Filter val_sims to only include parameter keys
val_sims_params = {k: v for k, v in val_sims.items() if k in parameter_names}

print({k: v.shape for k, v in post_draws.items()})
print({k: v.shape for k, v in val_sims_params.items()})

ecdf = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=parameter_names,
    difference=True,
    rank_type="distance"
)
ecdf.savefig(os.path.join(save_dir, 'calibration_ecdf_basic_integrative_model_sigma_mixed.png'))


sbc = calibration_histogram(
    estimates=post_draws, 
    targets=val_sims_params,
    variable_keys=parameter_names,

    num_bins=10,
    binomial_interval=0.99,
    label_fontsize=16,
    title_fontsize=18
)
sbc.savefig(os.path.join(save_dir, 'calibration_histogram_basic_integrative_model_sigma_mixed.png'))