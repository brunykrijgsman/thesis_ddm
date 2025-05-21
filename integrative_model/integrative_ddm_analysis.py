# =====================================================================================
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import integrative_ddm_sim as ddm
import bayesflow as bf
from bayesflow.networks import SetTransformer, CouplingFlow
from bayesflow.adapters import Adapter
from bayesflow.simulators import make_simulator
# from bayesflow.diagnostics import plot_posterior_predictive_check
from calibration_histogram import calibration_histogram
import keras

# Load checkpoint
CHECKPOINT_PATH = 'checkpoints/jax_integrative_ddm_checkpoint.keras'

# Create save directory
save_dir = 'Figures'
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
ddm.simulated_data_check(sim_draws)

# Define summary and inference networks
summary_network = bf.networks.SetTransformer(summary_dim=10)
inference_network = bf.networks.CouplingFlow()

# Define adapter
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

# Set the number of posterior draws you want to get
num_samples = 1000

# Simulate validation data (unseen during training)
val_sims = simulator.sample(300)

# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = approximator.sample(conditions=val_sims, num_samples=num_samples)

# post_draws is a dictionary of draws with one element per named parameters
post_draws.keys() 

# Plot recovery plot
f = bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=val_sims,
)
f.savefig('Figures/recovery_plot.png')

# =====================================================================================
# Posterior predictive check
# You need to define how to simulate new data given posterior samples
# Check your simulator's interface
# ppc_fig = plot_posterior_predictive_check(
#     make_simulator([ddm.prior, ddm.likelihood], meta_fn=meta), post_draws, val_sims,
#     stat_fn=np.mean,  # or np.std, skew, etc.
#     observed_key="choicert",
# )
# ppc_fig.savefig("Figures/posterior_predictive_check.png")

# =====================================================================================
# Simulation-based calibration
# Define variable names explicitly
parameter_names = ["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"]

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
ecdf.savefig(f'{save_dir}/calibration_ecdf.png')


sbc = calibration_histogram(
    estimates=post_draws, 
    targets=val_sims_params,
    variable_keys=parameter_names,

    num_bins=10,
    binomial_interval=0.99,
    label_fontsize=16,
    title_fontsize=18
)
sbc.savefig(f'{save_dir}/calibration_histogram.png')