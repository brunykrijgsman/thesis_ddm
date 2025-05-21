# =====================================================================================
# Import modules
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import bayesflow as bf
from bayesflow.simulators import make_simulator
from bayesflow.adapters import Adapter
import arviz as az

from calibration_histogram import calibration_histogram ⁠

# === Import your simulator module ===
import integrative_ddm_sim as ddm


# === Define checkpoint path ===
CHECKPOINT_PATH = 'checkpoints/jax_integrative_ddm_checkpoint.keras'

# === Set parameter names ===
param_names = ['alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma']

# === Create adapter (must match training) ===
adapter = (
    Adapter()
    .map_structure(lambda x: np.array(x), keys=["choicert", "z"])  # <- convert lists to arrays
    .broadcast("n_obs", to="choicert")
    .as_set(["choicert", "z"])
    .standardize(exclude=["n_obs"])
    .convert_dtype("float64", "float32")
    .concatenate(param_names, into="inference_variables")
    .concatenate(["choicert", "z"], into="summary_variables")
    .rename("n_obs", "inference_conditions")
)

# === Load simulator ===
simulator = make_simulator([ddm.prior, ddm.likelihood], meta_fn=lambda: dict(n_obs=100))

# === Load trained model ===
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

print("Loading trained model...")
approximator = keras.saving.load_model(CHECKPOINT_PATH)

# === Simulate validation data ===
n_param_sets = 1000
n_obs = 100
n_posterior_samples = 1000

print(f"Generating {n_param_sets} validation datasets...")
val_data = simulator.sample(n_param_sets, n_obs=n_obs)

# === Adapt data ===
adapted_data = adapter(val_data)
observations = adapted_data["summary_variables"]
conditions = adapted_data["inference_conditions"]

# === Perform inference ===
print("Running inference...")
posterior_samples = approximator.sample(
    observations=observations,
    conditions=conditions,
    num_samples=n_posterior_samples
)

# === Compute posterior means and reshape for ArviZ ===
posterior_means = {k: v.mean(axis=1) for k, v in posterior_samples.items()}
posterior_array = np.stack([posterior_samples[p] for p in param_names], axis=-1)
posterior_array = posterior_array.transpose(1, 0, 2)  # Shape: (n_param_sets, n_samples, n_params)

# === Convert to InferenceData ===
idata = az.from_dict(
    posterior={param_names[i]: posterior_array[:, :, i] for i in range(len(param_names))}
)

# === ArviZ diagnostics summary ===
summary = az.summary(idata, var_names=param_names)
print(summary)

# === Convergence checks ===
def check_convergence(summary_df, rhat_thresh=1.1, ess_thresh=400):
    failed_rhat = summary_df[summary_df['r_hat'] >= rhat_thresh]
    failed_ess = summary_df[summary_df['ess_bulk'] <= ess_thresh]

    passed = failed_rhat.empty and failed_ess.empty

    if passed:
        print("All parameters passed convergence diagnostics (R-hat < 1.1, ESS > 400).")
    else:
        print("Some parameters did NOT pass convergence diagnostics:")
        if not failed_rhat.empty:
            print(" - High R-hat:\n", failed_rhat[['r_hat']])
        if not failed_ess.empty:
            print(" - Low ESS:\n", failed_ess[['ess_bulk']])

        # Save for inspection
        os.makedirs("results", exist_ok=True)
        failed_summary = summary_df.loc[failed_rhat.index.union(failed_ess.index)]
        failed_summary.to_csv("results/failed_convergence_diagnostics.csv")

    return passed

convergence_passed = check_convergence(summary)

# === Save diagnostics ===
os.makedirs("results", exist_ok=True)
summary.to_csv("results/diagnostics_summary.csv")

# === Plot inferred parameter means vs. true ===
true_params = np.column_stack([val_data["parameters"][p] for p in param_names])
inferred_params = np.column_stack([posterior_means[p] for p in param_names])

for i, p in enumerate(param_names):
    plt.figure(figsize=(8, 4))
    sns.histplot(inferred_params[:, i], bins=30, kde=True, label="Inferred Mean", color="skyblue")
    plt.axvline(true_params[:, i].mean(), color='orange', linestyle='--', label='True Mean')
    plt.title(f"Inferred vs. True Parameter Mean: {p}")
    plt.xlabel(p)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{p}_posterior_mean_comparison.png")
    plt.close()

print("Validation complete. Results saved in 'results/' directory.")