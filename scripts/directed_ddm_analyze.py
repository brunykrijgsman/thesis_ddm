from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import from_csv
import scipy.io as sio

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from directed_model.analysis import (
    check_convergence,
    plot_trace_grids,
    generate_predicted_data,
    posterior_predictive_check,
    extract_parameter_samples,
)
from directed_model.simulation import simul_directed_ddm
from shared.plots import recovery_plot, compute_recovery_metrics

# =====================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze directed DDM results for a specific model')
parser.add_argument('--model', type=str, default='directed_ddm_base', 
                    help='Model name (default: directed_ddm_base)')
args = parser.parse_args()

model_name = args.model

# =====================================================================================
# Set up paths
DIRECTED_MODEL_DIR = PROJECT_ROOT / "directed_model"
DATA_DIR = DIRECTED_MODEL_DIR / "data"
RESULTS_DIR = DIRECTED_MODEL_DIR / "results" / model_name
FIGURES_DIR = DIRECTED_MODEL_DIR / "figures" / model_name

# Check if data file exists
data_file = DATA_DIR / f"{model_name}.mat"
if not data_file.exists():
    print(f"Error: Data file {data_file} does not exist!")
    sys.exit(1)

# Check if results directory exists
if not RESULTS_DIR.exists():
    print(f"Error: Results directory {RESULTS_DIR} does not exist!")
    sys.exit(1)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Analyzing model: {model_name}")
print(f"Data file: {data_file}")
print(f"Results directory: {RESULTS_DIR}")

# =====================================================================================
# Load data and results
# Load true parameters from the simulation
genparam = sio.loadmat(data_file)
print("Available keys in .mat file:", genparam.keys())
true_alpha = np.squeeze(genparam["alpha"])
true_tau = np.squeeze(genparam["tau"])
true_beta = np.squeeze(genparam["beta"])
true_eta = np.squeeze(genparam["eta"])
true_mu_z = np.squeeze(genparam["mu_z"])
true_sigma_z = np.squeeze(genparam["sigma_z"])
true_lambda = np.squeeze(genparam["lambda_param"])
true_b = np.squeeze(genparam["b"])
true_y = np.squeeze(genparam["y"])
true_z = np.squeeze(genparam["z"])

# Get number of participants
participants = np.squeeze(genparam["participant"]).astype(int)
nparts = participants.max()

# Load CmdStanMCMC results from CSV
csv_files = sorted(RESULTS_DIR.glob("*.csv"))
if not csv_files:
    print(f"No CSV files found in {RESULTS_DIR}, exiting.")
    sys.exit()
fit = from_csv([str(p) for p in csv_files])

# Extract posterior samples
df = fit.draws_pd()

# R-hat diagnostics
summary = fit.summary()

# =====================================================================================
# Print R-hat and ESS summary
check_convergence(summary)

# =====================================================================================
# Plot relevant parameter trace plots in grids
trace_figures = plot_trace_grids(
    df,
    fit,
    params_of_interest=("alpha", "tau", "beta", "eta", "mu_z", "sigma_z", "lambda", "b"),
    grid_cols=10,
)

# Save trace plot figures
for param_name, fig in trace_figures.items():
    fig.savefig(FIGURES_DIR / f"trace_plots_{param_name}.png", dpi=300)
    plt.close(fig)

# =====================================================================================
# Parameters to plot recovery for
params = {
    "alpha": true_alpha,
    "tau": true_tau,
    "beta": true_beta,
    "eta": true_eta,
    "mu_z": true_mu_z,
    "sigma_z": true_sigma_z,
    "lambda": true_lambda,
    "b": true_b,
}

# Initialize dictionaries for the recovery plot
post_draws = {}
val_sims = {}

for param_name, true_values in params.items():
    # Get posterior samples and reshape to (n_participants, n_samples)
    post_draws[param_name] = extract_parameter_samples(df, param_name, nparts)
    # Store true values
    val_sims[param_name] = true_values

# Create recovery plot
f = recovery_plot(post_draws, val_sims)
f.savefig(FIGURES_DIR / "recovery_plot_directed_ddm.png", dpi=300)
plt.close(f)

# =====================================================================================
# Posterior Predictive Checks with in-sample and new simulated out-of-sample data
from directed_model.simulation import simul_directed_ddm

# In-sample data (original training data)
train_y = true_y
train_z = true_z
train_participants = participants

# Generate completely new out-of-sample data using the same true parameters
print("Generating new out-of-sample data using true parameters...")
# Save current random state to restore later
current_state = np.random.get_state()
# Set a different seed for PPC simulation only
np.random.seed(5202)

n_new_trials = len(true_y)  # Same number of trials as original
test_y = []
test_z = []
test_participants = []

for p in range(nparts):
    # Use true parameters for this participant to simulate new data
    n_trials_per_participant = np.sum(participants == (p + 1))
    
    # Simulate new data using true parameters
    simulated_y, _, simulated_z = simul_directed_ddm(
        ntrials=n_trials_per_participant,
        alpha=true_alpha[p],
        tau=true_tau[p],
        beta=true_beta[p],
        eta=true_eta[p],
        lambda_param=true_lambda[p],
        mu_z=true_mu_z[p],
        sigma_z=true_sigma_z[p],
        b=true_b[p]
    )
    
    # Store the new simulated data
    test_y.extend(simulated_y)
    test_z.extend(simulated_z)
    test_participants.extend([p + 1] * n_trials_per_participant)

# Restore the original random state
np.random.set_state(current_state)

# Convert to arrays
test_y = np.array(test_y)
test_z = np.array(test_z)
test_participants = np.array(test_participants)

print(f"Generated {len(test_y)} new out-of-sample trials")

# In-sample predictions (on original training data)
predicted_y_train, predicted_z_train = generate_predicted_data(
    fit, df, train_participants, train_y, n_trials=len(train_z)
)

# Out-of-sample predictions (on newly simulated data)
predicted_y_test, predicted_z_test = generate_predicted_data(
    fit, df, test_participants, test_y, n_trials=len(test_z)
)

# Create combined posterior predictive check plot (2x2 subplots)
import seaborn as sns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Set font sizes
title_fontsize = 18
label_fontsize = 14
legend_fontsize = 12
tick_fontsize = 12

# In-sample PPC for y
sns.histplot(train_y, label='Observed', stat='density', color='orange', ax=axes[0, 0])
sns.kdeplot(predicted_y_train, label='Predicted', color='blue', ax=axes[0, 0])
axes[0, 0].set_title("In-Sample Choice/RT (y)", fontsize=title_fontsize, fontweight='bold')
axes[0, 0].set_xlabel("Signed RT", fontsize=label_fontsize)
axes[0, 0].set_ylabel("Density", fontsize=label_fontsize)
axes[0, 0].legend(fontsize=legend_fontsize)
axes[0, 0].tick_params(labelsize=tick_fontsize)
axes[0, 0].grid(False)

# In-sample PPC for z
sns.histplot(train_z, label='Observed', stat='density', color='orange', ax=axes[0, 1])
sns.kdeplot(predicted_z_train, label='Predicted', color='blue', ax=axes[0, 1])
axes[0, 1].set_title("In-Sample Latent Variable (z)", fontsize=title_fontsize, fontweight='bold')
axes[0, 1].set_xlabel("Latent Variable (z)", fontsize=label_fontsize)
axes[0, 1].set_ylabel("Density", fontsize=label_fontsize)
axes[0, 1].tick_params(labelsize=tick_fontsize)
axes[0, 1].grid(False)

# Out-of-sample PPC for y
sns.histplot(test_y, label='Observed', stat='density', color='orange', ax=axes[1, 0])
sns.kdeplot(predicted_y_test, label='Predicted', color='blue', ax=axes[1, 0])
axes[1, 0].set_title("Out-of-Sample Choice/RT (y)", fontsize=title_fontsize, fontweight='bold')
axes[1, 0].set_xlabel("Signed RT", fontsize=label_fontsize)
axes[1, 0].set_ylabel("Density", fontsize=label_fontsize)
axes[1, 0].tick_params(labelsize=tick_fontsize)
axes[1, 0].grid(False)

# Out-of-sample PPC for z
sns.histplot(test_z, label='Observed', stat='density', color='orange', ax=axes[1, 1])
sns.kdeplot(predicted_z_test, label='Predicted', color='blue', ax=axes[1, 1])
axes[1, 1].set_title("Out-of-Sample Latent Variable (z)", fontsize=title_fontsize, fontweight='bold')
axes[1, 1].set_xlabel("Latent Variable (z)", fontsize=label_fontsize)
axes[1, 1].set_ylabel("Density", fontsize=label_fontsize)
axes[1, 1].tick_params(labelsize=tick_fontsize)
axes[1, 1].grid(False)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "posterior_predictive_checks_combined.png", dpi=300)
plt.close(fig)

# Summary statistics
print(f"In-sample metrics for y:")
print(f"Mean observed y: {np.mean(train_y):.2f}, predicted: {np.mean(predicted_y_train):.2f}")
print(f"Variance observed y: {np.var(train_y):.2f}, predicted: {np.var(predicted_y_train):.2f}")

print(f"In-sample metrics for z:")
print(f"Mean observed z: {np.mean(train_z):.2f}, predicted: {np.mean(predicted_z_train):.2f}")
print(f"Variance observed z: {np.var(train_z):.2f}, predicted: {np.var(predicted_z_train):.2f}")

print(f"Out-of-sample metrics for y:")
print(f"Mean observed y: {np.mean(test_y):.2f}, predicted: {np.mean(predicted_y_test):.2f}")
print(f"Variance observed y: {np.var(test_y):.2f}, predicted: {np.var(predicted_y_test):.2f}")

print(f"Out-of-sample metrics for z:")
print(f"Mean observed z: {np.mean(test_z):.2f}, predicted: {np.mean(predicted_z_test):.2f}")
print(f"Variance observed z: {np.var(test_z):.2f}, predicted: {np.var(predicted_z_test):.2f}")

# =====================================================================================
# Extended Recovery Metrics
compute_recovery_metrics(post_draws, val_sims)