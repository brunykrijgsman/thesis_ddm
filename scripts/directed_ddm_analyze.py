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
    extract_parameter_samples,
)
from directed_model.analysis_ppc import posterior_predictive_check
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
# Posterior Predictive Checks
true_params = {
    "alpha": true_alpha,
    "tau": true_tau,
    "beta": true_beta,
    "eta": true_eta,
    "mu_z": true_mu_z,
    "sigma_z": true_sigma_z,
    "lambda": true_lambda,
    "b": true_b,
}

fig = posterior_predictive_check(fit, df, participants, true_y, true_z, nparts, conditions_data=None)

fig.savefig(FIGURES_DIR / "posterior_predictive_checks_combined.png", dpi=300)
plt.close(fig)

# =====================================================================================
# Extended Recovery Metrics
compute_recovery_metrics(post_draws, val_sims)