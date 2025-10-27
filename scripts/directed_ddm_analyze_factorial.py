"""
Analyze directed DDM results for factorial data

Usage:
> uv run scripts/directed_ddm_analyze_factorial.py --prefix ddmdata_
> uv run scripts/directed_ddm_analyze_factorial.py --prefix cross_directed_
"""

# =====================================================================================
# Import modules
from pathlib import Path
import sys
import argparse

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import from_csv
import scipy.io as sio
from directed_model.analysis import (
    check_convergence,
    plot_trace_grids,
    posterior_predictive_check,
    extract_parameter_samples,
)
from shared.plots import recovery_plot, compute_recovery_metrics

# =====================================================================================
# Set up paths
DIRECTED_MODEL_DIR = PROJECT_ROOT / "directed_model"
DATA_DIR = DIRECTED_MODEL_DIR / "data_new_sigma_z"
RESULTS_DIR = DIRECTED_MODEL_DIR / "results_new_sigma_z"
FIGURES_ROOT = DIRECTED_MODEL_DIR / "figures_new_sigma_z"
FIGURES_ROOT.mkdir(exist_ok=True)

# =====================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze directed DDM results for factorial data')
parser.add_argument('--prefix', type=str, default='ddmdata_', 
                    help='Glob prefix for data files (default: ddmdata_)')
args = parser.parse_args()

# Get all .mat files using the specified prefix
mat_files = sorted(DATA_DIR.glob(f"{args.prefix}*.mat"))

if not mat_files:
    print(f"No .mat files found with prefix '{args.prefix}'!")
    sys.exit()

print(f"Found {len(mat_files)} .mat files to process with prefix '{args.prefix}'")

# =====================================================================================
# Initialize combined lambda datasets
combined_lambda_estimates = {}  # estimates[condition_name] = np.ndarray
combined_lambda_targets = {}    # targets[condition_name] = np.ndarray

# Loop over all .mat files
for mat_path in mat_files:
    base_name = mat_path.stem
    condition_name = base_name.replace(f"{args.prefix}", "")
    
    # Handle cases where there's an extra 'ddm_data_' prefix after removing the main prefix
    if condition_name.startswith("ddm_data_"):
        condition_name = condition_name.replace("ddm_data_", "")
    
    print(f"\n=== Analyzing condition: {condition_name} ===")

    # Prepare directories
    result_path = RESULTS_DIR / base_name
    if not result_path.is_dir():
        print(f"Results directory not found for {condition_name}, skipping.")
        continue
        
    csv_files = list(result_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {result_path}, skipping.")
        continue

    fig_dir = FIGURES_ROOT / base_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load Stan results
    fit = from_csv([str(p) for p in csv_files])
    df = fit.draws_pd()
    summary = fit.summary()

    # Load simulation ground truth
    genparam = sio.loadmat(mat_path)
    nparts = int(genparam["nparts"].item())
    participants = np.squeeze(genparam["participant"]).astype(int)
    true_vals = {
        "alpha": np.squeeze(genparam["alpha"]),
        "tau": np.squeeze(genparam["tau"]),
        "beta": np.squeeze(genparam["beta"]),
        "eta": np.squeeze(genparam["eta"]),
        "mu_z": np.squeeze(genparam["mu_z"]),
        "sigma_z": np.squeeze(genparam["sigma_z"]),
        "lambda": np.squeeze(genparam["lambda_param"]),
        "b": np.squeeze(genparam["b"]),
        "y": np.squeeze(genparam["y"]),
        "z": np.squeeze(genparam["z"]),
    }

    # =====================================================================================
    # Run convergence diagnostics (R-Hat and ESS)
    check_convergence(summary)

    # Trace plots
    trace_params = ("alpha", "tau", "beta", "eta", "mu_z", "sigma_z", "lambda", "b")
    trace_figures = plot_trace_grids(df, fit, params_of_interest=trace_params, grid_cols=10)
    for param_name, fig in trace_figures.items():
        fig.savefig(fig_dir / f"trace_plots_{param_name}.png", dpi=300)
        plt.close(fig)

    # =====================================================================================
    # Parameter recovery
    post_draws = {}
    val_sims = {}
    for param_name in ["alpha", "tau", "beta", "eta", "mu_z", "sigma_z", "lambda", "b"]:
        post_draws[param_name] = extract_parameter_samples(df, param_name, nparts)
        val = true_vals[param_name]
        if np.ndim(val) == 0:
            val = np.repeat(val, nparts)
        val_sims[param_name] = val

    # Parameter recovery plots
    combined_lambda_estimates[condition_name] = post_draws["lambda"]
    combined_lambda_targets[condition_name] = val_sims["lambda"]

    # Regular recovery plot
    fig = recovery_plot(post_draws, val_sims)
    fig.savefig(fig_dir / f"{condition_name}_recovery_plot.png", dpi=300)
    plt.close(fig)

    # =====================================================================================
    # Posterior Predictive Checks
    true_params = {
        "alpha": true_vals["alpha"],
        "tau": true_vals["tau"],
        "beta": true_vals["beta"],
        "eta": true_vals["eta"],
        "mu_z": true_vals["mu_z"],
        "sigma_z": true_vals["sigma_z"],
        "lambda": true_vals["lambda"],
        "b": true_vals["b"],
    }

    fig = posterior_predictive_check(
        fit, df, participants, true_vals["y"], true_vals["z"], true_params, nparts
    )
    fig.savefig(fig_dir / "posterior_predictive_checks_combined.png", dpi=300)
    plt.close(fig)

    # =====================================================================================
    # Extended Recovery Metrics
    compute_recovery_metrics(post_draws, val_sims)

# =====================================================================================
# Process combined lambda datasets
# Create condition display title mapping for combined plots
# Create condition display title mapping for combined plots
condition_display_titles = {
    'SNR_low_COUP_low_DIST_gaussian': r'Gaussian, Low SNR, Low Coupling',
    'SNR_low_COUP_low_DIST_laplace': r'Laplace, Low SNR, Low Coupling',
    'SNR_low_COUP_low_DIST_uniform': r'Uniform, Low SNR, Low Coupling',

    'SNR_high_COUP_low_DIST_gaussian': r'Gaussian, High SNR, Low Coupling',
    'SNR_high_COUP_low_DIST_laplace': r'Laplace, High SNR, Low Coupling',
    'SNR_high_COUP_low_DIST_uniform': r'Uniform, High SNR, Low Coupling',

    'SNR_low_COUP_high_DIST_gaussian': r'Gaussian, Low SNR, High Coupling',
    'SNR_low_COUP_high_DIST_laplace': r'Laplace, Low SNR, High Coupling',
    'SNR_low_COUP_high_DIST_uniform': r'Uniform, Low SNR, High Coupling',  

    'SNR_high_COUP_high_DIST_gaussian': r'Gaussian, High SNR, High Coupling',
    'SNR_high_COUP_high_DIST_laplace': r'Laplace, High SNR, High Coupling',
    'SNR_high_COUP_high_DIST_uniform': r'Uniform, High SNR, High Coupling',

    # No SNR conditions
    'no_SNR_COUP_low_DIST_gaussian': r'Gaussian, Low Coupling',
    'no_SNR_COUP_low_DIST_laplace': r'Laplace, Low Coupling',
    'no_SNR_COUP_low_DIST_uniform': r'Uniform, Low Coupling',

    'no_SNR_COUP_high_DIST_gaussian': r'Gaussian, High Coupling',
    'no_SNR_COUP_high_DIST_laplace': r'Laplace, High Coupling',
    'no_SNR_COUP_high_DIST_uniform': r'Uniform, High Coupling',

}

print(f"\n===  Recovery Plot for all lambdas ===")
print(f"Conditions: {list(combined_lambda_estimates.keys())}")
fig_high = recovery_plot(combined_lambda_estimates, combined_lambda_targets, 
                        parameter_display_titles=condition_display_titles, fig_height=12, fig_width=15.5)
fig_high.savefig(FIGURES_ROOT / f"recovery_plot_combined_lambda_{args.prefix}.png", dpi=300)
plt.close(fig_high)
