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
from shared.plots import recovery_plot,compute_recovery_metrics

# =====================================================================================
# Set up paths
DIRECTED_MODEL_DIR = PROJECT_ROOT / "directed_model"
DATA_DIR = DIRECTED_MODEL_DIR / "data"
RESULTS_DIR = DIRECTED_MODEL_DIR / "results"
FIGURES_ROOT = DIRECTED_MODEL_DIR / "figures"
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

# Loop over all .mat files
for mat_path in mat_files:
    base_name = mat_path.stem
    condition_name = base_name.replace(f"{args.prefix}", "")
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

    fig_dir = FIGURES_ROOT / condition_name
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
    if "COUP_high" in condition_name:
        print(f"Splitting lambda plots for {condition_name}...")

        lambda_samples = post_draws["lambda"]  # Shape: (n_participants, n_draws)
        lambda_true = val_sims["lambda"]

        neg_idx = np.where(lambda_true < 0)[0]
        pos_idx = np.where(lambda_true >= 0)[0]

        if len(neg_idx) > 0:
            post_draws["lambda_negative"] = lambda_samples[neg_idx, :]
            val_sims["lambda_negative"] = lambda_true[neg_idx]

        if len(pos_idx) > 0:
            post_draws["lambda_positive"] = lambda_samples[pos_idx, :]
            val_sims["lambda_positive"] = lambda_true[pos_idx]

        # Remove original lambda to avoid triple-plotting
        del post_draws["lambda"]
        del val_sims["lambda"]

        # Create recovery plot with all parameters (including both lambda groups)
        fig = recovery_plot(post_draws, val_sims)
        fig.savefig(fig_dir / f"{condition_name}_recovery_plot.png", dpi=300)
        plt.close(fig)

    else:
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