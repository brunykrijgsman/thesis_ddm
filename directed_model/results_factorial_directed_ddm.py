# =====================================================================================
# Import modules
import os
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path
sys.path.append(os.path.dirname(current_dir))

# Import modules
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns # Delete?
from cmdstanpy import CmdStanMCMC, from_csv
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from analysis_utils import check_convergence, plot_trace_grids, generate_predicted_data, posterior_predictive_check, extract_parameter_samples, compute_credible_interval_coverage
from shared import plots

# =====================================================================================
# Parameters
mat_dir = os.path.join(current_dir, 'data')
results_dir = os.path.join(current_dir, 'Results')
figures_root = os.path.join(current_dir, 'Figures')
os.makedirs(figures_root, exist_ok=True)

# =====================================================================================
# Get all .mat files
mat_files = [f for f in os.listdir(mat_dir) if f.startswith('ddmdata_') and f.endswith('.mat')]

# Get only the high coupling condition (e.g., ddmdata_COUP_high.mat)
# mat_files = [f for f in os.listdir(mat_dir)
#              if f.startswith('ddmdata_') and f.endswith('.mat') and 'COUP_high' in f]

# Loop over all .mat files
for mat_file in mat_files:
    base_name = os.path.splitext(mat_file)[0]
    condition_name = base_name.replace('ddmdata_', '')

    print(f"\n=== Analyzing condition: {condition_name} ===")

    # Prepare directories
    mat_path = os.path.join(mat_dir, mat_file)
    result_path = os.path.join(results_dir, base_name)
    csv_files = [os.path.join(result_path, f) for f in os.listdir(result_path) if f.endswith('.csv')]
    fig_dir = os.path.join(figures_root, condition_name)
    os.makedirs(fig_dir, exist_ok=True)

    # Load Stan results
    fit = from_csv(csv_files)
    df = fit.draws_pd()
    summary = fit.summary()

    # Load simulation ground truth
    genparam = sio.loadmat(mat_path)
    nparts = int(genparam['nparts'].item())
    participants = np.squeeze(genparam['participant']).astype(int)
    true_vals = {
        'alpha': np.squeeze(genparam['alpha']),
        'tau': np.squeeze(genparam['tau']),
        'beta': np.squeeze(genparam['beta']),
        'eta': np.squeeze(genparam['eta']),
        'mu_z': np.squeeze(genparam['mu_z']),
        'sigma_z': np.squeeze(genparam['sigma_z']),
        'lambda': np.squeeze(genparam['lambda_param']),
        'b': np.squeeze(genparam['b']),
        'y': np.squeeze(genparam['y']),
        'z': np.squeeze(genparam['z'])
    }

    # =====================================================================================
    # Run convergence diagnostics (R-Hat and ESS)
    check_convergence(summary)

    # Trace plots
    trace_figures = plot_trace_grids(df, fit, params_of_interest=('alpha', 'tau', 'beta', 'eta', 'mu_z', 'sigma_z', 'lambda', 'b'), grid_cols=10)
    for param_name, fig in trace_figures.items():
        fig.savefig(os.path.join(fig_dir, f'{condition_name}_trace_plots_{param_name}.png'), dpi=300)
        plt.close(fig)

    # =====================================================================================
    # Parameter recovery
    post_draws = {}
    val_sims = {}
    for param_name in ['alpha', 'tau', 'beta', 'eta', 'mu_z', 'sigma_z', 'b', 'lambda']:
        post_draws[param_name] = extract_parameter_samples(df, param_name, nparts)
        val = true_vals[param_name]
        if np.ndim(val) == 0:
            val = np.repeat(val, nparts)  # Replicate the scalar to match number of participants
        val_sims[param_name] = val

    # Parameter recovery plots
    if 'COUP_high' in condition_name:
        print(f"Splitting lambda plots for {condition_name}...")

        lambda_samples = post_draws['lambda']  # Shape: (n_participants, n_draws)
        lambda_true = val_sims['lambda']

        neg_idx = np.where(lambda_true < 0)[0]
        pos_idx = np.where(lambda_true >= 0)[0]

        if len(neg_idx) > 0:
            post_draws['lambda_negative'] = lambda_samples[neg_idx, :]
            val_sims['lambda_negative'] = lambda_true[neg_idx]

        if len(pos_idx) > 0:
            post_draws['lambda_positive'] = lambda_samples[pos_idx, :]
            val_sims['lambda_positive'] = lambda_true[pos_idx]

        # Remove original lambda to avoid triple-plotting
        del post_draws['lambda']
        del val_sims['lambda']

        # Create recovery plot with all parameters (including both lambda groups)
        fig = plots.recovery(post_draws, val_sims)
        fig.savefig(os.path.join(fig_dir, f'{condition_name}_recovery_plot.png'), dpi=300)
        plt.close(fig)

    else:
        # Regular recovery plot
        fig = plots.recovery(post_draws, val_sims)
        fig.savefig(os.path.join(fig_dir, f'{condition_name}_recovery_plot.png'), dpi=300)
        plt.close(fig)

    # =====================================================================================
    # Posterior predictive checks
    n_trials = len(true_vals['z'])
    predicted_y, predicted_z = generate_predicted_data(fit, df, participants, true_vals['y'], n_trials=n_trials)

    # Plot for y
    fig_y = posterior_predictive_check(true_vals['y'], predicted_y, name='y')
    fig_y.savefig(os.path.join(fig_dir, f'{condition_name}_posterior_predictive_check_y.png'), dpi=300)
    plt.close(fig_y)
    
    # Plot for z
    fig_z = posterior_predictive_check(true_vals['z'], predicted_z, name='z')
    fig_z.savefig(os.path.join(fig_dir, f'{condition_name}_posterior_predictive_check_z.png'), dpi=300)
    plt.close(fig_z)

    # Print summary stats
    print(f"y Observed: {np.mean(true_vals['y']):.3f}, Predicted: {np.mean(predicted_y):.3f}")
    print(f"z Observed: {np.mean(true_vals['z']):.3f}, Predicted: {np.mean(predicted_z):.3f}")

    # =====================================================================================
    # Extended Recovery Metrics
    plots.compute_recovery_metrics(post_draws, val_sims)
