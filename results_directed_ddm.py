# Import modules
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanMCMC
import scipy.io as sio
from cmdstanpy import from_csv
from directed_ddm_utils import simul_directed_ddm

# Load CmdStanMCMC results from CSV
csv_dir = 'directed_ddm_fit_results'
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
fit = from_csv(csv_files)

# Load true parameters from the simulation
genparam = sio.loadmat('data/directed_ddm_simdata.mat')
true_alpha = np.squeeze(genparam['alpha'])
true_tau = np.squeeze(genparam['tau'])
true_beta = np.squeeze(genparam['beta'])
true_eta = np.squeeze(genparam['eta'])
true_mu_z = float(genparam['mu_z'][0, 0])
true_sigma_z = float(genparam['sigma_z'][0, 0])
true_lambda = float(genparam['lambda'][0, 0])
true_b = float(genparam['b'][0, 0])
true_y = np.squeeze(genparam['y'])
true_z = np.squeeze(genparam['z'])

# Get number of participants
participants = np.squeeze(genparam['participant']).astype(int)
nparts = participants.max()

# Load Stan input data to get participant count 
with open("directed_ddm_data.json", "r") as f:
    stan_data = json.load(f)

# Extract posterior samples 
df = fit.draws_pd()

# R-hat diagnostics
summary = fit.summary()

print(summary.columns)

# ---------- Convergence summary print functions ----------
def print_rhat_summary(summary_df):
    print("\nAll R-hat values:")
    rhat_sorted = summary_df[['R_hat']].sort_values(by='R_hat', ascending=False)
    print(rhat_sorted)

def print_ess_summary(summary_df):
    print("\nAll ESS values (bulk and tail):")
    ess_sorted = summary_df[['ESS_bulk', 'ESS_tail']].sort_values(by='ESS_bulk', ascending=True)
    print(ess_sorted)

# ---------- Convergence checks ----------
def check_convergence(summary_df, rhat_thresh=1.01, ess_thresh=400):
    rhat_issues = summary_df[summary_df['R_hat'] > rhat_thresh]
    if not rhat_issues.empty:
        print(f"\nParameters with R-hat > {rhat_thresh}:")
        print(rhat_issues[['R_hat']])
    else:
        print(f"\nAll parameters passed R-hat < {rhat_thresh}")

    ess_issues = summary_df[(summary_df['ESS_bulk'] < ess_thresh) | (summary_df['ESS_tail'] < ess_thresh)]
    if not ess_issues.empty:
        print(f"\nParameters with ESS < {ess_thresh}:")
        print(ess_issues[['ESS_bulk', 'ESS_tail']])
    else:
        print(f"\nAll parameters passed ESS > {ess_thresh}")

# ---------- Run diagnostics ----------
# print_rhat_summary(summary)
# print_ess_summary(summary)
# check_convergence(summary)

# --------------- Trace plots ---------------
import re
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import math
import re
from collections import defaultdict
import numpy as np

def plot_trace_grids(df, fit, params_of_interest=('alpha', 'tau', 'beta', 'eta'), grid_cols=5):
    param_cols = df.columns.tolist()
    num_chains = fit.chains

    # Group parameters like alpha[1], alpha[2], ...
    grouped_params = defaultdict(list)
    for col in param_cols:
        match = re.match(r"([a-zA-Z_]+)\[(\d+)\]", col)
        if match:
            base, idx = match.groups()
            if base in params_of_interest:
                grouped_params[base].append((int(idx), col))

    # Plot trace plots for each parameter group
    for param_name, items in grouped_params.items():
        items.sort()
        param_list = [col for _, col in items]
        num_params = len(param_list)

        grid_rows = math.ceil(num_params / grid_cols)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 2), sharex=True)
        axes = axes.flatten()

        for i, param_col in enumerate(param_list):
            values = df[param_col].values.reshape(num_chains, -1)

            for chain in range(num_chains):
                axes[i].plot(values[chain], label=f'Chain {chain + 1}', alpha=0.6)
            
            # Optional: plot posterior mean
            mean_val = df[param_col].mean()
            axes[i].axhline(mean_val, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

            axes[i].set_title(param_col, fontsize=9)
            axes[i].tick_params(labelsize=6)

        # Turn off unused subplots
        for j in range(len(param_list), len(axes)):
            axes[j].axis("off")

        # Add shared labels and title
        fig.text(0.5, 0.04, 'Iteration', ha='center')
        fig.text(0.04, 0.5, 'Parameter Value', va='center', rotation='vertical')
        fig.suptitle(f"Trace Plots for '{param_name}'", fontsize=14)
        plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])

        # Add legend only once
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        plt.show()

# Plot relevant parameter trace plots in grids
# plot_trace_grids(df, fit, params_of_interest=('alpha', 'tau', 'beta', 'eta'), grid_cols=10)

# --------------- Posterior Means (Recovery) ---------------
posterior_alpha = np.array([df[f'alpha[{i+1}]'].mean() for i in range(nparts)])
posterior_tau   = np.array([df[f'tau[{i+1}]'].mean() for i in range(nparts)])
posterior_beta  = np.array([df[f'beta[{i+1}]'].mean() for i in range(nparts)])
posterior_eta   = np.array([df[f'eta[{i+1}]'].mean() for i in range(nparts)])

# Create a dictionary for true values and posterior estimates (including z as individual-level)
recovery_pairs = {
    "Alpha":  (true_alpha, posterior_alpha),
    "Tau":    (true_tau, posterior_tau),
    "Beta":   (true_beta, posterior_beta),
    "Eta":    (true_eta, posterior_eta),
}

# Add 95% credible intervals for individual-level parameters (alpha, tau, beta, eta, z)
individual_recovery = {}
for name in ['alpha', 'tau', 'beta', 'eta']:
    # Get the posterior samples for each individual-level parameter
    samples = np.array([df[f'{name}[{i+1}]'].values for i in range(nparts)])
    
    # Compute mean and credible intervals
    mean = samples.mean(axis=1)
    ci_lower = np.percentile(samples, 5, axis=1)
    ci_upper = np.percentile(samples, 95, axis=1)
    
    individual_recovery[name] = {
        'mean': mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# Define group-level parameter names
group_param_names = ['mu_z', 'sigma_z', 'lambda', 'b']

# Extract true and posterior samples for group-level parameters
group_recovery = {}
for name in group_param_names:
    true_val = eval(f"true_{name}")
    samples = df[name]
    mean = samples.mean()
    ci_lower = np.percentile(samples, 5)
    ci_upper = np.percentile(samples, 95)
    group_recovery[name] = {
        'true': true_val,
        'mean': mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# Flexible recovery plot function with credible intervals
def plot_recovery(true_vals, estimated_vals, param_name, ci_lower=None, ci_upper=None):
    # Convert to arrays if they're floats or scalars
    true_vals = np.atleast_1d(true_vals)
    estimated_vals = np.atleast_1d(estimated_vals)

    # Create plot for individual parameters
    plt.figure(figsize=(7, 7))

    if ci_lower is not None and ci_upper is not None:
        # Plot estimated means with error bars (credible intervals)
        plt.errorbar(estimated_vals, true_vals, xerr=[estimated_vals - ci_lower, ci_upper - estimated_vals], fmt='o', label=f'{param_name} (CI)', color='dodgerblue')
    else:
        # Regular scatter plot
        sns.regplot(x=true_vals, y=estimated_vals, line_kws={'color': 'red'})

    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Estimated {param_name}")
    plt.title(f"Parameter Recovery: {param_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plotting all parameters with CI for both group-level and individual-level parameters
for param_name, (true_vals, estimated_vals) in recovery_pairs.items():
    if param_name in individual_recovery:
        # For individual-level parameters, include CI (credible intervals)
        ci_lower = individual_recovery[param_name]['ci_lower']
        ci_upper = individual_recovery[param_name]['ci_upper']
        plot_recovery(true_vals, estimated_vals, param_name, ci_lower, ci_upper)
    elif param_name in group_recovery:
        # For group-level parameters, include CI (credible intervals)
        ci_lower = group_recovery[param_name]['ci_lower']
        ci_upper = group_recovery[param_name]['ci_upper']
        plot_recovery(true_vals, estimated_vals, param_name, ci_lower, ci_upper)
    else:
        # For parameters without CI
        plot_recovery(true_vals, estimated_vals, param_name)


# --------------- Posterior Predictive Checks ---------------
# Generate predicted data based on the posterior samples for parameters
def generate_predicted_data(fit, df, n_trials):
    """
    Generate predicted data based on the posterior samples for parameters.
    """
    predicted_data = []

    lambda_sample = df['lambda'].mean() 
    mu_z_sample = df['mu_z'].mean()           
    sigma_z_sample = df['sigma_z'].mean()     

    for i in range(n_trials):
        participant = participants[i]
        alpha_sample = df[f'alpha[{participant}]'].mean()  
        tau_sample = df[f'tau[{participant}]'].mean()      
        beta_sample = df[f'beta[{participant}]'].mean()    
        eta_sample = df[f'eta[{participant}]'].mean()      
        b_sample = df['b'].mean()

        # Simulate latent z using the DDM with the sampled parameters
        _, _, simulated_z = simul_directed_ddm(
            ntrials=1,
            alpha=alpha_sample,
            tau=tau_sample,
            beta=beta_sample,
            eta=eta_sample,
            lambda_param=lambda_sample,
            mu_z=mu_z_sample,
            sigma_z=sigma_z_sample,
            b=b_sample
        )
        
        predicted_data.append(simulated_z[0])
    
    return np.array(predicted_data)

# Posterior Predictive Check Function
def posterior_predictive_check(observed_data, predicted_data):
    """
    This function performs a posterior predictive check by comparing the observed data
    with the predicted data generated from the model's posterior samples.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(observed_data, label='Observed', stat='density', kde=True, color='blue')
    sns.histplot(predicted_data, label='Predicted', stat='density', kde=True, color='orange')
    plt.title("Posterior Predictive Check")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Generate posterior predictive data
predicted_data = generate_predicted_data(fit, df, n_trials=len(true_z))

# Perform posterior predictive check
posterior_predictive_check(true_z, predicted_data)