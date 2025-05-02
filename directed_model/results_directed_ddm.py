# =====================================================================================
# Import modules
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns # Delete?
from cmdstanpy import CmdStanMCMC, from_csv
import scipy.io as sio
from directed_ddm_utils import simul_directed_ddm
import re
import math
from collections import defaultdict
import sys

# =====================================================================================
# Load CmdStanMCMC results from CSV
csv_dir = 'directed_ddm_fit_results'
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
fit = from_csv(csv_files)

# Create directory to save plots
save_dir = 'Figures'
# Create directory if it does not exist
os.makedirs(save_dir, exist_ok=True) 

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

# Extract posterior samples 
df = fit.draws_pd()

# R-hat diagnostics
summary = fit.summary()

# =====================================================================================
# Checking R-hat and ESS values
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

# Print R-hat and ESS summary
check_convergence(summary)

# =====================================================================================
# Plot trace plots for each parameter
def plot_trace_grids(df, fit, params_of_interest=('alpha', 'tau', 'beta', 'eta'), grid_cols=5, save_dir=save_dir):
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

        # Calculate the number of rows needed for the grid
        grid_rows = math.ceil(num_params / grid_cols)

        # Create the figure and axes
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 4, grid_rows * 2), sharex=True)
        axes = axes.flatten()

        # Plot each parameter
        for i, param_col in enumerate(param_list):
            values = df[param_col].values.reshape(num_chains, -1)

            # Plot each chain
            for chain in range(num_chains):
                axes[i].plot(values[chain], label=f'Chain {chain + 1}', alpha=0.6)
            
            # Plot posterior mean
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

        # Save the plot
        plt.savefig(os.path.join(save_dir, f'trace_plots_{param_name}.png'), dpi=300)
        plt.close()

# Plot relevant parameter trace plots in grids
plot_trace_grids(df, fit, params_of_interest=('alpha', 'tau', 'beta', 'eta'), grid_cols=10, save_dir=save_dir)

# =====================================================================================
# Compute posterior means for each parameter
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
    
    # Store the results
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
def plot_recovery(true_vals, estimated_vals, param_name, ci_lower=None, ci_upper=None, save_dir=save_dir):
    # Convert to arrays if they're floats or scalars
    true_vals = np.atleast_1d(true_vals)
    estimated_vals = np.atleast_1d(estimated_vals)

    # Calculate correlation between true and estimated values
    correlation = np.corrcoef(true_vals, estimated_vals)[0, 1]

    # Create plot for individual parameters
    plt.figure(figsize=(7, 7))

    # Plot estimated means with error bars (credible intervals)
    if ci_lower is not None and ci_upper is not None:
        plt.errorbar(estimated_vals, true_vals, xerr=[estimated_vals - ci_lower, ci_upper - estimated_vals], fmt='o', label=f'{param_name} (CI)', color='dodgerblue')
    else:
        # Regular scatter plot
        sns.regplot(x=true_vals, y=estimated_vals, line_kws={'color': 'red'})

    # Plot details
    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Estimated {param_name}")
    plt.title(f"Parameter Recovery: {param_name} (r = {correlation:.2f})")
    plt.grid(False)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'recovery_plot_{param_name}.png'), dpi=300)
    plt.close()

# Plotting all parameters with CI for both group-level and individual-level parameters
for param_name, (true_vals, estimated_vals) in recovery_pairs.items():
    if param_name in individual_recovery:
        # For individual-level parameters, include CI (credible intervals)
        ci_lower = individual_recovery[param_name]['ci_lower']
        ci_upper = individual_recovery[param_name]['ci_upper']
        plot_recovery(true_vals, estimated_vals, param_name, ci_lower, ci_upper, save_dir)
    elif param_name in group_recovery:
        # For group-level parameters, include CI (credible intervals)
        ci_lower = group_recovery[param_name]['ci_lower']
        ci_upper = group_recovery[param_name]['ci_upper']
        plot_recovery(true_vals, estimated_vals, param_name, ci_lower, ci_upper, save_dir)
    else:
        # For parameters without CI
        plot_recovery(true_vals, estimated_vals, param_name, save_dir)

# =====================================================================================
# Posterior Predictive Checks
def generate_predicted_data(fit, df, n_trials):
    """
    Generate predicted data based on the posterior samples for parameters.
    """
    # Generate predicted data
    predicted_y = []
    predicted_z = []

    # Compute mean values for parameters
    lambda_sample = df['lambda'].mean() 
    mu_z_sample = df['mu_z'].mean()           
    sigma_z_sample = df['sigma_z'].mean()     

    # Generate predicted data for each trial
    for i in range(n_trials):
        participant = participants[i]
        alpha_sample = df[f'alpha[{participant}]'].mean()  
        tau_sample = df[f'tau[{participant}]'].mean()      
        beta_sample = df[f'beta[{participant}]'].mean()    
        eta_sample = df[f'eta[{participant}]'].mean()      
        b_sample = df['b'].mean()
        y_sample = true_y[i]

        # Simulate signed RT and latent z using the DDM with the sampled parameters
        simulated_y, _, simulated_z = simul_directed_ddm(
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
        
        # Store the predicted values
        predicted_y.append(simulated_y[0])
        predicted_z.append(simulated_z[0])
    
    # Return the predicted values as arrays
    return np.array(predicted_y), np.array(predicted_z)

# Posterior Predictive Check Function
def posterior_predictive_check(observed_data, predicted_data, name, save_dir=save_dir):  
    """
    This function performs a posterior predictive check by comparing the observed data
    with the predicted data generated from the model's posterior samples.
    """
    # Create a figure for the posterior predictive check
    plt.figure(figsize=(8, 6))

    # Plot the observed data
    sns.histplot(observed_data, label='Observed', stat='density', kde=True, color='blue')

    # Plot the predicted data
    sns.histplot(predicted_data, label='Predicted', stat='density', kde=True, color='orange')

    # Add title and legend
    plt.title("Posterior Predictive Check")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'posterior_predictive_check_{name}.png'), dpi=300)
    plt.close()

# Checks for y and z
# Generate posterior predictive data
predicted_y, predicted_z = generate_predicted_data(fit, df, n_trials=len(true_z))

# Perform posterior predictive check for y
posterior_predictive_check(true_y, predicted_y, name='y', save_dir=save_dir)

# Summary statistics for y
print(f"Mean of observed y: {np.mean(true_y)}")
print(f"Mean of predicted y: {np.mean(predicted_y)}")
print(f"Variance of observed y: {np.var(true_y)}")
print(f"Variance of predicted y: {np.var(predicted_y)}")

# Perform posterior predictive check for z  
posterior_predictive_check(true_z, predicted_z, name='z', save_dir=save_dir)

# Summary statistics for z
print(f"Mean of observed z: {np.mean(true_z)}")
print(f"Mean of predicted z: {np.mean(predicted_z)}")
print(f"Variance of observed z: {np.var(true_z)}")
print(f"Variance of predicted z: {np.var(predicted_z)}")

# Compute Bayesian p-value
def compute_bayesian_p_value(observed_data, predicted_data):
    """
    Compute the Bayesian p-value by comparing the observed and predicted distributions.
    """
    # Compute the absolute difference between observed and predicted means
    observed_diff = np.abs(np.mean(observed_data) - np.mean(predicted_data))

    # Compute the Bayesian p-value by comparing how extreme the observed difference is
    p_value = np.mean(np.abs(predicted_data - np.mean(predicted_data)) >= observed_diff)
    
    return p_value

# Bayesian p-value for y
p_value_y = compute_bayesian_p_value(true_y, predicted_y)
print(f"Bayesian p-value for y: {p_value_y}")

# Bayesian p-value for z
p_value_z = compute_bayesian_p_value(true_z, predicted_z)
print(f"Bayesian p-value for z: {p_value_z}")
