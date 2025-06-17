# =====================================================================================
# Import modules
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
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
from shared import plots
from results_directed_ddm import generate_predicted_data, posterior_predictive_check

# =====================================================================================
# Load CmdStanMCMC results from CSV
csv_dir = 'directed_model/directed_ddm_fit_results'
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
fit = from_csv(csv_files)

# Create directory to save plots
save_dir = 'Figures'
# Create directory if it does not exist
os.makedirs(save_dir, exist_ok=True) 

# Load true parameters from the simulation
genparam = sio.loadmat('directed_model/data/directed_ddm_simdata.mat')
true_alpha = np.squeeze(genparam['alpha'])
true_tau = np.squeeze(genparam['tau'])
true_beta = np.squeeze(genparam['beta'])
true_eta = np.squeeze(genparam['eta'])
true_mu_z = np.squeeze(genparam['mu_z'])
true_sigma_z = np.squeeze(genparam['sigma_z'])
true_lambda = np.squeeze(genparam['lambda'])
true_b = np.squeeze(genparam['b'])
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
# Convergence checks
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
# Reshape posterior samples for each parameter
# Get number of samples
n_samples = len(df)

# Initialize dictionaries for the recovery plot
post_draws = {}
val_sims = {}

# Helper function to extract and reshape parameter samples
def extract_parameter_samples(df, param_name, n_participants):
    # Get all columns that start with the parameter name
    cols = [col for col in df.columns if col.startswith(f"{param_name}[")]
    # Sort columns to ensure correct participant order
    cols.sort(key=lambda x: int(re.findall(r'\[(\d+)\]', x)[0]))
    # Extract and reshape samples to (n_participants, n_samples)
    samples = df[cols].values.T
    return samples

# Process each parameter
parameters = {
    "alpha": true_alpha,
    "tau": true_tau,
    "beta": true_beta,
    "eta": true_eta,
    # "mu_z": true_mu_z,
    # "sigma_z": true_sigma_z,
    # "lambda": true_lambda,
    # "b": true_b,
    # "y": true_y,
    # "z": true_z
}

for param_name, true_values in parameters.items():
    # Get posterior samples and reshape to (n_participants, n_samples)
    post_draws[param_name] = extract_parameter_samples(df, param_name, nparts)
    # Store true values
    val_sims[param_name] = true_values

# Create recovery plot
f = plots.recovery(post_draws, val_sims)
f.savefig(f'{save_dir}/alt_recovery_plot.png')

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