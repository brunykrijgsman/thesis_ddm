# =====================================================================================
# Analysis utilities for directed DDM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from collections import defaultdict

from directed_model.simulation import simul_directed_ddm

# =====================================================================================
# Helper function to extract and reshape parameter samples for the recovery plots
def extract_parameter_samples(df, param_name, n_participants):
    # Get all columns that start with the parameter name
    cols = [col for col in df.columns if col.startswith(f"{param_name}[")]
    # Sort columns to ensure correct participant order
    cols.sort(key=lambda x: int(re.findall(r'\[(\d+)\]', x)[0]))
    # Extract and reshape samples to (n_participants, n_samples)
    samples = df[cols].values.T
    return samples

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

# =====================================================================================
# Plot trace plots for each parameter
def plot_trace_grids(df, fit, params_of_interest=('alpha', 'tau', 'beta', 'eta', 'mu_z', 'sigma_z', 'lambda', 'b'), grid_cols=5):
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

    figures = {}
    
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

        figures[param_name] = fig

    return figures

# =====================================================================================
def generate_predicted_data(fit, df, participants, true_y, n_trials):
    """
    Generate predicted data based on the posterior samples for parameters.
    """
    # Generate predicted data
    predicted_y = []
    predicted_z = []

    # Compute mean values for parameters
    # lambda_sample = df['lambda'].mean() 
    # mu_z_sample = df['mu_z'].mean()           
    # sigma_z_sample = df['sigma_z'].mean()     

    # Generate predicted data for each trial
    for i in range(n_trials):
        participant = participants[i]
        alpha_sample = df[f'alpha[{participant}]'].mean()  
        tau_sample = df[f'tau[{participant}]'].mean()      
        beta_sample = df[f'beta[{participant}]'].mean()    
        eta_sample = df[f'eta[{participant}]'].mean()  
        mu_z_sample = df[f'mu_z[{participant}]'].mean()
        sigma_z_sample = df[f'sigma_z[{participant}]'].mean()
        lambda_sample = df[f'lambda[{participant}]'].mean()
        b_sample = df[f'b[{participant}]'].mean()
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

# =====================================================================================
# Posterior Predictive Check Function
def posterior_predictive_check(observed_data, predicted_data, name):  
    """
    This function performs a posterior predictive check by comparing the observed data
    with the predicted data generated from the model's posterior samples.
    Returns the figure instead of saving it.
    """
    # Create a figure for the posterior predictive check
    fig = plt.figure(figsize=(8, 6))

    # Plot the observed data
    sns.histplot(observed_data, label='Observed', stat='density', kde=True, color='blue')

    # Plot the predicted data
    sns.histplot(predicted_data, label='Predicted', stat='density', kde=True, color='orange')

    # Add title and legend
    plt.title(f"Posterior Predictive Check - {name}")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    return fig

# =====================================================================================
# Flexible recovery plot function with credible intervals
def plot_recovery(true_vals, estimated_vals, param_name, ci_lower=None, ci_upper=None):
    # Convert to arrays if they're floats or scalars
    true_vals = np.atleast_1d(true_vals)
    estimated_vals = np.atleast_1d(estimated_vals)

    # Calculate correlation between true and estimated values
    correlation = np.corrcoef(true_vals, estimated_vals)[0, 1]

    # Create plot for individual parameters
    fig = plt.figure(figsize=(7, 7))

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
    
    return fig 