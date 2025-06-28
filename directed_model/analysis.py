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
# Comprehensive Posterior Predictive Check Function
def posterior_predictive_check(fit, df, participants, true_y, true_z, true_params, nparts):
    """
    Performs comprehensive posterior predictive checks with both in-sample and out-of-sample data.
    
    Parameters:
    - fit: CmdStanMCMC fit object
    - df: DataFrame with posterior samples
    - participants: Array of participant IDs for each trial
    - true_y: Array of observed choice/RT data
    - true_z: Array of observed latent variable data
    - true_params: Dictionary with true parameter values (alpha, tau, beta, eta, mu_z, sigma_z, lambda, b)
    - nparts: Number of participants
    
    Returns:
    - fig: Combined 2x2 subplot figure with in-sample and out-of-sample PPCs
    """
    
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

    test_y = []
    test_z = []
    test_participants = []

    for p in range(nparts):
        # Use true parameters for this participant to simulate new data
        n_trials_per_participant = np.sum(participants == (p + 1))
        
        # Simulate new data using true parameters
        simulated_y, _, simulated_z = simul_directed_ddm(
            ntrials=n_trials_per_participant,
            alpha=true_params["alpha"][p],
            tau=true_params["tau"][p],
            beta=true_params["beta"][p],
            eta=true_params["eta"][p],
            lambda_param=true_params["lambda"][p],
            mu_z=true_params["mu_z"][p],
            sigma_z=true_params["sigma_z"][p],
            b=true_params["b"][p]
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

    # Print summary statistics
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
