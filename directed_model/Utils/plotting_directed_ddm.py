# =====================================================================================
# Import modules
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math
import numpy as np

# =====================================================================================
# Trace plots function
def plot_trace_grids(df, fit, params_of_interest=('alpha', 'tau', 'beta', 'eta'), grid_cols=5, save_dir=save_dir):
    """
    Plot trace plots for the parameters of interest.
    """
    # Set default save directory if not provided
    if save_dir is None:
        save_dir = 'Figures'  # Default value if not passed

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

# =====================================================================================
# Function to plot recovery (true vs estimated values)
def plot_recovery(true_vals, estimated_vals, param_name, ci_lower=None, ci_upper=None, save_dir='Figures'):
    """
    Plot recovery (true vs estimated values) for a given parameter.
    """
    # Set default save directory if not provided
    if save_dir is None:
        save_dir = 'Figures'  # Default value if not passed

    true_vals = np.atleast_1d(true_vals)
    estimated_vals = np.atleast_1d(estimated_vals)

    # Calculate correlation
    correlation = np.corrcoef(true_vals, estimated_vals)[0, 1]

    # Create plot
    plt.figure(figsize=(7, 7))
    if ci_lower is not None and ci_upper is not None:
        plt.errorbar(estimated_vals, true_vals, xerr=[estimated_vals - ci_lower, ci_upper - estimated_vals], fmt='o', label=f'{param_name} (CI)', color='dodgerblue')
    else:
        sns.regplot(x=true_vals, y=estimated_vals, line_kws={'color': 'red'})

    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Estimated {param_name}")
    plt.title(f"Parameter Recovery: {param_name} (r = {correlation:.2f})")
    plt.grid(False)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'recovery_plot_{param_name}.png'), dpi=300)
    plt.close()

# =====================================================================================
# Posterior Predictive Check Function
def posterior_predictive_check(observed_data, predicted_data, name, save_dir=save_dir):  
    """
    This function performs a posterior predictive check by comparing the observed data
    with the predicted data generated from the model's posterior samples.
    """
    # Set default save directory if not provided
    if save_dir is None:
        save_dir = 'Figures'  # Default value if not passed

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
