from collections.abc import Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import binom

from bayesflow.utils import logging
from bayesflow.utils import prepare_plot_data, add_titles_and_labels, prettify_subplots
import bayesflow as bf

# =====================================================================================
# Training History and Loss Plotting Functions

class MockHistory:
    """Mock history object for bayesflow plotting compatibility."""
    def __init__(self, history_dict):
        self.history = history_dict

def plot_training_history(history_dict, figures_path):
    """
    Plot training history including loss curves and learning rate schedule.
    
    Parameters
    ----------
    history_dict : dict
        Dictionary containing training history with keys like 'loss', 'val_loss', 'lr'
    figures_path : str
        Full path to save the figure (including filename)
    
    Returns
    -------
    str
        Path to the saved plot file
    """
    # Loss plotting and stability analysis
    train_loss = np.array(history_dict['loss'])
    val_loss = np.array(history_dict['val_loss'])

    # Moving average window
    window = 10
    train_ma = pd.Series(train_loss).rolling(window, min_periods=1).mean()
    val_ma = pd.Series(val_loss).rolling(window, min_periods=1).mean()

    # Set font sizes
    title_fontsize = 18
    label_fontsize = 14
    legend_fontsize = 12
    tick_fontsize = 12

    # Plot both losses
    plt.figure(figsize=(15, 5))

    # Main loss plot
    plt.plot(train_loss, label="Train Loss", alpha=0.4)
    plt.plot(train_ma, label=f"Train MA ({window})", color='blue', linewidth=2)
    plt.plot(val_loss, label="Val Loss", alpha=0.4)
    plt.plot(val_ma, label=f"Val MA ({window})", color='orange', linewidth=2)
    plt.xlabel("Epoch", fontsize=label_fontsize)
    plt.ylabel("Loss", fontsize=label_fontsize)
    plt.title("Train vs. Validation Loss", fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(False)
    
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(figures_path, dpi=300)
    plt.close()
    
    return figures_path

def analyze_training_performance(history_dict, verbose=True):
    """
    Analyze training performance and detect potential issues.
    
    Parameters
    ----------
    history_dict : dict
        Dictionary containing training history
    verbose : bool, optional
        Whether to print analysis results
    
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    train_loss = np.array(history_dict['loss'])
    val_loss = np.array(history_dict['val_loss'])
    
    # Enhanced analysis
    N = 20
    results = {
        'train_loss_variance_last_N': np.var(train_loss[-N:]),
        'val_loss_variance_last_N': np.var(val_loss[-N:]),
        'final_train_loss': train_loss[-1],
        'final_val_loss': val_loss[-1],
        'best_val_loss': np.min(val_loss),
        'best_val_epoch': np.argmin(val_loss) + 1,
        'N_epochs_analyzed': N
    }
    
    # Check for overfitting
    if len(val_loss) > 10:
        recent_val_trend = np.polyfit(range(len(val_loss)//2, len(val_loss)), 
                                    val_loss[len(val_loss)//2:], 1)[0]
        results['overfitting_detected'] = recent_val_trend > 0
        results['val_loss_trend'] = recent_val_trend
    else:
        results['overfitting_detected'] = None
        results['val_loss_trend'] = None
    
    if verbose:
        print(f"Train loss variance (last {N} epochs): {results['train_loss_variance_last_N']:.6f}")
        print(f"Val   loss variance (last {N} epochs): {results['val_loss_variance_last_N']:.6f}")
        print(f"Final train loss: {results['final_train_loss']:.4f}")
        print(f"Final validation loss: {results['final_val_loss']:.4f}")
        print(f"Best validation loss: {results['best_val_loss']:.4f} at epoch {results['best_val_epoch']}")
        
        # Check for overfitting
        if results['overfitting_detected'] is not None:
            if results['overfitting_detected']:
                print("WARNING: Validation loss is trending upward - potential overfitting detected!")
            else:
                print("Validation loss appears stable or improving")
    
    return results

def plot_bayesflow_loss(history_dict, figures_path):
    """
    Create bayesflow-style loss plot.
    
    Parameters
    ----------
    history_dict : dict
        Dictionary containing training history
    figures_path : str
        Full path to save the figure (including filename)
    
    Returns
    -------
    str
        Path to the saved plot file
    """
    # Create a mock history object for bf.diagnostics.plots.loss
    mock_history = MockHistory(history_dict)
    
    # Plot loss using bayesflow
    plot = bf.diagnostics.plots.loss(history=mock_history)
    
    # Save plot
    plot.savefig(figures_path)
    
    return figures_path

# =====================================================================================
# Original plotting functions

def simulated_data_check(sim_data, rt_path, p300_path):
    """
    Plots the simulated data from BayesFlow simulator output
    
    Parameters
    ----------
    sim_data : dict
        Simulated data dictionary
    rt_path : str
        Full path to save RT distribution plot (including filename)
    p300_path : str
        Full path to save P300 distribution plot (including filename)
    """
    # Extract RTs and P300s from the dictionary structure
    choicert = sim_data["choicert"].flatten()
    z = sim_data["z"].flatten()

    # Plot RTs
    plt.figure(figsize=(8, 6))
    sns.kdeplot(choicert, fill=True, color='blue')
    plt.title("Simulated Choice RTd")
    plt.xlabel("Choice RT")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(rt_path, dpi=300)
    plt.close()

    # Plot P300s
    plt.figure(figsize=(8, 6))
    sns.kdeplot(z, fill=True, color='green')
    plt.title("Simulated P300 (z) Distribution")
    plt.xlabel("P300 Amplitude")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(p300_path, dpi=300)
    plt.close()

    print(f"RT plot saved to {rt_path}")
    print(f"P300 plot saved to {p300_path}")

def calibration_histogram(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    figsize: Sequence[float] = None,
    num_bins: int = 10,
    binomial_interval: float = 0.99,
    label_fontsize: int = 14,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "teal",
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
    """Creates and plots publication-ready histograms of rank statistics for simulation-based calibration
    (SBC) checks according to [1].

    Any deviation from uniformity indicates miscalibration and thus poor convergence
    of the networks or poor combination between generative model/networks.

    [1] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
    Validating Bayesian inference algorithms with simulation-based calibration.
    arXiv preprint arXiv:1804.06788.

    Parameters
    ----------
    estimates      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    targets     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    figsize          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None
    num_bins          : int, optional, default: 10
        The number of bins to use for each marginal histogram
    binomial_interval : float in (0, 1), optional, default: 0.99
        The width of the confidence interval for the binomial distribution
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    color        : str, optional, default '#a34f4f'
        The color to use for the histogram body
    num_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `estimates` and `targets`.
    """

    plot_data = prepare_plot_data(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
        num_col=num_col,
        num_row=num_row,
        figsize=figsize,
    )

    print(20 * "=")
    print(plot_data["axes"])

    estimates = plot_data.pop("estimates")
    targets = plot_data.pop("targets")

    # Determine the ratio of simulations to prior draw
    # num_params = plot_data['num_variables']
    num_sims = estimates.shape[0]
    num_draws = estimates.shape[1]

    ratio = int(num_sims / num_draws)

    # Log a warning if N/B ratio recommended by Talts et al. (2018) < 20
    # if ratio < 20:
        # logging.warning(
        #     "The ratio of simulations / posterior draws should be > 20 "
        #     f"for reliable variance reduction, but your ratio is {ratio}. "
        #     "Confidence intervals might be unreliable!"
        # )

    # Set num_bins automatically, if nothing provided
    if num_bins is None:
        num_bins = int(ratio / 2)
        # Attempt a fix if a single bin is determined so plot still makes sense
        if num_bins == 1:
            num_bins = 4

    # Compute ranks (using broadcasting)
    ranks = np.sum(estimates < targets[:, np.newaxis, :], axis=1)

    # Compute confidence interval and mean
    num_trials = int(targets.shape[0])
    # uniform distribution expected -> for all bins: equal probability
    # p = 1 / num_bins that a rank lands in that bin
    endpoints = binom.interval(binomial_interval, num_trials, 1 / num_bins)
    mean = num_trials / num_bins  # corresponds to binom.mean(N, 1 / num_bins)

    for j, ax in enumerate(plot_data["axes"].flat):
        if j >= ranks.shape[1] or j == 7:  # Skip the 8th plot (index 7)
            ax.set_visible(False)
            continue
        ax.axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.3)
        ax.axhline(mean, color="gray", zorder=0, alpha=0.9)
        sns.histplot(ranks[:, j], kde=False, ax=ax, color=color, bins=num_bins, alpha=0.95)
        ax.set_ylabel("")  
        ax.get_yaxis().set_ticks([])  
        ax.grid(False)
    prettify_subplots(plot_data["axes"], tick_fontsize)

    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        title=plot_data["variable_names"],
        xlabel="Rank Statistic",
        ylabel="Number of Simulations",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )
    plot_data["fig"].tight_layout()

    return plot_data["fig"]