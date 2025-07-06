# =====================================================================================
# Code for recovery plots copied from Michael Nunez. Adapted for our purposes.

# =====================================================================================
# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# =====================================================================================
# Compute credible interval coverage
def compute_credible_interval_coverage(post_samples, true_values, level=0.95):
    lower_bound = (1.0 - level) / 2.0
    upper_bound = 1.0 - lower_bound
    coverage = []

    for i in range(post_samples.shape[0]):  # participant level
        ci_lower = np.quantile(post_samples[i, :], lower_bound)
        ci_upper = np.quantile(post_samples[i, :], upper_bound)
        coverage.append(ci_lower <= true_values[i] <= ci_upper)

    return np.mean(coverage)

# =====================================================================================
def recovery_plot(estimates, targets, fig_width=15, fig_height=9, parameter_display_titles=None):
    """
    Parameter recovery plots: true vs. estimated.
    Includes median, mean, 95% and 99% credible intervals.

    Parameters
    ----------
    estimates : dict[str, np.ndarray]
        Posterior samples per parameter: shape (n_participants, n_samples)
    targets : dict[str, np.ndarray]
        True parameter values per parameter
    fig_width : int, default=15
        Figure width in inches
    fig_height : int, default=9
        Figure height in inches
    parameter_display_titles : dict[str, str], optional
        Mapping from parameter names to display titles. If None, uses default mapping.
    """
    # Default parameter display title mapping
    if parameter_display_titles is None:
        parameter_display_titles = {
            'alpha': r'$\alpha$: Boundary separation',
            'tau': r'$\tau$: Non-decision time',
            'beta': r'$\beta$: Starting point/bias',
            'mu_z': r'$\mu_z$: Mean of latent variable',
            'eta': r'$\eta$: Variance of drift rate',
            'mu_delta': r'$\mu_{\delta}$: Mean of drift rate',
            'eta_delta': r'$\eta_{\delta}$: Variance of drift rate',
            'sigma': r'$\sigma$: S.d. of latent signal',
            'sigma_z': r'$\sigma_z$: S.d. of latent signal',
            'lambda': r'$\lambda$: Coupling strength',
            'gamma': r'$\gamma$: Coupling strength',
            'b': r'$b$: Baseline drift rate'
        }
    
    # Identify parameters - use order from parameter_display_titles if provided
    if parameter_display_titles is not None:
        # Use the order from parameter_display_titles, but only include params that exist in estimates
        params = [param for param in parameter_display_titles.keys() if param in estimates]
    else:
        # Fall back to original order from estimates
        params = list(estimates.keys())
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), tight_layout=True)
    columns = 3
    rows = int(np.ceil(len(params) / columns))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    # Plot
    for i, param in enumerate(params):
        ax = fig.add_subplot(rows, columns, i + 1)

        # Store handles for legend
        h_95, h_99, h_median, h_mean, h_line = None, None, None, None, None

        # Plot each participant
        for v in range(estimates[param].shape[0]):
            # Compute percentiles
            bounds = stats.scoreatpercentile(estimates[param][v, :], (0.5, 2.5, 97.5, 99.5))

            # 95% and 99% CI
            for b in range(2):
                # Plot credible intervals
                credint = np.ones(100) * targets[param][v]
                y = np.linspace(bounds[b], bounds[-1 - b], 100)
                line = ax.plot(credint, y, color=Colors[b], linewidth=LineWidths[b])
                if v == 0:
                    if b == 0:
                       h_95 = line[0]
                    else:
                       h_99 = line[0]
            
            # Mark median
            mmedian = ax.plot(targets[param][v], np.median(estimates[param][v, :]), 'o', color=[0, 0, 0], markersize=10)
            if v == 0:
                h_median = mmedian[0]

            # Mark mean
            mmean = ax.plot(targets[param][v], np.mean(estimates[param][v, :]), '*', color=teal, markersize=10)
            if v == 0:
                h_mean = mmean[0]

        # Line y = x
        tempx = np.linspace(np.min(targets[param]), np.max(targets[param]), num=100)
        recoverline = ax.plot(tempx, tempx, color=orange, linewidth=3)
        h_line = recoverline[0]

        # Correlation between true values and posterior means
        posterior_means = np.mean(estimates[param], axis=1).flatten()
        true_vals = targets[param].flatten()
        r, _ = stats.pearsonr(true_vals, posterior_means)
        r_squared = r ** 2

        # Set axis labels and title based on parameter type
        ax.set_xlabel('True')
        ax.set_ylabel('Posterior')
        
        # Use display title if available, otherwise use parameter name
        display_name = parameter_display_titles.get(param, param)
        ax.set_title(f"{display_name} (r = {r:.2f}, R² = {r_squared:.2f})")
        
        # Add legend
        if i == 0:
            ax.legend([h_95, h_99, h_median, h_mean, h_line], ['95% CI', '99% CI', 'Median', 'Mean', 'y = x'], loc='upper left')

    return fig

# =====================================================================================
def compute_credible_intervals(gamma_estimates, ci=95):
    n_participants = gamma_estimates.shape[0]
    lower_bounds = np.zeros(n_participants)
    upper_bounds = np.zeros(n_participants)

    lower_p = (100 - ci) / 2
    upper_p = 100 - lower_p

    for i in range(n_participants):
        lower_bounds[i] = np.percentile(gamma_estimates[i, :], lower_p)
        upper_bounds[i] = np.percentile(gamma_estimates[i, :], upper_p)

    return lower_bounds, upper_bounds

# =====================================================================================
# Compute recovery metrics
def compute_recovery_metrics(post_draws, val_sims):
    print("{:<10} {:>10} {:>10} {:>10} {:>10}".format(
        "Param", "R^2", "NRMSE", "95% Cov.", "99% Cov."
    ))

    for param_name, true_values in val_sims.items():
        samples = post_draws[param_name].squeeze()
        true_values = true_values.squeeze()
        
        if true_values is None or samples is None:
            continue
        
        posterior_means = samples.mean(axis=1)

        if np.any(np.isnan(posterior_means)) or np.any(np.isnan(true_values)):
            print(f"Skipping {param_name} because of NaN values")
            continue

        # R²
        r, _ = pearsonr(true_values, posterior_means)
        r2 = r ** 2

        # NRMSE
        mse = mean_squared_error(true_values, posterior_means)
        rmse = np.sqrt(mse)
        range_true = np.max(true_values) - np.min(true_values)
        nrmse = rmse / range_true if range_true > 0 else np.nan

        # Coverage
        coverage_95 = compute_credible_interval_coverage(samples, true_values, level=0.95)
        coverage_99 = compute_credible_interval_coverage(samples, true_values, level=0.99)

        print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
            param_name, r2, nrmse, coverage_95, coverage_99
        ))