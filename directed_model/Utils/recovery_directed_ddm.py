# =====================================================================================
# Import modules
import numpy as np

# =====================================================================================
# Compute posterior means and credible intervals
def compute_posterior_means_and_cis(df, nparts, group_param_names, individual_param_names):
    """
    Compute posterior means and credible intervals for the parameters.
    """
    # Compute posterior means for individual parameters
    posterior = {}
    for param_name in individual_param_names:
        posterior[param_name] = np.array([df[f'{param_name}[{i+1}]'].mean() for i in range(nparts)])

    # Recovery for individual-level parameters
    individual_recovery = {}
    for name in individual_param_names:
        samples = np.array([df[f'{name}[{i+1}]'].values for i in range(nparts)])
        mean = samples.mean(axis=1)
        ci_lower = np.percentile(samples, 5, axis=1)
        ci_upper = np.percentile(samples, 95, axis=1)
        individual_recovery[name] = {'mean': mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

    # Recovery for group-level parameters
    group_recovery = {}
    for name in group_param_names:
        true_val = eval(f"true_{name}")
        samples = df[name]
        mean = samples.mean()
        ci_lower = np.percentile(samples, 5)
        ci_upper = np.percentile(samples, 95)
        group_recovery[name] = {'true': true_val, 'mean': mean, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

    return posterior, individual_recovery, group_recovery