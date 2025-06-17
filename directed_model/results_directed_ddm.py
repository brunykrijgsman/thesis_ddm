# =====================================================================================
# Import modules
import os
import sys

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path
sys.path.append(os.path.dirname(current_dir))

import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import from_csv
import scipy.io as sio
from analysis_utils import check_convergence, plot_trace_grids, generate_predicted_data, posterior_predictive_check, extract_parameter_samples, compute_credible_interval_coverage
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from shared import plots

# =====================================================================================
# Create directory to save plots
save_dir = os.path.join(current_dir, 'Figures')
# Create directory if it does not exist
os.makedirs(save_dir, exist_ok=True) 

# Load true parameters from the simulation
genparam = sio.loadmat(os.path.join(current_dir, 'data/directed_ddm_simdata.mat'))
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

# Load CmdStanMCMC results from CSV
csv_dir = os.path.join(current_dir, 'Results/directed_ddm_fit_results_new')
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
fit = from_csv(csv_files)

# Extract posterior samples 
df = fit.draws_pd()

# R-hat diagnostics
summary = fit.summary()

# =====================================================================================
# Print R-hat and ESS summary
check_convergence(summary)

# =====================================================================================
# Create directory for trace plots
trace_dir = os.path.join(save_dir, 'Trace Plots')
os.makedirs(trace_dir, exist_ok=True)

# Parameters of interest
params_of_interest = ('alpha', 'tau', 'beta', 'eta', 'mu_z', 'sigma_z', 'lambda', 'b')

# Participants to plot (first 10) for thesis and paper
participants_to_plot = range(1, 11)  # 1 through 10 inclusive

# Collect column names for these participants and parameters
columns_to_plot = []

for param in params_of_interest:
    for p in participants_to_plot:
        col_name = f"{param}[{p}]"
        if col_name in df.columns:
            columns_to_plot.append(col_name)

# Subset the dataframe to only these columns
df_subset = df[columns_to_plot]

# Plot traces with the subset df
trace_figures = plot_trace_grids(df_subset, fit, params_of_interest=params_of_interest, grid_cols=5)

# Save trace plot figures
for param_name, fig in trace_figures.items():
    fig.savefig(os.path.join(trace_dir, f'trace_plots_{param_name}_first10_new.png'), dpi=300)
    plt.close(fig)

# =====================================================================================
# Create directory for recovery plots
recovery_dir = os.path.join(save_dir, 'Recovery Plots')
os.makedirs(recovery_dir, exist_ok=True)

# Parameters to plot recovery for
params = {
    'alpha': true_alpha,
    'tau': true_tau, 
    'beta': true_beta,
    'eta': true_eta,
    'mu_z': true_mu_z,
    'sigma_z': true_sigma_z,
    'lambda': true_lambda,
    'b': true_b
}

# Initialize dictionaries for the recovery plot
post_draws = {}
val_sims = {}

for param_name, true_values in params.items():
    # Get posterior samples and reshape to (n_participants, n_samples)
    post_draws[param_name] = extract_parameter_samples(df, param_name, nparts)
    # Store true values
    val_sims[param_name] = true_values

# Create recovery plot
print(f"post_draws: {post_draws['lambda'].shape}")
print(f"val_sims: {val_sims['lambda'].shape}")
f = plots.recovery(post_draws, val_sims)
f.savefig(os.path.join(recovery_dir, 'recovery_plot_directed_ddm_new.png'))

# =====================================================================================
# Posterior Predictive Checks
ppc_dir = os.path.join(save_dir, 'Posterior Predictive Checks')
os.makedirs(ppc_dir, exist_ok=True)

# Generate posterior predictive data
predicted_y, predicted_z = generate_predicted_data(fit, df, participants, true_y, n_trials=len(true_z))

# Perform posterior predictive check for y
fig_y = posterior_predictive_check(true_y, predicted_y, name='y')
fig_y.savefig(os.path.join(ppc_dir, 'posterior_predictive_check_y_new.png'), dpi=300)
plt.close(fig_y)

# Summary statistics for y
print(f"Mean of observed y: {np.mean(true_y)}")
print(f"Mean of predicted y: {np.mean(predicted_y)}")
print(f"Variance of observed y: {np.var(true_y)}")
print(f"Variance of predicted y: {np.var(predicted_y)}")

# Perform posterior predictive check for z  
fig_z = posterior_predictive_check(true_z, predicted_z, name='z')
fig_z.savefig(os.path.join(ppc_dir, 'posterior_predictive_check_z_new.png'), dpi=300)
plt.close(fig_z)

# Summary statistics for z
print(f"Mean of observed z: {np.mean(true_z)}")
print(f"Mean of predicted z: {np.mean(predicted_z)}")
print(f"Variance of observed z: {np.var(true_z)}")
print(f"Variance of predicted z: {np.var(predicted_z)}")

# =====================================================================================
# Extended Recovery Metrics
print("{:<10} {:>10} {:>10} {:>10} {:>10}".format("Param", "R^2", "NRMSE", "95% Cov.", "99% Cov."))

for param_name, true_values in val_sims.items():
    print(f"post_draws_shape: {post_draws[param_name].shape}")
    print(f"val_sims_shape: {val_sims[param_name].shape}")

    samples = post_draws[param_name]
    posterior_means = samples.mean(axis=1)

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