from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import from_csv
import scipy.io as sio
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from directed_model.analysis import extract_parameter_samples

# =====================================================================================
# Parse command line arguments
parser = argparse.ArgumentParser(description='Directed DDM exploratory analysis')
parser.add_argument('--model', type=str, default='directed_ddm_base', 
                    help='Model name (default: directed_ddm_base)')
args = parser.parse_args()

model_name = args.model

# =====================================================================================
# Set up paths
DIRECTED_MODEL_DIR = PROJECT_ROOT / "directed_model"
DATA_DIR = DIRECTED_MODEL_DIR / "data"
RESULTS_DIR = DIRECTED_MODEL_DIR / "results" / model_name

fig_model_name = model_name.replace("ddm_data_", "").replace("ddmdata_", "").replace("cross_directed_data", "").replace("cross_directed_", "")
FIGURES_DIR = DIRECTED_MODEL_DIR / "figures" / fig_model_name

# Check if data file exists
data_file = DATA_DIR / f"{model_name}.mat"
if not data_file.exists():
    print(f"Error: Data file {data_file} does not exist!")
    sys.exit(1)

# Check if results directory exists
if not RESULTS_DIR.exists():
    print(f"Error: Results directory {RESULTS_DIR} does not exist!")
    sys.exit(1)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Analyzing model: {model_name}")
print(f"Data file: {data_file}")
print(f"Results directory: {RESULTS_DIR}")

# =====================================================================================
# Load data and results
# Load true parameters and neural signals from the simulation
genparam = sio.loadmat(data_file)
print("Available keys in .mat file:", genparam.keys())

# Extract neural signals (actual EEG-derived signals)
z_i = np.squeeze(genparam["z"])
n_trials = len(z_i)

# Get number of participants
participants = np.squeeze(genparam["participant"]).astype(int)
nparts = participants.max()

print(f"Number of trials: {n_trials}")
print(f"Number of participants: {nparts}")

# Load CmdStanMCMC results from CSV
csv_files = sorted(RESULTS_DIR.glob("*.csv"))
if not csv_files:
    print(f"No CSV files found in {RESULTS_DIR}, exiting.")
    sys.exit()
fit = from_csv([str(p) for p in csv_files])

# Extract posterior samples
df = fit.draws_pd()

# =====================================================================================
# Posterior Correlation Analysis for λ, b, η

def compute_and_plot_posterior_correlations(df, nparts, param_triplet=("lambda", "b", "eta")):
    """
    Computes and visualizes posterior correlations between a set of parameters.
    """
    correlation_results = {}
    high_corr_participants = []
    all_correlations = []

    for pid in range(nparts):
        samples = []
        for pname in param_triplet:
            samples.append(df[f"{pname}[{pid + 1}]"].values)  # Stan indexing starts at 1
        samples = np.array(samples)
        corr_matrix = np.corrcoef(samples)
        correlation_results[pid] = corr_matrix
        all_correlations.append(corr_matrix)

        # Check for high absolute correlations
        upper_triangle_indices = np.triu_indices(len(param_triplet), k=1)
        high_corr = np.any(np.abs(corr_matrix[upper_triangle_indices]) > 0.8)
        if high_corr:
            high_corr_participants.append(pid)

    # Create overall summary correlation heatmap (average across participants)
    mean_corr_matrix = np.mean(all_correlations, axis=0)
    corr_df = pd.DataFrame(mean_corr_matrix, index=param_triplet, columns=param_triplet)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax, 
                cbar_kws={'label': 'Correlation'})
    ax.set_title(f"Average Posterior Correlations Across {nparts} Participants")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "posterior_correlations_overall.png", dpi=300, bbox_inches='tight')
    print(f"Saved overall correlation heatmap to {FIGURES_DIR / 'posterior_correlations_overall.png'}")
    plt.close(fig)

    print(f"Number of participants with high posterior correlation (>|0.8|): {len(high_corr_participants)}")
    if high_corr_participants:
        print("Participants with high correlations:", [p + 1 for p in high_corr_participants])  # Convert to 1-based index

    return correlation_results, high_corr_participants


# Run correlation analysis
posterior_corrs, flagged_pids = compute_and_plot_posterior_correlations(df, nparts)

# =====================================================================================
# Extract model parameter estimates from posterior
# Get posterior samples for parameters of interest
lambda_samples = extract_parameter_samples(df, "lambda", nparts)  # Shape: (nparts, n_samples)
b_samples = extract_parameter_samples(df, "b", nparts)
eta_samples = extract_parameter_samples(df, "eta", nparts)

# For simplicity, use posterior means (you could also use random samples)
lambda_mean = np.mean(lambda_samples, axis=1)  # Mean for each participant
b_mean = np.mean(b_samples, axis=1)
eta_mean = np.mean(eta_samples, axis=1)

print(f"Lambda posterior means: {lambda_mean}")
print(f"B posterior means: {b_mean}")
print(f"Eta posterior means: {eta_mean}")

# =====================================================================================
# Generate posterior predictive samples of drift rates
n_samples = 50  # Number of posterior samples to simulate per trial

# For each trial, use the corresponding participant's parameters
participant_indices = participants - 1  # Convert to 0-based indexing

# Drift mean: deterministic part
mu_delta = lambda_mean[participant_indices] * z_i + b_mean[participant_indices]

# Generate samples using posterior uncertainty
delta_samples = np.zeros((n_trials, n_samples))
for i in range(n_trials):
    p_idx = participant_indices[i]
    # Sample from posterior distribution of parameters
    lambda_sample_idx = np.random.choice(lambda_samples.shape[1], size=n_samples)
    b_sample_idx = np.random.choice(b_samples.shape[1], size=n_samples)
    eta_sample_idx = np.random.choice(eta_samples.shape[1], size=n_samples)
    
    lambda_vals = lambda_samples[p_idx, lambda_sample_idx]
    b_vals = b_samples[p_idx, b_sample_idx]
    eta_vals = eta_samples[p_idx, eta_sample_idx]
    
    # Compute drift mean for each sample
    mu_delta_samples = lambda_vals * z_i[i] + b_vals
    
    # Sample drift rates
    delta_samples[i] = np.random.normal(loc=mu_delta_samples, scale=np.sqrt(eta_vals))

# =====================================================================================
# Plot results
plt.figure(figsize=(12, 8))
for i in range(n_trials):
    plt.plot([z_i[i]] * n_samples, delta_samples[i], 'o', alpha=0.1, color='steelblue')

# Add posterior mean trend line
z_range = np.linspace(z_i.min(), z_i.max(), 100)
# Use population-level posterior means for trend line
lambda_pop_mean = np.mean(lambda_mean)
b_pop_mean = np.mean(b_mean)
trend_line = lambda_pop_mean * z_range + b_pop_mean
plt.plot(z_range, trend_line, 'r-', linewidth=2, label=f'Population trend: δ = {lambda_pop_mean:.3f}z + {b_pop_mean:.3f}')

plt.title('Posterior Predictive Drift Samples vs Neural Input')
plt.xlabel('$z_i$ (Neural signal)')
plt.ylabel('$\delta_i$ (Drift rate samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
fig_path = FIGURES_DIR / "drift_vs_neural_exploratory.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {fig_path}")
plt.show()

# =====================================================================================
# =====================================================================================
# (1) λ Recovery Analysis (RMSE and R²)

# Load true λ values
if "lambda_param" in genparam:
    lambda_true = np.squeeze(genparam["lambda_param"])
else:
    print("Error: 'lambda_param' not found in .mat file!")
    sys.exit(1)

# Compute RMSE and R² across participants
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(lambda_true, lambda_mean))
r2 = r2_score(lambda_true, lambda_mean)

# Scatter plot of recovered vs. true λ
plt.figure(figsize=(7, 6))
sns.scatterplot(x=lambda_true, y=lambda_mean, s=60, color='navy')
plt.plot([lambda_true.min(), lambda_true.max()],
         [lambda_true.min(), lambda_true.max()],
         'r--', label='Identity Line')
plt.title(f"λ Recovery Across Participants\nRMSE={rmse:.3f}, R²={r2:.3f}")
plt.xlabel("True λ")
plt.ylabel("Recovered λ (Posterior Mean)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "lambda_recovery_scatter.png", dpi=300, bbox_inches='tight')
print(f"λ recovery scatter saved to {FIGURES_DIR / 'lambda_recovery_scatter.png'}")
plt.close()

# =====================================================================================
# (4) Posterior vs Prior Comparison for λ

from scipy.stats import norm

# Choose up to 4 example participants for plotting
example_pids = [0, 1, 2, 3] if nparts >= 4 else list(range(nparts))
prior_dist = norm(loc=0, scale=1)

fig, axes = plt.subplots(1, len(example_pids), figsize=(4 * len(example_pids), 4), sharey=True)

for i, pid in enumerate(example_pids):
    ax = axes[i]
    sns.histplot(lambda_samples[pid], bins=40, stat="density", kde=True,
                 color="skyblue", label="Posterior", ax=ax)
    
    x = np.linspace(-3, 3, 300)
    ax.plot(x, prior_dist.pdf(x), 'r--', label="Prior N(0,1)")
    ax.set_title(f"Participant {pid+1}\nTrue λ = {lambda_true[pid]:.3f}")
    ax.set_xlabel("λ")
    ax.legend()
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel("Density")
plt.suptitle("Posterior vs Prior for λ (Sample Participants)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIGURES_DIR / "posterior_vs_prior_lambda.png", dpi=300, bbox_inches='tight')
print(f"Posterior vs prior plot saved to {FIGURES_DIR / 'posterior_vs_prior_lambda.png'}")
plt.close()
