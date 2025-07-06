# =====================================================================================
# Initialize JAX backend
import os
os.environ["KERAS_BACKEND"] = "jax"

# =====================================================================================
# Imports
from pathlib import Path
import sys
import argparse
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
import statsmodels.api as sm
import keras
from bayesflow.simulators import make_simulator
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from integrative_model.simulation import prior, likelihood

# =====================================================================================
# Paths and config
CHECKPOINT = "checkpoint_integrative_ddm_seed_12_150epochs.keras"
CHECKPOINTS_DIR = PROJECT_ROOT / "integrative_model" / "checkpoints"
DATA_DIR = PROJECT_ROOT / "integrative_model" / "data"
CHECKPOINT_PATH = CHECKPOINTS_DIR / CHECKPOINT

# =====================================================================================
# Argument parser
parser = argparse.ArgumentParser(description='Test statistical entanglement between delta and gamma.')
parser.add_argument('--prefix', type=str, default='integrative_ddm_data_', 
                    help='Prefix for .mat files (default: integrative_ddm_data_)')
args = parser.parse_args()

# =====================================================================================
# Setup simulator and load model
def meta():
    return dict(n_obs=100)

simulator = make_simulator([prior, likelihood], meta_fn=meta)
approximator = keras.saving.load_model(CHECKPOINT_PATH)

# =====================================================================================
# Find data files
matlab_files = sorted(DATA_DIR.glob(f"{args.prefix}*_including_delta.mat"))
if not matlab_files:
    print(f"No .mat files found with pattern '{args.prefix}*_including_delta.mat'!")
    sys.exit()

# =====================================================================================
# Run entanglement analysis
for matlab_file in matlab_files:
    condition_name = matlab_file.stem.replace(f"{args.prefix}", "").replace("_including_delta", "")
    if condition_name.startswith("data_"):
        condition_name = condition_name.replace("data_", "")

    data = sio.loadmat(matlab_file)
    nparts = data["nparts"][0][0]
    ntrials = data["ntrials"][0][0]

    # Extract true values
    gamma_true = data["gamma"].T.flatten()                          # shape: (nparts,)
    # For _including_delta files, use mu_delta as the participant-level mean δᵢ
    mean_delta = data["mu_delta"].T.flatten()                       # participant-level mean δᵢ

    # Prepare input for inference
    reshaped_data = {
        "n_obs": np.full((nparts, 1), ntrials),
        "alpha": data["alpha"].T,
        "tau": data["tau"].T,
        "beta": data["beta"].T,
        "mu_delta": data["mu_delta"].T,
        "eta_delta": data["eta_delta"].T,
        "gamma": data["gamma"].T,
        "sigma": data["sigma"].T,
        "choicert": np.vstack(data["choicert"][0]).reshape((nparts, ntrials)),
        "z": np.vstack(data["z"][0]).reshape((nparts, ntrials)),
    }

    # Run inference
    post_draws = approximator.sample(conditions=reshaped_data, num_samples=ntrials)
    gamma_hat = np.mean(post_draws["gamma"], axis=1).flatten()

    # Compute error in estimated gamma
    error_gamma = gamma_hat - gamma_true

    # Mask for valid data
    valid_mask = ~np.isnan(mean_delta) & ~np.isnan(error_gamma)
    valid_delta = mean_delta[valid_mask]
    valid_error = error_gamma[valid_mask]

    # Correlation
    if len(valid_delta) > 1:
        r, p = pearsonr(valid_delta, valid_error)
        print(f"\n{condition_name}:")
        print(f"  Pearson r(δᵢ, error_γ) = {r:.4f}, p = {p:.3e}")
        rho, p_spear = spearmanr(valid_delta, valid_error)
        print(f"  Spearman rho(δᵢ, error_γ) = {rho:.4f}, p = {p_spear:.4e}")

        # Regression
        X = sm.add_constant(valid_delta)
        model = sm.OLS(valid_error, X).fit()
        print(f"  Regression slope: {model.params[1]:.4f}, p = {model.pvalues[1]:.4e}")
        print(f"  R² = {model.rsquared:.4f}")
    else:
        print(f"{condition_name}: Not enough valid participants for analysis.")
    