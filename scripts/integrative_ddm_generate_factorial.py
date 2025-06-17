# =====================================================================================
# Generate data across 2x2x3 factorial design using integrative DDM model:
# - SNR (low/high): controls sigma (P300 noise)
# - Coupling (low/high): controls gamma (DDM-P300 coupling)
# - Error distribution: laplace, gaussian, uniform
# =====================================================================================
# Import modules
import numpy as np
import scipy.io as sio
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from integrative_model.simulation import prior, likelihood

# Setup paths and directories
INTEGRATIVE_MODEL_DIR = PROJECT_ROOT / "integrative_model"
DATA_DIR = INTEGRATIVE_MODEL_DIR / "data"

# =====================================================================================
print("Generating factorial integrative DDM data...")

# Parameters for data generation
ntrials = 100
nparts = 100
seed = 2025

# Set seed for reproducibility
np.random.seed(seed)

# Conditions for simulation
snr_levels = ["low", "high"]
coupling_levels = ["low", "high"]
error_distributions = ["laplace", "gaussian", "uniform"]

# SNR controls sigma (P300 noise level)
sigma_conditions = {"low": 0.5, "high": 0}  # low SNR = high noise

# Coupling controls gamma (DDM-P300 coupling strength)
gamma_conditions = {
    "low": (-0.2, 0.2),  # weak coupling
    "high": (2.0, 3.0),  # strong coupling
}

# Initialize lists to store data and labels
all_data = []
condition_labels = []

# Generate data for each condition
for snr in snr_levels:
    for coupling in coupling_levels:
        for dist in error_distributions:

            # Generate condition key
            condition_key = f"SNR_{snr}_COUP_{coupling}_DIST_{dist}"
            print(f"\nGenerating data for condition: {condition_key}")

            # Initialize arrays for data storage
            choicert_all = np.zeros(ntrials * nparts)
            z_clean_all = np.zeros(ntrials * nparts)  # clean P300 signal
            z_noisy_all = np.zeros(ntrials * nparts)  # P300 with additional noise
            participant = np.zeros(ntrials * nparts)

            # Store parameters for each participant
            alpha_all = np.zeros(nparts)
            tau_all = np.zeros(nparts)
            beta_all = np.zeros(nparts)
            mu_delta_all = np.zeros(nparts)
            eta_delta_all = np.zeros(nparts)
            gamma_all = np.zeros(nparts)
            sigma_all = np.zeros(nparts)

            # Initialize index for tracking data
            indextrack = 0

            # Modify gamma based on coupling condition
            signs = np.random.choice([-1, 1], size=nparts)
            gamma_min, gamma_max = gamma_conditions[coupling]
            gamma_all = np.random.uniform(gamma_min, gamma_max, nparts) * signs

            # Generate data for each participant
            for p in range(nparts):

                # Sample base parameters from prior
                params = prior()

                # Modify gamma based on coupling condition
                params["gamma"] = gamma_all[p]

                # Modify sigma based on SNR condition
                params["sigma"] = params["sigma"] + sigma_conditions[snr]

                # Store participant parameters
                alpha_all[p] = params["alpha"]
                tau_all[p] = params["tau"]
                beta_all[p] = params["beta"]
                mu_delta_all[p] = params["mu_delta"]
                eta_delta_all[p] = params["eta_delta"]
                sigma_all[p] = params["sigma"]

                # Generate behavioral and P300 data
                sim_data = likelihood(
                    params["alpha"],
                    params["tau"],
                    params["beta"],
                    params["mu_delta"],
                    params["eta_delta"],
                    params["gamma"],
                    params["sigma"],
                    ntrials,
                )

                choicert = sim_data["choicert"]

                # Generate noisy P300 factors
                delta = np.random.normal(params["mu_delta"], params["eta_delta"])
                if dist == "gaussian":
                    z_noisy = np.random.normal(
                        params["gamma"] * delta, params["sigma"], ntrials
                    )
                elif dist == "laplace":
                    b_laplace = params["sigma"] / np.sqrt(2)
                    z_noisy = np.random.laplace(
                        params["gamma"] * delta, b_laplace, ntrials
                    )
                elif dist == "uniform":
                    a_uniform = (params["gamma"] * delta) - np.sqrt(3) * params[
                        "sigma"
                    ]
                    b_uniform = (params["gamma"] * delta) + np.sqrt(3) * params[
                        "sigma"
                    ]
                    z_noisy = np.random.uniform(a_uniform, b_uniform, ntrials)
                else:
                    raise ValueError(f"Unknown distribution: {dist}")

                # Store data
                start = indextrack
                end = indextrack + ntrials

                choicert_all[start:end] = choicert
                z_noisy_all[start:end] = z_noisy
                participant[start:end] = p + 1
                indextrack += ntrials

            # Convert choicert to rt and accuracy for compatibility
            rt_all = np.abs(choicert_all)
            acc_all = (np.sign(choicert_all) + 1) / 2  # Convert to 0/1
            y_all = np.sign(choicert_all) * rt_all  # Signed RT

            # Compute minRT after the full loop
            minRT = np.zeros(nparts)
            for p in range(nparts):
                rts_p = rt_all[participant == (p + 1)]
                valid_rts = rts_p[np.isfinite(rts_p)]
                if len(valid_rts) > 0:
                    minRT[p] = np.min(valid_rts)
                else:
                    minRT[p] = np.nan

            # Store generated parameters and data
            genparam = dict(
                # Integrative model parameters
                alpha=alpha_all,
                tau=tau_all,
                beta=beta_all,
                mu_delta=mu_delta_all,
                eta_delta=eta_delta_all,
                gamma=gamma_all,
                sigma=sigma_all,
                # Data
                choicert=choicert_all,
                rt=rt_all,
                acc=acc_all,
                y=y_all,
                z=z_noisy_all,
                participant=participant,
                # Metadata
                nparts=nparts,
                ntrials=ntrials,
                N=len(rt_all),
                minRT=minRT,
                condition=condition_key,
                snr_level=snr,
                coupling_level=coupling,
                error_dist=dist,
            )

            # Save data to mat file
            DATA_DIR.mkdir(exist_ok=True)
            file_path = DATA_DIR / f"integrative_ddm_data_{condition_key}.mat"
            sio.savemat(file_path, genparam)
            print(f"Saved: {file_path}")

            # Store data and labels
            all_data.append(genparam)
            condition_labels.append((snr, coupling, dist))

print(f"\nGenerated data for {len(all_data)} conditions:")
for i, (snr, coupling, dist) in enumerate(condition_labels):
    print(f"  {i+1}. SNR: {snr}, Coupling: {coupling}, Distribution: {dist}")
print("\nData generation complete!")