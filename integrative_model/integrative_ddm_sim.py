# =====================================================================================
# Import modules
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from time import time
import matplotlib.pyplot as plt
import sys
from numba import njit

# Configure environment
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"

# =====================================================================================
# Prior
def ddm_prior(batch_size):
    """
    Prior distribution for the parameters of the DDM + P300 model.
    Samples from the prior 'batch_size' times.
    ----------
    Returns:
        samples: np.ndarray -- samples from the prior
    """

    # Parameters: alpha, tau, beta, mu_delta, eta_delta, gamma, sigma
    # Alpha ~ U(0.8, 2.0)
    # Tau ~ U(0.15, 0.6)
    # Beta ~ U(0.3, 0.7)
    # Mu_delta ~ N(0, 1)
    # Eta_delta ~ U(0, 2)
    # Gamma ~ N(1.25, 0.5)
    # Sigma ~ U(0, 2)
    low =  [0.8,  0.15, 0.3, -3.0, 0.0, -5.0, 0.1]
    high = [2.0,  0.6, 0.7, 3.0, 2.0,  5.0, 1.5]
    samples = np.random.uniform(low=low, high=high, size=(batch_size, len(low)))
    
    # Return samples 
    return samples.astype(np.float32)

# =====================================================================================
# DDM simulation
@njit
def simulate_ddm(alpha, tau, delta, beta, dt=0.001, dc=1.0):
    """ 
    Simulates one DDM trial 
    ----------
    Returns:
        choicert: float -- choice RT
    """
    evidence = alpha * beta
    n_steps = 0.

    # Simulate DDM until 0 or alpha is reached
    while 0 < evidence < alpha:
        evidence += delta * dt + np.sqrt(dt) * dc * np.random.normal()
        n_steps += 1

    # Calculate RT
    rt = n_steps * dt

    # Calculate choice RT
    choicert = tau + rt if evidence >= alpha else -tau - rt

    # Return choice RT
    return choicert

# P300 simulation
@njit
def simulate_trial(params):
    """ 
    Simulates one full trial including choicert and p300 (z)
    ----------
    Returns:
        choicert: float -- choice RT
        z: float -- P300 response
    """
    alpha, tau, beta, mu_delta, eta_delta, gamma, sigma = params

    # Trial-wise drift rate
    delta = mu_delta + eta_delta * np.random.normal()

    # Simulate behavioral data
    choicert = simulate_ddm(alpha, tau, delta, beta)

    # Simulate P300
    z = gamma * delta + sigma * np.random.normal()

    return choicert, z

# Simulate multiple datasets for BayesFlow training 
def batch_simulator(prior_samples, n_obs):
    """ 
    Simulates multiple datasets for BayesFlow training 
    ----------
    Returns:
        sim_data: np.ndarray -- simulated data
    """
    # Number of parameter sets and initializing arrays
    n_sim = prior_samples.shape[0]
    sim_choicert = np.empty((n_sim, n_obs), dtype=np.float32)
    sim_z = np.empty((n_sim, n_obs), dtype=np.float32)

    # Simulate data for each parameter set
    for i in range(n_sim):
        for j in range(n_obs):
            choicert, z = simulate_trial(prior_samples[i])
            sim_choicert[i, j] = choicert
            sim_z[i, j] = z

    # Combine RT and P300 into BayesFlow's expected input shape (n_sim, n_trials, 2)
    sim_data = np.stack([sim_choicert, sim_z], axis=-1)
    
    # Return data 
    return sim_data

# =====================================================================================
# Plotting
def simulated_data_check(sim_data, name_prefix= 'sim', save_dir= 'Figures'):
    """
    Plots the simulated data
    """

    os.makedirs(save_dir, exist_ok=True)

    # Extract RTs and P300s
    choicert = sim_data[:, :, 0].flatten()
    z = sim_data[:, :, 1].flatten()

    # Plot RTs
    plt.figure(figsize=(8, 6))
    sns.kdeplot(choicert, fill=True, color='blue')
    plt.title("Simulated Choice RT")
    plt.xlabel("Choice RT")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name_prefix}_rt_distribution.png'), dpi=300)
    plt.close()

    # Plot P300s
    plt.figure(figsize=(8, 6))
    sns.kdeplot(z, fill=True, color='green')
    plt.title("Simulated P300 (z) Distribution")
    plt.xlabel("P300 Amplitude")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name_prefix}_P300_distribution.png"), dpi=300)
    plt.close()

    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    # Simulate for 100 participants for 100 trials
    batch_size = 100
    n_trials = 100
    prior_samples = prior(batch_size)
    sim_data = batch_simulator(prior_samples, n_trials)
    
    # Visual check
    simulated_data_check(sim_data, name_prefix='sim', save_dir='Figures')