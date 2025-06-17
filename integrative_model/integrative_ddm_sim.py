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
from scipy.stats import truncnorm

# Configure environment
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"

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

# P300 simulator
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
    delta = np.random.normal(mu_delta, eta_delta)

    # Simulate behavioral data
    choicert = simulate_ddm(alpha, tau, delta, beta)

    # Simulate P300
    z = np.random.normal(gamma * delta, sigma)

    return choicert, z

# Simulate multiple datasets for BayesFlow training 
@njit
def batch_simulator(params, n_obs):
    """ 
    Simulates multiple datasets for BayesFlow training 
    ----------
    Returns:
        sim_choicert: np.ndarray -- simulated choice RT
        sim_z: np.ndarray -- simulated P300 response
    """
    # Number of parameter sets and initializing arrays
    sim_choicert = np.empty(n_obs, dtype=np.float32)
    sim_z = np.empty(n_obs, dtype=np.float32)

    # Simulate data for each parameter set
    for i in range(n_obs):
        sim_choicert[i], sim_z[i] = simulate_trial(params)

    # Return data 
    return sim_choicert, sim_z

# Prior distribution
def prior():
    """
    Prior distribution for the parameters of the DDM + P300 model.
    Returns a dictionary containing samples from the (truncated) normal and uniform priors.
    """

    # Helper function to sample from a truncated normal
    def truncated_normal(mean, std, low, high):
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)

    # Old sample informed parameters
    # alpha = truncated_normal(1.0, 0.5, 0.001, 3.0)       # N(1, .5^2), truncated to [.001, 3]
    # beta = truncated_normal(0.5, 0.25, 0.001, 0.99)      # N(0.5, .25^2), truncated to [.001, .99]
    # Fixed tau (previous overestimated)
    # tau = truncated_normal(0.3, 0.1, 0.1, 0.9)           # Truncated normal distribution between 0.1 and 0.6
    # mu_delta = np.random.normal(0.0, 1)                  # N(0, 1^2), no truncation
    # log_eta_delta = np.random.normal(np.log(0.5), 0.3)   # Log-normal distribution with mean 0.5 and standard deviation 0.3
    # eta_delta = np.exp(log_eta_delta)                    # Exponential transformation to ensure positive values
    # gamma = np.random.normal(0.0, 1.0)                   # N(0, 1^2), no truncation
    # Fixed sigma (previous underestimated)
    # sigma = (np.abs(np.random.normal(0.0, 1)))           # Absolute value of normal distribution (only positive values)

    # Best sample uniform parameters
    # alpha = truncated_normal(1.5, 0.5, 0.001, 3.0)
    # tau = np.random.uniform(0.1, 0.6)
    # beta = np.random.uniform(0.1, 0.9)
    # beta = truncated_normal(0.5, 0.20, 0.01, 0.99)
    # mu_delta = np.random.normal(0.0, 1.0)
    # eta_delta = np.random.uniform(0.0, 2.0)
    # log_eta_delta = np.random.normal(np.log(0.5), 0.3)
    # eta_delta = np.exp(log_eta_delta)
    # gamma = np.random.uniform(-5.0, 5.0)
    # gamma = np.random.normal(0.0, 1.0)
    # sigma = np.random.uniform(0.5, 2.0)

    # Uniform sample parameters
    alpha = np.random.uniform(.5,2.0)
    tau = np.random.uniform(.1,1)
    # beta = np.random.uniform(.1,.9)
    beta = truncated_normal(0.5, 0.25, 0.001, 0.99) # new beta
    mu_delta = np.random.normal(0,1)
    eta_delta = np.random.uniform(0,2)
    gamma = np.random.uniform(-3, 3)
    # sigma = np.random.uniform(0,2)
    sigma = (np.abs(np.random.normal(0.5, 0.5)))
    
    # Return dictionary of parameters
    return dict(
        alpha=alpha,
        tau=tau,
        beta=beta,
        mu_delta=mu_delta,
        eta_delta=eta_delta,
        gamma=gamma,
        sigma=sigma
    )


# Likelihood function
def likelihood(alpha, tau, beta, mu_delta, eta_delta, gamma, sigma, n_obs):
    """ 
    Simulates one full trial including choicert and p300 (z)
    ----------
    Args:
        alpha: float -- drift rate
        tau: float -- non-decision time
        beta: float -- boundary separation
        mu_delta: float -- mean of the drift rate
        eta_delta: float -- standard deviation of the drift rate
        gamma: float -- P300 response
        sigma: float -- standard deviation of the P300 response
        N: int -- number of trials
    Returns:
        choicert: float -- choice RT
        z: float -- P300 response
    """

    params = np.array([alpha, tau, beta, mu_delta, eta_delta, gamma, sigma])
    choicert, z = batch_simulator(params, n_obs)

    return dict(choicert=choicert, z=z)

