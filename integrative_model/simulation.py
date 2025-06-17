# =====================================================================================
# Import modules
import numpy as np
from numba import njit
from scipy.stats import truncnorm

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
    
    # Uniform sample parameters
    alpha = np.random.uniform(.5,2.0)
    tau = np.random.uniform(.1,1)
    beta = np.random.uniform(.1,.9)
    mu_delta = np.random.normal(0,1)
    eta_delta = np.random.uniform(0,2)
    gamma = np.random.uniform(-3, 3)
    # sigma = np.random.uniform(0,.2)
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

