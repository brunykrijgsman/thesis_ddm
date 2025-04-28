import numpy as np

def simul_directed_ddm(ntrials=100, alpha=1, tau=0.4, beta=0.5, eta=0.3, 
                      varsigma=1, mu_z=0, sigma_z=1, lambda_param=0.7, b=0.5,
                      nsteps=1000, step_length=0.01):
    """
    Simulates data according to a directed drift diffusion model with P300 influence.
    
    Parameters
    ----------
    ntrials : int, optional
        Number of trials to simulate (default=100)
    alpha : float, optional
        Boundary separation (default=1)
    tau : float, optional
        Non-decision time in seconds (default=0.4)
    beta : float, optional
        Starting point bias as proportion of boundary (default=0.5)
    eta : float, optional
        Trial-to-trial variability in drift rate (default=0.3)
    varsigma : float, optional
        Within-trial variability in drift rate (diffusion coefficient) (default=1)
    mu_z : float, optional
        Mean of latent P300 factor (default=0)
    sigma_z : float, optional
        Standard deviation of latent P300 factor (default=1)
    lambda_param : float, optional
        Scaling factor for P300 influence (default=0.7)
    b : float, optional
        Baseline drift adjustment (default=0.5)
    nsteps : int, optional
        Number of steps for simulation (default=300)
    step_length : float, optional
        Time step size in seconds (default=0.01)
        
    Returns
    -------
    ndarray result
        Array of response times (in seconds) multiplied by choice (-1 or 1)
        where negative values indicate incorrect responses and positive values 
        indicate correct responses
    ndarray random_walks
        Array of random walks for plotting
    """
    
    # Initialize output arrays
    rts = np.zeros(ntrials)
    choice = np.zeros(ntrials)
    
    # Generate latent P300 factors
    z = np.random.normal(mu_z, sigma_z, ntrials)
    
    # Calculate individual drift rates including P300 influence and trial-to-trial variability
    drift_rates = lambda_param * z + b + np.random.normal(0, eta, ntrials)
    
    # Initialize arrays for storing random walks
    random_walks = np.zeros((nsteps, ntrials))
    
    # Simulation loop
    for n in range(ntrials):
        drift = drift_rates[n]
        random_walk = np.zeros(nsteps)
        random_walk[0] = beta * alpha
        
        for s in range(1, nsteps):
            # Update position with drift and noise
            random_walk[s] = random_walk[s-1] + np.random.normal(
                drift * step_length, 
                varsigma * np.sqrt(step_length)
            )
            
            # Check for boundary crossings
            if random_walk[s] >= alpha:
                random_walk[s:] = alpha  # Set remaining path to boundary
                rts[n] = s * step_length + tau
                choice[n] = 1
                break
            elif random_walk[s] <= 0:
                random_walk[s:] = 0  # Set remaining path to boundary
                rts[n] = s * step_length + tau
                choice[n] = -1
                break
            elif s == (nsteps - 1):
                rts[n] = np.nan
                choice[n] = np.nan
        
        random_walks[:, n] = random_walk
                
    # Combine RTs and choices into signed response times
    result = rts * choice
    
    # Returns results and random walks for plotting 
    return result, random_walks 