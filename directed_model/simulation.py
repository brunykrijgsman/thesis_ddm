import numpy as np
import scipy.io as sio
import json
import cmdstanpy
from pathlib import Path

def prior(nparts):
    """
    Generate prior parameters for a directed drift diffusion model.
    """
    
    alpha = np.random.uniform(0.8, 2, size=nparts)
    tau = np.random.uniform(0.15, 0.6, size=nparts)
    beta = np.random.uniform(0.3, 0.7, size=nparts)
    mu_z = np.random.normal(0, 1, size=nparts)
    sigma_z = np.abs(np.random.normal(0.5, 0.5, size=nparts))
    b = np.random.normal(0, 1, size=nparts)
    eta = np.random.uniform(0, 1, size=nparts)
    lambda_param = np.random.uniform(-3, 3, size=nparts)

    return dict(
        alpha=alpha,
        tau=tau,
        beta=beta,
        mu_z=mu_z,
        sigma_z=sigma_z,
        b=b,
        eta=eta,
        lambda_param=lambda_param
    )

def simul_directed_ddm(ntrials=100, alpha=1, tau=0.4, beta=0.5, eta=0.3, 
                      varsigma=1, mu_z=0, noise_distribution='gaussian', sigma_z=1, lambda_param=0.7, b=0.5,
                      nsteps=10000, step_length=0.001):
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
    noise_distribution : str, optional
        Distribution of noise (default='gaussian')
    sigma_z : float, optional
        Standard deviation of latent P300 factor (default=1)
    lambda_param : float, optional
        Scaling factor for P300 influence (default=0.7)
    b : float, optional
        Baseline drift adjustment (default=0.5)
    nsteps : int, optional
        Number of steps for simulation (default=300)
    step_length : float, optional
        Time step size in seconds (default=0.001)
    Returns
    -------
    ndarray
        Array of response times (in seconds) multiplied by choice (-1 or 1)
        where negative values indicate incorrect responses and positive values 
        indicate correct responses
    ndarray
        Array of random walks for plotting
    ndarray
        Array of latent z values for each trial
    """
    
    # Initialize output arrays
    rts = np.zeros(ntrials)
    choice = np.zeros(ntrials)
    
    # Generate latent P300 factors
    if noise_distribution == 'gaussian' or noise_distribution == 'base':
        z_value = np.random.normal(mu_z, sigma_z, ntrials)
    elif noise_distribution == 'laplace':
        # Laplace (var = 2*b^2) → b = sigma / sqrt(2)
        b_laplace = sigma_z / np.sqrt(2)
        z_value = np.random.laplace(mu_z, b_laplace, ntrials)
    elif noise_distribution == 'uniform':
        # Uniform (var = (b - a)^2 / 12) → range = sqrt(12)*sigma
        a_uniform = mu_z -np.sqrt(3) * sigma_z
        b_uniform = mu_z + np.sqrt(3) * sigma_z
        z_value = np.random.uniform(a_uniform, b_uniform, ntrials)
    else:
        raise ValueError(f"Unknown distribution: {noise_distribution}")

    # Calculate individual drift rates including P300 influence and trial-to-trial variability
    drift_rates = np.random.normal(lambda_param * z_value + b, eta, ntrials)
    
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
                # Assign max RT if no boundary hit
                rts[n] = s * step_length + tau  
                # Indicates no decision made
                choice[n] = np.nan
        
        random_walks[:, n] = random_walk
                
    # Combine RTs and choices into signed response times
    result = rts * choice
    
    # Returns results, random walks, and z values
    return result, random_walks, z_value 


def generate_directed_ddm_data(ntrials=100, nparts=100, 
                               snr='base', coupling='base', dist='base'):
    """
    Generate directed DDM data for a single condition or base case.
    
    Parameters:
    -----------
    ntrials : int
        Number of trials per participant
    nparts : int  
        Number of participants
    seed : int
        Random seed for reproducibility
    snr : str or None
        Signal-to-noise ratio condition: 'low', 'high', or 'base'
    coupling : str or None
        Coupling condition: 'low', 'high', or 'base' 
    dist : str or None
        Error distribution: 'laplace', 'gaussian', 'uniform', or 'base'

    Returns:
    --------
    dict : Dictionary containing all generated data and parameters
    """
    
    # Generate base parameters from priors
    priors = prior(nparts)
    alpha = priors['alpha']
    tau = priors['tau']
    beta = priors['beta']
    mu_z = priors['mu_z']
    sigma_z = priors['sigma_z']
    b = priors['b']
    eta = priors['eta']
    lambda_param = priors['lambda_param']
    
    # Handle lambda parameter based on coupling condition
    if coupling == 'base':
        lambda_param = lambda_param
    elif coupling == 'low':  # weak coupling [-0.2, 0.2]
        lambda_param = np.random.uniform(-0.2, 0.2, size=nparts)
    elif coupling == 'high':  # strong coupling [-3,-2] U [2,3]
        signs = np.random.choice([-1, 1], size=nparts)
        magnitudes = np.random.uniform(2, 3, size=nparts)
        lambda_param = signs * magnitudes
    else:
        raise ValueError(f"Unknown coupling condition: {coupling}")
    
    condition_key = f"SNR_{snr}_COUP_{coupling}_DIST_{dist}"
    print(f"\nGenerating data for condition: {condition_key}")
    
    # Initialize data arrays
    rt = np.zeros(ntrials * nparts)
    acc = np.zeros(ntrials * nparts)
    y = np.zeros(ntrials * nparts)
    participant = np.zeros(ntrials * nparts)
    z_all = np.zeros(ntrials * nparts)
    
    indextrack = 0
    
    # Generate data for each participant
    for p in range(nparts):
        # Adjust sigma_z for SNR condition
        if snr == 'low':
            sigma_z_condition = sigma_z[p] + 0.5
        elif snr == 'high' or snr == 'base':
            sigma_z_condition = sigma_z[p]
        else:
            raise ValueError(f"Unknown SNR condition: {snr}")
        
        # Simulate response time and accuracy
        signed_rt, _, z_sim = simul_directed_ddm(
            ntrials=ntrials,
            alpha=alpha[p],
            tau=tau[p],
            beta=beta[p],
            eta=eta[p],
            mu_z=mu_z[p],
            sigma_z=sigma_z_condition,
            noise_distribution=dist,
            lambda_param=lambda_param[p],
            b=b[p]
        )

        accuracy = np.sign(signed_rt)
        response_time = np.abs(signed_rt)
        
        # Store results
        start = indextrack
        end = indextrack + ntrials
        y[start:end] = accuracy * response_time
        rt[start:end] = response_time
        acc[start:end] = (accuracy + 1) / 2
        participant[start:end] = p + 1
        z_all[start:end] = z_sim
        indextrack += ntrials
    
    # Compute min RT per participant
    minRT = np.zeros(nparts)
    for p in range(nparts):
        rts_p = rt[participant == (p + 1)]
        valid_rts = rts_p[np.isfinite(rts_p)]
        if len(valid_rts) > 0:
            minRT[p] = np.min(valid_rts)
        else:
            minRT[p] = np.nan
    
    # Create data dictionary
    genparam = dict(
        alpha=alpha,
        beta=beta,
        tau=tau,
        mu_z=mu_z,
        sigma_z=sigma_z,
        lambda_param=lambda_param,
        b=b,
        eta=eta,
        rt=rt,
        acc=acc,
        y=y,
        participant=participant,
        nparts=nparts,
        ntrials=ntrials,
        N=len(rt),
        minRT=minRT,
        z=z_all,
        condition=condition_key
    )
    
    return genparam


def save_simulation_data(data_dict, filepath):
    """
    Save simulation data to a .mat file.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing simulation data
    filepath : Path
        Path to the file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    sio.savemat(filepath, data_dict)
    print(f"Saved: {filepath}")
    
    return filepath 

def fit_directed_ddm(mat_file_path, chains=4, parallel_chains=4, iter_sampling=1000, 
                     iter_warmup=500, seed=2025, show_console=True):
    """
    Fit directed DDM model to data from a .mat file.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file containing the data
    chains : int
        Number of MCMC chains (default=4)
    parallel_chains : int
        Number of parallel chains (default=4)
    iter_sampling : int
        Number of sampling iterations (default=1000)
    iter_warmup : int
        Number of warmup iterations (default=500)
    seed : int
        Random seed (default=2025)
    show_console : bool
        Whether to show console output (default=True)
        
    Returns:
    --------
    fit : cmdstanpy.CmdStanMCMC
        The fitted model object
    """
    
    # Get the directory of this file for finding the .stan file
    file_dir = Path(__file__).resolve().parent
    
    # Load Stan model
    model = cmdstanpy.CmdStanModel(stan_file=str(file_dir / 'directed_ddm.stan'))
    
    # Load data from .mat file
    genparam = sio.loadmat(mat_file_path)
    y = np.squeeze(genparam['y'])
    z = np.squeeze(genparam['z'])
    participant = np.squeeze(genparam['participant']).astype(int)
    minRT = np.squeeze(genparam['minRT'])
    nparts = int(genparam['nparts'].item())
    
    # Clean data (remove invalid trials)
    valid = ~np.isnan(y) & ~np.isinf(y)
    y = y[valid]
    z = z[valid]
    participant = participant[valid]
    N = len(y)

    # Format for Stan
    data_dict = {
        'N': N,
        'nparts': nparts,
        'y': y.tolist(),
        'participant': participant.tolist(),
        'minRT': minRT.tolist(),
        'z': z.tolist()
    }

    # Create a temporary JSON file for Stan
    base = Path(mat_file_path).stem
    json_file = file_dir / f"data/{base}_temp.json"
    with open(json_file, "w") as jfile:
        json.dump(data_dict, jfile)

    # Fit model 
    fit = model.sample(
        data=json_file,
        chains=chains,  
        parallel_chains=parallel_chains,  
        iter_sampling=iter_sampling,  
        iter_warmup=iter_warmup,  
        seed=seed,
        show_console=show_console
    )
    
    # Clean up temporary JSON file
    if json_file.exists():
        json_file.unlink()
    
    return fit 