# =====================================================================================
# Import modules
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from scipy import stats
from directed_model.directed_ddm_utils import simul_directed_ddm

# =====================================================================================
# Simulations

# Get current filename without extension and use it for data file
current_file = os.path.splitext(os.path.basename(__file__))[0]
data_filename = os.path.join(os.path.dirname(__file__), 'data', f'{current_file}data.mat')

# Simulation parameters 
if not os.path.exists(data_filename) or True:
    # Number of simulated participants
    nparts = 100
    # Number of trials for one participant
    ntrials = 100
    # Number of total trials in each simulation
    ntotal = ntrials * nparts
    # Set random seed
    np.random.seed(2025)

    # Parameters for simulation 
    alpha = np.random.uniform(0.8, 2, size=nparts)  # Uniform from 0.8 to 2
    tau = np.random.uniform(.15, .6, size=nparts)  # Uniform from .15 to .6 seconds
    beta = np.random.uniform(.3, .7, size=nparts)  # Uniform from .3 to .7 
    mu_z = np.random.normal(0, 1, size=nparts)  # Normal distribution with mean 0 and SD 1
    sigma_z = np.abs(np.random.normal(0.5, 0.5, size=nparts))  # Absolute value of normal distribution with mean 0.5 and SD 0.5
    lambda_param = np.random.uniform(-3, 3, size=nparts)  # Uniform from -3 to 3
    b = np.random.normal(0, 1, size=nparts)  # Normal distribution with mean 0 and SD 1
    eta = np.random.uniform(0, 1, size=nparts)  # Trial-to-trial variability uniform from 0 to 1
    
    # Simulate data
    rt = np.zeros(ntotal)
    acc = np.zeros(ntotal)
    y = np.zeros(ntotal)
    participant = np.zeros(ntotal)
    z_all = np.zeros(ntotal)  
    
    indextrack = 0 
    for p in range(nparts):

        # Simulate data
        signed_rt, random_walks, z_i = simul_directed_ddm(ntrials=ntrials, alpha=alpha[p], 
                                                          tau=tau[p], beta=beta[p], mu_z=mu_z[p], 
                                                          sigma_z=sigma_z[p], lambda_param=lambda_param[p], 
                                                          b=b[p], eta=eta[p]
                                                          )
        accuracy = np.sign(signed_rt)
        response_time = np.abs(signed_rt)
        
        # Initialize tracker
        start = indextrack
        end = indextrack + ntrials

        # Store results
        y[start:end] = accuracy * response_time
        rt[start:end] = response_time
        acc[start:end] = (accuracy + 1) / 2
        participant[start:end] = p + 1   
        z_all[start:end] = z_i 
        indextrack += ntrials

        # Check for invalid RTs
        if np.any(np.isnan(response_time)) or np.any(np.isinf(response_time)):
            print(f"Participant {p+1}: Invalid RTs found.")
            print(f"Parameters: alpha={alpha[p]}, tau={tau[p]}, beta={beta[p]}, mu_z={mu_z[p]}, sigma_z={sigma_z[p]}, lambda={lambda_param[p]}, b={b[p]}, eta={eta[p]}")

        # Compute minRT after the full loop
        minRT = np.zeros(nparts)
        for p in range(nparts):
            rts_p = rt[participant == (p + 1)]
            valid_rts = rts_p[np.isfinite(rts_p)]
            if len(valid_rts) > 0:
                minRT[p] = np.min(valid_rts)
            else:
                minRT[p] = np.nan  # just in case

    # Generate data dictionary
    genparam = dict()
    genparam['tau'] = tau
    genparam['beta'] = beta
    genparam['alpha'] = alpha
    genparam['mu_z'] = mu_z
    genparam['sigma_z'] = sigma_z
    genparam['lambda'] = lambda_param
    genparam['b'] = b
    genparam['eta'] = eta
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['participant'] = participant
    genparam['nparts'] = nparts
    genparam['ntrials'] = ntrials
    genparam['N'] = ntotal
    genparam['minRT'] = minRT
    genparam['z'] = z_all
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
    
    # Save with dynamic filename
    sio.savemat(data_filename, genparam)
else:
    genparam = sio.loadmat(data_filename)
