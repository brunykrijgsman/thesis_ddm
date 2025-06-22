import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Function to convert directed data to integrative data for recovery plotting and analysis
def directed_to_integrative_ddm(directed_data):
    """
    Reshape directed data to integrative ddm data.
    """
    
    # FROM: 'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'
    # TO: 'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'

    # Extract parameters for correlation computation
    nparts = int(np.squeeze(directed_data['nparts']))
    z_i = np.ravel(directed_data['z'])
    participant = np.ravel(directed_data['participant'])  # Assumes participant labels are 1-indexed
    lambda_param = np.ravel(directed_data['lambda_param'])
    b = np.ravel(directed_data['b'])
    eta = np.ravel(directed_data['eta'])

    # Generate delta per trial and compute correlation between delta and z
    delta = np.zeros_like(z_i)
    corr_delta_z = np.zeros(nparts)
    for p in range(nparts):
        idx = participant == (p + 1)
        z_p = z_i[idx]
        noise = np.random.normal(loc=0, scale=eta[p], size=len(z_p))
        delta_p = lambda_param[p] * z_p + b[p] + noise
        delta[idx] = delta_p
        if len(z_p) > 1:
            corr_delta_z[p] = np.corrcoef(z_p, delta_p)[0, 1]
        else:
            corr_delta_z[p] = 0

    # print(f"Empirical correlation between delta and z: {corr_delta_z}")
    # print(f"Shape of corr_delta_z: {np.array(corr_delta_z).shape}")
    
    # Compute the transformation parameters
    # mu_delta = b (because mu_z = 0)
    mu_delta = directed_data['b']

    # eta_delta = sqrt(lambda_param^2 * sigma_z^2 + eta^2)
    eta_delta = np.sqrt(directed_data['lambda_param']**2 * directed_data['sigma_z']**2 + directed_data['eta']**2)

    # gamma = corr_delta_z * (sigma_z / sqrt(lambda^2 * sigma_z^2 + eta^2))
    gamma = corr_delta_z * (directed_data['sigma_z'] / np.sqrt(directed_data['lambda_param']**2 * directed_data['sigma_z']**2 + directed_data['eta']**2))

    # sigma = sqrt(sigma_z^2 * (1 - corr_delta_z^2))
    sigma = directed_data['sigma_z'] * (np.sqrt(1 - corr_delta_z**2))

    new_integrative_data = {
        # Core parameters stay the same
        'alpha': directed_data['alpha'],
        'beta': directed_data['beta'],
        'tau': directed_data['tau'],
        'mu_delta': mu_delta,
        'eta_delta': eta_delta,
        'gamma': gamma,
        'sigma': sigma,

        # Other parameters
        'rt': directed_data['rt'],
        'acc': directed_data['acc'],
        'choicert': directed_data['y'],
        'z': directed_data['z'],
        'participant': directed_data['participant'],
        'nparts': directed_data['nparts'],
        'ntrials': directed_data['ntrials'],
        'N': directed_data['N'],
        'minRT': directed_data['minRT'],
        'condition': directed_data['condition'],
        
        # Add the computed correlation 
        'cor_delta_z_theoretical': corr_delta_z,
    }

    condition = np.squeeze(new_integrative_data['condition'])
    print(f"condition: {condition}")

    DATA_DIR = PROJECT_ROOT / 'integrative_model' / 'data'
    file_path = DATA_DIR / f"cross_integrative_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_integrative_data)
    print(f"Saved: {file_path}")


# Function to convert integrative data to directed data for recovery plotting and analysis
def integrative_to_directed_ddm(integrative_data):
    """
     Reshape integrative data to directed data.
     """

    # FROM: 'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'
    # TO: 'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'

    # Extract parameters for correlation computation
    nparts = int(np.squeeze(integrative_data['nparts']))
    ntrials = int(np.squeeze(integrative_data['ntrials']))
    z_i = np.ravel(integrative_data['z'])
    mu_delta = np.ravel(integrative_data['mu_delta'])
    eta_delta = np.ravel(integrative_data['eta_delta'])
    participant = np.ravel(integrative_data['participant'])

    # Generate delta per trial
    delta = np.zeros_like(z_i)
    corr_delta_z = np.zeros(nparts)
    for p in range(nparts):
        idx = participant == (p + 1)
        z_p = z_i[idx]
        delta_p = np.random.normal(loc=mu_delta[p], scale=eta_delta[p], size=len(z_p))
        delta[idx] = delta[p]
        if len(z_p) > 1:
            corr_delta_z[p] = np.corrcoef(z_p, delta_p)[0, 1]
        else:
            corr_delta_z[p] = 0
    
    print(f"Empirical correlation between delta and z: {corr_delta_z}")

    # mu_z = 0 (assumption in directed model)
    mu_z = np.zeros((1, nparts))

    # sigma_z = sqrt(gamma^2 * eta_delta^2 + sigma^2)
    sigma_z = np.sqrt(integrative_data['gamma']**2 * integrative_data['eta_delta']**2 + integrative_data['sigma']**2)

    # lambda_param = eta_delta / sqrt(gamma^2 * eta_delta^2 + sigma^2)
    lambda_param = corr_delta_z * (integrative_data['eta_delta'] / np.sqrt(integrative_data['gamma']**2 * integrative_data['eta_delta']**2 + integrative_data['sigma']**2))

    # eta = sqrt(eta_delta^2 – gamma^2 * sigma_z^2)
    eta = np.sqrt(integrative_data['eta_delta']**2 - integrative_data['gamma']**2 * sigma_z**2)

    # b = mu_delta * (1 - lambda * gamma)
    b_value = integrative_data['mu_delta'] * (1 - lambda_param * integrative_data['gamma'])

    new_directed_data = {
        # Core parameters stay the same
        'alpha': integrative_data['alpha'],
        'beta': integrative_data['beta'],
        'tau': integrative_data['tau'],
        'mu_z': mu_z,
        'sigma_z': sigma_z,
        'lambda_param': lambda_param,
        'b': b_value,
        'eta': eta,

        'rt': integrative_data['rt'],
        'acc': integrative_data['acc'],
        'y': integrative_data['choicert'],
        'z': integrative_data['z'],
        'participant': integrative_data['participant'],
        'nparts': integrative_data['nparts'],
        'ntrials': integrative_data['ntrials'],
        'N': integrative_data['N'],
        'minRT': integrative_data['minRT'],
        'condition': integrative_data['condition'],
        
        # Add the computed correlation
        'cor_delta_z_theoretical': corr_delta_z
    }

    condition = np.squeeze(new_directed_data['condition'])
    print(f"condition: {condition}")

    DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data'
    file_path = DATA_DIR / f"cross_directed_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_directed_data)
    print(f"Saved: {file_path}")
