import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Function to convert directed data to integrative data for recovery plotting and analysis
def directed_to_integrative_ddm(directed_data):
    """
    Reshape directed data to integrative DDM data and compute theoretical correlations.

    FROM: 'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'
    TO:   'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'
    """
    
    # Extract parameters
    nparts = int(np.squeeze(directed_data['nparts']))
    z_i = np.ravel(directed_data['z'])
    participant = np.ravel(directed_data['participant'])  # Assumes participant labels are 1-indexed
    lambda_param = np.ravel(directed_data['lambda_param'])
    b = np.ravel(directed_data['b'])
    eta = np.ravel(directed_data['eta'])
    sigma_z = np.ravel(directed_data['sigma_z'])

    # Compute theoretical correlation
    theoretical_corr_delta_z = lambda_param * sigma_z / np.sqrt(lambda_param**2 * sigma_z**2 + eta**2)

    # Compute transformation parameters
    # mu_delta = b
    mu_delta = directed_data['b'] 
    
    # eta_delta = sqrt(lambda_param^2 * sigma_z^2 + eta^2)
    eta_delta = np.sqrt(lambda_param**2 * sigma_z**2 + eta**2)
    
    # sigma = sigma_z * np.sqrt(1 - corr_delta_z_theoretical**2)
    sigma = sigma_z * np.sqrt(1 - theoretical_corr_delta_z**2)
    
    # gamma = theoretical_corr_delta_z * (sigma_z / eta_delta)
    gamma = theoretical_corr_delta_z * (sigma_z / eta_delta)

    new_integrative_data = {
        # Core parameters
        'alpha': directed_data['alpha'],
        'beta': directed_data['beta'],
        'tau': directed_data['tau'],
        
        # New parameters
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

        # Report theoretical correlation
        'theoretical_corr_delta_z': theoretical_corr_delta_z,
    }

    condition = np.squeeze(new_integrative_data['condition'])
    DATA_DIR = PROJECT_ROOT / 'integrative_model' / 'data_new_sigma'
    file_path = DATA_DIR / f"cross_integrative_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_integrative_data)
    print(f"Saved: {file_path}")


# Function to convert integrative data to directed data
def integrative_to_directed_ddm(integrative_data):
    """
    Reshape integrative data to directed DDM data and compute theoretical correlations.

    FROM: 'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'
    TO:   'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'
    """

    # Extract parameters
    nparts = int(np.squeeze(integrative_data['nparts']))
    z_i = np.ravel(integrative_data['z'])
    mu_delta = np.ravel(integrative_data['mu_delta'])
    eta_delta = np.ravel(integrative_data['eta_delta'])
    participant = np.ravel(integrative_data['participant'])

    # Compute theoretical correlation
    sigma_z = np.sqrt(integrative_data['gamma']**2 * eta_delta**2 + integrative_data['sigma']**2)
    theoretical_corr_delta_z = (integrative_data['gamma'] * eta_delta) / sigma_z

    # Compute transformation parameters
    # mu_z = 0
    mu_z = np.zeros((1, nparts))
    
    # lambda_param = theoretical_corr_delta_z * (eta_delta / sigma_z)
    lambda_param = theoretical_corr_delta_z * (eta_delta / sigma_z)
    
    # eta = eta_delta * np.sqrt(1 - theoretical_corr_delta_z**2)
    eta = eta_delta * np.sqrt(1 - theoretical_corr_delta_z**2)
    
    # b_value = mu_delta * (1 - lambda_param * integrative_data['gamma'])
    b_value = mu_delta * (1 - lambda_param * integrative_data['gamma'])

    new_directed_data = {
        # Core parameters
        'alpha': integrative_data['alpha'],
        'beta': integrative_data['beta'],
        'tau': integrative_data['tau'],

        # Transformed parameters
        'mu_z': mu_z,
        'sigma_z': sigma_z,
        'lambda_param': lambda_param,
        'b': b_value,
        'eta': eta,

        # Other parameters
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

        # Report theoretical correlation
        'theoretical_corr_delta_z': theoretical_corr_delta_z,
    }

    condition = np.squeeze(new_directed_data['condition'])
    DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data_new_sigma_z'
    file_path = DATA_DIR / f"cross_directed_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_directed_data)
    print(f"Saved: {file_path}")
