import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def directed_to_integrative_ddm(directed_data):
    """
    Reshape directed data to integrative ddm data.
    """
    
    # FROM: 'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'
    # TO: 'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'

    nparts = np.squeeze(directed_data['nparts'])

    new_integrative_data = {
        # Core parameters stay the same
        'alpha': directed_data['alpha'],
        'beta': directed_data['beta'],
        'tau': directed_data['tau'],

        # mu_delta = b (because mu_z = 0)
        'mu_delta': directed_data['b'],

        # eta_delta^2 = lambda_param^2 * sigma_z^2 + eta^2
        # eta_delta = sqrt(lambda_param^2 * sigma_z^2 + eta^2)
        'eta_delta': np.sqrt(directed_data['lambda_param']**2 * directed_data['sigma_z']**2 + directed_data['eta']**2),

        # gamma = 1 / lambda_param
        'gamma': 1 / directed_data['lambda_param'],

        # sigma = sqrt(sigma_z^2 - (1 / lambda_param)^2 * eta_delta^2)
        'sigma': np.sqrt(directed_data['sigma_z']**2 - (1 / directed_data['lambda_param'])**2 * directed_data['eta_delta']**2),

        # Other parameters
        'rt': directed_data['rt'],
        'acc': directed_data['acc'],
        'y': directed_data['y'],
        'z': directed_data['z'],
        'participant': directed_data['participant'],
        'nparts': directed_data['nparts'],
        'ntrials': directed_data['ntrials'],
        'N': directed_data['N'],
        'minRT': directed_data['minRT'],
        'condition': directed_data['condition'],
    }

    condition = np.squeeze(new_integrative_data['condition'])
    print(f"condition: {condition}")

    DATA_DIR = PROJECT_ROOT / 'integrative_model' / 'data'
    file_path = DATA_DIR / f"cross_integrative_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_integrative_data)
    print(f"Saved: {file_path}")



def integrative_to_directed_ddm(integrative_data):
    """
    Reshape integrative data to directed data.
    """

    # FROM: 'alpha', 'tau', 'beta', 'mu_delta', 'eta_delta', 'gamma', 'sigma'
    # TO: 'alpha', 'tau', 'beta', 'mu_z', 'sigma_z', 'lambda_param', 'b', 'eta'
    
    nparts = np.squeeze(integrative_data['nparts'])

    new_directed_data = {
        # Core parameters stay the same
        'alpha': integrative_data['alpha'],
        'beta': integrative_data['beta'],
        'tau': integrative_data['tau'],

        # mu_z = 0,
        'mu_z': 0.0,

        # sigma_z^2 = gamma^2 * eta_delta^2 + sigma^2
        # sigma_z = sqrt(gamma^2 * eta_delta^2 + sigma^2)
        'sigma_z': np.sqrt(integrative_data['gamma']**2 * integrative_data['eta_delta']**2 + integrative_data['sigma']**2),

        # lambda = 1 / gamma
        'lambda_param': 1 / integrative_data['gamma'],

        # b = mu_delta, because mu_z = 0
        'b': integrative_data['mu_delta'], # - integrative_data['lambda_param'] * integrative_data['mu_z'],

        # eta = sqrt(eta_delta^2 - (1 / gamma)^2 * (gamma^2 * eta_delta^2 + sigma^2))
        'eta': np.sqrt(integrative_data['eta_delta']**2 - (1 / integrative_data['gamma'])**2 * 
        (integrative_data['gamma']**2 * integrative_data['eta_delta']**2 + integrative_data['sigma']**2)),

        'rt': integrative_data['rt'],
        'acc': integrative_data['acc'],
        'y': integrative_data['y'],
        'z': integrative_data['z'],
        'participant': integrative_data['participant'],
        'nparts': integrative_data['nparts'],
        'ntrials': integrative_data['ntrials'],
        'N': integrative_data['N'],
        'minRT': integrative_data['minRT'],
        'condition': integrative_data['condition'],
    }

    print(f"integrative_data[sigma]: {integrative_data['sigma']}")
    print(f"new_directed_data[sigma_z]: {new_directed_data['sigma_z']}")
    print(f"new_directed_data[eta]: {new_directed_data['eta']}")

    condition = np.squeeze(new_directed_data['condition'])
    print(f"condition: {condition}")

    DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data'
    file_path = DATA_DIR / f"cross_directed_ddm_data_{condition}.mat"
    sio.savemat(file_path, new_directed_data)
    print(f"Saved: {file_path}")

if __name__ == "__main__":
    # directed_data = sio.loadmat(PROJECT_ROOT / 'directed_model' / 'data' / 'directed_ddm_base.mat')
    # directed_to_integrative_ddm(directed_data)

    integrative_data = sio.loadmat(PROJECT_ROOT / 'integrative_model' / 'data' / 'integrative_ddm_data_SNR_high_COUP_high_DIST_gaussian.mat')
    integrative_to_directed_ddm(integrative_data)

