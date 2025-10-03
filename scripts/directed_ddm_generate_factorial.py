# =====================================================================================
# Generate data across 2x2x3 factorial design:
# - SNR (low/high): controls sigma_z
# - Coupling (low/high): controls lambda_param
# - Error distribution: laplace, gaussian, uniform
# =====================================================================================
# Import modules
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from directed_model.simulation_new_sigma_z import generate_directed_ddm_data, save_simulation_data

DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data_new_sigma_z'

ntrials = 100
nparts = 100
seed = 2025

# =====================================================================================

# Set seed for reproducibility
np.random.seed(seed)

# Conditions for simulation
snr_levels = ['low', 'high'] 
coupling_levels = ['low', 'high']
error_distributions = ['laplace', 'gaussian', 'uniform']

# Generate data for each condition
for snr in snr_levels:
    for coupling in coupling_levels:
        for dist in error_distributions:
            
            # Generate data using the utility function
            data_dict = generate_directed_ddm_data(
                ntrials=ntrials,
                nparts=nparts,
                snr=snr,
                coupling=coupling,
                dist=dist
            )
            
            # Create filename for this condition
            condition_key = f"SNR_{snr}_COUP_{coupling}_DIST_{dist}"
            filename = f"ddmdata_{condition_key}.mat"
            filepath = DATA_DIR / filename
            
            # Save data using the utility function
            save_simulation_data(data_dict, filepath)


