import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from directed_model.simulation import generate_directed_ddm_data, save_simulation_data

np.random.seed(2025)

DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data'

model_name = 'directed_ddm_base'
filename = f'{model_name}.mat'
nparts = 100
ntrials = 100

genparam = generate_directed_ddm_data(
    ntrials=ntrials,
    nparts=nparts,
    snr='base',
    coupling='base', 
    dist='base'
)

filename =  DATA_DIR / filename
save_simulation_data(genparam, filename)





