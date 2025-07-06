import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from shared.cross_validation import directed_to_integrative_ddm, integrative_to_directed_ddm

# Process all directed model files
DIRECTED_DATA_DIR = PROJECT_ROOT / 'directed_model' / 'data'
directed_mat_files = sorted(DIRECTED_DATA_DIR.glob("ddmdata_*.mat"))

print(f"Processing {len(directed_mat_files)} directed model files:")
for mat_file in directed_mat_files:
    directed_data = sio.loadmat(mat_file)
    directed_to_integrative_ddm(directed_data)

# Process all integrative model files
INTEGRATIVE_DATA_DIR = PROJECT_ROOT / 'integrative_model' / 'data'
integrative_mat_files = sorted(INTEGRATIVE_DATA_DIR.glob("integrative_ddm_data_*.mat"))

print(f"\nProcessing {len(integrative_mat_files)} integrative model files:")
for mat_file in integrative_mat_files:
    integrative_data = sio.loadmat(mat_file)
    integrative_to_directed_ddm(integrative_data)
