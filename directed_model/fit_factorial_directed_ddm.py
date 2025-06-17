# =====================================================================================
# Import modules
from glob import glob
import os
from fit_directed_ddm import fit_model

# Get the directory of this file
file_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================================================================
# Loop through all generated datasets
mat_files = sorted(glob(os.path.join(file_dir, "data/ddmdata_*.mat")))

for f in mat_files:
    print(f"\n--- Fitting model to {f} ---")
    
    # Get output name from filename
    base = os.path.splitext(os.path.basename(f))[0]
    
    # Fit model using the reusable function
    fit = fit_model(f, base)
    
    print(f"Done. Results saved for {base}")
