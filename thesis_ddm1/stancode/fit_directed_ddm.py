import numpy as np
import scipy.io as sio
import cmdstanpy
import os

# Load simulation data
data_filename = 'data/directed_ddm_simdata.mat'
genparam = sio.loadmat(data_filename)

# Extract necessary variables
y = np.squeeze(genparam['y'])
participant = np.squeeze(genparam['participant']).astype(int)
nparts = int(genparam['nparts'].item())
N = len(y)
minRT = np.squeeze(genparam['minRT'])
z = np.squeeze(genparam['z'])
# Remove invalid trials (NaN or Inf)
valid_indices = ~np.isnan(y) & ~np.isinf(y)
y = y[valid_indices]
participant = participant[valid_indices]

# Update N
N = len(y)

# Prepare data dictionary for CmdStanPy
stan_data = {
    'N': N,
    'nparts': nparts,
    'y': y.tolist(),
    'participant': participant.tolist(),
    'minRT': minRT.tolist(),
    'z': z.tolist()
}

# Save data as JSON for CmdStanPy
import json
with open("directed_ddm_data.json", "w") as f:
    json.dump(stan_data, f)

print('Data cleaned and saved!')

# Compile and fit model
stan_model = cmdstanpy.CmdStanModel(stan_file='directed_ddm.stan')

# Sample from the posterior
fit = stan_model.sample(
    data="directed_ddm_data.json",
    chains=4,
    parallel_chains=4,
    iter_sampling=1000,
    iter_warmup=500,
    seed=2025,
    show_console=True
)

print('bye')

# Save results
fit.save_csvfiles('directed_ddm_fit_results')