# Script to fit the directed DDM model to the simulated data
# This script is used to fit the model to the simulated data and save the results
# The results are saved in the Results directory

# =====================================================================================
# Import modules
import numpy as np
import scipy.io as sio
import cmdstanpy
import os
import json

# =====================================================================================
def fit_model(data_filename, output_name=None):
    """
    Fit the directed DDM model to data from a .mat file
    
    Parameters:
    -----------
    data_filename : str
        Path to the .mat file containing the data
    output_name : str, optional
        Name for the output directory. If None, uses the base name of the data file
    
    Returns:
    --------
    fit : cmdstanpy.CmdStanMCMC
        The fitted model object
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load simulation data
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
    z = z[valid_indices]

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
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(data_filename))[0]
    
    json_filename = os.path.join(file_dir, 'data', f'{output_name}.json')
    with open(json_filename, "w") as f:
        json.dump(stan_data, f)

    print(f'Data cleaned and saved to {json_filename}!')

    # Compile and fit model
    stan_model = cmdstanpy.CmdStanModel(stan_file=os.path.join(file_dir, 'directed_ddm.stan'))

    # Sample from the posterior
    fit = stan_model.sample(
        data=json_filename,
        chains=4,
        parallel_chains=4,
        iter_sampling=1000,
        iter_warmup=500,
        seed=2025,
        show_console=True
    )

    print('Fitting complete!')

    # Save results
    results_dir = os.path.join(file_dir, 'Results', output_name)
    os.makedirs(results_dir, exist_ok=True)
    fit.save_csvfiles(dir=results_dir)
    
    print(f'Results saved to {results_dir}')
    
    return fit

# =====================================================================================
if __name__ == "__main__":
    # When running as main, fit model from the default data file
    file_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_file = os.path.join(file_dir, 'data/directed_ddm_simdata.mat')
    
    print("Fitting directed DDM model to default simulation data...")
    fit = fit_model(default_data_file, 'directed_ddm_fit_results')
    print('Done!')