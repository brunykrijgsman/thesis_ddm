# =====================================================================================
# Generate data across 2x2x3 factorial design:
# - SNR (low/high): controls sigma_z (P300 noise)
# - Coupling (low/high): controls lambda_param (DDM-P300 coupling)
# - Error distribution: laplace, gaussian, uniform
# =====================================================================================
# Import modules
import numpy as np
import scipy.io as sio
import os
from directed_model.directed_ddm_utils import simul_directed_ddm

# Get the directory of this file
file_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================================================================
def generate_factorial_ddm_data(ntrials=100, nparts=100, seed=2025):
    """
    Generate data across 2x2x3 factorial design:
    - SNR (low/high): controls sigma_z (P300 noise)
    - Coupling (low/high): controls lambda_param (DDM-P300 coupling)
    - Error distribution: laplace, gaussian, uniform
    -------------------------------------------------------------------------------------
    Returns:
        - List of data dictionaries, one per condition.
        - List of condition labels as tuples.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    # Informed parameters for simulation
    alpha = np.random.uniform(0.8, 2, size=nparts)
    tau = np.random.uniform(.15, .6, size=nparts)
    beta = np.random.uniform(.3, .7, size=nparts)
    mu_z = np.random.normal(0, 1, size=nparts)
    sigma_z = np.abs(np.random.normal(0.5, 0.5, size=nparts))
    lambda_param = np.random.uniform(-3, 3, size=nparts)  # Uniform from -3 to 3
    b = np.random.uniform(0, 1, size=nparts)
    eta = np.random.uniform(0, 1, size=nparts)

    # Conditions for simulation
    snr_levels = ['low', 'high'] 
    coupling_levels = ['low', 'high']
    error_distributions = ['laplace', 'gaussian', 'uniform']

    # Initialize lists to store data and labels
    all_data = []
    condition_labels = []

    # Factorial loop
    for snr in snr_levels:
        for coupling in coupling_levels:
            for dist in error_distributions:
                
                # Coupling values: Sample lambda_param depending on condition
                if coupling == 'low':  # Weak coupling [-0.2, 0.2] around 0 can be negative or positive
                    lambda_param_condition = np.random.uniform(-0.2, 0.2, size=nparts)
                else:  # Strong coupling [-3,-2] U [2,3] can be negative or positive
                    signs = np.random.choice([-1, 1], size=nparts)  # randomly choose negative or positive range
                    magnitudes = np.random.uniform(2, 3, size=nparts)
                    lambda_param_condition = signs * magnitudes
                
                # Condition label
                condition_key = f"SNR_{snr}_COUP_{coupling}_DIST_{dist}"
                print(f"\nGenerating data for condition: {condition_key}")

                # Initialize arrays to store data
                rt = np.zeros(ntrials * nparts)
                acc = np.zeros(ntrials * nparts)
                y = np.zeros(ntrials * nparts)
                participant = np.zeros(ntrials * nparts)
                z_all = np.zeros(ntrials * nparts)
                sigma_z_condition_all = np.zeros(nparts) # Store condition-level sigma_z

                # Initialize index for tracking data
                indextrack = 0

                # Generate data for each participant
                for p in range(nparts):
                    # Adjust sigma_z for SNR
                    sigma_z_condition = sigma_z[p] + 0.5 if snr == 'low' else sigma_z[p]
                    sigma_z_condition_all[p] = sigma_z_condition

                    # Simulate response time and accuracy
                    signed_rt, _, z = simul_directed_ddm(
                        ntrials=ntrials,
                        alpha=alpha[p],
                        tau=tau[p],
                        beta=beta[p],
                        eta=eta[p],
                        mu_z=mu_z[p],
                        sigma_z=sigma_z_condition,
                        noise_distribution=dist,
                        lambda_param=lambda_param_condition[p],
                        b=b[p]
                    )

                    # Store data
                    accuracy = np.sign(signed_rt)
                    response_time = np.abs(signed_rt)

                    start = indextrack
                    end = indextrack + ntrials
                    y[start:end] = accuracy * response_time
                    rt[start:end] = response_time
                    acc[start:end] = (accuracy + 1) / 2
                    participant[start:end] = p + 1
                    z_all[start:end] = z
                    indextrack += ntrials

                # Compute min RT per participant
                minRT = np.zeros(nparts)
                for p in range(nparts):
                    rts_p = rt[participant == (p + 1)]
                    valid_rts = rts_p[np.isfinite(rts_p)]
                    minRT[p] = np.min(valid_rts) if len(valid_rts) > 0 else np.nan
                    
                # Store generated parameters
                genparam = dict(
                    alpha=alpha,
                    beta=beta,
                    tau=tau,
                    mu_z=mu_z,
                    sigma_z=sigma_z_condition_all,
                    lambda_param=lambda_param_condition,
                    b=b,
                    eta=eta,
                    rt=rt,
                    acc=acc,
                    y=y,
                    participant=participant,
                    nparts=nparts,
                    ntrials=ntrials,
                    N=len(rt),
                    minRT=minRT,
                    z=z_all,
                    condition=condition_key
                )

                # Save data to mat file
                data_dir = os.path.join(file_dir, 'data')
                os.makedirs(data_dir, exist_ok=True)
                filename = os.path.join(data_dir, f"ddmdata_{condition_key}.mat")
                sio.savemat(filename, genparam)
                print(f"Saved: {filename}")

                # Append to results
                all_data.append(genparam)
                condition_labels.append((snr, coupling, dist))

    return all_data, condition_labels

# =====================================================================================
# Main function
if __name__ == "__main__":
    generate_factorial_ddm_data()