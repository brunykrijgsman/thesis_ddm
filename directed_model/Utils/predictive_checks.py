# =====================================================================================
# Import modules
import numpy as np

# =====================================================================================
# Generate posterior predictive data
# Posterior Predictive Checks
def generate_predicted_data(fit, df, n_trials):
    """
    Generate predicted data based on the posterior samples for parameters.
    """
    # Generate predicted data
    predicted_y = []
    predicted_z = []

    # Compute mean values for parameters
    lambda_sample = df['lambda'].mean() 
    mu_z_sample = df['mu_z'].mean()           
    sigma_z_sample = df['sigma_z'].mean()     

    # Generate predicted data for each trial
    for i in range(n_trials):
        participant = participants[i]
        alpha_sample = df[f'alpha[{participant}]'].mean()  
        tau_sample = df[f'tau[{participant}]'].mean()      
        beta_sample = df[f'beta[{participant}]'].mean()    
        eta_sample = df[f'eta[{participant}]'].mean()      
        b_sample = df['b'].mean()
        y_sample = true_y[i]

        # Simulate signed RT and latent z using the DDM with the sampled parameters
        simulated_y, _, simulated_z = simul_directed_ddm(
            ntrials=1,
            alpha=alpha_sample,
            tau=tau_sample,
            beta=beta_sample,
            eta=eta_sample,
            lambda_param=lambda_sample,
            mu_z=mu_z_sample,
            sigma_z=sigma_z_sample,
            b=b_sample
        )
        
        # Store the predicted values
        predicted_y.append(simulated_y[0])
        predicted_z.append(simulated_z[0])
    
    # Return the predicted values as arrays
    return np.array(predicted_y), np.array(predicted_z)