# =====================================================================================
# Analysis utilities for directed DDM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from directed_model.simulation import simul_directed_ddm

# =====================================================================================
def generate_predicted_data(fit, df, participants, true_y, n_trials):
    """
    Generate predicted data based on the posterior samples for parameters.
    """
    # Generate predicted data
    predicted_y = []
    predicted_z = []

    # Generate predicted data for each trial
    for i in range(n_trials):
        participant = participants[i]
        alpha_sample = df[f'alpha[{participant}]'].mean()  
        tau_sample = df[f'tau[{participant}]'].mean()      
        beta_sample = df[f'beta[{participant}]'].mean()    
        eta_sample = df[f'eta[{participant}]'].mean()  
        mu_z_sample = df[f'mu_z[{participant}]'].mean()
        sigma_z_sample = df[f'sigma_z[{participant}]'].mean()
        lambda_sample = df[f'lambda[{participant}]'].mean()
        b_sample = df[f'b[{participant}]'].mean()
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

# =====================================================================================
def posterior_predictive_check(fit, df, participants, true_y, true_z, nparts, conditions_data=None):
    """
    Posterior Predictive Check using a fixed baseline model. Handles both in-sample and out-of-sample conditions.
    """

    # Handle in-sample check if no conditions provided
    if conditions_data is None or len(conditions_data) == 0:
        print("Running in-sample posterior predictive check using baseline model...\n")

        # Predict using the original data (true_y and true_z) and posterior means
        predicted_y, predicted_z = generate_predicted_data(
            fit, df, participants, true_y, n_trials=len(true_z)
        )

        conditions_data = [{
            'condition_y': true_y,
            'condition_z': true_z,
            'condition_participants': participants,
            'condition_name': 'In-Sample (Baseline)'
        }]

    # Store predictions and observations per condition
    condition_results = []
    for i, condition in enumerate(conditions_data):
        test_y = condition['condition_y']
        test_z = condition['condition_z']
        test_participants = condition['condition_participants']
        condition_name = condition['condition_name']

        print(f"Processing condition '{condition_name}' with {len(test_y)} trials")

        predicted_y_test, predicted_z_test = generate_predicted_data(
            fit, df, test_participants, test_y, n_trials=len(test_z)
        )

        condition_results.append({
            'name': condition_name,
            'observed_y': test_y,
            'observed_z': test_z,
            'predicted_y': predicted_y_test,
            'predicted_z': predicted_z_test
        })

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    for i, result in enumerate(condition_results):
        cname = result['name']

        # y plot
        sns.histplot(result['observed_y'], label=f'Observed {cname}', stat='density',
                     color='orange', alpha=0.6, ax=axes[0])
        sns.kdeplot(result['predicted_y'], label=f'Predicted {cname}',
                    color='blue', linewidth=2, ax=axes[0])

        # z plot
        sns.histplot(result['observed_z'], label=f'Observed {cname}', stat='density',
                     color='orange', alpha=0.6, ax=axes[1])
        sns.kdeplot(result['predicted_z'], label=f'Predicted {cname}',
                    color='blue', linewidth=2, ax=axes[1])

    # Format y plot
    axes[0].set_title("Posterior Predictive Check: Choice/RT (y)", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Signed RT", fontsize=14)
    axes[0].set_ylabel("Density", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(labelsize=12)

    # Format z plot
    axes[1].set_title("Posterior Predictive Check: Latent Variable (z)", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Latent Variable (z)", fontsize=14)
    axes[1].set_ylabel("Density", fontsize=14)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()

    # Print summaries
    print(f"\n{'='*60}\nPOSTERIOR PREDICTIVE CHECK SUMMARY\n{'='*60}")
    for result in condition_results:
        print(f"\n{result['name']} - PPC Metrics:")

        # select only the trials with choice
        y_obs = result['observed_y']
        y_pred = result['predicted_y']

        # Separate by choice (sign of y)
        rt_obs_neg = np.abs(y_obs[y_obs < 0])
        rt_obs_pos = y_obs[y_obs > 0]
        rt_pred_neg = np.abs(y_pred[y_pred < 0])
        rt_pred_pos = y_pred[y_pred > 0]

        # Conditional means
        mean_rt_obs_neg = np.mean(rt_obs_neg) if len(rt_obs_neg) > 0 else np.nan
        mean_rt_obs_pos = np.mean(rt_obs_pos) if len(rt_obs_pos) > 0 else np.nan
        mean_rt_pred_neg = np.mean(rt_pred_neg) if len(rt_pred_neg) > 0 else np.nan
        mean_rt_pred_pos = np.mean(rt_pred_pos) if len(rt_pred_pos) > 0 else np.nan

        # Print results
        print(f"  Conditional RTs:")
        print(f"  Neg choices    mean RT obs = {mean_rt_obs_neg:.2f}, pred = {mean_rt_pred_neg:.2f}")
        print(f"                 var RT obs = {np.var(rt_obs_neg):.2f}, var RT pred = {np.var(rt_pred_neg):.2f}")
        print(f"  Pos choices    mean RT obs = {mean_rt_obs_pos:.2f}, pred = {mean_rt_pred_pos:.2f}")
        print(f"                 var RT obs = {np.var(rt_obs_pos):.2f}, var RT pred = {np.var(rt_pred_pos):.2f}")
        print(f"  z (Latent):    mean obs = {np.mean(result['observed_z']):.2f}, mean pred = {np.mean(result['predicted_z']):.2f}")
        print(f"                 var obs = {np.var(result['observed_z']):.2f}, var pred = {np.var(result['predicted_z']):.2f}")

    return fig