// Model for directed DDM defined in Stan.
// This model is used to fit the parameters of the directed DDM to the data.
// The model is defined in the following functions:
// - ratcliff_lpdf: The log-likelihood function for the Wiener diffusion model.
// - simulate_single_trial: A function that simulates a single trial of the directed DDM.

// ------------------------------------------------------------------------------
// The model is defined in the following parameters:
// - alpha: The boundary separation.
// - ndt: The non-decision time.
// - beta: The bias/starting point parameter.
// - mu_z: The mean of the latent variable.
// - sigma_z: The standard deviation of the latent variable.
// - lambda: The lambda/scaling parameter.
// - b: The intercept parameter.
// - eta: The eta parameter.
// - z: The latent variable.

// ------------------------------------------------------------------------------
functions { 
    real ratcliff_lpdf(real Y, real boundary, real ndt, real bias, real drift, real sddrift) {
        real X;
        X = (abs(Y) - ndt); // Remove non-decision time
        if (abs(Y) > ndt) { 
            if (Y >= 0) {
            return wiener_lpdf(abs(Y) | boundary, ndt, bias, drift) +
                    (((boundary * (1 - bias) * sddrift)^2 + 2 * drift * boundary * (1 - bias) - (drift^2) * X) /
                    (2 * (sddrift^2) * X + 2)) - log(sqrt((sddrift^2) * X + 1)) - 
                    drift * boundary * (1 - bias) + (drift^2) * X * 0.5;
            } 
            else {
            return wiener_lpdf(abs(Y) | boundary, ndt, 1 - bias, -drift) +
                    (((boundary * bias * sddrift)^2 - 2 * drift * boundary * bias - (drift^2) * X) /
                    (2 * (sddrift^2) * X + 2)) - log(sqrt((sddrift^2) * X + 1)) + 
                    drift * boundary * bias + (drift^2) * X * 0.5;
            }
        }
        else {
            return wiener_lpdf(ndt | boundary, ndt, bias, drift, sddrift);
        }
   }
}

// ------------------------------------------------------------------------------
data {
    int<lower=1> N_trials;            // Total number of trials
    int<lower=1> N_parts;             // Number of participants
    array[N_trials] real y;           // Signed RTs (acc * RT)
    array[N_trials] int participant;  // Participant index

    // Priors
    real mu_z_prior;
    real<lower=0> sigma_z_prior;
    real lambda_prior;
    real b_prior;
    real eta_prior;
}

// ------------------------------------------------------------------------------
parameters {
    vector<lower=0.01, upper=5>[N_parts] alpha;
    vector<lower=0.001, upper=0.3>[N_parts] ndt;
    vector<lower=0.1, upper=0.9>[N_parts] beta;
    vector<lower=-5, upper=5>[N_parts] mu_z;
    vector<lower=0>[N_parts] sigma_z;
    vector[N_parts] lambda;
    vector[N_parts] b;
    vector<lower=0>[N_parts] eta;
    array[N_trials] real z;
}

// ------------------------------------------------------------------------------
transformed parameters {
    array[N_trials] real delta;
    
    for (i in 1:N_trials) {
        delta[i] = lambda[participant[i]] * z[i] + b[participant[i]];
    }
}

// ------------------------------------------------------------------------------
model {
    // Priors
    alpha ~ normal(1, 1) T[0.33,];
    ndt ~ normal(0.3, 0.1) T[0.001,0.3];
    beta ~ beta(2, 2);
    mu_z ~ normal(mu_z_prior, sigma_z_prior);
    sigma_z ~ normal(1, 0.5);
    lambda ~ normal(lambda_prior, 0.5);
    b ~ normal(b_prior, 2);
    eta ~ normal(eta_prior, 0.25);
    
    // Latent variable model
    z ~ normal(mu_z[participant], sigma_z[participant]);

    // Wiener likelihood
    for (i in 1:N_trials) {
        // Log density for DDM process
        y[i] ~ ratcliff(alpha[participant[i]], ndt[participant[i]], beta[participant[i]], delta[i], eta[participant[i]]);
    }
}

// ------------------------------------------------------------------------------
generated quantities {
    // Wiener likelihood
    vector[N_trials] log_lik;
    for (i in 1:N_trials) {
        // Log density for DDM process
         log_lik[i] = ratcliff_lpdf(y[i] | alpha[participant[i]], ndt[participant[i]], beta[participant[i]], delta[i], eta[participant[i]]);
   }
}
