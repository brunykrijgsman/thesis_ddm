// Model for directed DDM defined in Stan.
// This model is used to fit the parameters of the directed DDM to the data.
// The model is defined in the following functions:
// - ratcliff_lpdf: The log-likelihood function for the Wiener diffusion model.

// ------------------------------------------------------------------------------
// The model is defined in the following parameters:
// - alpha: The boundary separation.
// - ndt: The non-decision time.
// - beta: The bias/starting point parameter.
// - mu_z: The mean of the latent variable.
// - sigma_z: The standard deviation of the latent variable.
// - lambda: The lambda/scaling parameter.
// - b: The bias parameter.
// - eta: The eta parameter.
// - z: The latent variable.

// ------------------------------------------------------------------------------
functions { 
   real ratcliff_lpdf(real Y, real boundary, real ndt, real bias, real drift, real sddrift) { 
       real X;
       X = (abs(Y) - ndt); // Remove non-decision time
       if (Y >= 0) {
           return wiener_lpdf(abs(Y) | boundary, ndt, bias, drift) +
                  (((boundary * (1 - bias) * sddrift)^2 + 2 * drift * boundary * (1 - bias) - (drift^2) * X) /
                  (2 * (sddrift^2) * X + 2)) - log(sqrt((sddrift^2) * X + 1)) - 
                  drift * boundary * (1 - bias) + (drift^2) * X * 0.5;
       } else {
           return wiener_lpdf(abs(Y) | boundary, ndt, 1 - bias, -drift) +
                  (((boundary * bias * sddrift)^2 - 2 * drift * boundary * bias - (drift^2) * X) /
                  (2 * (sddrift^2) * X + 2)) - log(sqrt((sddrift^2) * X + 1)) + 
                  drift * boundary * bias + (drift^2) * X * 0.5;
       }
   }
}

// ------------------------------------------------------------------------------
data {
    int<lower=1> N;             // Total number of trials
    int<lower=1> nparts;        // Number of participants
    array[N] real y;            // Signed RTs (acc * RT)
    array[N] int participant;   // Participant index

    // Priors
    real mu_z_prior;
    real<lower=0> sigma_z_prior;
    real lambda_prior;
    real b_prior;
    real eta_prior;
}

// ------------------------------------------------------------------------------
parameters {
    // Parameters for each participant
    vector[nparts] alpha;
    vector[nparts] ndt;
    vector[nparts] beta;
    vector[nparts] mu_z;
    vector<lower=0>[nparts] sigma_z;
    vector[nparts] lambda;
    vector[nparts] b;
    vector<lower=0>[nparts] eta;
    array[N] real z;
}

// ------------------------------------------------------------------------------
transformed parameters {
    array[N] real delta;
    for (i in 1:N) {
        delta[i] = lambda[participant[i]] * z[i] + b[participant[i]];
    }
}

// ------------------------------------------------------------------------------
model {
    // Priors
    alpha ~ normal(1, 0.2);
    beta ~ beta(2, 2);
    ndt ~ beta(2, 5);
    mu_z ~ normal(mu_z_prior, sigma_z_prior);
    sigma_z ~ normal(1, 0.5) T[0,]; 
    lambda ~ normal(lambda_prior, 0.5) T[0,];
    b ~ normal(b_prior, 1);
    eta ~ normal(0.5, 0.25) T[0,];

    // Latent variable model
    z ~ normal(mu_z[participant], sigma_z[participant]);

    // DDM likelihood
    for (i in 1:N) {
    target += ratcliff_lpdf(y[i] | alpha[participant[i]], beta[participant[i]], ndt[participant[i]], delta[i], eta[participant[i]]);
    }
}

// ------------------------------------------------------------------------------
generated quantities {
    vector[N] log_lik;
    for (i in 1:N) {
        log_lik[i] = ratcliff_lpdf(y[i] | alpha[participant[i]], beta[participant[i]], ndt[participant[i]], delta[i], eta[participant[i]]);
    }
}