// ------------------------------------------------------------------------------
// Extended DDM model with drift rate as a function of latent variable z.
// The model is defined in the following parameters:
// - alpha: The boundary separation.
// - tau: The non-decision time.
// - beta: The bias/starting point parameter.
// - mu_z: The mean of the latent variable.
// - sigma_z: The standard deviation of the latent variable.
// - lambda: The lambda/scaling parameter.
// - b: The intercept parameter.
// - eta: The eta parameter.

// ------------------------------------------------------------------------------
// Changes:
// - Added a constraint on tau_raw to ensure it is between 0.5 and 1 so tau is positive.
// - Added a constraint on eta to ensure it is above 0.1 (positive).
// - Added a constraint on sigma_z to ensure it is above 0.1 (positive).

data {
    int<lower=1> N;                     // Total number of trials
    int<lower=1> nparts;                // Number of participants
    array[N] real y;                    // Signed RTs (acc * RT)
    array[N] int participant;           // Participant index
    array[nparts] real minRT;           // Minimum RT per participant
}

parameters {
  // DDM parameters
  vector<lower=0.1, upper=5>[nparts] alpha;  // Threshold separation (per participant)
  vector<lower=0, upper=0.99>[nparts] beta;  // Starting point bias (per participant)
  vector<lower=0.5, upper=1>[nparts] tau_raw;  // Scaled non-decision time (per participant)

  // Latent variable (subject-level)
  real mu_z;
  real<lower=0.1> sigma_z;

  // Trial-level variables
  vector[N] z;                      // Latent variable per trial
  real lambda;                      // Effect of z on drift rate
  real b;                           // Drift rate baseline
  vector<lower=0.1>[nparts] eta;    // Within-trial noise
}

// ------------------------------------------------------------------------------
transformed parameters {
    array[N] real delta;
    vector[nparts] tau;
    
    // Calculate delta 
    for (i in 1:N) {
        delta[i] = lambda * z[i] + b;  
    }

    // Scale tau by minRT
    for (i in 1:nparts) {
        tau[i] = tau_raw[i] * minRT[i] * 0.98;
    }
}

model {
  // Priors
  alpha ~ normal(1, 1);
  beta ~ beta(2, 2);
  tau_raw ~ beta(2, 2);    
  mu_z ~ normal(0, 1);
  sigma_z ~ normal(0, 1);
  lambda ~ normal(0, 1);
  b ~ normal(0, 1);

  // Within-trial noise
  for(i in 1:nparts){
    eta[i]~ normal(0,.2);
  }

  // Latent variable
  z ~ normal(mu_z, sigma_z);

  // Model for delta
  for (i in 1:N) {
      delta[i] ~ normal(lambda * z[i] + b, eta[participant[i]]);
  }

  // Log-likelihood for each trial
  for (i in 1:N) {
      if (abs(y[i]) > tau[participant[i]]) {
          if (y[i] > 0) {
              target += wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], beta[participant[i]], delta[i]);
          } else {
              target += wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], 1 - beta[participant[i]], -delta[i]);
          }
      } else {
          // Adjust RT slightly to avoid issues
          target += wiener_lpdf(tau[participant[i]] | alpha[participant[i]], tau[participant[i]], beta[participant[i]], delta[i]);
      }
  }
}   

generated quantities {
    vector[N] log_lik;
    for (i in 1:N) {
        // Log density for DDM process
        if (y[i] > 0) {
            log_lik[i] = wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], beta[participant[i]], delta[i]);
        } else {
            log_lik[i] = wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], 1 - beta[participant[i]], -delta[i]);
        }
    }
}

