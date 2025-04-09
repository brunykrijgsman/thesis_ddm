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
data {
    int<lower=1> N;                     // Total number of trials
    int<lower=1> nparts;                // Number of participants
    array[N] real y;                    // Signed RTs (acc * RT)
    array[N] int participant;           // Participant index
    array[nparts] real minRT;           // Minimum RT per participant
}

parameters {
  // Subject-level DDM parameters
  vector<lower=0.001, upper=3>[nparts] alpha;   // Threshold separation (per participant)
  vector<lower=0.001, upper=0.99>[nparts] beta; // Starting point bias (per participant)
  vector<lower=0.5, upper=1>[nparts] tau_raw;   // Raw non-decision time truncated (per participant)
  vector<lower=0.001, upper=5>[nparts] eta;     // Within-trial noise ensure eta > 0 (per participant)

  // Group-level parameters
  real mu_z;                         // Latent variable mean
  real<lower=0.1> sigma_z;           // Latent variable SD ensure sigma_z > 0
  real<lower=0> lambda;              // Effect of z on drift rate
  real<lower=0> b;                   // Drift rate baseline

  // Trial-level variables
  vector[N] z;                      // Latent variable per trial
}

// ------------------------------------------------------------------------------
transformed parameters {
    // Delta estimates from lambda, z, b, and eta
    array[N] real delta;
    for (i in 1:N) {
        delta[i] = lambda * z[i] + b;
        }
    
    // Non-decision time, scaled to stay below RT (with 2% margin)
    vector<lower=0>[nparts] tau;
    for (i in 1:nparts) {
        tau[i] = tau_raw[i] * minRT[i] * 0.98;
        }
}

// ------------------------------------------------------------------------------
model {
  // Priors
  alpha ~ normal(1,.5); 
  beta ~ normal(.5,.25) T[0,1]; 
  tau_raw ~ normal(.3, .1); 
  // tau ~ normal(0.5,0.25) T[0,]; 
  // tau_raw ~ normal(0,.2)T[0,];    
  mu_z ~ normal(0,1); 
  sigma_z ~ normal(0,1) T[0, ]; 
  lambda ~ normal(0,.2) T[0,];
  b ~ normal(0,.2) T[0,];

  // Within-trial noise
  for(i in 1:nparts){
    eta[i]~ normal(0,.2)T[0,];
    }

  // Latent variable
  z ~ normal(mu_z, sigma_z);

  // Likelihood for delta
  for (i in 1:N) {
    delta[i] ~ normal(lambda * z[i] + b, eta[participant[i]]);
  }

  // Log-likelihood for each trial
  for (i in 1:N) {
    if (y[i] > 0) {
        target += wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], beta[participant[i]], delta[i], eta[participant[i]]);
    } else {
        target += wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], 1 - beta[participant[i]], -delta[i], eta[participant[i]]);
          }
  }
}   

// ------------------------------------------------------------------------------
generated quantities {
    vector[N] log_lik;
    for (i in 1:N) {
        // Log density for DDM process
        if (y[i] > 0) {
            log_lik[i] = wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], beta[participant[i]], delta[i], eta[participant[i]]);
        } else {
            log_lik[i] = wiener_lpdf(abs(y[i]) | alpha[participant[i]], tau[participant[i]], 1 - beta[participant[i]], -delta[i], eta[participant[i]]);
        }
    }
}