# Date            Programmers                         Descriptions of Change
# =====================================================================================
# 13-Feb-2025     Bruny Krijgsman                     Added relevant parameters based on 
#                                                     the specified model
# 12-Mar-2025     Bruny Krijgsman                     Implemented simulation function for
#                                                     plotting
# =====================================================================================
# Simulates random walk, response time distribution, and P300 amplitude distribution 
# to plot for paper.
# =====================================================================================
# Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from directed_ddm_utils import simul_directed_ddm

# Simulation Parameters
ntrials = 1000
alpha = 1  # Boundary parameter (threshold)
ndt = .4  # Non-decision time in seconds
beta = .5  # Relative start point, proportion of boundary
eta = .3  # Additional trial-to-trial variability in drift rate
varsigma = 1  # Accumulation variance (diffusion coefficient)
mu_z = 0  # Mean of latent factor (normal distribution)
sigma_z = 1  # Variance of latent factor (normal distribution)
lambda_param = 0.7  # Scaling factor for P300 influence
b = 0.5  # Intercept/Baseline drift adjustment
nsteps = 300  # nsteps*step_length is seconds after ndt
step_length = .01

# Set random seed for reproducibility
np.random.seed(21)

# Use the simul_directed_ddm function to generate data
signed_rts, random_walks = simul_directed_ddm(
    ntrials=ntrials,
    alpha=alpha,
    tau=ndt,
    beta=beta,
    eta=eta,
    varsigma=varsigma,
    mu_z=mu_z,
    sigma_z=sigma_z,
    lambda_param=lambda_param,
    b=b,
    nsteps=nsteps,
    step_length=step_length
)

# Extract RTs and choices from signed RTs
rts = np.abs(signed_rts)
choice = np.sign(signed_rts)

# Generate z values again with same seed for plotting
z = np.random.normal(mu_z, sigma_z, ntrials)

# Plotting parameters
plot_time = np.linspace(0, nsteps * step_length, nsteps)

# =====================================================================================
# Plotting
# Plot random walks for first 100 trials  
plt.figure(figsize=(10, 5))
for n in range(100):
    plt.plot(plot_time + ndt, random_walks[:, n], color='blue')
plt.xlim(ndt, 2)
plt.xlabel('Time (secs)')
plt.ylabel('Drift')
plt.title('Random Walks (first 100 trials)')
plt.show()

# Plot Response Time Distributions
plt.figure(figsize=(10, 5))
sns.kdeplot(signed_rts, shade=True, color='blue')
plt.xlabel('Correct and Incorrect Response Times (secs)')
plt.ylabel('Density')
plt.title('Response Time Distribution')
plt.show()

# Plot P300 Distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(z, shade=True, color='red')
plt.xlabel('P300 Amplitude (µV)')
plt.ylabel('Density')
plt.title('P300 Amplitude Distribution')
plt.show()

# Print Summary Statistics
print(f"Mean Response Time: {np.nanmean(rts):.3f} sec")
print(f"Missing Responses: {np.sum(np.isnan(rts))}")
print(f"Minimum response time: {np.nanmin(rts):.3f} sec")
print(f"Maximum response time: {np.nanmax(rts):.3f} sec")
print(f"Mean P300 Amplitude: {np.nanmean(z):.3f} µV")