# Date            Programmers                         Descriptions of Change
# =====================================================================================
# 13-Feb-2025     Bruny Krijgsman                     Added P300 peak activation in simulated data
# 16-Feb-2025     Bruny Krijgsman                     Added mu_z and sigma_z to the model

# =====================================================================================
# Directed single-trial model with 8 parameters (α, τ, β, μ_z, σ, λ, b, η)
# r_i, x_i ~ DDM(α, τ, δ_i, β, η)
# z_i ~ N(μ_z, σ²)
# δ_i = λz_i + b + N(0, η)

# =====================================================================================
# Modules
import numpy as np
import pyjags
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from scipy import stats
import pyhddmjagsutils as phju
from bk_directed_ddm_utils import simul_directed_ddm
import sys
# =====================================================================================
# Simulations

# Get current filename without extension and use it for data file
current_file = os.path.splitext(os.path.basename(__file__))[0]
data_filename = f'data/{current_file}_simdata.mat'

# Simulation parameters 
if not os.path.exists(data_filename):
    # Number of simulated participants
    nparts = 100
    # Number of trials for one participant
    ntrials = 100
    # Number of total trials in each simulation
    ntotal = ntrials * nparts
    # Set random seed
    np.random.seed(2025)

    alpha = np.random.uniform(.8, 1.4, size=nparts)  # Uniform from .8 to 1.4 evidence units
    ndt = np.random.uniform(.15, .6, size=nparts)  # Uniform from .15 to .6 seconds
    beta = np.random.uniform(.3, .7, size=nparts)  # Uniform from .3 to .7 * alpha
    mu_z = np.random.uniform(-2, 2, size=nparts)  # Mean of latent variable
    sigma_z = np.random.uniform(0.5, 2, size=nparts)  # SD of latent variable
    lambda_param = np.random.uniform(0.5, 2, size=nparts)  # Scaling parameter
    b = np.random.uniform(-2, 2, size=nparts)  # Baseline drift
    eta = np.random.uniform(0, 1, size=nparts)  # Trial-to-trial variability
    
    rt = np.zeros(ntotal)
    acc = np.zeros(ntotal)
    y = np.zeros(ntotal)
    participant = np.zeros(ntotal)  # Participant index
    indextrack = np.arange(ntrials)
    for p in range(nparts):
        tempout, _ = simul_directed_ddm(ntrials=ntrials, alpha=alpha[p], tau=ndt[p], beta=beta[p],
                                   mu_z=mu_z[p], sigma_z=sigma_z[p],
                                   lambda_param=lambda_param[p], b=b[p],
                                   eta=eta[p])
        accuracy = np.sign(tempout)
        response_time = np.abs(tempout)

        # Update z_i (P300 simulation)
        z_i = np.random.normal(mu_z[p], sigma_z[p], size=ntrials)
        delta_i = lambda_param[p] * z_i + b[p] + np.random.normal(0, eta[p], size=ntrials)
        
        y[indextrack] = accuracy * response_time
        rt[indextrack] = response_time

        if np.sum(np.isnan(rt)) > 0:
            print(f"Participant {p} has {np.sum(np.isnan(rt))} NaN RTs")

        acc[indextrack] = (accuracy + 1) / 2
        participant[indextrack] = p + 1
        indextrack += ntrials

    genparam = dict()
    genparam['ndt'] = ndt
    genparam['beta'] = beta
    genparam['alpha'] = alpha
    genparam['mu_z'] = mu_z
    genparam['sigma_z'] = sigma_z
    genparam['lambda'] = lambda_param
    genparam['b'] = b
    genparam['eta'] = eta
    genparam['rt'] = rt
    genparam['acc'] = acc
    genparam['y'] = y
    genparam['participant'] = participant
    genparam['nparts'] = nparts
    genparam['ntrials'] = ntrials
    genparam['N'] = ntotal
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save with dynamic filename
    sio.savemat(data_filename, genparam)
else:
    genparam = sio.loadmat(data_filename)

# sys.exit()
# =====================================================================================
# JAGS code

# Set random seed
np.random.seed(2025)

tojags = '''
model {
    ##########
    # Parameter priors
    ##########
    for (p in 1:nparts) {
        # Boundary parameter (speed-accuracy tradeoff) per participant
        alpha[p] ~ dnorm(1, pow(.5,-2))T(0, 3)

        # Non-decision time per participant
        ndt[p] ~ dnorm(.5, pow(.25,-2))T(0, 1)

        # Start point bias towards choice A per participant
        beta[p] ~ dnorm(.5, pow(.25,-2))T(0, 1)

        # Mean of latent variable per participant
        mu_z[p] ~ dnorm(0, pow(2, -2))
        
        # SD of latent variable per participant
        sigma_z[p] ~ dnorm(1, pow(.5,-2))T(0, 3)
        
        # Scaling parameter for the drift rate per trial per participant
        lambda[p] ~ dnorm(1, pow(.5,-2))
        
        # Baseline drift per participant
        b[p] ~ dnorm(0, pow(2, -2))

        # Trial-to-trial variability per participant
        eta[p] ~ dnorm(0.5, pow(.25,-2))T(0, 2)
    }

    ##########
    # Model likelihood
    ##########
    for (i in 1:N) {
        # Latent variable (P300)
        z[i] ~ dnorm(mu_z[participant[i]], pow(sigma_z[participant[i]], -2))
        
        # Drift rate
        delta[i] <- lambda[participant[i]] * z[i] + b[participant[i]] + eta[participant[i]]
        
        # DDM process (acc*RT)
        y[i] ~ dwiener(alpha[participant[i]], ndt[participant[i]], 
                       beta[participant[i]], delta[i])
    }
}
'''

# =====================================================================================
# pyjags code

# Make sure $LD_LIBRARY_PATH sees /usr/local/lib
# Make sure that the correct JAGS/modules-4/ folder contains wiener.so and wiener.la
pyjags.modules.load_module('wiener')
pyjags.modules.load_module('dic')
pyjags.modules.list_modules()

nchains = 6
burnin = 2000  # Note that scientific notation breaks pyjags
nsamps = 10000

modelfile = f'jagscode/{current_file}.jags'
os.makedirs(os.path.dirname(modelfile), exist_ok=True)
f = open(modelfile, 'w')
f.write(tojags)
f.close()

# Track these variables
trackvars = ['alpha', 'ndt', 'beta', 'mu_z', 'sigma_z', 
             'lambda', 'b', 'eta']

N = np.squeeze(genparam['N'])

print(genparam['rt'])

# Fit model to data
y = np.squeeze(genparam['y'])
rt = np.squeeze(genparam['rt'])
participant = np.squeeze(genparam['participant'])
nparts = np.squeeze(genparam['nparts'])
ntrials = np.squeeze(genparam['ntrials'])

print(rt)

# Check ranges for response times and signed response times
print("\nRT range:")
print(f"RT length: {len(rt)}")
print(f"Number of NaN RTs: {np.sum(np.isnan(rt))}")
print(f"Min RT: {np.min(rt)}")
print(f"Max RT: {np.max(rt)}")
print("\nSigned RT range (y):")
print(f"Number of NaN y: {np.sum(np.isnan(y))}")
print(f"Min y: {np.min(y)}")
print(f"Max y: {np.max(y)}")

minrt = np.zeros(nparts)
for p in range(0, nparts):
    minrt[p] = np.min(rt[(participant == (p + 1))])

initials = []
for c in range(0, nchains):
    chaininit = {
        'alpha': np.random.uniform(.5, 2., size=nparts),
        'ndt': np.random.uniform(.1, .5, size=nparts),
        'beta': np.random.uniform(.2, .8, size=nparts),
        'mu_z': np.random.uniform(-2, 2, size=nparts),
        'sigma_z': np.random.uniform(0.5, 2, size=nparts),
        'lambda': np.random.uniform(0.5, 2, size=nparts),
        'b': np.random.uniform(-2, 2, size=nparts),
        'eta': np.random.uniform(0, 1, size=nparts)
    }
    for p in range(0, nparts):
        chaininit['ndt'][p] = np.random.uniform(0., minrt[p] / 2)
    initials.append(chaininit)
print('Fitting ''single-trial directed'' model ...')
threaded = pyjags.Model(file=modelfile, init=initials,
                        data=dict(y=y, N=N, nparts=nparts,
                                  participant=participant),
                        chains=nchains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(nsamps, vars=trackvars, thin=10)
savestring = (f'modelfits/{current_file}.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

sys.exit()

# =====================================================================================
# Diagnostics
samples = sio.loadmat(savestring)
diags = phju.diagnostic(samples)

# Posterior distributions
plt.figure()
phju.jellyfish(samples['alpha'])
plt.title('Posterior distributions of boundary parameter')
plt.savefig('figures/alpha_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ndt'])
plt.title('Posterior distributions of the non-decision time parameter')
plt.savefig('figures/ndt_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['beta'])
plt.title('Posterior distributions of the start point parameter')
plt.savefig('figures/beta_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['mu_z'])
plt.title('Posterior distributions of the mean of the latent variable')
plt.savefig('figures/mu_z_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['sigma_z'])
plt.title('Posterior distributions of the standard deviation of the latent variable')
plt.savefig('figures/sigma_z_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['lambda'])
plt.title('Posterior distributions of the scaling parameter')
plt.savefig('figures/lambda_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['b'])
plt.title('Posterior distributions of the baseline drift')
plt.savefig('figures/b_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['eta'])
plt.title('Posterior distributions of the trial-to-trial variability')
plt.savefig('figures/eta_posteriors_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['P300noise'])
plt.title('Posterior distributions of the noise in the observed P300 slope')
plt.savefig('figures/P300noise_posteriors_simpleCPP.png', format='png', bbox_inches="tight")


# =====================================================================================
# Recovery
plt.figure()
phju.recovery(samples['alpha'], genparam['alpha'])
plt.title('Recovery of boundary parameter')
plt.savefig('figures/alpha_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['ndt'], genparam['ndt'])
plt.title('Recovery of the non-decision time parameter')
plt.savefig('figures/ndt_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['beta'], genparam['beta'])
plt.title('Recovery of the start point parameter')
plt.savefig('figures/beta_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['mu_z'], genparam['mu_z'])
plt.title('Recovery of the mean of the latent variable')
plt.savefig('figures/mu_z_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['sigma_z'], genparam['sigma_z'])
plt.title('Recovery of the standard deviation of the latent variable')
plt.savefig('figures/sigma_z_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['lambda'], genparam['lambda'])
plt.title('Recovery of the scaling parameter')
plt.savefig('figures/lambda_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['b'], genparam['b'])
plt.title('Recovery of the baseline drift')
plt.savefig('figures/b_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['eta'], genparam['eta'])
plt.title('Recovery of the trial-to-trial variability')
plt.savefig('figures/eta_recovery_simpleCPP.png', format='png', bbox_inches="tight")

plt.figure()
phju.recovery(samples['P300noise'], genparam['P300noise'])
plt.title('Recovery of the noise in the observed P300 slope')
plt.savefig('figures/P300noise_recovery_simpleCPP.png', format='png', bbox_inches="tight")

# =====================================================================================
# Recovery plots nicely formatting for tutorial
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True) #sudo apt install texlive-latex-extra cm-super dvipng


def recoverysub(possamps, truevals, ax):  # Parameter recovery subplots
    """Plots true parameters versus 99% and 95% credible intervals of recovered
    parameters. Also plotted are the median (circles) and mean (stars) of the posterior
    distributions.

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is the
    number of chains, the second to last dimension is the number of samples in
    each chain, all other dimensions must match the dimensions of truevals

    truevals : ndarray of true parameter values
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))
    truevals = np.reshape(truevals, (nvars))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    for v in range(0, nvars):
        # Compute percentiles
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Plot credible intervals
            credint = np.ones(100) * truevals[v]
            y = np.linspace(bounds[b], bounds[-1 - b], 100)
            lines = ax.plot(credint, y, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark median
                mmedian = ax.plot(truevals[v], np.median(alldata[v, :]), 'o', markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = ax.plot(truevals[v], np.mean(alldata[v, :]), '*', markersize=10, color=teal)
    # Plot line y = x
    tempx = np.linspace(np.min(truevals), np.max(
        truevals), num=100)
    recoverline = ax.plot(tempx, tempx, linewidth=3, color=orange)



fontsize = 12

fig = plt.figure(figsize=(9,10),dpi=300)
gs = gridspec.GridSpec(3, 2)

ax1 = plt.subplot(gs[0, 0:2])
recoverysub(samples['mu_z'], genparam['mu_z'],ax1)
ax1.set_xlabel('Simulated $\\mu_{z}$ ($\\mu V$)', fontsize=fontsize)
ax1.set_ylabel('Posterior of $\\mu_{z}$ ($\\mu V$)', fontsize=fontsize)

ax2 = plt.subplot(gs[1, 0])
recoverysub(samples['sigma_z'], genparam['sigma_z'],ax2)
ax2.set_xlabel('Simulated $\\sigma_{z}$ ($\\mu V$)', fontsize=fontsize)
ax2.set_ylabel('Posterior of $\\sigma_{z}$ ($\\mu V$)', fontsize=fontsize)

ax3 = plt.subplot(gs[1, 1])
recoverysub(samples['lambda'], genparam['lambda'],ax3)
ax3.set_xlabel('Simulated $\\lambda$ ($\\mu V$)', fontsize=fontsize)
ax3.set_ylabel('Posterior of $\\lambda$ ($\\mu V$)', fontsize=fontsize)

ax4 = plt.subplot(gs[2, 0])
recoverysub(samples['b'], genparam['b'],ax4)
ax4.set_xlabel('Simulated $b$ ($\\mu V$)', fontsize=fontsize)
ax4.set_ylabel('Posterior of $b$ ($\\mu V$)', fontsize=fontsize)

ax5 = plt.subplot(gs[2, 1])
recoverysub(samples['eta'], genparam['eta'],ax5)
ax5.set_xlabel('Simulated $\\eta$', fontsize=fontsize)
ax5.set_ylabel('Posterior of $\\eta$', fontsize=fontsize)

plt.savefig('figures/All_recovery_simpleCPP.png', dpi=300, format='png', bbox_inches="tight")