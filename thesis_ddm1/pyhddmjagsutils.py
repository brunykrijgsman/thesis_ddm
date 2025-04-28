# pyhddmjagsutils.py - Definitions for simulation, model diagnostics, and parameter recovery
#
# Copyright (C) 2024 Michael D. Nunez, <m.d.nunez@uva.nl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 06/29/20(US)  Michael Nunez                             Original code
# 12/04/20(US)  Michael Nunez             Update explanation of summary output
# 01/14/21(US)  Michael Nunez            Add simuldiff2ndt() and flipstanout()
# 02/10/21(US)  Michael Nunez                 Add logwienerpdf
# 07/19/21(US)  Michael Nunez               Remove unnecessary imports
# 25-April-2022 Michael Nunez                   Remove use of numba
# 13-March-2023 Michael Nunez                 Add recovery_scatter()
# 31-July-2023  Michael Nunez       ERC interview flag for recovery_scatter()
# 06-Sept-23    Michael Nunez  In recovery_scatter() print pearson correlation
# 15-Nov-23     Michael Nunez            grantB1 x&y limit changes
# 21-Feb-24     Michael Nunez        Removing unused and unfinished functions
# 06-March-24   Michael Nunez  Additional true_param input in plot_posterior2d


# Modules
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
import warnings
import matplotlib.pyplot as plt

### Definitions ###

# Simulate diffusion models
def simulratcliff(N=100, Alpha=1, Tau=.4, Nu=1, Beta=.5, rangeTau=0, rangeBeta=0, Eta=.3, Varsigma=1):
    """
    SIMULRATCLIFF  Generates data according to a drift diffusion model with optional trial-to-trial variability


    Reference:
    Tuerlinckx, F., Maris, E.,
    Ratcliff, R., & De Boeck, P. (2001). A comparison of four methods for
    simulating the diffusion process. Behavior Research Methods,
    Instruments, & Computers, 33, 443-456.

    Parameters
    ----------
    N: a integer denoting the size of the output vector
    (defaults to 100 experimental trials)

    Alpha: the mean boundary separation across trials  in evidence units
    (defaults to 1 evidence unit)

    Tau: the mean non-decision time across trials in seconds
    (defaults to .4 seconds)

    Nu: the mean drift rate across trials in evidence units per second
    (defaults to 1 evidence units per second, restricted to -5 to 5 units)

    Beta: the initial bias in the evidence process for choice A as a proportion of boundary Alpha
    (defaults to .5 or 50% of total evidence units given by Alpha)

    rangeTau: Non-decision time across trials is generated from a uniform
    distribution of Tau - rangeTau/2 to  Tau + rangeTau/2 across trials
    (defaults to 0 seconds)

    rangeZeta: Bias across trials is generated from a uniform distribution
    of Zeta - rangeZeta/2 to Zeta + rangeZeta/2 across trials
    (defaults to 0 evidence units)

    Eta: Standard deviation of the drift rate across trials
    (defaults to 3 evidence units per second, restricted to less than 3 evidence units)

    Varsigma: The diffusion coefficient, the standard deviation of the
    evidence accumulation process within one trial. It is recommended that
    this parameter be kept fixed unless you have reason to explore this parameter
    (defaults to 1 evidence unit per second)

    Returns
    -------
    Numpy array with reaction times (in seconds) multiplied by the response vector
    such that negative reaction times encode response B and positive reaction times
    encode response A 
    
    
    Converted from simuldiff.m MATLAB script by Joachim Vandekerckhove
    See also http://ppw.kuleuven.be/okp/dmatoolbox.
    """

    if (Nu < -5) or (Nu > 5):
        Nu = np.sign(Nu) * 5

    if (Eta > 3):
        eta = 3

    if (Eta == 0):
        Eta = 1e-16

    # Initialize output vectors
    result = np.zeros(N)
    T = np.zeros(N)
    XX = np.zeros(N)

    # Called sigma in 2001 paper
    D = np.power(Varsigma, 2) / 2

    # Program specifications
    eps = 2.220446049250313e-16  # precision from 1.0 to next double-precision number
    delta = eps

    for n in range(0, N):
        r1 = np.random.normal()
        mu = Nu + r1 * Eta
        bb = Beta - rangeBeta / 2 + rangeBeta * np.random.uniform()
        zz = bb * Alpha
        finish = 0
        totaltime = 0
        startpos = 0
        Aupper = Alpha - zz
        Alower = -zz
        radius = np.min(np.array([np.abs(Aupper), np.abs(Alower)]))
        while (finish == 0):
            lambda_ = 0.25 * np.power(mu, 2) / D + 0.25 * D * np.power(np.pi, 2) / np.power(radius, 2)
            # eq. formula (13) in 2001 paper with D = sigma^2/2 and radius = Alpha/2
            F = D * np.pi / (radius * mu)
            F = np.power(F, 2) / (1 + np.power(F, 2))
            # formula p447 in 2001 paper
            prob = np.exp(radius * mu / D)
            prob = prob / (1 + prob)
            dir_ = 2 * (np.random.uniform() < prob) - 1
            l = -1
            s2 = 0
            while (s2 > l):
                s2 = np.random.uniform()
                s1 = np.random.uniform()
                tnew = 0
                told = 0
                uu = 0
                while (np.abs(tnew - told) > eps) or (uu == 0):
                    told = tnew
                    uu = uu + 1
                    tnew = told + (2 * uu + 1) * np.power(-1, uu) * np.power(s1, (F * np.power(2 * uu + 1, 2)))
                    # infinite sum in formula (16) in BRMIC,2001
                l = 1 + np.power(s1, (-F)) * tnew
            # rest of formula (16)
            t = np.abs(np.log(s1)) / lambda_
            # is the negative of t* in (14) in BRMIC,2001
            totaltime = totaltime + t
            dir_ = startpos + dir_ * radius
            ndt = Tau - rangeTau / 2 + rangeTau * np.random.uniform()
            if ((dir_ + delta) > Aupper):
                T[n] = ndt + totaltime
                XX[n] = 1
                finish = 1
            elif ((dir_ - delta) < Alower):
                T[n] = ndt + totaltime
                XX[n] = -1
                finish = 1
            else:
                startpos = dir_
                radius = np.min(np.abs([Aupper, Alower] - startpos))

    result = T * XX
    return result



def diagnostic(insamples):
    """
    Returns two versions of Rhat (measure of convergence, less is better with an approximate
    1.10 cutoff) and Neff, number of effective samples). Note that 'rhat' is more diagnostic than 'oldrhat' according to 
    Gelman et al. (2014).

    Reference for preferred Rhat calculation (split chains) and number of effective sample calculation: 
        Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2014). 
        Bayesian data analysis (Third Edition). CRC Press:
        Boca Raton, FL

    Reference for original Rhat calculation:
        Gelman, A., Carlin, J., Stern, H., & Rubin D., (2004).
        Bayesian Data Analysis (Second Edition). Chapman & Hall/CRC:
        Boca Raton, FL.


    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.

    Returns
    -------
    dict:
        rhat, oldrhat, neff, posterior mean, and posterior std for each variable. Prints maximum Rhat and minimum
        Neff across all variables
    """

    result = {}  # Initialize dictionary
    maxrhatsold = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float) * np.inf
    allkeys = {}  # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}

            possamps = insamples[key]

            # Number of chains
            nchains = possamps.shape[-1]

            # Number of samples per chain
            nsamps = possamps.shape[-2]

            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])

            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps / 2), nchains * 2,))
            newc = 0
            for c in range(nchains):
                possampsnew[..., newc] = np.take(np.take(possamps, np.arange(0, int(nsamps / 2)), axis=-2), c, axis=-1)
                possampsnew[..., newc + 1] = np.take(np.take(possamps, np.arange(int(nsamps / 2), nsamps), axis=-2), c,
                                                     axis=-1)
                newc += 2

            # Index of variables
            varindx = np.arange(nvars).reshape(possamps.shape[0:-2])

            # Reshape data
            alldata = np.reshape(possamps, (nvars, nsamps, nchains))

            # Mean of each chain for rhat
            chainmeans = np.mean(possamps, axis=-2)
            # Mean of each chain for rhatnew
            chainmeansnew = np.mean(possampsnew, axis=-2)
            # Global mean of each parameter for rhat
            globalmean = np.mean(chainmeans, axis=-1)
            globalmeannew = np.mean(chainmeansnew, axis=-1)
            result[key]['mean'] = globalmean
            result[key]['std'] = np.std(allsamps, axis=-1)
            globalmeanext = np.expand_dims(
                globalmean, axis=-1)  # Expand the last dimension
            globalmeanext = np.repeat(
                globalmeanext, nchains, axis=-1)  # For differencing
            globalmeanextnew = np.expand_dims(
                globalmeannew, axis=-1)  # Expand the last dimension
            globalmeanextnew = np.repeat(
                globalmeanextnew, nchains * 2, axis=-1)  # For differencing
            # Between-chain variance for rhat
            between = np.sum(np.square(chainmeans - globalmeanext),
                             axis=-1) * nsamps / (nchains - 1.)
            # Mean of the variances of each chain for rhat
            within = np.mean(np.var(possamps, axis=-2), axis=-1)
            # Total estimated variance for rhat
            totalestvar = (1. - (1. / nsamps)) * \
                          within + (1. / nsamps) * between
            # Rhat (original Gelman-Rubin statistic)
            temprhat = np.sqrt(totalestvar / within)
            maxrhatsold[keyindx] = np.nanmax(temprhat)  # Ignore NANs
            allkeys[keyindx] = key
            result[key]['oldrhat'] = temprhat
            # Between-chain variance for rhatnew
            betweennew = np.sum(np.square(chainmeansnew - globalmeanextnew),
                                axis=-1) * (nsamps / 2) / ((nchains * 2) - 1.)
            # Mean of the variances of each chain for rhatnew
            withinnew = np.mean(np.var(possampsnew, axis=-2), axis=-1)
            # Total estimated variance
            totalestvarnew = (1. - (1. / (nsamps / 2))) * \
                             withinnew + (1. / (nsamps / 2)) * betweennew
            # Rhatnew (Gelman-Rubin statistic from Gelman et al., 2013)
            temprhatnew = np.sqrt(totalestvarnew / withinnew)
            maxrhatsnew[keyindx] = np.nanmax(temprhatnew)  # Ignore NANs
            result[key]['rhat'] = temprhatnew
            # Number of effective samples from Gelman et al. (2013) 286-288
            neff = np.empty(possamps.shape[0:-2])
            for v in range(0, nvars):
                whereis = np.where(varindx == v)
                rho_hat = []
                rho_hat_even = 0
                rho_hat_odd = 0
                t = 2
                while ((t < nsamps - 2) & (float(rho_hat_even) + float(rho_hat_odd) >= 0)):
                    variogram_odd = np.mean(
                        np.mean(np.power(alldata[v, (t - 1):nsamps, :] - alldata[v, 0:(nsamps - t + 1), :], 2),
                                axis=0))  # above equation (11.7) in Gelman et al., 2013
                    rho_hat_odd = 1 - np.divide(variogram_odd,
                                                2 * totalestvar[whereis])  # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_odd)
                    variogram_even = np.mean(
                        np.mean(np.power(alldata[v, t:nsamps, :] - alldata[v, 0:(nsamps - t), :], 2),
                                axis=0))  # above equation (11.7) in Gelman et al., 2013
                    rho_hat_even = 1 - np.divide(variogram_even,
                                                 2 * totalestvar[whereis])  # Equation (11.7) in Gelman et al., 2013
                    rho_hat.append(rho_hat_even)
                    t += 2
                rho_hat = np.asarray(rho_hat)
                neff[whereis] = np.divide(nchains * nsamps,
                                          1 + 2 * np.sum(rho_hat))  # Equation (11.8) in Gelman et al., 2013
            result[key]['neff'] = np.round(neff)
            minneff[keyindx] = np.nanmin(np.round(neff))
            keyindx += 1

            # Geweke statistic?
    # print("Maximum old Rhat was %3.2f for variable %s" % (np.max(maxrhatsold),allkeys[np.argmax(maxrhatsold)]))
    maxrhatkey = allkeys[np.argmax(maxrhatsnew)]
    maxrhatindx = np.unravel_index(np.argmax(result[maxrhatkey]['rhat']), result[maxrhatkey]['rhat'].shape)
    print("Maximum Rhat was %3.2f for variable %s at index %s" % (np.max(maxrhatsnew), maxrhatkey, maxrhatindx))
    minneffkey = allkeys[np.argmin(minneff)]
    minneffindx = np.unravel_index(np.argmin(result[minneffkey]['neff']), result[minneffkey]['neff'].shape)
    print("Minimum number of effective samples was %d for variable %s at index %s" % (
    np.min(minneff), minneffkey, minneffindx))
    return result


def summary(insamples):
    """
    Returns parameter estimates for each posterior distribution (mean and median posteriors) as well as 95% and 99%
    credible intervals (.5th, 2.5th, 97.5th, 99.5th percentiles)

    Parameters
    ----------
    insamples: dic
        Sampled values of monitored variables as a dictionary where keys
        are variable names and values are numpy arrays with shape:
        (dim_1, dim_n, iterations, chains). dim_1, ..., dim_n describe the
        shape of variable in JAGS model.
    """

    result = {}  # Initialize dictionary
    maxrhats = np.zeros((len(insamples.keys())), dtype=float)
    maxrhatsnew = np.zeros((len(insamples.keys())), dtype=float)
    minneff = np.ones((len(insamples.keys())), dtype=float) * np.inf
    allkeys = {}  # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            result[key] = {}

            possamps = insamples[key]

            # Number of chains
            nchains = possamps.shape[-1]

            # Number of samples per chain
            nsamps = possamps.shape[-2]

            # Number of variables per key
            nvars = np.prod(possamps.shape[0:-2])

            # Reshape data
            allsamps = np.reshape(possamps, possamps.shape[:-2] + (nchains * nsamps,))

            # Reshape data to preduce R_hatnew
            possampsnew = np.empty(possamps.shape[:-2] + (int(nsamps / 2), nchains * 2,))
            newc = 0
            for c in range(nchains):
                possampsnew[..., newc] = np.take(np.take(possamps, np.arange(0, int(nsamps / 2)), axis=-2), c, axis=-1)
                possampsnew[..., newc + 1] = np.take(np.take(possamps, np.arange(int(nsamps / 2), nsamps), axis=-2), c,
                                                     axis=-1)
                newc += 2

            result[key]['mean'] = np.mean(allsamps, axis=-1)
            result[key]['std'] = np.std(allsamps, axis=-1)
            result[key]['median'] = np.quantile(allsamps, 0.5, axis=-1)
            result[key]['95lower'] = np.quantile(allsamps, 0.025, axis=-1)
            result[key]['95upper'] = np.quantile(allsamps, 0.975, axis=-1)
            result[key]['99lower'] = np.quantile(allsamps, 0.005, axis=-1)
            result[key]['99upper'] = np.quantile(allsamps, 0.995, axis=-1)
    return result


def flipstanout(insamples):
    result = {}  # Initialize dictionary
    allkeys = {}  # Initialize dictionary
    keyindx = 0
    for key in insamples.keys():
        if key[0] != '_':
            possamps = insamples[key]
            transamps = np.moveaxis(possamps, 0, -1)
            bettersamps = np.moveaxis(transamps, 0, -1)
            if len(bettersamps.shape) == 2:
                reshapedsamps = np.reshape(bettersamps, (1,) + bettersamps.shape[0:2])
                result[key] = reshapedsamps
            else:
                result[key] = bettersamps
    return result


def jellyfish(possamps):  # jellyfish plots
    """Plots posterior distributions of given posterior samples in a jellyfish
    plot. Jellyfish plots are posterior distributions (mirrored over their
    horizontal axes) with 99% and 95% credible intervals (currently plotted
    from the .5% and 99.5% & 2.5% and 97.5% percentiles respectively.
    Also plotted are the median and mean of the posterior distributions"

    Parameters
    ----------
    possamps : ndarray of posterior chains where the last dimension is
    the number of chains, the second to last dimension is the number of samples
    in each chain, all other dimensions describe the shape of the parameter
    """

    # Number of chains
    nchains = possamps.shape[-1]

    # Number of samples per chain
    nsamps = possamps.shape[-2]

    # Number of dimensions
    ndims = possamps.ndim - 2

    # Number of variables to plot
    nvars = np.prod(possamps.shape[0:-2])

    # Index of variables
    varindx = np.arange(nvars).reshape(possamps.shape[0:-2])

    # Reshape data
    alldata = np.reshape(possamps, (nvars, nchains, nsamps))
    alldata = np.reshape(alldata, (nvars, nchains * nsamps))

    # Plot properties
    LineWidths = np.array([2, 5])
    teal = np.array([0, .7, .7])
    blue = np.array([0, 0, 1])
    orange = np.array([1, .3, 0])
    Colors = [teal, blue]

    # Initialize ylabels list
    ylabels = ['']

    for v in range(0, nvars):
        # Create ylabel
        whereis = np.where(varindx == v)
        newlabel = ''
        for l in range(0, ndims):
            newlabel = newlabel + ('_%i' % whereis[l][0])

        ylabels.append(newlabel)

        # Compute posterior density curves
        kde = stats.gaussian_kde(alldata[v, :])
        bounds = stats.scoreatpercentile(alldata[v, :], (.5, 2.5, 97.5, 99.5))
        for b in range(0, 2):
            # Bound by .5th percentile and 99.5th percentile
            x = np.linspace(bounds[b], bounds[-1 - b], 100)
            p = kde(x)

            # Scale distributions down
            maxp = np.max(p)

            # Plot jellyfish
            upper = .25 * p / maxp + v + 1
            lower = -.25 * p / maxp + v + 1
            lines = plt.plot(x, upper, x, lower)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark mode
                wheremaxp = np.argmax(p)
                mmode = plt.plot(np.array([1., 1.]) * x[wheremaxp],
                                 np.array([lower[wheremaxp], upper[wheremaxp]]))
                plt.setp(mmode, linewidth=3, color=orange)
                # Mark median
                mmedian = plt.plot(np.median(alldata[v, :]), v + 1, 'ko')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(np.mean(alldata[v, :]), v + 1, '*')
                plt.setp(mmean, markersize=10, color=teal)

    # Display plot
    plt.setp(plt.gca(), yticklabels=ylabels, yticks=np.arange(0, nvars + 1))


def recovery(possamps, truevals):  # Parameter recovery plots
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
            lines = plt.plot(credint, y)
            plt.setp(lines, color=Colors[b], linewidth=LineWidths[b])
            if b == 1:
                # Mark median
                mmedian = plt.plot(truevals[v], np.median(alldata[v, :]), 'o')
                plt.setp(mmedian, markersize=10, color=[0., 0., 0.])
                # Mark mean
                mmean = plt.plot(truevals[v], np.mean(alldata[v, :]), '*')
                plt.setp(mmean, markersize=10, color=teal)
    # Plot line y = x
    tempx = np.linspace(np.min(truevals), np.max(
        truevals), num=100)
    recoverline = plt.plot(tempx, tempx)
    plt.setp(recoverline, linewidth=3, color=orange)


# Using a better Estimated versus True parameter plot"""

def recovery_scatter(theta_true, theta_est, param_names,
                      figsize=(20, 4), font_size=12, color='blue', 
                      ercinterview=False, alpha=0.4,grantB1=False):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values.

    Parameters
    ----------
    theta_true: np.array
        Array of true parameters.
    theta_est: np.array
        Array of estimated parameters.
    param_names: list(str)
        List of parameter names for plotting.
    dpi: int, default:300
        Dots per inch (dpi) for the plot.
    figsize: tuple(int, int), default: (20,4)
        Figure size.
    show: boolean, default: True
        Controls if the plot will be shown
    filename: str, default: None
        Filename if plot shall be saved
    font_size: int, default: 12
        Font size

    """


    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Determine n_subplots dynamically
    if ercinterview:
        n_col = int(np.ceil(len(param_names) / 6))
        n_row = int(np.ceil(len(param_names) / n_col))
    else:
        n_row = int(np.ceil(len(param_names) / 6))
        n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_true[:, j], theta_est[:, j], color=color, alpha=alpha)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute R2
        r2 = r2_score(theta_true[:, j], theta_est[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=font_size)

        # Compute pearson correlation
        pearson, pvalue = stats.pearsonr(theta_true[:, j], theta_est[:, j])
        axarr[j].text(0.7, 0.1, '$\\rho$={:.3f}'.format(pearson),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=font_size)
        
        axarr[j].set_xlabel('True %s' % param_names[j],fontsize=font_size)
        if (np.mod(j,6) == 0) or ercinterview:
            # Label plot
            axarr[j].set_ylabel('Estimated parameters',fontsize=font_size)
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)

        # if grantB1:
        #     axarr[0].set_xlim(-5.5, 5.5)
        #     axarr[0].set_ylim(-5.5, 5.5)
        #     axarr[0].set_xticks([-4, -2, 0, 2, 4])
        #     axarr[0].set_yticks([-4, -2, 0, 2, 4])
        #     axarr[0].set_aspect('equal', adjustable='box')
        #     axarr[1].set_xlim(0, 2.1)
        #     axarr[1].set_ylim(0, 2.1)
        #     axarr[1].set_xticks([0.5, 1, 1.5, 2])
        #     axarr[1].set_yticks([0.5, 1, 1.5, 2])
        #     axarr[1].set_aspect('equal', adjustable='box')

    
    # Adjust spaces
    f.tight_layout()


def plot_posterior2d(posteriors1, posteriors2, 
    param_names=['parameter1, parameter2'], true_params=None,
    figsize=(20, 4), font_size=12, color='blue', alpha=0.25, color2='red',
    highlight=None, color3='#332288'):
    """ Plots a scatter plot of a joint posterior distributions

    Parameters
    ----------
    posteriors1: np.array
        Array of marginal posteriors
    posteriors2: np.array
        Array of marginal posteriors
    true_params: np.array
        Array of true parameter values
    param_names: list(str)
        List of two parameter names
    figsize: tuple(int, int), default: (20,4)
        Figure size.
    font_size: int, default: 12
        Font size

    """

    # Raise an error if the posteriors are different
    if posteriors1.shape != posteriors2.shape:
        raise ValueError("Posterior arrays have different shapes")

    if posteriors1.ndim > 2:
        raise ValueError("First posterior array has more than 2 dimensions")

    if posteriors2.ndim > 2:
        raise ValueError("Second posterior array has more than 2 dimensions")

    if posteriors1.shape[0] > 20:
        raise ValueError("The function would be making more than 20 subplots")

    if true_params is not None:
        if true_params.ndim > 2:
            raise ValueError("true_params array has more than 2 dimensions")
        if true_params.shape[1] != 2:
            raise ValueError("true_params should have a second dimension of length 2")

    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Number of plots (number of joint posteriors)
    num_plots = posteriors1.shape[0]

    # Determine n_subplots dynamically
    n_row = int(np.ceil(num_plots / 6))
    n_col = int(np.ceil(num_plots / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(num_plots):
        # Place highlighted subplot first
        if highlight is not None and (highlight<num_plots):
            if (j > highlight):
                k = j
            elif (j < highlight):
                k = j+1
            else:
                k = 0
        else:
            k = j
        
        # Plot analytic vs estimated
        axarr[k].scatter(posteriors1[j, :], posteriors2[j, :], color=color, 
            alpha=alpha)

        if true_params is not None:
            axarr[k].scatter(true_params[j, 0], true_params[j, 1], color=color2)
        
        # Compute correlation
        pearson, pvalue = stats.pearsonr(posteriors1[j, :], posteriors2[j, :])
        axarr[k].text(0.1, 0.8, '$\\rho$={:.3f}'.format(pearson),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[k].transAxes, 
                     size=font_size)
        
        
        if np.mod(k,6) == 0:
            # Label y-axis
            axarr[k].set_ylabel(f'{param_names[1]}',fontsize=font_size)

        if k >= (n_row-1)*6:
            # Label x-axis
            axarr[k].set_xlabel(f'{param_names[0]}', fontsize=font_size)

        if j==highlight:
            for spine in axarr[k].spines.values():
                spine.set_visible(True)
                spine.set_color(color3)
                spine.set_linewidth(2)
        else:
            axarr[k].spines['right'].set_visible(False)
            axarr[k].spines['top'].set_visible(False) 
    # Adjust spaces
    f.tight_layout()



def rsquared_pred(trueval, predval):
    """
    RSQUARED_PRED  Calculates R^2_prediction for data and statistics derived from data
    """
    divisor = np.sum(np.isfinite(trueval)) - 1
    # Mean squared error of prediction
    MSEP = np.nansum(np.power(trueval - predval, 2)) / divisor
    # Variance estimate of the true values
    vartrue = np.nansum(np.power(trueval - np.nanmean(trueval), 2)) / divisor
    # R-squared definition
    rsquared = 1 - (MSEP / vartrue)
    return rsquared