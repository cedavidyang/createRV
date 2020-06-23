import scipy.optimize as opt
from scipy.special import gamma

import numpy as np
from scipy import stats

def lognstats(mu, sigma, inputtype=None):
    if inputtype is None:    # mean and std to logmean and logstd
        cov = sigma/mu
        logmean = np.log(mu/np.sqrt(1.+cov**2))
        logstd = np.sqrt(np.log(1.+cov**2))
        return [logmean,logstd]
    elif inputtype == 'log':
        mean = np.exp(mu+sigma**2/2.)
        std = np.sqrt( (np.exp(sigma**2)-1.)*np.exp(2*mu+sigma**2) )
        return [mean,std]


def wblstats(mu, sigma, inputtype=None):
    if inputtype is None:
        def wblfunc(x):
            y1 = x[0]*gamma(1+1./x[1]) - mu
            y2 = x[0]**2*(gamma(1+2./x[1])-gamma(1+1./x[1])**2) - sigma**2
            return [y1,y2]
        x0 = np.array([mu, 1.2/(sigma/mu)])
        rootres = opt.root(wblfunc, x0)
        [scale, c] = rootres.x
        return [scale,c]
    elif inputtupe == 'wbl':
        c = mu; scale=sigma
        mean,var = stats.weibull_min.stats(c, scale=scale)
        return [mean, np.sqrt(var)]


def gblstats(mu, sigma, inputtype=None):
    """gumbel_r for maximum value (e.g. load effects)"""
    if inputtype is None:
        scale = np.sqrt(6)*sigma/np.pi
        loc = mu-scale*np.euler_gamma
        return [loc, scale]
    elif inputtupe == 'gbl':
        loc = mu; scale=sigma
        mean = loc+scale*np.euler_gamma
        std = np.pi*scale/np.sqrt(6)
        return [mean, std]


def gammastats(mu, sigma, inputtype=None):
    """gamma distribution"""
    if inputtype is None:
        scale = std_**2/mean_
        a = mean_ / scale
        return [a, scale]
    elif inputtupe == 'gamma_scale':
        a = mu; scale=sigma
        mean_ = a*scale
        std_ = np.sqrt(a*scale**2)
        return [mean_, std_]

