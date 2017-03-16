import numpy as np
from scipy import stats
from scipy.special import erf
import sys

class TruncatedRv(object):
    def __init__(self, rv, lb, ub, rvname=None):
        self.rv = rv
        self.lb = lb
        self.ub = ub
        self.rvname = rvname

    def pdf(self, x):
        rv = self.rv
        ub = self.ub
        lb = self.lb
        rvpdf = rv.pdf(x) / (rv.cdf(ub)-rv.cdf(lb))
        try:
            rvpdf[np.logical_or(x<lb, x>ub)] = 0.
        except TypeError:
            if x<lb or x>ub:
                rvpdf = 0.
        return rvpdf

    def cdf(self, x):
        rv = self.rv
        ub = self.ub
        lb = self.lb
        rvcdf = (rv.cdf(x)-rv.cdf(lb)) / (rv.cdf(ub)-rv.cdf(lb))
        try:
            rvcdf[x<lb] = 0.
            rvcdf[x>ub] = 1.
        except TypeError:
            if x<lb:
                rvcdf = 0.
            elif x>ub:
                rvcdf = 1.
        return rvcdf

    def ppf(self, x):
        rv = self.rv
        ub = self.ub
        lb = self.lb
        rv0cdf = x*(rv.cdf(ub)-rv.cdf(lb))+rv.cdf(lb)
        rvppf = rv.ppf(rv0cdf)
        return rvppf

    def rvs(self, size=None):
        rvcdf = np.random.rand(size)
        rvsmp = self.ppf(rvcdf)
        return rvsmp

    def moment(self, order, nsmp=1e4):
        l = self.lb; u = self.ub
        rvname = self.rvname
        if rvname.lower() == "normal" and order<=3:
            dnmntr = self.rv.cdf(u) - self.rv.cdf(l)
            mu,var = self.rv.stats()
            sgm = np.sqrt(var)
            k0 = sgm/np.sqrt(2*np.pi)
            k1 = (u-mu)/sgm/np.sqrt(2)
            k2 = (l-mu)/sgm/np.sqrt(2)
            #mm0 = 0.5*(erf(k1) - erf(k2))
            mm0 = stats.norm.cdf(np.sqrt(2)*k1)-stats.norm.cdf(np.sqrt(2)*k2)
            if order == 1:
                mm = k0*(np.exp(-k2**2)-np.exp(-k1**2))+mu*mm0
            elif order == 2:
                if np.isposinf(k1):
                    mm = k0*((l+mu)*np.exp(-k2**2)-0.0)+(mu**2+sgm**2)*mm0
                elif np.isneginf(k2):
                    mm = k0*(0.0-(u+mu)*np.exp(-k1**2))+(mu**2+sgm**2)*mm0
                else:
                    mm = k0*((l+mu)*np.exp(-k2**2)-(u+mu)*np.exp(-k1**2))+(mu**2+sgm**2)*mm0
            elif order == 3:
                mm = k0*((2*sgm**2+mu**2+l*mu+l**2)*np.exp(-k2**2) -
                         (2*sgm**2+mu**2+u*mu+u**2)*np.exp(-k1**2))+mu*(mu**2+3*sgm**2)*mm0
            mm = mm/dnmntr
        elif rvname.lower() == "lognormal":
            dnmntr = self.rv.cdf(u) - self.rv.cdf(l)
            m,v = self.rv.stats()
            mu = np.log(m/np.sqrt(1+v/m**2))
            sgm = np.sqrt(np.log(1+v/m**2))
            N = order
            k1 = (np.log(u)-mu)/sgm/np.sqrt(2) - N*sgm/np.sqrt(2)
            k2 = (np.log(l)-mu)/sgm/np.sqrt(2) - N*sgm/np.sqrt(2)
            d = 2*(stats.norm.cdf(np.sqrt(2)*k1)-stats.norm.cdf(np.sqrt(2)*k2))
            mm = 0.5*np.exp(N*mu+N**2*sgm**2/2)*d
            mm = mm/dnmntr
        else:
            nsmp = int(nsmp)
            smp = self.rvs(size=nsmp)
            mm = np.mean( smp**(int(order)) )
        return mm

    def stats(self, moments='mv', nsmp=1e4):
        sts = []
        useCloseMoments = self.rvname.lower() == "normal" or\
                          self.rvname.lower() == "lognormal"
        if useCloseMoments:
            for moment in moments:
                if moment == 'm':
                    sts.append(np.array(self.moment(1)))
                elif moment == 'v':
                    ex1 = self.moment(1)
                    ex2 = self.moment(2)
                    sts.append(np.array(ex2-ex1**2))
        else:
            nsmp = int(nsmp)
            smp = self.rvs(size=nsmp)
            for moment in moments:
                if moment == 'm':
                    sts.append(np.array(np.mean(smp)))
                elif moment == 'v':
                    sts.append(np.array(np.var(smp)))

        if len(sts) == 1:
            return sts[0]
        else:
            return tuple(sts)

    def mean(self):
        return self.stats(moments='m')


class NegExpon(object):
    def __init__(self, loc=0, scale=1.):
        self.rv = stats.expon(loc=loc, scale=scale)
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        y = 2*self.loc - x
        return self.rv.pdf(y)

    def cdf(self, x):
        y = 2*self.loc - x
        return self.rv.cdf(y)

    def ppf(self, x):
        y = self.rv.ppf(x)
        return 2*self.loc - y

    def rvs(self, size=None):
        y = self.rv.rvs(size)
        return 2*self.loc - y

    def moment(self, order):
        if order>2:
            print "moments of higher order than 2 is not supported"
            sys.exit(1)
        if order == 1:
            return 2*self.loc - self.rv.moment(1)
        elif order == 2:
            mm1 = self.rv.moment(1)
            mm2 = self.rv.moment(2)
            k = self.loc
            return mm2-4.*k*mm1 + 4.*k**2

    def stats(self, moments='mv'):
        keys = moments
        ystats = self.rv.stats(moments)
        ydict = dict(zip(keys, ystats))
        sts = []
        for moment in moments:
            if moment == 'm':
                sts.append(2*self.loc - ydict['m'])
            elif moment == 'v':
                sts.append(ydict['v'])
            elif moment == 's':
                sts.append(2*self.loc - ydict['s'])
            elif moment == 'k':
                sts.append(ydict['k'])
        if len(sts) == 1:
            return sts[0]
        else:
            return tuple(sts)

    def mean(self):
        return 2*self.loc-self.rv.mean()
