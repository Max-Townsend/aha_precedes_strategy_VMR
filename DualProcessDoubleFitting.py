"""
Created on Sat Mar 19 12:51:10 2022
@author: 44796
"""
import numpy as np
import pandas as pd
import DataManipulation as DM
from optimparallel import minimize_parallel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import vonmises
import multiprocessing as mp
from numba import njit
from cmaes import CMA
@njit
def logBesselI0(x):
    x = np.abs(x)
    if x < 3.75:
        t = (x / 3.75)**2
        i0 = 1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492 + t*(0.2659732 + t*(0.0360768 + t*0.0045813)))))
        return np.log(i0)
    else:
        t = 3.75 / x
        poly = 0.39894228 + t*(0.01328592 + t*(0.00225319 + t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706 + t*(0.02635537 + t*(-0.01647633 + t*0.00392377)))))))
        return x - 0.5 * np.log(x) + np.log(poly)
@njit
def vonMisesLogPdf(theta, kappa):
    if kappa <= 0.0:
        return -np.log(360.0)
    return kappa * np.cos(theta) - np.log(360.0) - logBesselI0(kappa)
@njit
def computeNegLl(params, aims, rots, imps):
    lr1, retain1, lr2, retain2, noise1, noise2 = params
    logLikelihood = 0.0
    mOut1 = 0.0
    mOut2 = 0.0
    n = len(aims)
    for t in range(n):
        rot = rots[t] if t > 0 else 0.0
        aim = aims[t]
        imp = imps[t]
        if not np.isnan(imp):
            angularError = (((imp) - mOut2 + 180.0) % 360.0 - 180.0)
            theta = np.deg2rad(angularError)
            kappa2 = 1.0 / (noise2 * (np.pi / 180.0)**2) if noise2 > 0.0 else 0.0
            logLik = vonMisesLogPdf(theta, kappa2)
            logLikelihood += logLik
        if not np.isnan(aim):
            angularError = (((aim) - mOut1 + 180.0) % 360.0 - 180.0)
            theta = np.deg2rad(angularError)
            kappa1 = 1.0 / (noise1 * (np.pi / 180.0)**2) if noise1 > 0.0 else 0.0
            logLik = vonMisesLogPdf(theta, kappa1)
            logLikelihood += logLik
                                                            
        pe1 = rot + mOut1
        delta1 = -pe1 * lr1
        mOut1 = (mOut1 * retain1) + delta1
        pe2 = rot + mOut2
        delta2 = -pe2 * lr2
        mOut2 = (mOut2 * retain2) + delta2
    return -logLikelihood
@njit
def generateData(lr1, retain1, lr2, retain2, noise1, noise2, rots, trials):
    n = len(trials)
    states = np.empty(n)
    kappa1 = 1.0 / (noise1 * (np.pi / 180.0)**2) if noise1 > 0.0 else 0.0
    kappa2 = 1.0 / (noise2 * (np.pi / 180.0)**2) if noise2 > 0.0 else 0.0
    mOut1 = 0.0
    mOut2 = 0.0
    if noise1 > 0.0:
        angularNoise1 = np.rad2deg(np.random.vonmises(0.0, kappa1))
        mOut1 += angularNoise1
    if noise2 > 0.0:
        angularNoise2 = np.rad2deg(np.random.vonmises(0.0, kappa2))
        mOut2 += angularNoise2
    for i in range(n):
        t = trials[i]
        states[i] = mOut1 + mOut2
        rot = rots[t]
                                                            
        pe1 = rot + mOut1
        delta1 = -pe1 * lr1
        mOut1 = (mOut1 * retain1) + delta1
        pe2 = rot + mOut2
        delta2 = -pe2 * lr2
        mOut2 = (mOut2 * retain2) + delta2
        if noise1 > 0.0:
            angularNoise1 = np.rad2deg(np.random.vonmises(0.0, kappa1))
            mOut1 += angularNoise1
        if noise2 > 0.0:
            angularNoise2 = np.rad2deg(np.random.vonmises(0.0, kappa2))
            mOut2 += angularNoise2
        mOut1 = ((mOut1 + 180.0) % 360.0 - 180.0)
        mOut2 = ((mOut2 + 180.0) % 360.0 - 180.0)
    return states
                                      
def _fit_worker(args):
    pp, i, dat, fitPhase, imp, flipRot, method, bounds, initial = args
    if fitPhase is not None:
        pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    else:
        pDat = dat[(dat['participantNum'] == pp)]
    bNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == bNums[0]]
    aims = pDat['aim'].values.astype(np.float64)
    rots = pDat['rotation'].values.astype(np.float64)
    if flipRot:
        rots = -rots
    if imp:
        imps = pDat['imp'].values.astype(np.float64)
    else:
        imps = np.zeros_like(aims)
    def local_fitPP(params):
        return computeNegLl(params, aims, rots, imps)
                              
    boundsArray = np.array(bounds)
    maxRestarts = 1
    popSize = 512 * 10
    sigma = popSize // 2
    bestValue = np.inf
    bestX = None
    restart = 0
    while restart < maxRestarts:
        np.random.seed(33 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=999 + restart)
        es.tolfun = 1e-2
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 25:
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [local_fitPP(x) for x in xSamples]
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                if currentBest[1] < bestValue * 0.999995:
                    sinceBest = 0
                else:
                    sinceBest += 1
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                sinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.999995:
                    sinceBest = 0
                else:
                    sinceBest += 1
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        restart += 1
    res = minimize(local_fitPP, bestX, method='L-BFGS-B', bounds=bounds)
    if res.fun < bestValue:
        bestValue = res.fun
        bestX = res.x
    res.x = bestX
    res.fun = bestValue
                              
    lr1, retain1, lr2, retain2, noise1, noise2 = res.x
    totErr = []
    m1 = []
    m2 = []
    sts = []
    mOut1 = 0
    mOut2 = 0
    liks_total = []
    for t in range(len(aims)):
        rot = rots[t] if t > 0 else 0.0
        m_total = mOut1 + mOut2
        sts.append(m_total)
        m1.append(mOut1)
        m2.append(mOut2)
        aim = aims[t]
        imp_val = imps[t]
        h_total = np.nan
        if np.isnan(aim) and np.isnan(imp_val):
            h_total = np.nan
        elif np.isnan(aim):
            h_total = imp_val
        elif np.isnan(imp_val):
            h_total = aim
        else:
            h_total = aim + imp_val
        if not np.isnan(h_total):
            angular_error = (((h_total) - m_total + 180.0) % 360.0 - 180.0)
            if not np.isnan(aim) and not np.isnan(imp_val):
                      
                kappa1 = 1.0 / (noise1 * (np.pi / 180.0)**2) if noise1 > 0.0 else float('inf')
                kappa2 = 1.0 / (noise2 * (np.pi / 180.0)**2) if noise2 > 0.0 else float('inf')
                if kappa1 == float('inf') and kappa2 == float('inf'):
                    kappa = float('inf')
                elif kappa1 == float('inf'):
                    kappa = kappa2
                elif kappa2 == float('inf'):
                    kappa = kappa1
                else:
                    kappa = (kappa1 * kappa2) / (kappa1 + kappa2)
            elif not np.isnan(aim):
                kappa = 1.0 / (noise1 * (np.pi / 180.0)**2) if noise1 > 0.0 else float('inf')
            else:
                kappa = 1.0 / (noise2 * (np.pi / 180.0)**2) if noise2 > 0.0 else float('inf')
            if kappa == float('inf'):
                lik = 1.0 if angular_error == 0.0 else 0.0
            else:
                lik = (vonmises.pdf(np.deg2rad(angular_error), kappa) * (np.pi / 180.0)) + 1e-12
            liks_total.append(lik)
            totErr.append(angular_error)
                                                            
        pe1 = rot + mOut1
        delta1 = -pe1 * lr1
        mOut1 = (mOut1 * retain1) + delta1
        pe2 = rot + mOut2
        delta2 = -pe2 * lr2
        mOut2 = (mOut2 * retain2) + delta2
    logLikelihood = np.sum(np.log(liks_total)) if liks_total else 0.0
    numSamp = len(liks_total)
    k = 6
    BIC = k * np.log(numSamp) - 2 * logLikelihood if numSamp > 0 else float('inf')
    return {
        'x': res.x,
        'fun': res.fun,
        'errors': totErr,
        'mOut1': m1,
        'mOut2': m2,
        'mStates': sts,
        'allAims': aims,
        'allImps': imps,
        'BIC': BIC
    }
class fitShell:
    def __init__(self,df='none', conVal='none',condition='none',fitLength=320,fitPhase=None,imp=True,
                 rmse=False,flipRot=False,method='Powell'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.allAims = [[]]
        self.dat = df
        self.BICs = []
        self.fitLength = fitLength
        self.fitPhase = fitPhase
        self.imp = imp
        self.errors = []
        self.indiErrors = []
        self.rmse = rmse
        self.flipRot = flipRot
        self.method = method
 
    def fitRot(self,lrUB=1):
        if self.condition is not None:
            self.dat = self.df[self.df[self.condition] == self.conVal]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[] for _ in uniqP]
        self.mOut1 = [[] for _ in uniqP]
        self.mOut2 = [[] for _ in uniqP]
        self.allAims = [[] for _ in uniqP]
        self.allImps = [[] for _ in uniqP]
        self.xs = []
        self.errors = []
        bounds = [(0,1.),(0,1.),(0.,1),(.95,1.),(1,100000.),(1,100000.)]
        initial = [1.0,0.95,0.005,0.95,500.,500.]
        args_list = [(pp, i, self.dat, self.fitPhase, self.imp, self.flipRot, self.method, bounds, initial) for i, pp in enumerate(uniqP)]
        with mp.Pool() as pool:
            results = pool.map(_fit_worker, args_list)
        for i, res in enumerate(results):
            self.xs.append(res['x'])
            self.negLL[i] = res['fun']
            self.errors.append(res['errors'])
            self.BICs[i] = res['BIC']
            self.mStates[i] = res['mStates']
            self.mOut1[i] = res['mOut1']
            self.mOut2[i] = res['mOut2']
            self.allAims[i] = res['allAims']
            self.allImps[i] = res['allImps']
                                                
            plt.figure()
            trials = range(len(self.allAims[i]))
            alpha_ind = 0.5
            plt.plot(trials, self.allAims[i], label='Human Aim', alpha=alpha_ind)
            plt.plot(trials, self.allImps[i], label='Human Imp', alpha=alpha_ind)
            plt.plot(trials, self.mOut1[i], label='Model Process 1', alpha=alpha_ind)
            plt.plot(trials, self.mOut2[i], label='Model Process 2', alpha=alpha_ind)
                                 
            human_total = []
            for a, im in zip(self.allAims[i], self.allImps[i]):
                if np.isnan(a) and np.isnan(im):
                    human_total.append(np.nan)
                elif np.isnan(a):
                    human_total.append(im)
                elif np.isnan(im):
                    human_total.append(a)
                else:
                    human_total.append(a + im)
            plt.plot(trials, human_total, label='Human Total', alpha=1)
            plt.plot(trials, self.mStates[i], label='Model Total', alpha=1)
            plt.xlabel('Trial')
            plt.ylabel('Degrees')
            plt.title(f'Participant {uniqP[i]} Processes')
            plt.legend()
            filename = f'participant_{uniqP[i]}_plot.png'
            plt.savefig(filename)
            plt.clf()
            plt.close()
     
 
    def genDat(self,params,rots,trials=np.arange(40)):
        lr1,retain1,lr2,retain2 = params
        noise1 = 0
        noise2 = 0
        return generateData(lr1, retain1, lr2, retain2, noise1, noise2, rots, trials)