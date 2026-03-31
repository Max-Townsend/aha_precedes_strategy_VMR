"""
Created on Sat Mar 19 12:51:10 2022
@author: 44796
"""
import numpy as np
import pandas as pd
import DataManipulation as DM
from optimparallel import minimize_parallel
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from scipy.stats import norm
import multiprocessing as mp
from numba import njit
from scipy.stats.qmc import LatinHypercube
from cmaes import CMA
from scipy.stats import qmc

SEED = 42
np.random.seed(SEED)

@njit
def gaussianLogPdf(x, sigma):
    if sigma <= 0.0:
        return -1e100                                         
    return -np.log(sigma * np.sqrt(2 * np.pi)) - (x ** 2) / (2 * sigma ** 2)

@njit
def computeNegLl(params, aims, rots, imps):
    lr, retain, noise = params
    logLikelihood = 0.0
    mOut = 0.0
    n = len(aims)
    for t in range(n):
        rot = rots[t] if t > 0 else 0.0
        aim = aims[t]
        imp = imps[t]
        if not np.isnan(imp):
            pe = rot + mOut
            mOut = (mOut * retain) + (-pe * lr)
        if not np.isnan(aim):
            angularError = ((aim - mOut + 180.0) % 360.0 - 180.0)
            logLik = gaussianLogPdf(angularError, noise)
            logLikelihood += logLik
    return -logLikelihood



def transform(u, bounds):
    return np.array([b[0] + u[j] * (b[1] - b[0]) for j, b in enumerate(bounds)])

def _fitWorker(pp, i, dat, fitPhase, imp, flipRot, method, bounds, initial, n_starts=500):
    if fitPhase is not None:
        pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    else:
        pDat = dat[(dat['participantNum'] == pp)]
    bNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == bNums[0]]
    aims = pDat['aim'].values.astype(np.float64)
    rots = pDat['rotation'].values.astype(np.float64)
   
    original_rotations = pDat['rotation'].values.astype(np.float64)
    target_positions = pDat['targetPosition'].values.astype(np.float64)
   
    if flipRot:
        rots = -rots
    if imp:
        imps = pDat['imp'].values.astype(np.float64)
    else:
        imps = np.zeros_like(aims)
    def localFitPp_orig(p):
        return computeNegLl(p, aims, rots, imps)
    def localFitPp(u):
        p = transform(u, bounds)
        return computeNegLl(p, aims, rots, imps)
    n_params = len(bounds)
    best_fun = np.inf
    sampler = qmc.LatinHypercube(d=n_params, seed=SEED)
    norm_samples = sampler.random(n=n_starts)
    
    best_fun = np.inf
    best_p = None
    candidates = [] 
    
    for i in range(n_starts):
        x0 = transform(norm_samples[i], bounds)
        
                              
        res = minimize(
            localFitPp_orig, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-5, 'gtol': 1e-4}
        )
        
        if res.fun < best_fun:
            best_res = res
            best_fun = res.fun
            best_p = res.x.copy()
            print(f"Quick best: start {i}, f={best_fun}")
        
                                               
        candidates.append((res.fun, res.x))
        if len(candidates) > 20:
            candidates = sorted(candidates, key=lambda c: c[0])[:20]
                                    
    for fun, x in sorted(candidates, key=lambda c: c[0])[:5]:
        polished = minimize(
            localFitPp_orig, x, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )
        if polished.fun < best_fun:
            best_res = polished
            best_fun = polished.fun
            best_p = polished.x
    best_fun = best_res.fun
    res = best_res
                              
    lr, retain, noise = res.x
    totErr = []
    sts = []
    mOut = 0
    liks = []
    for t in range(len(aims)):
        rot = rots[t]
        aim = aims[t]
        impVal = imps[t]
        rot = rot if t > 0 else 0
        if not np.isnan(impVal):
            pe = rot + mOut
            mOut = (mOut * retain) + (-pe * lr)
        sts.append(mOut)
        if not np.isnan(aim):
            angularError = ((aim - mOut + 180) % 360 - 180)
            lik = (norm.pdf(angularError, 0, noise)) + 1e-12
            liks.append(lik)
            totErr.append(angularError)
    numSamp = len(totErr)
    logLikelihood = np.sum(np.log(liks))
    k = 3
    bic = k * np.log(numSamp) - 2 * logLikelihood
    totErr = np.array(totErr)
    rmse = np.sqrt(np.mean(totErr**2))
    ssRes = np.sum(totErr**2)
    validAims = np.array([aims[t] for t in range(len(aims)) if not np.isnan(aims[t])])
    meanAim = np.mean(validAims)
    ssTot = np.sum((validAims - meanAim)**2)
    rSquared = 1 - (ssRes / ssTot) if ssTot != 0 else 0
                                                                       
    return {
        'pp': pp,          
        'targetPosition': target_positions,          
        'rotation': original_rotations,                        
        'x': res.x,
        'fun': res.fun,
        'errors': totErr,
        'mStates': sts,
        'allAims': aims,
        'bic': bic,
        'rmse': rmse,
        'rSquared': rSquared
    }

class FitShell:
    def __init__(self, df='none', conVal='none', condition='none', fitLength=320, fitPhase='rotation', imp=True,
                 rmse=False, flipRot=False, method='CMA', n_starts=500):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.allAims = [[]]
        self.dat = df
        self.bics = []
        self.fitLength = fitLength
        self.fitPhase = fitPhase
        self.imp = imp
        self.errors = []
        self.indiErrors = []
        self.rmse = rmse
        self.flipRot = flipRot
        self.method = method
        self.n_starts = n_starts
        self.participants = []                     
        self.targetPositions = []                                          
        self.rotations = []                                             
   
    def fitRot(self, lrUb=1):
        if self.fitPhase is not None:
            self.dat = self.df[self.df[self.condition] == self.conVal]
        else:
            self.dat = self.df
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        self.participantNums = uniqP
                               
        n = len(uniqP)
        self.bics = np.zeros(n)
        self.negLl = np.ones(n)*100000
        self.mStates = [[] for _ in uniqP]
        self.allAims = [[] for _ in uniqP]
        self.xs = []
        self.errors = []
        self.rmses = np.zeros(n)
        self.rSquareds = np.zeros(n)
                            
        self.participants = []
        self.targetPositions = []
        self.rotations = []
        bounds = [(0,1.),(0,1.),(1,100.)]
        initial = [0.5,0.95,5.]
        argsList = [(pp, i, self.dat, self.fitPhase, self.imp, self.flipRot, self.method, bounds, initial, self.n_starts)
                    for i, pp in enumerate(uniqP)]
        with mp.Pool(processes=14) as pool:
            results = pool.starmap(_fitWorker, argsList)
        for i, res in enumerate(results):
            pp = uniqP[i]
            self.participants.append(res['pp'])
            self.targetPositions.append(res['targetPosition'])
            self.rotations.append(res['rotation'])
            self.xs.append(res['x'])
            self.negLl[i] = res['fun']
            self.errors.append(res['errors'])
            self.bics[i] = res['bic']
            self.mStates[i] = res['mStates']
            self.allAims[i] = res['allAims']
            self.rmses[i] = res['rmse']
            self.rSquareds[i] = res['rSquared']
       
    def fitPp(self, params):
        lr, retain, noise = params
        if self.fitPhase != None:
            pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        else:
            pDat = self.dat[(self.dat['participantNum'] == self.pp)]
        bNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == bNums[0]]
        aims = pDat['aim'].values.astype(np.float64)
        rots = pDat['rotation'].values.astype(np.float64)
        if self.flipRot:
            rots = -rots
        if self.imp:
            imps = pDat['imp'].values.astype(np.float64)
        else:
            imps = np.zeros_like(aims)
        return computeNegLl(params, aims, rots, imps)


