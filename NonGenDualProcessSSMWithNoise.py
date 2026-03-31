"""
Created on Sat Mar 19 12:51:10 2022
@author: 44796
"""
import numpy as np
import pandas as pd
import DataManipulation as DM
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from scipy.stats import norm
import multiprocessing as mp
from numba import njit
from scipy.stats import qmc
SEED = 42
np.random.seed(SEED)
@njit
def gaussianLogPdf(x, sigma):
    if sigma <= 0.0:
        return -np.log(360.0)
    return -0.5 * np.log(2 * np.pi * sigma**2) - (x ** 2) / (2 * sigma ** 2)
@njit
def computeNegLl(params, aims, rots, imps, targets, is_washout):
    lr1, retain1, noise, lr2, retain2, sigma2 = params
    lr1 += lr2
    lr1 = min(1, lr1)
    logLikelihood = 0.0
    n = len(aims)
    angles = np.arange(0.0, 360.0)
    nAngles = len(angles)
    mOut1 = 0.0
    x2Vec = np.zeros(nAngles)
    for t in range(n):
        rot = rots[t]
        aim = aims[t]
        imp = imps[t]
        target = targets[t]
        if np.isnan(target):
            target = 0                                                             
        washout_t = is_washout[t] if is_washout is not None else False
        raw_center = target + mOut1
        idx = int(np.round(raw_center)) % 360
        center = raw_center % 360.0                      
        out1 = mOut1
        out2 = x2Vec[idx]
        out1_for_pred = 0.0 if washout_t else out1
        predicted = out1_for_pred + out2
  
        if not np.isnan(imp) and not np.isnan(aim):
            angularError = (((aim + imp) - predicted + 180.0) % 360.0 - 180.0)
            logLik = gaussianLogPdf(angularError, noise)
            logLikelihood += logLik
                          
        centerIdx = idx
        out2 = x2Vec[centerIdx]
        pe = rot + out2
        angular_diff = np.minimum(np.abs(angles - center), 360.0 - np.abs(angles - center))
        g2Vec = np.exp(-(angular_diff**2) / (2 * sigma2**2))
        x2Vec = retain2 * x2Vec - lr2 * pe * g2Vec
        newOut2 = x2Vec[centerIdx]
                          
        peRem = rot + out2 + out1
        delta1 = -peRem * lr1 if not washout_t else 0.0
        mOut1 = retain1 * mOut1 + delta1
        mOut1 = ((mOut1 + 180.0) % 360.0 - 180.0)
    return -logLikelihood


def transform(u, bounds):
    return np.array([b[0] + u[j] * (b[1] - b[0]) for j, b in enumerate(bounds)])
def _fitWorker(args):
    pp, i, dat, fitPhase, imp, flipRot, bounds, initial = args
    if fitPhase is not None:
        pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    else:
        pDat = dat[(dat['participantNum'] == pp)]
    bNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == bNums[0]]
    aims = pDat['aim'].values.astype(np.float64)
    rots = pDat['rotation'].values.astype(np.float64)
    is_washout = (pDat['phase'] == 'washout').values.astype(np.bool_)
    targets = pDat['targetPosition'].values.astype(np.float64)
  
                                                                    
    original_rotations = pDat['rotation'].values.astype(np.float64)                       
    target_positions = pDat['targetPosition'].values.astype(np.float64)
  
    if flipRot:
        rots = -rots
    if imp:
        imps = pDat['imp'].values.astype(np.float64)
    else:
        imps = np.zeros_like(aims)
    def localFitPp(u):
        p = transform(u, bounds)
        return computeNegLl(p, aims, rots, imps, targets, is_washout)
    def localFitPp_orig(p):
        return computeNegLl(p, aims, rots, imps, targets, is_washout)
    n_params = len(bounds)
    best_fun = np.inf
    n_starts = 500                                                             
    sampler = qmc.LatinHypercube(d=n_params, seed=SEED)
    norm_samples = sampler.random(n=n_starts)
   
    best_fun = np.inf
    best_p = None
    candidates = []
    best_res = None
    for i in range(n_starts):
        x0 = transform(norm_samples[i], bounds)
       
                              
        res = minimize(
            localFitPp_orig, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-5, 'gtol': 1e-4}
        )
       
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_res = res
            best_fun = res.fun
            best_p = res.x.copy()
            print(f"Quick best: start {i}, f={best_fun}")
       
                                               
        if np.isfinite(res.fun):
            candidates.append((res.fun, res.x))
        if len(candidates) > 20:
            candidates = sorted(candidates, key=lambda c: c[0])[:20]
                                    
    for fun, x in sorted(candidates, key=lambda c: c[0])[:5]:
        polished = minimize(
            localFitPp_orig, x, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )
        if np.isfinite(polished.fun) and polished.fun < best_fun:
            best_res = polished
            best_fun = polished.fun
            best_p = polished.x
                                                     
    if best_res is None:
        print(f"Warning: No valid finite fit found for participant {pp}. Using initial parameters.")
        initial_array = np.array(initial)
        best_fun = localFitPp_orig(initial_array) if np.isfinite(localFitPp_orig(initial_array)) else float('inf')
                                    
        class DummyRes:
            def __init__(self, x, fun):
                self.x = x
                self.fun = fun
        best_res = DummyRes(initial_array, best_fun)
    best_fun = best_res.fun
    res = best_res
    lr1, retain1, noise, lr2, retain2, sigma2 = res.x
    lr1 += lr2
    lr1 = min(1, lr1)
    totErr = []
    observed = []
    m1 = []
    m2 = []
    sts = []
    angles = np.arange(0.0, 360.0)
    nAngles = len(angles)
    mOut1 = 0.0
    x2Vec = np.zeros(nAngles)
    liks = []
    ese = []                              
    for t in range(len(aims)):
        rot = rots[t]
        aim = aims[t]
        impVal = imps[t]
        target = targets[t]
        if np.isnan(target):
            target = 0                   
        washout_t = is_washout[t]
        raw_center = target + mOut1
        idx = int(np.round(raw_center)) % 360
        center = raw_center % 360.0                      
        out1 = mOut1
        out2 = x2Vec[idx]
        out1_for_pred = 0.0 if washout_t else out1
        predicted = out1_for_pred + out2
        sts.append(predicted)
        m1.append(out1_for_pred)
        m2.append(out2)
        if not np.isnan(impVal) and not np.isnan(aim):
            obs = aim + impVal
            angularError = (((obs - predicted + 180) % 360 - 180))
            lik = norm.pdf(angularError, loc=0, scale=noise) + 1e-12
            liks.append(lik)
            totErr.append(angularError)
            observed.append(obs)
            ese.append(angularError**2 + noise**2)
                
        centerIdx = idx
        out2 = x2Vec[centerIdx]
        pe = rot + out2
        angular_diff = np.minimum(np.abs(angles - center), 360.0 - np.abs(angles - center))
        g2Vec = np.exp(-(angular_diff**2) / (2 * sigma2**2))
        x2Vec = retain2 * x2Vec - lr2 * pe * g2Vec
        newOut2 = x2Vec[centerIdx]
        peRem = rot + out2 + out1
        delta1 = -peRem * lr1 if not washout_t else 0.0
        mOut1 = retain1 * mOut1 + delta1
        mOut1 = ((mOut1 + 180.0) % 360.0 - 180.0)
    numSamp = len(totErr)
    logLikelihood = np.nansum(np.log(liks)) if liks else 0.0
    k = 6
    BIC = k * np.log(numSamp) - 2 * logLikelihood if numSamp > 0 else np.nan
                                
    if numSamp > 0:
        totErr = np.array(totErr)
        observed = np.array(observed)
        rmse = np.sqrt(np.nanmean(totErr**2)) if len(totErr) > 0 else np.nan
        if numSamp > 1:
            meanObs = np.nanmean(observed)
            ssTot = np.nansum((observed - meanObs)**2)
            ssRes = np.nansum(totErr**2)
            rSquared = 1 - (ssRes / ssTot) if ssTot > 0 else np.nan
        else:
            rSquared = np.nan
    else:
        rmse = np.nan
        rSquared = np.nan
                                             
    rmse_dist = np.sqrt(np.mean(ese)) if ese else np.nan
    ssTot = np.nansum((observed - np.nanmean(observed))**2) if numSamp > 0 else 1.0
    rSquared_dist = 1 - np.sum(ese) / ssTot if ssTot > 0 else (np.nan if np.sum(ese) > 0 else 1.0)
                                                                                                                     
    return {
        'pp': pp,          
        'targetPosition': target_positions,                                                
        'rotation': original_rotations,                                                         
        'is_washout': is_washout,          
        'x': res.x,
        'fun': res.fun,
        'errors': totErr,
        'mOut1': m1,
        'mOut2': m2,
        'mStates': sts,
        'allAims': aims,
        'allImps': imps,
        'BIC': BIC,
        'rmse': rmse,
        'rSquared': rSquared,
        'rmse_dist': rmse_dist,
        'rSquared_dist': rSquared_dist
    }
class FitShell:
    def __init__(self, df='none', conVal='none', condition='none', fitLength=320, fitPhase=None, imp=True,
                 rmse=False, flipRot=False, method='CMA'):
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
                                                   
        self.rmses_dist = []
        self.rSquareds_dist = []
                                                                          
        self.participants = []                                              
        self.targetPositions = []                                                       
        self.rotations = []                                                          
        self.is_washouts = []                                            
    def fitRot(self, lrUB=1):
        if self.condition is not None:
            self.dat = self.df[self.df[self.condition] == self.conVal]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
                               
        n_pp = len(uniqP)
        self.bics = np.zeros(n_pp)
        self.negLL = np.ones(n_pp) * 100000
        self.rmses = np.full(n_pp, np.nan)
        self.rSquareds = np.full(n_pp, np.nan)
        self.mStates = [[] for _ in uniqP]
        self.mOut1 = [[] for _ in uniqP]
        self.mOut2 = [[] for _ in uniqP]
        self.allAims = [[] for _ in uniqP]
        self.allImps = [[] for _ in uniqP]
        self.errors = []
                                
        self.rmses_dist = np.full(n_pp, np.nan)
        self.rSquareds_dist = np.full(n_pp, np.nan)
                                                                    
        self.participants = []                                 
        self.targetPositions = []                                       
        self.rotations = []                                          
        self.is_washouts = []          
        self.xs = []
        bounds = [(0,1), (.95,1.), (1,100.), (0.001, 0.5), (0.8, 0.9999), (1.,90.)]
        initial = [.8, 0.99, 5., 0.005, 0.99, 30.]
        argsList = [(pp, i, self.dat, self.fitPhase, self.imp, self.flipRot, bounds, initial)
                    for i, pp in enumerate(uniqP)]
        with mp.Pool(processes=20) as pool:
            results = pool.map(_fitWorker, argsList)
        for i, res in enumerate(results):
            pp = uniqP[i]
                                          
            self.participants.append(pp)
                                                                   
            self.targetPositions.append(res['targetPosition'])
            self.rotations.append(res['rotation'])
            self.is_washouts.append(res['is_washout'])          
                             
            self.xs.append(res['x'])
            self.negLL[i] = res['fun']
            self.errors.append(res['errors'])
            self.bics[i] = res['BIC']
            self.rmses[i] = res['rmse']
            self.rSquareds[i] = res['rSquared']
            self.mStates[i] = res['mStates']
            self.mOut1[i] = res['mOut1']
            self.mOut2[i] = res['mOut2']
            self.allAims[i] = res['allAims']
            self.allImps[i] = res['allImps']
                                    
            self.rmses_dist[i] = res['rmse_dist']
            self.rSquareds_dist[i] = res['rSquared_dist']
            if False:
                fig, ax = plt.subplots(figsize=(15, 6))
                trials = range(len(self.allAims[i]))
  
                ax.scatter(trials, self.mOut1[i], color='orange', s=20, alpha=0.7, edgecolors='orange',
                           linewidths=0.8, facecolors='none', label='Model Process 1')
                ax.scatter(trials, self.mOut2[i], color='dodgerblue', s=20, alpha=0.4, edgecolors='dodgerblue',
                           linewidths=0.8, facecolors='none', label='Model Process 2')
  
                ax.scatter(trials, self.allImps[i], color='blue', marker='x', s=20, alpha=0.4,
                           linewidths=0.8, label='Human Implicit')
                ax.scatter(trials, self.allAims[i], color='darkred', marker='x', s=30, alpha=0.4,
                           linewidths=0.8, label='Human Explicit')
  
                humanTotal = []
                for a, im in zip(self.allAims[i], self.allImps[i]):
                    if np.isnan(a):
                        humanTotal.append(im)
                    else:
                        humanTotal.append(a + im)
                ax.plot(trials, humanTotal, color='black', alpha=1, label='Human Total')
                ax.plot(trials, self.mStates[i], color='brown', alpha=1, lw=3, label='Model Total')
  
                ax.hlines(y=0, xmin=0, xmax=max(trials), linewidth=2, color='green', alpha=1)
  
                ax.set_xlabel('Trial')
                ax.set_ylabel('Degrees')
                ax.set_title(f'Model vs Human for Participant {pp}')
  
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='orange', label='Model Process 1', markersize=6,
                           linestyle='None', alpha=0.5, fillstyle='none'),
                    Line2D([0], [0], marker='o', color='dodgerblue', label='Model Process 2', markersize=6,
                           linestyle='None', alpha=0.4, fillstyle='none'),
                    Line2D([0], [0], marker='x', color='blue', label='Human Implicit', markersize=6,
                           linestyle='None', alpha=0.4),
                    Line2D([0], [0], marker='x', color='darkred', label='Human Explicit', markersize=6,
                           linestyle='None', alpha=0.4),
                    Line2D([0], [0], color='black', label='Human Total', linewidth=2, alpha=1),
                    Line2D([0], [0], color='brown', label='Model Total', linewidth=3, alpha=1)
                ]
                ax.legend(handles=legend_elements)
  
                filename = f'participant_{pp}_plot.png'
                plt.savefig(filename, dpi=200)
                plt.clf()
                plt.close()
      

