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
from scipy.stats.qmc import LatinHypercube
from cmaes import CMA
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
    lr1, retain1, lr2, retain2, sigma2, noise = params
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
        washout_t = is_washout[t]
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

@njit
def generateData(lr1, retain1, lr2, retain2, sigma2, noise, rots, trials, targets, is_washout):
    n = len(trials)
    lr1 += lr2
    lr1 = min(1, lr1)
    states = np.empty(n)
    angles = np.arange(0.0, 360.0)
    nAngles = len(angles)
    mOut1 = 0.0
    x2Vec = np.zeros(nAngles)
    if noise > 0.0:
        angularNoise1 = np.random.normal(0.0, noise / np.sqrt(2))
        mOut1 += angularNoise1
        mOut1 = ((mOut1 + 180.0) % 360.0 - 180.0)
        angularNoise2 = np.random.normal(0.0, noise / np.sqrt(2), nAngles)
        x2Vec += angularNoise2
    for i in range(n):
        t = trials[i]
        target = targets[i]
        rot = rots[t]
        washout_i = is_washout[i]
        raw_center = target + mOut1
        idx = int(np.round(raw_center)) % 360
        center = raw_center % 360.0
        out1 = mOut1
        out2 = x2Vec[idx]
        out1_for_pred = 0.0 if washout_i else out1
        states[i] = out1_for_pred + out2
        centerIdx = idx
        out2 = x2Vec[centerIdx]
        pe = rot + out2
        angular_diff = np.minimum(np.abs(angles - center), 360.0 - np.abs(angles - center))
        g2Vec = np.exp(-(angular_diff**2) / (2 * sigma2**2))
        x2Vec = retain2 * x2Vec - lr2 * pe * g2Vec
        newOut2 = x2Vec[centerIdx]
        peRem = rot + out2 + out1
        delta1 = -peRem * lr1 if not washout_i else 0.0
        mOut1 = retain1 * mOut1 + delta1
        if noise > 0.0:
            angularNoise1 = np.random.normal(0.0, noise / np.sqrt(2))
            mOut1 += angularNoise1
            angularNoise2 = np.random.normal(0.0, noise / np.sqrt(2), nAngles)
            x2Vec += angularNoise2
        mOut1 = ((mOut1 + 180.0) % 360.0 - 180.0)
    return states

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
    targets = pDat['targetPosition'].values.astype(np.float64)
                                                           
    original_rotations = pDat['rotation'].values.astype(np.float64)
    target_positions = pDat['targetPosition'].values.astype(np.float64)
    is_washout = (pDat['phase'] == 'washout').values.astype(np.bool_)
   
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
    best_x = None
    total_evals = 0
    max_evals = 50000                  
    base_popsize = 4 + int(3 * np.log(n_params))
    norm_bounds = np.array([(0.0, 1.0) for _ in range(n_params)])
    

    n_large = 11
    pop_sequence = []
    for k in range(1, n_large + 1):
        large_pop = int(base_popsize * (1.5 ** k))
        pop_sequence.append(large_pop)
        pop_sequence.append(base_popsize)
    n_restarts = len(pop_sequence)
    
                                                            
    sampler = qmc.LatinHypercube(d=n_params, seed=SEED)
    base_samples = sampler.random(n=n_restarts)
    
    for i in range(n_restarts):
        if total_evals >= max_evals:
            pass      
        
        current_popsize = pop_sequence[i]
        
                                                      
        mean = base_samples[i] if i < len(base_samples) else np.random.uniform(0, 1, n_params)
      
        sigma = 0.3
        es = CMA(mean=mean, sigma=sigma, bounds=norm_bounds, population_size=current_popsize, seed=SEED + i)
        es.tolFun = 1e-7
      
        since_best = 0
        best_in_run = np.inf
        while not es.should_stop() and since_best < (200/np.log(current_popsize)):
            x_samples = [es.ask() for _ in range(es.population_size)]
            f_values = [localFitPp(x) for x in x_samples]               
            solutions = list(zip(x_samples, f_values))
            es.tell(solutions)
          
            current_best_idx = np.argmin(f_values)
            current_best = (x_samples[current_best_idx], f_values[current_best_idx])
            total_evals += current_popsize              
          
            improved_global = False
            if current_best[1] < best_fun:
                print(f"pp {pp}, restart {i}, evals {total_evals}, f={current_best[1]}")
                best_fun = current_best[1]
                best_x = current_best[0].copy()
                improved_global = True
          
            if current_best[1] < best_in_run:
                if current_best[1] < best_in_run * 0.9999: 
                    since_best = 0
                else:
                    since_best += 1
                best_in_run = current_best[1]
            else:
                since_best += 1
          
        print(f"pp {pp}, restart {i}, since_best {since_best}, total_evals {total_evals}, run_best {best_in_run}, global_best {best_fun}")
      
    best_p = transform(best_x, bounds)
    best_res = minimize(
        localFitPp_orig, best_p, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-9}
    )
    best_fun = best_res.fun
    best_x = transform(best_res.x, norm_bounds)
    res = best_res
    lr1, retain1, lr2, retain2, sigma2, noise = res.x
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
                               
        n = len(uniqP)
        self.bics = np.zeros(n)
        self.negLL = np.ones(n) * 100000
        self.rmses = np.full(n, np.nan)
        self.rSquareds = np.full(n, np.nan)
        self.mStates = [[] for _ in uniqP]
        self.mOut1 = [[] for _ in uniqP]
        self.mOut2 = [[] for _ in uniqP]
        self.allAims = [[] for _ in uniqP]
        self.allImps = [[] for _ in uniqP]
        self.xs = []
        self.errors = []
        self.rmses_dist = np.full(n, np.nan)
        self.rSquareds_dist = np.full(n, np.nan)
                        
        self.participants = []
        self.targetPositions = []
        self.rotations = []
        self.is_washouts = []
        bounds = [(0,1), (.5,1.), (0.0001,.5), (.95,1.), (1.,90.), (1,100.)]
        initial = [.8, 0.99, 0.005, 0.99, 22., 5.]
        argsList = [(pp, i, self.dat, self.fitPhase, self.imp, self.flipRot, bounds, initial)
                    for i, pp in enumerate(uniqP)]
        with mp.Pool(processes=20) as pool:
            results = pool.map(_fitWorker, argsList)
        for i, res in enumerate(results):
            pp = uniqP[i]
                                                   
            self.participants.append(res['pp'])
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
               
    def genDat(self, params, rots, trials=np.arange(40), targets=None, is_washout=None):
        lr1, retain1, lr2, retain2, sigma2, noise = params
        if targets is None:
            targets = np.zeros_like(trials)
        return generateData(lr1, retain1, lr2, retain2, sigma2, noise, rots, trials, targets, is_washout)