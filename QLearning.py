

import numpy as np
import pandas as pd
import DataManipulation as DM
from optimparallel import minimize_parallel
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from scipy.stats import norm
import multiprocessing as mp
from numba import njit
import os
import time
from scipy.stats.qmc import LatinHypercube
from scipy.stats import qmc
from cmaes import CMA

SEED = 42
np.random.seed(SEED)

n_actions = 361                                         
n_mag = 181
action_centers = np.arange(-180, 181, dtype=np.float64)
action_mags = np.arange(0, 181, dtype=np.float64)

@njit(cache=True)
def angular_dist(x):
    return np.abs(((x + 180.0) % 360.0 - 180.0))

@njit(cache=True)
def signed_angular_dist(x):
    return ((x + 180.0) % 360.0 - 180.0)

@njit(cache=True)
def gaussianLogPdf(x, sigma):
    if sigma <= 0.0:
        return np.full_like(x, -1e30)
    return -np.log(sigma * np.sqrt(2 * np.pi)) - (x ** 2) / (2 * sigma ** 2)

@njit(cache=True)
def scalar_gaussianLogPdf(x, sigma):
    if sigma <= 0.0:
        return -1e30
    return -np.log(sigma * np.sqrt(2 * np.pi)) - (x ** 2) / (2 * sigma ** 2)

@njit(cache=True)
def safe_exp(z):
    threshold = -700.0
    below = z < threshold
    result = np.zeros_like(z)
    result[~below] = np.exp(z[~below])
    return result

@njit(cache=True)
def logsumexp(arr):
    if arr.size == 0:
        return -1e30
    m = np.max(arr)
    s = np.sum(safe_exp(arr - m))
    if s == 0:
        return m - 1e10
    return m + np.log(s)

@njit(cache=True)
def get_action_index(action):
    rounded = round(action)
    shifted = rounded + 180
    if shifted < 0:
        shifted = 0
    elif shifted > n_actions - 1:
        shifted = n_actions - 1
    return int(shifted)

@njit(cache=True)
def computeNegLl(params, aims, rots):
    alpha = params[0]
    beta = params[1]
    noise = params[2]
    beta_dir = params[3]
    if (alpha <= 0.0 or beta <= 0.0 or beta_dir <= 0.0 or noise <= 0.0):
        return 1e100
    logLikelihood = 0.0
    n = len(aims)
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)
    for t in range(n):
        rot = rots[t]
        aim = aims[t]
                                                          
        max_q_mag = np.max(Q_mag)
        exp_q_mag = safe_exp(beta * (Q_mag - max_q_mag))
        sum_exp_mag = np.sum(exp_q_mag)
        probs_mag = exp_q_mag / sum_exp_mag if sum_exp_mag > 0 else np.full(n_mag, 1.0 / n_mag)
                                                           
        max_q_dir = np.max(Q_dir)
        exp_q_dir = safe_exp(beta_dir * (Q_dir - max_q_dir))
        sum_exp_dir = np.sum(exp_q_dir)
        probs_dir = exp_q_dir / sum_exp_dir if sum_exp_dir > 0 else np.array([0.5, 0.5])
                      
        m_mag = np.dot(probs_mag, action_mags)
        mOut = probs_dir[0] * m_mag + probs_dir[1] * (-m_mag)
        has_aim = not np.isnan(aim)
        if has_aim:
                                          
            unnorm1 = np.full(n_actions, -1e30)
            for dirr in range(2):
                p_d = probs_dir[dirr]
                signn = 1.0 if dirr == 0 else -1.0
                for k in range(n_mag):
                    act = signn * action_mags[k]
                    angular_error1 = signed_angular_dist(aim - act)
                    g_log1 = scalar_gaussianLogPdf(angular_error1, noise)
                    log_p_action = np.log(p_d + 1e-300) + np.log(probs_mag[k] + 1e-300)
                    j = get_action_index(act)
                    unnorm1[j] = log_p_action + g_log1
            log_em1 = logsumexp(unnorm1)
            logLikelihood += log_em1
                                                                     
        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_pos:
                max_r_pos = r_k
        td_e_pos = max_r_pos - Q_dir[0]
        Q_dir[0] += alpha * td_e_pos

        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_neg:
                max_r_neg = r_k
        td_e_neg = max_r_neg - Q_dir[1]
        Q_dir[1] += alpha * td_e_neg

                                                                                           
        for k in range(n_mag):
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot)
            r_pos = -err_pos
            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot)
            r_neg = -err_neg
            max_r_mag = max(r_pos, r_neg)
            td_e = max_r_mag - Q_mag[k]
            Q_mag[k] += alpha * td_e
    return -logLikelihood


def _fitWorker(pp, i, dat, fitPhase, flipRot, bounds, initial, n_starts=500):
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

    def transform(u, bounds):
        return np.array([b[0] + u[j] * (b[1] - b[0]) for j, b in enumerate(bounds)])

    def localFitPp(u):
        p = transform(u, bounds)
        return computeNegLl(p, aims, rots)

    def localFitPp_orig(p):
        return computeNegLl(p, aims, rots)

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
                                          
    alpha, beta, noise, beta_dir = best_res.x
    totErr = []
    sts = []
    log_liks = []
    model_predictive_policies = []
    model_predictive_dirs = []
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)
    n = len(aims)
    for t in range(n):
        rot = rots[t]
        aim = aims[t]
        max_q_mag = np.max(Q_mag)
        exp_q_mag = safe_exp(beta * (Q_mag - max_q_mag))
        sum_exp_mag = np.sum(exp_q_mag)
        probs_mag = exp_q_mag / sum_exp_mag if sum_exp_mag > 0 else np.full(n_mag, 1.0 / n_mag)
        max_q_dir = np.max(Q_dir)
        exp_q_dir = safe_exp(beta_dir * (Q_dir - max_q_dir))
        sum_exp_dir = np.sum(exp_q_dir)
        probs_dir = exp_q_dir / sum_exp_dir if sum_exp_dir > 0 else np.array([0.5, 0.5])
        model_predictive_dirs.append(probs_dir.copy())
        m_mag = np.dot(probs_mag, action_mags)
        mOut = probs_dir[0] * m_mag + probs_dir[1] * (-m_mag)
        sts.append(mOut)

        full_policy = np.zeros(n_actions)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                j = get_action_index(act)
                full_policy[j] += p_d * probs_mag[k]
        model_predictive_policies.append(full_policy)

        has_aim = not np.isnan(aim)
        if has_aim:
            unnorm1 = np.full(n_actions, -1e30)
            for dirr in range(2):
                p_d = probs_dir[dirr]
                signn = 1.0 if dirr == 0 else -1.0
                for k in range(n_mag):
                    act = signn * action_mags[k]
                    angular_error1 = signed_angular_dist(aim - act)
                    g_log1 = scalar_gaussianLogPdf(angular_error1, noise)
                    log_p_action = np.log(p_d + 1e-300) + np.log(probs_mag[k] + 1e-300)
                    j = get_action_index(act)
                    unnorm1[j] = log_p_action + g_log1
            log_em1 = logsumexp(unnorm1)
            log_liks.append(log_em1)
            angularError = signed_angular_dist(aim - mOut)
            totErr.append(angularError)

                 
        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_pos:
                max_r_pos = r_k
        td_e_pos = max_r_pos - Q_dir[0]
        Q_dir[0] += alpha * td_e_pos

        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]
            err = angular_dist(act + rot)
            r_k = -err
            if r_k > max_r_neg:
                max_r_neg = r_k
        td_e_neg = max_r_neg - Q_dir[1]
        Q_dir[1] += alpha * td_e_neg

        for k in range(n_mag):
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot)
            r_pos = -err_pos
            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot)
            r_neg = -err_neg
            max_r_mag = max(r_pos, r_neg)
            td_e = max_r_mag - Q_mag[k]
            Q_mag[k] += alpha * td_e

    numSamp = len(totErr)
    logLikelihood = np.sum(log_liks) if numSamp > 0 else 0.0
    k = 4
    bic = k * np.log(numSamp) - 2 * logLikelihood if numSamp > 0 else 0
    totErr = np.array(totErr)
    rmse = np.sqrt(np.mean(totErr**2)) if numSamp > 0 else 0
    ssRes = np.sum(totErr**2) if numSamp > 0 else 0
    observed = np.array([aims[t] for t in range(len(aims)) if not np.isnan(aims[t])])
    rSquared = 0
    if len(observed) > 0:
        meanObs = np.mean(observed)
        signed_obs = np.array([signed_angular_dist(o - meanObs) for o in observed])
        ssTot = np.sum(signed_obs**2)
        rSquared = 1 - (ssRes / ssTot) if ssTot != 0 else 0

    return {
        'x': best_res.x.tolist(),
        'fun': best_res.fun,
        'errors': totErr,
        'mStates': sts,
        'allAims': aims,
        'model_predictive_policies': model_predictive_policies,
        'model_predictive_dirs': model_predictive_dirs,
        'bic': bic,
        'rmse': rmse,
        'rSquared': rSquared
    }

class FitShell:
    def __init__(self, df='none', conVal='none', condition='none', fitLength=320, fitPhase='rotation',
                 rmse=False, flipRot=False, method='de', n_starts=500):
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
        self.errors = []
        self.indiErrors = []
        self.rmse = rmse
        self.flipRot = flipRot
        self.method = method
        self.model_predictive_policies = [[]]
        self.model_predictive_dirs = [[]]
        self.n_starts = n_starts

    def fitRot(self, lrUb=1):
        self.dat = self.df

        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]

        uniqP = self.dat['participantNum'].unique()
        self.participantNums = uniqP

        self.bics = np.zeros(len(uniqP))
        self.negLl = np.ones(len(uniqP)) * 100000
        self.mStates = [[] for _ in uniqP]
        self.allAims = [[] for _ in uniqP]
        self.model_predictive_policies = [[] for _ in uniqP]
        self.model_predictive_dirs = [[] for _ in uniqP]
        self.xs = []
        self.errors = []
        self.rmses = np.zeros(len(uniqP))
        self.rSquareds = np.zeros(len(uniqP))

        bounds = [(1e-4, 1.), (0.01, 20.), (1., 100.), (1e-300, 20)]
        initial = [0.1, 10.0, 3., 10.0]

        argsList = [(pp, i, self.dat, self.fitPhase, self.flipRot, bounds, initial, self.n_starts) for i, pp in enumerate(uniqP)]

        print(f"Starting fitting for {len(uniqP)} participants")
        start_time = time.time()
        with mp.Pool(processes=10) as pool:
            results = pool.starmap(_fitWorker, argsList)
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed / 60:.1f} minutes.")

        for i, res in enumerate(results):
            self.xs.append(res['x'])
            self.negLl[i] = res['fun']
            self.errors.append(res['errors'])
            self.bics[i] = res['bic']
            self.mStates[i] = res['mStates']
            self.allAims[i] = res['allAims']
            self.model_predictive_policies[i] = res['model_predictive_policies']
            self.model_predictive_dirs[i] = res['model_predictive_dirs']
            self.rmses[i] = res['rmse']
            self.rSquareds[i] = res['rSquared']

    def plot_all(self):
        os.makedirs("tempFigures", exist_ok=True)
        for local_i in range(len(self.participantNums)):
            plt.figure(figsize=(10, 6))
            plt.plot(self.mStates[local_i], label='Model explicit')
            plt.plot(self.allAims[local_i], label='Human aim')
            plt.legend()
            plt.title(f'Participant {self.participantNums[local_i]} - Rotation {self.conVal}')
            plt.savefig(f'tempFigures/{self.conVal}{local_i}.png', dpi=100)
            plt.close()

    def fitPp(self, params):
        if self.fitPhase is not None:
            pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        else:
            pDat = self.dat[(self.dat['participantNum'] == self.pp)]
        bNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == bNums[0]]
        aims = pDat['aim'].values.astype(np.float64)
        rots = pDat['rotation'].values.astype(np.float64)
        if self.flipRot:
            rots = -rots
        return computeNegLl(params, aims, rots)



