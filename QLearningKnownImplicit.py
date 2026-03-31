import numpy as np
import pandas as pd
import DataManipulation as DM
from optimparallel import minimize_parallel
from scipy.optimize import minimize, differential_evolution
import matplotlib
matplotlib.use('Agg')
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
n_implicit = 360
action_centers = np.arange(-180, 181, dtype=np.float64)
action_mags = np.arange(0, 181, dtype=np.float64)
angles = np.arange(n_implicit, dtype=np.float64)
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
    """Clamp to avoid denorm underflow; error <1e-300."""
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
def computeNegLl(params, aims, rots, imps, targetPositions, use_implicit=False):
    alpha = params[0]
    beta = params[1]
    noise = params[2]
    beta_dir = params[3]
    a = params[4] if use_implicit else 0.0
    b = params[5] if use_implicit else 0.0
    sigma_gen = params[6] if use_implicit else 0.0
    if (alpha <= 0.0 or beta <= 0.0 or beta_dir <= 0.0 or noise <= 0.0 or
        (use_implicit and (a <= 0.0 or b <= 0.0 or sigma_gen <= 0.0))):
        return 1e100
    logLikelihood = 0.0
    n = len(aims)
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)
    xVec = np.zeros(n_implicit)
    for t in range(n):
        rot = rots[t]
        aim = aims[t]
        imp_val = imps[t]
        target_pos = targetPositions[t]
                                                          
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
        forced_mOut = 0.0 if rot == 0.0 else mOut
                                                   
        planned_angle = np.mod(forced_mOut + 180.0, 360.0)
        implicit_comp = 0.0
        if use_implicit:
            effective_angle = np.mod(planned_angle + target_pos, 360.0)
            idxMu = int(np.round(effective_angle)) % n_implicit
            implicit_comp = xVec[idxMu]
                                    
        compensation_error = signed_angular_dist(forced_mOut + rot + implicit_comp)
        has_aim = not np.isnan(aim)
        has_imp = not np.isnan(imp_val)
        has_data = has_aim if not use_implicit else (has_aim and has_imp)
        if has_data:
            target_t = aim if not use_implicit else (aim + imp_val)
            if rot == 0.0:
                angular_error_forced = signed_angular_dist(target_t - (0.0 + implicit_comp))
                log_mix = scalar_gaussianLogPdf(angular_error_forced, noise)
            else:
                unnorm1 = np.full(n_actions, -1e30)
                for dirr in range(2):
                    p_d = probs_dir[dirr]
                    signn = 1.0 if dirr == 0 else -1.0
                    for k in range(n_mag):
                        act = signn * action_mags[k]
                        angular_error1 = signed_angular_dist(target_t - (act + implicit_comp))
                        g_log1 = scalar_gaussianLogPdf(angular_error1, noise)
                        log_p_action = np.log(p_d + 1e-300) + np.log(probs_mag[k] + 1e-300)
                        j = get_action_index(act)
                        unnorm1[j] = log_p_action + g_log1
                log_mix = logsumexp(unnorm1)
            logLikelihood += log_mix
                                                                     
        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]           
            err = angular_dist(act + rot + implicit_comp)
            r_k = -err
            if r_k > max_r_pos:
                max_r_pos = r_k
        td_e_pos = max_r_pos - Q_dir[0]
        Q_dir[0] += alpha * td_e_pos
                              
        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]           
            err = angular_dist(act + rot + implicit_comp)
            r_k = -err
            if r_k > max_r_neg:
                max_r_neg = r_k
        td_e_neg = max_r_neg - Q_dir[1]
        Q_dir[1] += alpha * td_e_neg
                                                                                           
        for k in range(n_mag):
                                                  
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot + implicit_comp)
            r_pos = -err_pos
            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot + implicit_comp)
            r_neg = -err_neg
                                                  
            max_r_mag = max(r_pos, r_neg)
            td_e = max_r_mag - Q_mag[k]
            Q_mag[k] += alpha * td_e
                                                                       
        if use_implicit:
            eT = rot + implicit_comp
            effective_angle = np.mod(planned_angle + target_pos, 360.0)
            d = np.empty(n_implicit)
            for ii in range(n_implicit):
                diff = angles[ii] - effective_angle
                d[ii] = signed_angular_dist(diff)
            close_mask = np.abs(d) < (4.0 * sigma_gen)
            g = np.zeros(n_implicit)
            for ii in range(n_implicit):
                if close_mask[ii]:
                    g[ii] = np.exp( - (d[ii]**2) / (2.0 * sigma_gen **2) )
            for ii in range(n_implicit):
                xVec[ii] = a * xVec[ii] - b * g[ii] * eT
                                                                   
        prev_error = compensation_error
    return -logLikelihood


def _fitWorker(args):
    pp, i, dat, fitPhase, imp, flipRot, bounds, initial, a_fixed, b_fixed, sigma_gen_fixed = args
    if fitPhase is not None:
        pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    else:
        pDat = dat[(dat['participantNum'] == pp)]
    bNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == bNums[0]]
    aims = pDat['aim'].values.astype(np.float64)
    rots = pDat['rotation'].values.astype(np.float64)
    targetPositions = pDat['targetPosition'].values.astype(np.float64)
    if flipRot:
        rots = -rots
    if imp:
        imps = pDat['imp'].values.astype(np.float64)
    else:
        imps = np.full_like(aims, np.nan)
    def transform(u, bounds):
        return np.array([b[0] + u[j] * (b[1] - b[0]) for j, b in enumerate(bounds)])
    def localFitPp(u):
        p = transform(u, bounds)
        if imp:
            params_full = np.append(p, [a_fixed, b_fixed, sigma_gen_fixed])
            return computeNegLl(params_full, aims, rots, imps, targetPositions, imp)
        else:
            return computeNegLl(p, aims, rots, imps, targetPositions, imp)
    def localFitPp_orig(p):
        if imp:
            params_full = np.append(p, [a_fixed, b_fixed, sigma_gen_fixed])
            return computeNegLl(params_full, aims, rots, imps, targetPositions, imp)
        else:
            return computeNegLl(p, aims, rots, imps, targetPositions, imp)
    n_params = len(bounds)
    best_fun = np.inf
    n_starts = 500                                                              
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
                              
    alpha = best_res.x[0]
    beta = best_res.x[1]
    noise = best_res.x[2]
    beta_dir = best_res.x[3]
    a = a_fixed if imp else 0.0
    b = b_fixed if imp else 0.0
    sigma_gen = sigma_gen_fixed if imp else 0.0
    totErr = []
    sts = []
    log_liks = []
    observed = []
    model_explicits = []
    model_implicits = []
    human_explicits = []
    human_implicits = []
    all_model_explicits = []
    all_model_implicits = []
    all_mstates = []
    valid_trials = []
    model_predictive_policies = []
    model_predictive_dirs = []
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)
    xVec = np.zeros(n_implicit)
    n = len(aims)
    prev_error = 0.0
    for t in range(n):
        rot = rots[t]
        aim = aims[t]
        impVal = imps[t]
        target_pos = targetPositions[t]
                                                          
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
        forced_mOut = 0.0 if rot == 0.0 else mOut
                                  
        full_policy = np.zeros(n_actions)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                j = get_action_index(act)
                full_policy[j] += p_d * probs_mag[k]
        model_predictive_policies.append(full_policy)
                                                   
        planned_angle = np.mod(forced_mOut + 180.0, 360.0)
        implicit_comp = 0.0
        if imp:
            effective_angle = np.mod(planned_angle + target_pos, 360.0)
            idxMu = int(np.round(effective_angle)) % n_implicit
            implicit_comp = xVec[idxMu]
                                        
        all_model_explicits.append(forced_mOut)
        all_model_implicits.append(implicit_comp)
        all_mstates.append(forced_mOut + implicit_comp)
                                    
        compensation_error = signed_angular_dist(forced_mOut + rot + implicit_comp)
        has_aim = not np.isnan(aim)
        has_imp = not np.isnan(impVal)
        has_data = has_aim if not imp else (has_aim and has_imp)
        if has_data:
            valid_trials.append(t)
            human_explicit = aim
            if imp:
                human_implicit = impVal
                target_t = aim + impVal
            else:
                human_implicit = 0.0
                target_t = aim
            model_explicit = forced_mOut
            if imp:
                model_implicit = implicit_comp
                total_mOut = forced_mOut + implicit_comp
            else:
                model_implicit = 0.0
                total_mOut = forced_mOut
   
            observed.append(target_t)
            sts.append(total_mOut)
            if rot == 0.0:
                diff = signed_angular_dist(target_t - (0.0 + implicit_comp))
                log_mix = scalar_gaussianLogPdf(diff, noise)
            else:
                unnorm1 = np.full(n_actions, -1e30)
                for dirr in range(2):
                    p_d = probs_dir[dirr]
                    signn = 1.0 if dirr == 0 else -1.0
                    for k in range(n_mag):
                        act = signn * action_mags[k]
                        angular_error1 = signed_angular_dist(target_t - (act + implicit_comp))
                        g_log1 = scalar_gaussianLogPdf(angular_error1, noise)
                        log_p_action = np.log(p_d + 1e-300) + np.log(probs_mag[k] + 1e-300)
                        j = get_action_index(act)
                        unnorm1[j] = log_p_action + g_log1
                log_mix = logsumexp(unnorm1)
            log_liks.append(log_mix)
            angularError = signed_angular_dist(target_t - total_mOut)
            totErr.append(angularError)
        else:
            human_explicit = np.nan
            human_implicit = np.nan
        human_explicits.append(human_explicit)
        human_implicits.append(human_implicit)
        model_explicits.append(forced_mOut)
        model_implicits.append(implicit_comp)
                                                                     
        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]           
            err = angular_dist(act + rot + implicit_comp)
            r_k = -err
            if r_k > max_r_pos:
                max_r_pos = r_k
        td_e_pos = max_r_pos - Q_dir[0]
        Q_dir[0] += alpha * td_e_pos
                              
        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]           
            err = angular_dist(act + rot + implicit_comp)
            r_k = -err
            if r_k > max_r_neg:
                max_r_neg = r_k
        td_e_neg = max_r_neg - Q_dir[1]
        Q_dir[1] += alpha * td_e_neg
                                                                                           
        for k in range(n_mag):
                                                  
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot + implicit_comp)
            r_pos = -err_pos
            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot + implicit_comp)
            r_neg = -err_neg
                                                  
            max_r_mag = max(r_pos, r_neg)
            td_e = max_r_mag - Q_mag[k]
            Q_mag[k] += alpha * td_e
                                                              
        if imp:
            eT = rot + implicit_comp
            effective_angle = np.mod(planned_angle + target_pos, 360.0)
            d = np.zeros(n_implicit)
            for ii in range(n_implicit):
                diff = angles[ii] - effective_angle
                d[ii] = signed_angular_dist(diff)
            close_mask = np.abs(d) < (4.0 * sigma_gen)
            g = np.zeros(n_implicit)
            for ii in range(n_implicit):
                if close_mask[ii]:
                    g[ii] = np.exp( - (d[ii]**2) / (2.0 * sigma_gen **2) )
            for ii in range(n_implicit):
                xVec[ii] = a * xVec[ii] - b * g[ii] * eT
                           
        prev_error = compensation_error
    numSamp = len(log_liks)
    logLikelihood = np.sum(log_liks) if numSamp > 0 else 0.0
    k = 4
    bic = k * np.log(numSamp) - 2 * logLikelihood if numSamp > 0 else 0
    totErr = np.array(totErr)
    rmse = np.sqrt(np.mean(totErr**2)) if len(totErr) > 0 else 0
    ssRes = np.sum(totErr**2)
    validObs = np.array(observed)
    if len(validObs) > 0:
        meanObs = np.mean(validObs)
        signed_obs = signed_angular_dist(validObs - meanObs)
        ssTot = np.sum(signed_obs**2)
        rSquared = 1 - (ssRes / ssTot) if ssTot != 0 else 0
    else:
        rSquared = 0
    return {
        'x': best_res.x,
        'fun': best_res.fun,
        'errors': totErr,
        'mStates': sts,
        'allAims': aims,
        'model_explicits': model_explicits,
        'model_implicits': model_implicits,
        'all_model_explicits': all_model_explicits,
        'all_model_implicits': all_model_implicits,
        'all_mstates': all_mstates,
        'valid_trials': valid_trials,
        'human_explicits': aims,
        'human_implicits': imps,
        'model_predictive_policies': model_predictive_policies,
        'model_predictive_dirs': model_predictive_dirs,
        'bic': bic,
        'rmse': rmse,
        'rSquared': rSquared
    }
                                                                           
                 
                                                                           
def _plotWorker(args):
    (i, all_model_explicits, all_model_implicits, all_mstates, human_explicits, human_implicits, valid_trials, conVal, participantNums_i, temp_dir) = args
    import matplotlib.pyplot as plt
    n_trials = len(all_model_explicits)
    fig, ax = plt.subplots(figsize=(10, 6))
                           
    ax.plot(range(n_trials), all_model_explicits, 'b-', label='Model Explicit', alpha=0.7)
    ax.plot(range(n_trials), all_model_implicits, 'g-', label='Model Implicit', alpha=0.7)
    ax.plot(range(n_trials), all_mstates, 'm-', label='Model Total', alpha=0.7)
                                                                     
    human_explicits_arr = np.array(human_explicits)
    human_implicits_arr = np.array(human_implicits)
    human_total = human_explicits_arr + human_implicits_arr
    ax.scatter(range(n_trials), human_explicits_arr, color='red', marker='o', s=20, label='Human Explicit', alpha=0.8)
    ax.scatter(range(n_trials), human_total, color='orange', marker='s', s=20, label='Human Total', alpha=0.8)
                                      
    ax.scatter(range(n_trials), human_implicits_arr, color='purple', marker='^', s=20, label='Human Implicit', alpha=0.8)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
                     
    plt.savefig(f'{temp_dir}/{conVal}{participantNums_i}_{i}', dpi=100)
    plt.close()
class FitShell:
    def __init__(self, df='none', conVal='none', condition='none', fitLength=320, fitPhase='rotation', imp=True,
                 rmse=False, flipRot=False, method='de', a=0.95, b=0.05, sigma_gen=5.0):
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
        self.a = a
        self.b = b
        self.sigma_gen = sigma_gen
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
        self.bics = np.zeros(len(uniqP))
        self.negLl = np.ones(len(uniqP))*100000
        self.mStates = [[] for _ in uniqP]
        self.allAims = [[] for _ in uniqP]
        self.model_explicits = [[] for _ in uniqP]
        self.model_implicits = [[] for _ in uniqP]
        self.all_model_explicits = [[] for _ in uniqP]
        self.all_model_implicits = [[] for _ in uniqP]
        self.all_mstates = [[] for _ in uniqP]
        self.valid_trials = [[] for _ in uniqP]
        self.human_explicits = [[] for _ in uniqP]
        self.human_implicits = [[] for _ in uniqP]
        self.model_predictive_policies = [[] for _ in uniqP]
        self.model_predictive_dirs = [[] for _ in uniqP]
        self.xs = []
        self.errors = []
        self.rmses = np.zeros(len(uniqP))
        self.rSquareds = np.zeros(len(uniqP))
        bounds = [(1e-4,1.), (0.01,20.), (1.,100.), (1e-300,20)]                                        
        initial = [0.1, 10.0, 3., 10.0]
        argsList = [(pp, i, self.dat, self.fitPhase, self.imp, self.flipRot, bounds, initial, self.a, self.b, self.sigma_gen) for i, pp in enumerate(uniqP)]
        print(f"Starting fitting for {len(uniqP)} participants.")
        os.makedirs("tempFigures", exist_ok=True)
        temp_dir = "tempFigures"
        start_time = time.time()
        with mp.Pool(processes=24) as pool:
            results = pool.map(_fitWorker, argsList)
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.1f} minutes.")
        for i, res in enumerate(results):
            self.xs.append(res['x'])
            self.negLl[i] = res['fun']
            self.errors.append(res['errors'])
            self.bics[i] = res['bic']
            self.mStates[i] = res['mStates']
            self.allAims[i] = res['allAims']
            self.model_explicits[i] = res['model_explicits']
            self.model_implicits[i] = res['model_implicits']
            self.all_model_explicits[i] = res['all_model_explicits']
            self.all_model_implicits[i] = res['all_model_implicits']
            self.all_mstates[i] = res['all_mstates']
            self.valid_trials[i] = res['valid_trials']
            self.human_explicits[i] = res['human_explicits']
            self.human_implicits[i] = res['human_implicits']
            self.model_predictive_policies[i] = res['model_predictive_policies']
            self.model_predictive_dirs[i] = res['model_predictive_dirs']
            self.rmses[i] = res['rmse']
            self.rSquareds[i] = res['rSquared']
                           
        plot_args_list = [
            (
                i,
                self.all_model_explicits[i],
                self.all_model_implicits[i],
                self.all_mstates[i],
                self.human_explicits[i],
                self.human_implicits[i],
                self.valid_trials[i],
                self.conVal,
                self.participantNums[i],
                temp_dir
            )
            for i in range(len(self.participantNums))
        ]
        print("Starting parallel plotting...")
        plot_start = time.time()
        with mp.Pool(processes=20) as plot_pool:
            plot_pool.map(_plotWorker, plot_args_list)
        plot_elapsed = time.time() - plot_start
        print(f"Plotting completed in {plot_elapsed/60:.1f} minutes.")
    def fitPp(self, params):
        if self.fitPhase != None:
            pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        else:
            pDat = self.dat[(self.dat['participantNum'] == self.pp)]
        bNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == bNums[0]]
        aims = pDat['aim'].values.astype(np.float64)
        rots = pDat['rotation'].values.astype(np.float64)
        targetPositions = pDat['targetPosition'].values.astype(np.float64)
        if self.flipRot:
            rots = -rots
        if self.imp:
            imps = pDat['imp'].values.astype(np.float64)
        else:
            imps = np.full_like(aims, np.nan)
        if self.imp:
            params = np.append(params, [self.a, self.b, self.sigma_gen])
        return computeNegLl(params, aims, rots, imps, targetPositions, self.imp)
