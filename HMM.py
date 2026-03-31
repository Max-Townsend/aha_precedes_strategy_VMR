import numpy as np
import pandas as pd
import DataManipulation as DM
from optimparallel import minimize_parallel
from scipy.optimize import minimize, differential_evolution
from scipy.special import logit
import matplotlib
matplotlib.use('Agg')                                              
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from scipy.stats import qmc
from cmaes import CMA
SEED = 42
np.random.seed(SEED)
import multiprocessing as mp
from multiprocessing import Manager
from numba import njit
import math
import time
from types import SimpleNamespace
import os
n_actions = 361
action_centers = np.arange(-180, 181, dtype=np.float64)
n_mag = 181
action_mags = np.arange(0, 181, dtype=np.float64)
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
def softmax(log_probs):
    max_log_p = np.max(log_probs)
    exp_p = safe_exp(log_probs - max_log_p)
    sum_p = np.sum(exp_p)
    if sum_p > 0:
        return exp_p / sum_p
    else:
        return np.array([0.5, 0.5])
                                                                           
                                                                                    
                                                                             
                                                                                             
                                                                                                
                                                                                  
                                                                           
@njit(cache=True)
def computeNegLl(params, aims, rots, targetPositions, annealing_mode=0):                                     
    noise = params[0]
    kappa = params[1]
    noise_context = params[2]
    alpha = params[3]                           
    beta = params[4]                                          
    beta_dir = params[5]                                     
    if (noise <= 0.0 or kappa < 0.0 or kappa >= 1.0 or noise_context <= 0.0
        or alpha <= 0.0 or beta <= 0.0 or beta_dir <= 0.0):
        return 1e100
    logLikelihood = 0.0
    n = len(aims)
                  
    si = 0.0
    pi = np.array([1.0,0])
    A = np.zeros((2, 2))
    A[0,0] = 1.0 - kappa; A[0,1] = kappa
    A[1,0] = kappa; A[1,1] = 1.0 - kappa
    log_A = np.log(A + 1e-301)
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)                                        
    prev_rot = 0.0
    for t in range(n):
                    
        rot = rots[t]
        aim = aims[t]
        target_pos = targetPositions[t]
        has_aim = not np.isnan(aim)
                            
        log_pi_pred = np.empty(2)
        for sp in range(2):
            log_trans = np.log(pi + 1e-300) + log_A[:, sp]
            log_pi_pred[sp] = logsumexp(log_trans)
        pi_pred = softmax(log_pi_pred)
                                                
        if annealing_mode == 1:                                 
            beta_t = beta * (t / (n - 1.0)) if n > 1 else beta
        elif annealing_mode == 2:                                
            beta_t = beta * pi_pred[1]
        else:                    
            beta_t = beta
                        
        mOut0 = 0.0
                                                          
        max_q_mag = np.max(Q_mag)
        exp_q_mag = safe_exp(beta_t * (Q_mag - max_q_mag))
        sum_exp_mag = np.sum(exp_q_mag)
        probs_mag = exp_q_mag / sum_exp_mag if sum_exp_mag > 0 else np.full(n_mag, 1.0 / n_mag)
                                                           
        max_q_dir = np.max(Q_dir)
        exp_q_dir = safe_exp(beta_dir * (Q_dir - max_q_dir))
        sum_exp_dir = np.sum(exp_q_dir)
        probs_dir = exp_q_dir / sum_exp_dir if sum_exp_dir > 0 else np.array([0.5, 0.5])
                                                            
                                                                    
        m_mag = np.dot(probs_mag, action_mags)
        mOut1 = probs_dir[0] * m_mag + probs_dir[1] * (-m_mag)
                      
        mean_mOut = pi_pred[0] * mOut0 + pi_pred[1] * mOut1
        full_comp = signed_angular_dist(rot)
                                               
        has_data = has_aim
        if has_data:
            target_t = aim
            diff0 = signed_angular_dist(target_t - mOut0)
            log_em0 = scalar_gaussianLogPdf(diff0, noise)
                                                   
            unnorm1 = np.full(n_actions, -1e30)
            for dirr in range(2):
                p_d = probs_dir[dirr]
                p_mags = probs_mag
                signn = 1.0 if dirr == 0 else -1.0
                for k in range(n_mag):
                    act = signn * action_mags[k]
                    angular_error1 = signed_angular_dist(target_t - act)
                    g_log1 = scalar_gaussianLogPdf(angular_error1, noise)
                    log_p_action = np.log(p_d + 1e-300) + np.log(p_mags[k] + 1e-300)
                    j = get_action_index(act)
                    unnorm1[j] = log_p_action + g_log1
            log_em1 = logsumexp(unnorm1)
            log_marg = logsumexp(log_pi_pred + np.array([log_em0, log_em1]))
            logLikelihood += log_marg
                                      
        diff_context0 = signed_angular_dist(mOut0 + full_comp)
        log_em_context0 = scalar_gaussianLogPdf(diff_context0, noise_context)
                                               
        unnorm_context1 = np.full(n_actions, -1e30)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            p_mags = probs_mag
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                diff_context1 = signed_angular_dist(act + full_comp)
                g_log_context1 = scalar_gaussianLogPdf(diff_context1, noise_context)
                log_p_action = np.log(p_d + 1e-300) + np.log(p_mags[k] + 1e-300)
                j = get_action_index(act)
                unnorm_context1[j] = log_p_action + g_log_context1
        log_em_context1 = logsumexp(unnorm_context1)
        log_alpha = log_pi_pred + np.array([log_em_context0, log_em_context1])
        pi = softmax(log_alpha)
                                                                     
                              
        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]           
            err = angular_dist(act + rot)
            r_k = -err    
            if r_k > max_r_pos:
                max_r_pos = r_k
        td_e_pos = max_r_pos - Q_dir[0]
        Q_dir[0] += alpha * pi[1] * td_e_pos
                              
        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]           
            err = angular_dist(act + rot)
            r_k = -err    
            if r_k > max_r_neg:
                max_r_neg = r_k
        td_e_neg = max_r_neg - Q_dir[1]
        Q_dir[1] += alpha * pi[1] * td_e_neg
                                                                                           
        for k in range(n_mag):
                                                  
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot)
            r_pos = -err_pos    
      
            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot)
            r_neg = -err_neg    
      
                                                  
            max_r_mag = max(r_pos, r_neg)
            td_e = max_r_mag - Q_mag[k]
      
            Q_mag[k] += alpha * pi[1] * td_e
        prev_rot = rot
    return -logLikelihood

                                                                           
                                            
                                                          
                                                                           
def _fitWorker(pp, i, dat, fitPhase, flipRot, bounds, initial, annealing_mode,n_starts=500):
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
    def transform(u, bounds):
        return np.array([b[0] + u[j] * (b[1] - b[0]) for j, b in enumerate(bounds)])
    def localFitPp(u):
        p = transform(u, bounds)
        return computeNegLl(p, aims, rots, targetPositions, annealing_mode)
    def localFitPp_orig(p):
        return computeNegLl(p, aims, rots, targetPositions, annealing_mode)
    n_params = len(bounds)
    best_fun = np.inf
    sampler = qmc.LatinHypercube(d=n_params, seed=SEED)
    norm_samples = sampler.random(n=n_starts)
    
    best_fun = np.inf
    best_p = None
    candidates = [] 
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    buffer = 1e-300
    
    for i in range(n_starts):
        x0 = transform(norm_samples[i], bounds)
        x0 = np.maximum(np.minimum(x0, ub - buffer), lb + buffer)
        try:
            res = minimize(
                localFitPp_orig, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-5, 'gtol': 1e-4}
            )
        except ValueError as e:
            if "bound constraints" in str(e):
                continue
            else:
                raise
        if res.fun < best_fun:
            best_res = res
            best_fun = res.fun
            best_p = res.x.copy()
            print(f"{pp} Quick best: start {i}, f={best_fun}")
        
                                               
        candidates.append((res.fun, res.x))
        if len(candidates) > 20:
            candidates = sorted(candidates, key=lambda c: c[0])[:20]
                                    
    for fun, x in sorted(candidates, key=lambda c: c[0])[:5]:
        x = np.maximum(np.minimum(x, ub - buffer), lb + buffer)
        try:
            polished = minimize(
                localFitPp_orig, x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
        except ValueError as e:
            if "bound constraints" in str(e):
                continue
            else:
                raise
        if polished.fun < best_fun:
            best_res = polished
            best_fun = polished.fun
            best_p = polished.x
    best_fun = best_res.fun
    noise = best_res.x[0]
    kappa = best_res.x[1]
    noise_context = best_res.x[2]
    alpha = best_res.x[3]
    beta = best_res.x[4]
    beta_dir = best_res.x[5]
    totErr = []
    log_liks = []
    observed = []
    pi_preds = []
    human_explicits = []
    model_explicits_valid = []
    valid_trials = []
    expected_aims = []
    model_predictive_pis = []
    model_predictive_policies = []
    model_predictive_dirs = []
    all_model_explicits = []
    n = len(aims)
    pi = np.array([1.0,0])
    A = np.zeros((2,2))
    A[0,0] = 1.0 - kappa; A[0,1] = kappa
    A[1,0] = kappa; A[1,1] = 1.0 - kappa
    log_A = np.log(A + 1e-301)
    Q_mag = np.zeros(n_mag)
    Q_dir = np.zeros(2)                                        
    prev_rot = 0.0
    for t in range(n):
                            
        log_pi_pred = np.empty(2)
        for sp in range(2):
            log_trans = np.log(pi + 1e-300) + log_A[:, sp]
            log_pi_pred[sp] = logsumexp(log_trans)
        rot = rots[t]
        pi_pred = softmax(log_pi_pred)
        pi_preds.append(pi_pred.copy())
        model_predictive_pis.append(pi_pred.copy())
                                                
        if annealing_mode == 1:          
            beta_t = beta * (t / (n - 1.0)) if n > 1 else beta
        elif annealing_mode == 2:             
            beta_t = beta * pi_pred[1]
        else:
            beta_t = beta
                                                          
        max_q_mag = np.max(Q_mag)
        exp_q_mag = safe_exp(beta_t * (Q_mag - max_q_mag))
        sum_exp_mag = np.sum(exp_q_mag)
        probs_mag = exp_q_mag / sum_exp_mag if sum_exp_mag > 0 else np.full(n_mag, 1.0 / n_mag)
                                                           
        max_q_dir = np.max(Q_dir)
        exp_q_dir = safe_exp(beta_dir * (Q_dir - max_q_dir))
        sum_exp_dir = np.sum(exp_q_dir)
        probs_dir = exp_q_dir / sum_exp_dir if sum_exp_dir > 0 else np.array([0.5, 0.5])
        model_predictive_dirs.append(probs_dir.copy())
                        
        mOut0 = 0.0
                                                            
        m_mag = np.dot(probs_mag, action_mags)
        mOut1 = probs_dir[0] * m_mag + probs_dir[1] * (-m_mag)
                                  
        full_policy = np.zeros(n_actions)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            p_mags = probs_mag
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                j = get_action_index(act)
                full_policy[j] += p_d * p_mags[k]
        model_predictive_policies.append(full_policy)
                      
        mean_mOut = pi_pred[0] * mOut0 + pi_pred[1] * mOut1
        mean_mOut = np.mod(mean_mOut + 180.0, 360.0) - 180.0
        all_model_explicits.append(mean_mOut)
                                 
        aim = aims[t]
        has_aim = not np.isnan(aim)
        has_data = has_aim
        if has_data:
            valid_trials.append(t)
            human_explicit = aim
            target_t = human_explicit
            diff0 = signed_angular_dist(target_t - mOut0)
            log_em0 = scalar_gaussianLogPdf(diff0, noise)
                                                   
            unnorm1 = np.full(n_actions, -1e30)
            for dirr in range(2):
                p_d = probs_dir[dirr]
                p_mags = probs_mag
                signn = 1.0 if dirr == 0 else -1.0
                for k in range(n_mag):
                    act = signn * action_mags[k]
                    angular_error1 = signed_angular_dist(target_t - act)
                    g_log1 = scalar_gaussianLogPdf(angular_error1, noise)
                    log_p_action = np.log(p_d + 1e-300) + np.log(p_mags[k] + 1e-300)
                    j = get_action_index(act)
                    unnorm1[j] = log_p_action + g_log1
            log_em1 = logsumexp(unnorm1)
            log_marg = logsumexp(log_pi_pred + np.array([log_em0, log_em1]))
            log_liks.append(log_marg)
            observed.append(target_t)
                                                                            
            pred_total = mean_mOut
            angularError = signed_angular_dist(target_t - pred_total)
            totErr.append(angularError)
            expected_aims.append(pred_total)
        else:
            human_explicit = np.nan
        model_explicits_valid.append(mean_mOut)
        human_explicits.append(human_explicit)
                        
        full_comp = signed_angular_dist(rot)
        diff_context0 = signed_angular_dist(mOut0 + full_comp)
        log_em_context0 = scalar_gaussianLogPdf(diff_context0, noise_context)
                                               
        unnorm_context1 = np.full(n_actions, -1e30)
        for dirr in range(2):
            p_d = probs_dir[dirr]
            p_mags = probs_mag
            signn = 1.0 if dirr == 0 else -1.0
            for k in range(n_mag):
                act = signn * action_mags[k]
                diff_context1 = signed_angular_dist(act + full_comp)
                g_log_context1 = scalar_gaussianLogPdf(diff_context1, noise_context)
                log_p_action = np.log(p_d + 1e-300) + np.log(p_mags[k] + 1e-300)
                j = get_action_index(act)
                unnorm_context1[j] = log_p_action + g_log_context1
        log_em_context1 = logsumexp(unnorm_context1)
        log_alpha = log_pi_pred + np.array([log_em_context0, log_em_context1])
        pi_post = softmax(log_alpha)
        pi = pi_post
                                                         
                              
        max_r_pos = -1e30
        for k in range(n_mag):
            act = action_mags[k]           
            err = angular_dist(act + rot)
            r_k = -err    
            if r_k > max_r_pos:
                max_r_pos = r_k
        td_e_pos = max_r_pos - Q_dir[0]
        Q_dir[0] += alpha * pi_post[1] * td_e_pos
   
                              
        max_r_neg = -1e30
        for k in range(n_mag):
            act = -action_mags[k]           
            err = angular_dist(act + rot)
            r_k = -err    
            if r_k > max_r_neg:
                max_r_neg = r_k
        td_e_neg = max_r_neg - Q_dir[1]
        Q_dir[1] += alpha * pi_post[1] * td_e_neg
   
                                
        for k in range(n_mag):
                                                  
            act_pos = action_mags[k]
            err_pos = angular_dist(act_pos + rot)
            r_pos = -err_pos    
        
            act_neg = -action_mags[k]
            err_neg = angular_dist(act_neg + rot)
            r_neg = -err_neg    
        
                                                  
            max_r_mag = max(r_pos, r_neg)
            td_e = max_r_mag - Q_mag[k]
        
            Q_mag[k] += alpha * pi_post[1] * td_e
        prev_rot = rot
                                      
    numSamp = len(totErr)
    logLikelihood = np.sum(log_liks) if numSamp > 0 else 0.0
    k = 6                    
    bic = k * np.log(numSamp) - 2 * logLikelihood if numSamp > 0 else 0
    totErr = np.array(totErr)
    rmse = np.sqrt(np.mean(totErr**2)) if numSamp > 0 else 0
    ssRes = np.sum(totErr**2) if numSamp > 0 else 0
    observed_arr = np.array(observed)
    rSquared = 0
    if len(observed_arr) > 0:
        meanObs = np.mean(observed_arr)
        signed_obs = np.array([signed_angular_dist(o - meanObs) for o in observed_arr])
        ssTot = np.sum(signed_obs**2)
        rSquared = 1 - (ssRes / ssTot) if ssTot != 0 else 0
    full_x = list(best_res.x)
    mStates = model_explicits_valid if numSamp > 0 else []
    return {
        'x': full_x,
        'fun': best_res.fun,
        'errors': totErr,
        'mStates': mStates,
        'allAims': aims,
        'pi_preds': pi_preds,
        'valid_trials': valid_trials,
        'model_explicits': model_explicits_valid,
        'all_model_explicits': all_model_explicits,
        'human_explicits': aims,
        'expected_aims': expected_aims,
        'model_predictive_pis': model_predictive_pis,
        'model_predictive_policies': model_predictive_policies,
        'model_predictive_dirs': model_predictive_dirs,
        'bic': bic,
        'rmse': rmse,
        'rSquared': rSquared
    }
                                                                           
                 
                                                                           
def _plotWorker(plot_args):
    i, pp, x, pi_preds, model_predictive_pis, model_predictive_policies, model_predictive_dirs, human_explicits = plot_args
    noise = x[0]
    n_trials = len(pi_preds)
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))                         
                                 
    if pi_preds:
        pi_preds_arr = np.array(pi_preds).T
        im = axs[0].imshow(pi_preds_arr, aspect='auto', cmap='viridis', extent=[0, n_trials, 0, 2])
        axs[0].set_xlabel('Trial')
        axs[0].set_ylabel('State (0: Naive, 1: Compensate)')
        axs[0].set_title('Model State Probabilities')
        plt.colorbar(im, ax=axs[0], label='Probability')
                                         
    if model_predictive_dirs:
        dirs_arr = np.array(model_predictive_dirs)[:, 1]               
        axs[1].plot(np.arange(n_trials), dirs_arr, color='blue', linewidth=2)
        axs[1].set_xlabel('Trial')
        axs[1].set_ylabel('P(Negative Direction)')
        axs[1].set_title('Model Sign/Dir Probabilities')
        axs[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                                                            
    aim_edges = np.arange(-180.0, 181.0, 6.0)
    num_aim_bins = len(aim_edges) - 1
    hist_norm = np.zeros((n_trials, num_aim_bins))
    for t in range(n_trials):
        pi_pred = np.array(model_predictive_pis[t])
        full_policy = np.array(model_predictive_policies[t])                           
        for k in range(num_aim_bins):
            low = aim_edges[k]
            high = aim_edges[k + 1]
                     
            em0 = (norm.cdf(high, 0, noise) - norm.cdf(low, 0, noise) +
                   norm.cdf(high + 360, 0, noise) - norm.cdf(low + 360, 0, noise) +
                   norm.cdf(high - 360, 0, noise) - norm.cdf(low - 360, 0, noise))
                                  
            i1 = norm.cdf(high, action_centers, noise) - norm.cdf(low, action_centers, noise)
            i2 = norm.cdf(high + 360, action_centers, noise) - norm.cdf(low + 360, action_centers, noise)
            i3 = norm.cdf(high - 360, action_centers, noise) - norm.cdf(low - 360, action_centers, noise)
            em1_contribs = i1 + i2 + i3
            em1 = np.dot(full_policy, em1_contribs)
            prob_k = pi_pred[0] * em0 + pi_pred[1] * em1
            hist_norm[t, k] = prob_k
    hist_norm = np.ma.masked_where(hist_norm < 1e-6, hist_norm)
    im1 = axs[2].imshow(hist_norm.T, origin='lower', aspect='auto', cmap='Greys', extent=[0, n_trials, -180, 180], interpolation='nearest')
    plt.colorbar(im1, ax=axs[2], label='Probability Mass')
    axs[2].scatter(np.arange(n_trials), np.array(human_explicits), color='red', marker='x', s=30, alpha=0.8, label='Human Aim')
    axs[2].axhline(y=0, color='green', linewidth=2)
    axs[2].set_xlabel('Trial')
    axs[2].set_ylabel('Aim (degrees)')
    axs[2].set_title('Model Predictive Density vs Human Aim')
    legend_elements1 = [
        Patch(facecolor='gray', alpha=0.5, label='Model Predictive Density'),
        Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=8, label='Human Aim')
    ]
    axs[2].legend(handles=legend_elements1, loc='upper right')
    plt.tight_layout()
                     
    plt.savefig(f'tempFigures/{pp}_{i}', dpi=200)
    plt.close()
                                                                           
                                                 
                                                       
                                                                           
class FitShell:
    def __init__(self, df='none', conVal='none', condition='none',
                 fitLength=320, fitPhase='rotation',
                 rmse=False, flipRot=False, method='de', annealing_mode=0, n_starts=500,quickPlots=False):
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
        self.n_starts = n_starts
        self.method = method
        self.annealing_mode = annealing_mode                                                                
        self.quickPlots = quickPlots
        
    def fitRot(self, lrUb=1):
                                     
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
        self.pi_preds = [[] for _ in uniqP]
        self.valid_trials = [[] for _ in uniqP]
        self.model_explicits = [[] for _ in uniqP]
        self.all_model_explicits = [[] for _ in uniqP]
        self.human_explicits = [[] for _ in uniqP]
        self.expected_aims = [[] for _ in uniqP]
        self.model_predictive_pis = [[] for _ in uniqP]
        self.model_predictive_policies = [[] for _ in uniqP]
        self.model_predictive_dirs = [[] for _ in uniqP]
        self.xs = []
        self.errors = []
        self.rmses = np.zeros(len(uniqP))
        self.rSquareds = np.zeros(len(uniqP))
                                                                          
        bounds = [(1, 30), (1e-300, 0.05), (.1, 100), (1e-6, 1), (1e-300, 2), (1e-300, 1)]
        initial = [1.1, 0.05, 10.0, 0.1, 1.0, 1.0]
        argsList = [(pp, i, self.dat, self.fitPhase,
                     self.flipRot, bounds, initial, self.annealing_mode, self.n_starts)
                    for i, pp in enumerate(uniqP)]
        print(f"Starting static fitting for {len(uniqP)} participants. Annealing mode: {self.annealing_mode}")
        os.makedirs("tempFigures", exist_ok=True)
        start_time = time.time()
        with mp.Pool(processes=8) as pool:
            results = pool.starmap(_fitWorker, argsList)
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.1f} minutes.")
        for i, res in enumerate(results):
            self.xs.append(res['x'])
            self.negLl[i] = res['fun']
            self.errors.append(res['errors'])
            self.bics[i] = res['bic']
            self.mStates[i] = res['mStates']
            self.allAims[i] = res['allAims']
            self.pi_preds[i] = res['pi_preds']
            self.valid_trials[i] = res['valid_trials']
            self.model_explicits[i] = res['model_explicits']
            self.all_model_explicits[i] = res['all_model_explicits']
            self.human_explicits[i] = res['human_explicits']
            self.expected_aims[i] = res['expected_aims']
            self.model_predictive_pis[i] = res['model_predictive_pis']
            self.model_predictive_policies[i] = res['model_predictive_policies']
            self.model_predictive_dirs[i] = res['model_predictive_dirs']
            self.rmses[i] = res['rmse']
            self.rSquareds[i] = res['rSquared']
                           
        if self.quickPlots:
            plot_args_list = [(i, self.participantNums[i], self.xs[i], self.pi_preds[i], self.model_predictive_pis[i], self.model_predictive_policies[i], self.model_predictive_dirs[i], self.human_explicits[i]) for i in range(len(self.participantNums))]
            print("Starting parallel plotting...")
            plot_start = time.time()
            with mp.Pool(processes=20) as plot_pool:
                plot_pool.map(_plotWorker, plot_args_list)
            plot_elapsed = time.time() - plot_start
            print(f"Plotting completed in {plot_elapsed/60:.1f} minutes.")
