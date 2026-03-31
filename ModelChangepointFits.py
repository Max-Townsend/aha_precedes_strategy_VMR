import numpy as np
import os
import pandas as pd
import multiprocessing as mp
import sys
import pickle
import scipy.ndimage as ndimage
from scipy.stats import norm
import ruptures as rpt
import warnings; warnings.filterwarnings("ignore")
def ruptures_changepoint(aims, bl, pen=1, min_size=1, debug=False):
    """
    Ruptures for detecting changepoints in aim sequence (post-baseline abs aims).
    - aims: np.array of aims (univariate, shape (n,)).
    - bl: baseline length.
    - pen: Penalty value to tune number of changepoints (lower = more sensitive).
    - min_size: Minimum segment length to allow early detection.
    - Returns: list of dicts for changepoints, each with 'position', 'p_value' (None), 'max_gain' (cost as proxy).
    """
    n = len(aims)
    if n < 2:
        return []
                                           
    original_indices = np.arange(n)
    valid_mask = ~np.isnan(aims)
    if np.sum(valid_mask) < 2:
        return []
    aims_clean = aims[valid_mask].reshape(-1, 1)                                 
    original_indices = original_indices[valid_mask]
                             
    if np.std(aims_clean) == 0:
        return []
    algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(aims_clean)
    cps = algo.predict(pen=pen)                                        
    if debug:
        print(f"Ruptures detected breakpoints: {cps}")
                                                                                      
    cps_positions = [original_indices[cp] for cp in cps[:-1]] if cps[:-1] else []
                              
    cps_positions = [pos for pos in cps_positions if pos > bl]
    if not cps_positions:
        return []
    cost_func = algo.cost                                
    costs = []
    prev = 0
    for cp in cps:
        segment = aims_clean[prev:cp]
        if len(segment) > 0:
            cost = cost_func.error(0, len(segment))
            costs.append(cost)
        prev = cp
                                                                                                                
    cps_list = [{'position': pos, 'p_value': None, 'max_gain': sum(costs)/len(costs) if costs else 0}
                for pos in sorted(cps_positions)]
                                                                                
    return cps_list[:1] if cps_list else []
def participant_worker(pId, datasetName, rotation, ssm_fit, hmm_fit, baseline_length, washout_length, debug, n_participants, n_simulations):
    rows = []
    if debug: print(f"Processing {datasetName}, rot{rotation}, p{pId}...")
  
                                     
    try:
        policies_full = np.array(hmm_fit.model_predictive_policies)
        pi_preds = np.array(hmm_fit.pi_preds)
    except (AttributeError, ValueError):
        if debug: print(f" No model_predictive_policies or pi_preds for p{pId}; skipping.")
        return rows
  
    try:
        policies_part = policies_full[pId]
        state0_probs = pi_preds[pId, :, 0]
        state1_probs = pi_preds[pId, :, 1]
    except (IndexError, ValueError):
        if debug: print(f" Invalid participant index {pId}; skipping.")
        return rows
  
    n_trials = policies_part.shape[0]
    if n_trials == 0:
        if debug: print(f" No trials for p{pId}; skipping.")
        return rows
  
                                            
    bl = baseline_length
    max_t = n_trials - (washout_length + min(40, bl))
    if max_t > n_trials:
        max_t = n_trials
                    
                                          
                                        
                                        
                     
                     
  
                   
    sigma = 0.0
    if datasetName != 'BondTaylor':
        if hasattr(hmm_fit, 'xs') and pId < len(hmm_fit.xs):
            try:
                sigma = float(hmm_fit.xs[pId][0])
                if np.isnan(sigma):
                    sigma = 0.0
            except:
                sigma = 0.0
    sigma = max(sigma, 0.1)
  
                                
    support_size = max(int(6 * sigma) + 1, 3)
    if support_size % 2 == 0:
        support_size += 1
    half_support = support_size // 2
    support = np.arange(-half_support, half_support + 1)
    kernel = norm.pdf(support, 0, sigma)
    kernel /= np.sum(kernel)
  
                                    
    angles = np.arange(-180, 181)
    orig_idx = (180 + angles) % 360
  
                                
    delta = np.zeros(361)
    delta[180] = 1.0
  
              
    if sigma > 0.001:
        convolved0 = ndimage.convolve1d(delta, kernel, mode='wrap')
        policies_recentered = policies_part[:, orig_idx]
        convolved1_all = ndimage.convolve1d(policies_recentered, kernel, axis=1, mode='wrap')
    else:
        convolved0 = delta
        policies_recentered = policies_part[:, orig_idx]
        convolved1_all = policies_recentered
  
                            
    state0_p = state0_probs.reshape(-1, 1)
    state1_p = state1_probs.reshape(-1, 1)
    marginal_all = state0_p * convolved0[None, :] + state1_p * convolved1_all
    marginal_sums = np.sum(marginal_all, axis=1, keepdims=True)
    marginal_all = np.divide(marginal_all, marginal_sums, where=(marginal_sums > 0), out=np.zeros_like(marginal_all))
  
                                 
    for run in range(n_simulations):
                       
        sampled_aims = np.zeros(n_trials)
        for t in range(n_trials):
            probs = marginal_all[t]
            sampled_angle = np.random.choice(angles, p=probs)
            sampled_aims[t] = abs(sampled_angle)
  
                                                 
        min_t = bl
        aim = sampled_aims
        if min_t < n_trials:
            post_aims = aim
            cp_rels = ruptures_changepoint(post_aims, bl=bl, pen=0.8, min_size=1, debug=debug)
            for cp in cp_rels:
                row = {
                    'dataset': datasetName,
                    'rotation': rotation,
                    'participant_id': pId,
                    'changepoint_position': cp['position'],
                    'relative_position': cp['position'] - baseline_length,
                    'p_value': cp['p_value'],
                    'max_gain': cp['max_gain'],
                    'simulation_run': run
                }
                rows.append(row)
  
    if debug and cp_rels:
        print(f" Model-based Aha! trials: {[cp['position'] - baseline_length for cp in cp_rels]}")
    if datasetName == 'Brudner':
        print(pId, [cp['position'] - bl for cp in cp_rels])
  
    return rows
def generate_per_participant_model_changepoints(debug=False, max_participants=10000, n_simulations=20):
    """
    Compute per-participant model changepoints for each dataset/rotation (first {max_participants} only).
    Saves to model_changepoints.csv.
    """
    changepoints_df = pd.DataFrame(columns=['dataset', 'rotation', 'participant_id', 'changepoint_position', 'relative_position', 'p_value', 'max_gain', 'simulation_run'])
    for dsIdx, datasetName in enumerate(datasetLabels):
        modelLists = datasets[dsIdx]                                    
        rotations = rotationMap[dsIdx]
        for rotIdx, rotation in enumerate(rotations):
                                        
            ssm_fit = modelLists[0][rotIdx] if rotIdx < len(modelLists[0]) else None
            hmm_fit = modelLists[2][rotIdx] if rotIdx < len(modelLists[2]) else None
            if ssm_fit is None or hmm_fit is None:
                if debug: print(f"Skipping {datasetName} rot{rotation}: Missing SSM or HMM fit.")
                continue
                                      
            n_participants = None
            if hasattr(hmm_fit, 'model_predictive_policies'):
                n_participants = len(hmm_fit.model_predictive_policies)
            elif hasattr(hmm_fit, 'pi_preds'):
                n_participants = len(hmm_fit.pi_preds)
            if n_participants is None:
                if debug: print(f"Skipping {datasetName} rot{rotation}: No participant data.")
                continue
                                             
            n_participants = min(max_participants, n_participants)
            baseline_length = 64 if datasetName == 'Brudner' else 40
            washout_length = baseline_length
                                                  
            with mp.Pool() as pool:
                worker_args = [(pId, datasetName, rotation, ssm_fit, hmm_fit, baseline_length, washout_length, debug, n_participants, n_simulations) for pId in range(n_participants)]
                all_rows = pool.starmap(participant_worker, worker_args)
                                   
            flat_rows = [row for sublist in all_rows for row in sublist]
            if flat_rows:
                changepoints_df = pd.concat([changepoints_df, pd.DataFrame(flat_rows)], ignore_index=True)
                                     
    csv_path = 'model_changepoints.csv'
    changepoints_df.to_csv(csv_path, index=False)
    print(f"Model changepoints saved to: {csv_path}")
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Please provide the path to the pickle file containing the global variables.")
    pickle_path = sys.argv[1]
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    datasets = data['datasets']
    datasetLabels = data['datasetLabels']
    hasImp = data['hasImp']
    rotationMap = data['rotationMap']
    rots = data['rots']
    generate_per_participant_model_changepoints(debug=False, max_participants=10000, n_simulations=1000)