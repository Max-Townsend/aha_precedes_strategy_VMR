from scipy.stats import t as tDist
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patheffects import withStroke
import ruptures as rpt
import pandas as pd
import multiprocessing as mp
import sys
import pickle
import math
import warnings; warnings.filterwarnings("ignore")
                                                
plot_dir = 'indiPlotsNoModel'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
print(f"Saving plots to: {plot_dir}")
def normalize_angle(angle):
    """
    Normalize angle(s) to [-180, 180).
    - Works on scalars or arrays.
    - Handles positives/negatives/multi-turns.
    - Preserves NaN.
    """
    if np.isscalar(angle):
        if np.isnan(angle):
            return np.nan
                                                      
        angle = angle - 360 * np.round(angle / 360)
        return ((angle + 180) % 360) - 180
    else:
        normalized = np.full_like(angle, np.nan)
        mask = ~np.isnan(angle)
                       
        unwrapped = angle[mask] - 360 * np.round(angle[mask] / 360)
        normalized[mask] = ((unwrapped + 180) % 360) - 180
        return normalized
def ruptures_changepoint(aims, bl, pen=1, min_size=1, debug=False):
    """
    Ruptures for detecting changepoints in aim sequence (post-baseline abs aims).
    - aims: np.array of aims (univariate, shape (n,)).
    - bl: baseline length.
    - pen: Penalty value to tune number of changepoints (lower = more sensitive).
    - min_size: Minimum segment length to allow early detection.
    - Returns: list of dicts for changepoints, each with 'position', 'p_value' (None), 'max_gain' (cost as proxy).
    """
    """
    # Compute baseline stats
    baseline_aims = aims[:bl]
    valid_baseline = ~np.isnan(baseline_aims)
    if np.sum(valid_baseline) < 2:
        return []
    mean_bl = np.mean(baseline_aims[valid_baseline])
    sd_bl = np.std(baseline_aims[valid_baseline])
    if sd_bl == 0:
        return []
    pen = max(pen * sd_bl**2, 0.2)
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
def participant_worker(pId, datasetName, rotation, impFlag, ssm_fit, hmm_fit, baseline_length, washout_length, debug, save_dir, n_participants):
    rows = []
    if debug: print(f"Generating plot for {datasetName}, rot{rotation}, p{pId}...")
                                                                          
    human_aims = np.array([])
    try:
        if hasattr(ssm_fit, 'allAims'):
            human_aims = np.array(ssm_fit.allAims[pId]).ravel()
        elif hasattr(ssm_fit, 'human_explicits'):
            human_aims = np.array(ssm_fit.human_explicits[pId]).ravel()
        elif hasattr(hmm_fit, 'human_explicits'):
            human_aims = np.array(hmm_fit.human_explicits[pId]).ravel()
        elif hasattr(hmm_fit, 'allAims'):
            human_aims = np.array(hmm_fit.allAims[pId]).ravel()
        human_aims = human_aims % 360.0                        
    except (IndexError, AttributeError, TypeError):
        human_aims = np.array([])
                                                            
    n_trials = len(human_aims)
    if n_trials == 0:
        if debug: print(f" No human aims data for p{pId}; skipping.")
        return rows
                                                 
    human_imps = np.array([])
    if impFlag:
        try:
            if hasattr(ssm_fit, 'allImps'):
                human_imps = np.array(ssm_fit.allImps[pId]).ravel()
            elif hasattr(ssm_fit, 'human_implicits'):
                human_imps = np.array(ssm_fit.human_implicits[pId]).ravel()
            elif hasattr(hmm_fit, 'human_implicits'):
                human_imps = np.array(hmm_fit.human_implicits[pId]).ravel()
            elif hasattr(hmm_fit, 'allImps'):
                human_imps = np.array(hmm_fit.allImps[pId]).ravel()
            human_imps = human_imps % 360.0
            human_imps = human_imps[:n_trials]                               
                             
            human_imps[human_imps == 0] = np.nan
        except (IndexError, AttributeError, TypeError):
            human_imps = np.array([])
    else:
        human_imps = np.array([])
    if debug:
        print(f" human_aims len: {len(human_aims)}, human_imps len: {len(human_imps)}")
                                               
    human_aims_signed = normalize_angle(human_aims)
    human_imps_signed = normalize_angle(human_imps) if len(human_imps) > 0 else np.array([])
                                        
    aha_trials = []
    bl = baseline_length
    min_t = bl
    max_t = n_trials - (washout_length + min(40, bl))
    aim = np.abs(human_aims_signed)
    rot_part = None
    try:
        rot_full = np.asarray(ssm_fit.rotations)
        if rot_full.ndim == 2 and rot_full.shape[0] == n_participants:
            rot_part = rot_full[pId, :n_trials]
        else:
            rot_part = np.asarray(ssm_fit.rotations[pId])[:n_trials]
        rot_part = normalize_angle(rot_part)
    except (AttributeError, IndexError, ValueError):
        rot_part = np.full(n_trials, rotation)
    if min_t <= max_t:
        full_baseline_aims = aim[:bl]
        valid_baseline = ~np.isnan(full_baseline_aims)
        n_base_full = np.sum(valid_baseline)
        if n_base_full < 2:
            aha_trials = []
        else:
                                                    
            post_aims = aim[:max_t]
                                                    
            cp_rels = ruptures_changepoint(post_aims, bl=bl, pen=0.8, min_size=1, debug=debug)
            aha_trials = [cp_rels[0]['position']] if cp_rels else []                                 
                               
            for cp in cp_rels:
                row = {
                    'dataset': datasetName,
                    'rotation': rotation,
                    'participant_id': pId,
                    'changepoint_position': cp['position'],
                    'relative_position': cp['position'] - baseline_length,
                    'p_value': cp['p_value'],
                    'max_gain': cp['max_gain']
                }
                rows.append(row)
    if debug and aha_trials:
        print(f" Human-based Aha! trials: {[t - baseline_length for t in aha_trials]}")
    if datasetName == 'Brudner':
        print(pId, [t - bl for t in aha_trials])
                                                                         
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
    ax.set_title(f'{datasetName} | Rot {rotation}° | Participant {pId}', fontsize=14)
    ax.set_facecolor('white')                               
                                                            
    ax.axhline(y=rotation, color='grey', linestyle='--', zorder=1, alpha=0.7, linewidth=3)
    ax.axhline(y=-rotation, color='grey', linestyle='--', zorder=1, alpha=0.7, linewidth=3)
    ax.axhline(y=0, color='grey', linestyle='--', zorder=1, alpha=0.7, linewidth=3)
    ax.axvline(x=0, color='grey', linestyle='--', zorder=1, alpha=0.7, linewidth=3)
    shift = baseline_length
    washout_start = n_trials - washout_length
    washout_onset = washout_start - shift
    has_washout = n_trials > baseline_length + washout_length
    if has_washout:
        ax.axvline(x=washout_onset, color='grey', linestyle='--', zorder=1, alpha=0.7, linewidth=3)
                                                                        
    ax.set_ylim(-2.1 * abs(rotation), 2.1 * abs(rotation))
    y_top = ax.get_ylim()[1]
    colors = ['cyan', 'magenta', 'yellow']
    dark_colors = ['darkcyan', 'darkmagenta', 'goldenrod']
    x_offset = 1.6
    for i, trial in enumerate(aha_trials):
        if not (0 <= trial < n_trials and trial < washout_start):
            continue
        trial_new = trial - shift
        color = colors[i % len(colors)]
        dark_color = dark_colors[i % len(colors)]
        cp_line = ax.axvline(x=trial_new, color=color, linestyle='-', linewidth=6, zorder=3, alpha=0.8)
        cp_line.set_path_effects([withStroke(linewidth=10, foreground=dark_color)])
                  
        label = "Aha!" if i == 0 else f"CP{i+1}"
        s = f'~"{label}" trial: {trial - baseline_length}'
        x_text = trial_new + x_offset
        ha = 'left'
        if x_text > (n_trials - shift) - 5:
            x_text = trial_new - x_offset
            ha = 'right'
        y_text = y_top * (0.9 - 0.05 * i)
        ax.text(
            x=x_text,
            y=y_text,
            s=s,
            ha=ha,
            va='bottom',
            fontsize=80,
            color=color,
            weight='bold',
            clip_on=False
        )
                                                 
    trials = np.arange(n_trials) - shift
    linewidth = 4.5
    ax.scatter(trials, human_aims_signed, color='blue', s=180, alpha=0.8, label='Human Aims (Explicit)', zorder=8, edgecolor='darkblue', linewidth=3)
    if impFlag and len(human_imps_signed) > 0:
        nan_mask = ~np.isnan(human_imps_signed)
        ax.scatter(trials[nan_mask], human_imps_signed[nan_mask], color='#ff2400', s=120, alpha=0.8, label='Human Implicits', zorder=8, edgecolor='darkred', linewidth=5)
    ax.set_xlim(-shift, n_trials - shift)
                                                                  
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim Angle (°)')
    ax.set_yticks(np.arange(-2 * abs(rotation), 2 * abs(rotation) + 1, 30))
    sns.despine()
    ax.legend(loc='upper left')
                                                       
    if has_washout:
        dist1 = shift                            
        dist2 = washout_onset                                   
        step = math.gcd(int(dist1), int(dist2))
        intervals = dist2 / step
        if datasetName == 'Brudner' and intervals == int(intervals) and int(intervals) % 2 == 1:
            current_gcd = step
            for candidate in range(current_gcd, 0, -1):
                if current_gcd % candidate == 0:
                    cand_intervals = dist2 // candidate
                    if cand_intervals % 2 == 0:
                        step = candidate
                        break
        if step >= 10:
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(step))
        else:
                                       
            current_xticks = ax.get_xticks()
            additional_ticks = [0, washout_onset]
            new_xticks = np.unique(np.append(current_xticks, additional_ticks))
            ax.set_xticks(new_xticks)
    else:
                    
        current_xticks = ax.get_xticks()
        additional_ticks = [0]
        new_xticks = np.unique(np.append(current_xticks, additional_ticks))
        ax.set_xticks(new_xticks)
                          
    save_path = os.path.join(save_dir, f"{datasetName}_rot{rotation}_p{pId}.svg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    del fig, ax                       
    if debug: print(f" Saved: {save_path}")
    return rows
def generate_per_participant_plots(save_dir=plot_dir, debug=False, max_participants=10, num_samples=200):
    """
    Generate per-participant plots for each dataset/rotation (first {max_participants} only).
    - Only human data (aims + implicits if present) + Aha! trial marker.
    - No model aims/implicits, no policy heatmap.
    Saves each as {datasetName}_rot{rotation}_p{participantId}.svg.
    """
    changepoints_df = pd.DataFrame(columns=['dataset', 'rotation', 'participant_id', 'changepoint_position', 'relative_position', 'p_value', 'max_gain'])
    for dsIdx, datasetName in enumerate(datasetLabels):
        modelLists = datasets[dsIdx]                                    
        rotations = rotationMap[dsIdx]
        impFlag = hasImp[dsIdx]
        if True:                           
            for rotIdx, rotation in enumerate(rotations):
                                            
                ssm_fit = modelLists[0][rotIdx] if rotIdx < len(modelLists[0]) else None
                hmm_fit = modelLists[2][rotIdx] if rotIdx < len(modelLists[2]) else None
                if ssm_fit is None or hmm_fit is None:
                    if debug: print(f"Skipping {datasetName} rot{rotation}: Missing SSM or HMM fit.")
                    continue
                                                                          
                n_participants = None
                if hasattr(ssm_fit, 'allAims'):
                    n_participants = len(ssm_fit.allAims)
                elif hasattr(ssm_fit, 'human_explicits'):
                    n_participants = len(ssm_fit.human_explicits)
                elif hasattr(hmm_fit, 'human_explicits'):
                    n_participants = len(hmm_fit.human_explicits)
                elif hasattr(hmm_fit, 'allAims'):
                    n_participants = len(hmm_fit.allAims)
                if n_participants is None:
                    if debug: print(f"Skipping {datasetName} rot{rotation}: No participant data.")
                    continue
                                                              
                n_participants = min(max_participants, n_participants)
                baseline_length = 64 if datasetName == 'Brudner' else 40
                washout_length = baseline_length
                                                      
                with mp.Pool() as pool:
                    worker_args = [(pId, datasetName, rotation, impFlag, ssm_fit, hmm_fit, baseline_length, washout_length, debug, save_dir, n_participants) for pId in range(n_participants)]
                    all_rows = pool.starmap(participant_worker, worker_args)
                                       
                flat_rows = [row for sublist in all_rows for row in sublist]
                if flat_rows:
                    changepoints_df = pd.concat([changepoints_df, pd.DataFrame(flat_rows)], ignore_index=True)
                                     
    csv_path = 'changepoints.csv'
    changepoints_df.to_csv(csv_path, index=False)
    print(f"All per-participant plots generated and saved to indiPlotsNoModel/")
    print(f"Changepoints saved to: {csv_path}")
                
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
    generate_per_participant_plots(debug=False, max_participants=100, num_samples=500)