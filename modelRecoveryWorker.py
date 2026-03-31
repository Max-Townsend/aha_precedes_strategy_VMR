import numpy as np
import importlib
import NonGenSSMWithNoise as NonGenSSM
import QLearning
import HMM
import pandas as pd

importlib.reload(NonGenSSM)
importlib.reload(QLearning)
importlib.reload(HMM)

action_centers = np.arange(-180, 181)
allCGData = pd.read_csv('CGData_.csv',low_memory=False)
CGDat = allCGData[(allCGData["targetCount"] == 8) &
        (allCGData["hasAutocorrection"] == 0)]

def get_template_df(CGDat, pp_num, rot_val):
    template = CGDat[(CGDat['participantNum'] == pp_num) &
                     (CGDat['blockNum'] == 0) &
                     (CGDat['blockRot'] == rot_val)].copy()
                        
    template['aim'] = np.nan
                                                                       
    required_cols = ['aim', 'rotation', 'targetPosition', 'participantNum', 'blockNum', 'phase', 'blockRot']
    for col in required_cols:
        if col not in template.columns:
            template[col] = np.nan                          
    return template

                                                
def sim_from_ssm(model, pp_idx, template_df):
    params = model.xs[pp_idx]                       
    noise = params[2]
    m_states = np.array(model.mStates[pp_idx])                           
    sim_aims = m_states + np.random.normal(0, noise, len(m_states))
    sim_aims = ((sim_aims + 180) % 360 - 180)
    sim_df = template_df.copy()
    sim_df['aim'] = sim_aims
    sim_df['participantNum'] = 999999                        
    return sim_df

                                                                                       
def sim_from_qlearn(model, pp_idx, template_df):
    params = model.xs[pp_idx]                                  
    noise = params[2]
    policies = model.model_predictive_policies[pp_idx]                                   
    sim_aims = []
    for policy in policies:
        act_idx = np.random.choice(len(action_centers), p=policy)
        act = action_centers[act_idx]
        aim = act + np.random.normal(0, noise)
        sim_aims.append(aim)
    sim_aims = np.array(sim_aims)
    sim_aims = ((sim_aims + 180) % 360 - 180)
    sim_df = template_df.copy()
    sim_df['aim'] = sim_aims
    sim_df['participantNum'] = 999999         
    return sim_df

                                                                                                                                 
def sim_from_hmm(model, pp_idx, template_df):
    params = model.xs[pp_idx]                                                        
    noise = params[0]
    pi_preds = model.pi_preds[pp_idx]                                          
    policies = model.model_predictive_policies[pp_idx]                                                 
    sim_aims = []
    for pi_pred, policy in zip(pi_preds, policies):
        state = np.random.choice([0, 1], p=pi_pred)
        if state == 0:
            aim = np.random.normal(0, noise)
        else:
            act_idx = np.random.choice(len(action_centers), p=policy)
            act = action_centers[act_idx]
            aim = act + np.random.normal(0, noise)
        sim_aims.append(aim)
    sim_aims = np.array(sim_aims)
    sim_aims = ((sim_aims + 180) % 360 - 180)
    sim_df = template_df.copy()
    sim_df['aim'] = sim_aims
    sim_df['participantNum'] = 999999         
    return sim_df

                                                    
def fit_worker(args):
    sim_df, true_model_name, n_starts = args                        
             
    ssm_fit = NonGenSSM.FitShell(df=sim_df, condition='none', conVal='none', fitPhase=None, imp=False, n_starts=n_starts)
    ssm_fit.fitRot()
    ssm_bic = ssm_fit.bics[0] if len(ssm_fit.bics) > 0 else np.inf
   
                   
    qlearn_fit = QLearning.FitShell(df=sim_df, condition='none', conVal='none', fitPhase=None, n_starts=n_starts)
    qlearn_fit.fitRot()
    qlearn_bic = qlearn_fit.bics[0] if len(qlearn_fit.bics) > 0 else np.inf
   
             
    hmm_fit = HMM.FitShell(df=sim_df, condition='none', conVal='none', fitPhase=None, annealing_mode=0, n_starts=n_starts)
    hmm_fit.fitRot()
    hmm_bic = hmm_fit.bics[0] if len(hmm_fit.bics) > 0 else np.inf
   
    bics = [ssm_bic, qlearn_bic, hmm_bic]
    recovered_idx = np.argmin(bics)
    rot = sim_df['blockRot'].iloc[0]
    return true_model_name, recovered_idx, rot