import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

class RLConf:
    def __init__(self, eta=0.1, init_var=1000, beta=1.0, sigma_null=10.0, pi0=0.01, kappa=1.0):
        self.eta = eta                                             
        self.init_var = init_var
        self.beta = beta                                              
        self.sigma_null = sigma_null
        self.pi0 = pi0
        self.kappa = kappa

def simulate_adaptation(model, theta=45.0, n_trials=20, explore_scale=0.05, conf_thresh=0.5, baseline_alpha=0.1, Q=0.01, seed=42, w_sin=0.0, w_cos=0.0, targets=None):
    np.random.seed(seed)
    phis = []
    explore_sigmas = []
    aim = 0.0
    mu = 0.0                                             
    precision = 1 / model.init_var
    baseline = 0.0                                             
    p_pert = model.pi0
    mu_theta = 0.0
    precision_theta = 1 / model.init_var
    if np.isscalar(theta):
        theta = np.full(n_trials, theta)                      
    else:
        theta = np.array(theta)                                            
        assert len(theta) == n_trials, "theta must be scalar or array of length n_trials"
    if targets is None:
        targets = np.zeros(n_trials)
    else:
        targets = np.array(targets)
        assert len(targets) == n_trials, "targets must be array of length n_trials"
    for trial in range(n_trials):
                                                                           
        if Q > 0:
            var = 1 / precision if precision > 1e-100 else model.init_var
            prior_var = var + Q
            precision = 1 / prior_var
                                 
        if Q > 0:
            var_theta = 1 / precision_theta if precision_theta > 1e-100 else model.init_var
            prior_var_theta = var_theta + Q
            precision_theta = 1 / prior_var_theta
        var = 1 / precision if precision > 1e-100 else model.init_var
        var_theta = 1 / precision_theta if precision_theta > 1e-100 else model.init_var
        explore_sigma = explore_scale * np.sqrt(var)
        explore_sigmas.append(explore_sigma)
        delta_aim = np.random.normal(0, explore_sigma)
        new_aim = aim + delta_aim                             
        phis.append(aim)                  
        cursor = new_aim + theta[trial]
                                                                    
        theta_obs = cursor - new_aim                  
        error = abs(cursor)
        R = -error                    
        advantage = R - baseline
                                                                      
        eta_t = model.eta * (1 + model.beta * advantage / (abs(baseline) + 1e-8))
        eta_t = max(eta_t, 0.0001)                                         
                                                    
        lik_null = norm.pdf(theta_obs, 0, model.sigma_null) + 1e-120
        lik_pert = norm.pdf(theta_obs, mu_theta, np.sqrt(var_theta)) + 1e-120
        p_pert = (p_pert * lik_pert) / (p_pert * lik_pert + (1 - p_pert) * lik_null)
                                                                              
        new_precision_theta = precision_theta + model.eta
        new_mu_theta = (mu_theta * precision_theta + theta_obs * model.eta) / new_precision_theta
        mu_theta = new_mu_theta
        precision_theta = new_precision_theta
                         
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * R
                                                                      
        var = 1 / precision if precision > 1e-100 else model.init_var
                                                           
        sin_val = np.sin(np.deg2rad(targets[trial]))
        cos_val = np.cos(np.deg2rad(targets[trial]))
        logit = w_sin * sin_val + w_cos * cos_val
        p_flip = (1-p_pert) * (1 / (1 + np.exp(-logit)))
        s = -1 if np.random.rand() < p_flip else 1
                                                         
        if p_pert > conf_thresh:
                                                                   
            grad = advantage * (new_aim - aim) / var
            mu += eta_t * grad * model.kappa * s
                                                              
            new_precision = precision + eta_t
            precision = new_precision
            aim = mu * model.kappa                      
                                    
    return np.array(phis), np.array(explore_sigmas)

def de_objective(params, aims, n_trials, theta, seed, targets):
    eta, init_var, explore_scale, conf_thresh, beta, baseline_alpha, Q, executionVar, sigma_null, pi0, kappa, w_sin, w_cos = params
    model = RLConf(eta=eta, init_var=init_var, beta=beta, sigma_null=sigma_null, pi0=pi0, kappa=kappa)
    phis, explores = simulate_adaptation(model, theta=theta, n_trials=n_trials, explore_scale=explore_scale,
                                        conf_thresh=conf_thresh, baseline_alpha=baseline_alpha, Q=Q, seed=seed, w_sin=w_sin, w_cos=w_cos, targets=targets)
    mask = ~np.isnan(aims)
    valid_aims = aims[mask]
    valid_phis = phis[mask]
    valid_explores = explores[mask]
    numSamp = len(valid_aims)
    if numSamp == 0:
        return np.inf
    modelStd = np.sqrt(executionVar) + valid_explores
    liks = norm.pdf(valid_aims, valid_phis, modelStd) + 1e-12
    logLikelihood = np.sum(np.log(liks))
    return -logLikelihood

class fitShell:
    def __init__(self, df, fitLen=320, fitPhase='rotation', method='L-BFGS-B', condition='rotation', conVal=30):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.fitPhase = fitPhase
        self.allAims = []
        self.method = method
        self.theta = -conVal                                                   
        self.seed = 42                                  

    def fitRot(self):
        if self.condition != 'none':
            dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            dat = self.df
        uniqP = dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP)) * 100000
        self.mStates = [[]] * len(uniqP)
        self.allAims = [[]] * len(uniqP)
        self.xs = [[]] * len(uniqP)
        i = 0
        for pp in uniqP:
            phase_dat = dat[(dat['participantNum'] == pp) & (dat['phase'] == self.fitPhase)]
            blockNums = phase_dat['blockNum'].unique()
            block_dat = phase_dat[phase_dat['blockNum'] == blockNums[0]]
            aims = block_dat['aim'].values
            targets = block_dat['targetPosition'].values
            n_trials = len(aims)
            bounds = [(-10, 1e2), (0, 1e3), (1e-6, 10), (0.1, 0.99), (1e-6, 1e4), (0, 1), (0, 1e2), (1, 3e2), (1, 5000), (1e-6, 1), (0, 1), (-10, 10), (-10, 10)]
            res = differential_evolution(de_objective, bounds=bounds, args=(aims, n_trials, self.theta, self.seed, targets), workers=-1, seed=self.seed)
                                                                         
            eta, init_var, explore_scale, conf_thresh, beta, baseline_alpha, Q, executionVar, sigma_null, pi0, kappa, w_sin, w_cos = res.x
            model = RLConf(eta=eta, init_var=init_var, beta=beta, sigma_null=sigma_null, pi0=pi0, kappa=kappa)
            phis, explores = simulate_adaptation(model, theta=self.theta, n_trials=n_trials, explore_scale=explore_scale,
                                                conf_thresh=conf_thresh, baseline_alpha=baseline_alpha, Q=Q, seed=self.seed, w_sin=w_sin, w_cos=w_cos, targets=targets)
            mask = ~np.isnan(aims)
            valid_aims = aims[mask]
            valid_phis = phis[mask]
            valid_explores = explores[mask]
            numSamp = len(valid_aims)
            if numSamp > 0:
                modelStd = np.sqrt(executionVar) + valid_explores
                liks = norm.pdf(valid_aims, valid_phis, modelStd) + 1e-12
                logLikelihood = np.sum(np.log(liks))
                k = 13                                                             
                BIC = k * np.log(numSamp) - 2 * logLikelihood
                negLL = -logLikelihood
            else:
                BIC = np.inf
                negLL = np.inf
            if negLL < self.negLL[i]:
                self.negLL[i] = negLL
                self.BICs[i] = BIC
                self.mStates[i] = phis.tolist()
                self.allAims[i] = aims.tolist()
                self.xs[i] = res.x.tolist()
                          
            print(i, 'out of', len(uniqP), ' , BIC: ,', BIC, end='\r')
                                    
            i += 1
        print()                                               

    def genDat(self, params, rots, trials=np.arange(-5, 35, 1)):
        if len(params) == 11:
            eta, init_var, explore_scale, conf_thresh, beta, baseline_alpha, Q, _, sigma_null, pi0, kappa = params
            w_sin = 0.0
            w_cos = 0.0
        else:
            eta, init_var, explore_scale, conf_thresh, beta, baseline_alpha, Q, _, sigma_null, pi0, kappa, w_sin, w_cos = params
        n_trials = len(trials)
        model = RLConf(eta=eta, init_var=init_var, beta=beta, sigma_null=sigma_null, pi0=pi0, kappa=kappa)
        phis, _ = simulate_adaptation(model, theta=rots, n_trials=n_trials, explore_scale=explore_scale,
                                      conf_thresh=conf_thresh, baseline_alpha=baseline_alpha, Q=Q, seed=self.seed, w_sin=w_sin, w_cos=w_cos, targets=None)
        return phis