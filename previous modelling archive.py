"""
Created on Sat Mar 19 12:51:10 2022
@author: 44796
"""



import numpy as np
import math
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
import numba as nb
from numba import njit, float64, types
from numba.experimental import jitclass
np.random.seed(99)
@njit
def norm_pdf(x, mu, sigma):
    if sigma <= 0.0:
        return 0.0
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    return inv_sqrt_2pi / sigma * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
@njit
def wrap_angle(x):
    return (x + 180.0) % 360.0 - 180.0
spec = [
    ('sigma0', float64),
    ('sigma1', float64),
    ('hazard', float64),
    ('rotation', float64),
    ('betaDecay', float64),
    ('betaConfidence', float64),
    ('activation_surprise', float64),
    ('mu', float64),
    ('kappa', float64),
    ('alpha', float64),
    ('beta', float64),
    ('alpha0', float64),
    ('beta0', float64),
    ('transMat', float64[:,:]),
    ('prevStatePosterior', float64[:]),
    ('trialCount', types.int64),
    ('uCos', float64),
    ('uSin', float64),
    ('w', float64),
    ('pFlip', float64),
    ('s1_active', types.boolean),
    ('cum_surprise', float64),
]
@jitclass(spec)
class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, hazard=0.01, rotation=30.0, betaDecay=0.01, betaConfidence=0.01, activation_surprise=5.0):
        self.sigma0 = sigma0                                  
        self.sigma1 = sigma0                    
        self.hazard = hazard
        self.rotation = rotation
        self.betaDecay = betaDecay
        self.betaConfidence = betaConfidence
        self.activation_surprise = activation_surprise
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.alpha0 = 20.0
        self.beta0 = self.sigma0 ** 2 * (self.alpha0 - 1)
        self.transMat = np.array([[1 - self.hazard, self.hazard],
                                  [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
        self.uCos = 0.0
        self.uSin = 0.0
        self.w = 0.0
        self.pFlip = 0.0
        self.s1_active = False
        self.cum_surprise = 0.0
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior if self.s1_active else np.array([1.0, 0.0])
        pS0 = predState[0]
        pS1 = predState[1]
        deltaG = self.mu
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        var0 = self.beta0 / (self.alpha0 - 1) if self.alpha0 > 1 else self.sigma0 ** 2
        if self.w == 0:
            pFlip = 0.0
        else:
            radCurrent = np.deg2rad(currentTarget)
            cosC = np.cos(radCurrent)
            sinC = np.sin(radCurrent)
            meanCos = self.uCos / self.w
            meanSin = self.uSin / self.w
            weightedSim = cosC * meanCos + sinC * meanSin
            baseFlip = (1 - weightedSim) / 2
            pFlip = baseFlip * np.exp(-self.betaConfidence * self.kappa)
            self.pFlip = pFlip
        return {'pS0': pS0, 'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = wrap_angle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior if self.s1_active else np.array([1.0, 0.0])
        pS0 = predState[0]
        pS1 = predState[1]
        var0 = self.beta0 / (self.alpha0 - 1) if self.alpha0 > 1 else self.sigma0 ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        lik0 = norm_pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        lik1 = 0.0
        if self.s1_active:
            if margVar1 > 1e100 or np.isnan(margVar1):
                lik1 = 1e-300
            else:
                lik1 = (1 - self.pFlip) * norm_pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm_pdf(deltaObs, -self.mu, np.sqrt(margVar1))
                lik1 = max(lik1, 1e-300)
        if not self.s1_active:
            surprise = -math.log(lik0) if lik0 > 0 else 1e10
            self.cum_surprise += surprise
            if self.cum_surprise > self.activation_surprise:
                self.s1_active = True
                                                           
                self.mu = 0.0
                self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
                self.alpha = 3.0
                self.beta = self.sigma1 ** 2 * (self.alpha - 1)
                                                         
                predState = np.array([1 - self.hazard, self.hazard])
                pS0 = predState[0]
                pS1 = predState[1]
                                                                       
                eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
                thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
                margVar1 = eSigma2 + thetaVar
                if margVar1 > 1e100 or np.isnan(margVar1):
                    lik1 = 1e-300
                else:
                    lik1 = (1 - self.pFlip) * norm_pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm_pdf(deltaObs, -self.mu, np.sqrt(margVar1))
                    lik1 = max(lik1, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w0 = post[0]
        w1 = post[1]
        if w1 > 0:
            newKappa = self.kappa + w1
            newMu = (self.kappa * self.mu + w1 * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w1
            res = deltaObs - self.mu
            betaAdd = 0.5 * w1 * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = wrap_angle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
        if w0 > 0:
            newAlpha0 = self.alpha0 + 0.5 * w0
            betaAdd0 = 0.5 * w0 * (deltaObs ** 2)
            newBeta0 = self.beta0 + betaAdd0
            self.alpha0 = newAlpha0
            self.beta0 = newBeta0
            if self.alpha0 > 1:
                self.sigma0 = np.sqrt(self.beta0 / (self.alpha0 - 1))
        radCurrent = np.deg2rad(currentTarget)
        cosCurrent = np.cos(radCurrent)
        sinCurrent = np.sin(radCurrent)
        expDecay = np.exp(-self.betaDecay)
        self.uCos = cosCurrent + expDecay * self.uCos
        self.uSin = sinCurrent + expDecay * self.uSin
        self.w = 1 + expDecay * self.w
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS1 = predDict['pS1']
        expectedAim = pS1 * ((1 - pFlip) * (-deltaG) + pFlip * deltaG)
        expectedAim = wrap_angle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logBeta, logBetaConfidence, logActivationSurprise = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        hazard = np.exp(logHazard * 10)
        betaDecay = np.exp(logBeta)
        betaConfidence = np.exp(logBetaConfidence)
        activation_surprise = np.exp(logActivationSurprise)
        stepper = BayesianStepper(sigma0, sigma1Init, hazard, self.rotation, betaDecay, betaConfidence, activation_surprise)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS0 = predDict['pS0']
            pS1 = predDict['pS1']
            deltaG = predDict['deltaG']
            varG = predDict['varG']
            var0 = predDict['var0']
            pFlip = predDict['p_flip']
            aim = self.allAims[trial]
            pdf0 = norm.pdf(aim, 0, np.sqrt(var0)) + 1e-300
            pdfNoflip = norm.pdf(aim, -deltaG, np.sqrt(varG)) + 1e-300
            pdfFlip = norm.pdf(aim, deltaG, np.sqrt(varG)) + 1e-300
            pdf1 = (1 - pFlip) * pdfNoflip + pFlip * pdfFlip
            pdf = pS0 * pdf0 + pS1 * pdf1
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 6,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1), np.log(1e2)),            
        (np.log(1), np.log(1e2)),                 
        (np.log(1e-9), np.log(1e-1)),            
        (np.log(1e-6), np.log(1e1)),                
        (np.log(1e-4), np.log(1e3)),                    
        (np.log(10), np.log(1e4)),                        
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 1
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts:                                        
        popSize = 128
        sigma = 64     
        np.random.seed(999 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=999 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 50:                                      
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [objFunc(x) for x in xSamples]                    
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                print(pp,restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                if currentBest[1] < bestValue * 0.9995:
                    globalSinceBest = 0
                else:
                    pass                     
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                globalSinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.9995:
                    sinceBest = 0
                else:
                    pass               
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize(objFunc, bestX, bounds=boundsSingle, method='L-BFGS-B')
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 6
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logBeta, logBetaConfidence, logActivationSurprise = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    betaDecay = np.exp(logBeta)
    betaConfidence = np.exp(logBetaConfidence)
    activation_surprise = np.exp(logActivationSurprise)
    xs = [sigma0, sigma1Init, hazard, betaDecay, betaConfidence, activation_surprise]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, hazard, conVal, betaDecay, betaConfidence, activation_surprise)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(pp,logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, betaDecay, betaConfidence, activation_surprise = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, betaDecay, betaConfidence, activation_surprise)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        varG = predDict['varG']
        var0 = predDict['var0']
        if not np.isnan(pFlip):
            states = np.random.choice([0, 1], size=numSamples, p=[pS0, pS1])
            samples = np.zeros(numSamples)
            for j in range(numSamples):
                s = states[j]
                if s == 0:
                    samples[j] = np.random.normal(0, np.sqrt(var0))
                elif s == 1:
                    flip = np.random.binomial(1, pFlip)
                    mean = deltaG if flip else -deltaG
                    samples[j] = np.random.normal(mean, np.sqrt(varG))
            samples = (samples + 180) % 360 - 180
        else:
            samples = np.array([])
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
               
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, betaDecay, betaConfidence, activation_surprise = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, betaDecay, betaConfidence, activation_surprise)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    data = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pFlip = predDict['p_flip']
        varG = predDict['varG']
        var0 = predDict['var0']
        components = [
            {'type': 'S0', 'mean': 0, 'std': np.sqrt(var0), 'weight': pS0},
            {'type': 'No Flip', 'mean': -deltaG, 'std': np.sqrt(varG), 'weight': pS1 * (1 - pFlip)},
            {'type': 'Flip', 'mean': deltaG, 'std': np.sqrt(varG), 'weight': pS1 * pFlip}
        ]
        for comp in components:
            if comp['weight'] > 1e-6 and comp['std'] > 0:
                n = int(np.round(comp['weight'] * numSamples))
                if n > 0:
                    samples = np.random.normal(comp['mean'], comp['std'], n)
                    samples = (samples + 180) % 360 - 180
                    for s in samples:
                        data.append({'trial': t, 'aim': s, 'component': comp['type']})
    df = pd.DataFrame(data)
    palette = {'S0': 'green', 'No Flip': 'blue', 'Flip': 'red'}
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=0.5, legend=False, ax=ax)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
               
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)]                               
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
    
                                         
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as pool:
            results = pool.map(fitSingle, dataList)
    
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues)                                 
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0                              
    return 1 - (ssRes / ssTot)












import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
from numba import njit
from numba.typed import Dict
from numba.types import float64, int64
np.random.seed(99)

@njit
def wrapAngle(x):
    return (x + 180) % 360 - 180

@njit
def gaussianPdf(x, mu, sigma):
    if sigma <= 0:
        return 0.0
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

@njit
def computeNegLoglik(params, allAims, mask, deltaObsArray, targets, uniqueThetas):
    logSigma0, logHazard, logEta = params
    sigma0 = np.exp(logSigma0)
    hazard = np.exp(logHazard * 10)
    eta = np.exp(logEta)
    
                              
    m = len(uniqueThetas)
    thetaToIdx = Dict.empty(key_type=float64, value_type=int64)
    for i in range(m):
        thetaToIdx[uniqueThetas[i]] = i
    
    muV = np.zeros(m)
    sigmaV = np.zeros((m, m))
    
    transMat = np.array([[1 - hazard, hazard], [hazard, 1 - hazard]])
    prevStatePosterior = np.array([1.0, 0.0])
    alpha0 = 20.0
    beta0 = sigma0 ** 2 * (alpha0 - 1)
    mu = 0.0
    sigma1 = sigma0
    kappa = sigma1 ** 2 / 1e12 if sigma1 > 0 else 1e-6
    alpha = 3.0
    beta = sigma1 ** 2 * (alpha - 1)
    var0 = sigma0 ** 2
    logLikelihood = 0.0
    numTrials = len(allAims)
    trialCount = 0
    
    for idx in range(numTrials):
        currentTarget = targets[idx]
        
                       
        predState = transMat.T @ prevStatePosterior
        pS0 = predState[0]
        pS1 = predState[1]
        deltaG = mu
        eSigma2 = beta / (alpha - 1) if alpha > 1 else sigma1 ** 2
        thetaVar = eSigma2 / kappa if kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        pFlip = 0.0
        if m > 0 and currentTarget in thetaToIdx:
            idxT = thetaToIdx[currentTarget]
            v = muV[idxT]
            pFlip = (1 - v) / 2
            pFlip = max(min(pFlip, 1.0), 0.0)
        
                                    
        aim = allAims[idx]
        pdf0 = gaussianPdf(aim, 0.0, np.sqrt(var0)) + 1e-300
        pdfNoFlip = gaussianPdf(aim, -deltaG, np.sqrt(varG)) + 1e-300
        pdfFlip = gaussianPdf(aim, deltaG, np.sqrt(varG)) + 1e-300
        pdf1 = (1 - pFlip) * pdfNoFlip + pFlip * pdfFlip
        pdf = pS0 * pdf0 + pS1 * pdf1
        if np.isinf(pdf) or np.isnan(pdf):
            pdf = 1e-300
        if mask[idx]:
            logLikelihood += np.log(pdf)
        
                          
        deltaObs = deltaObsArray[idx]
        deltaObs = wrapAngle(deltaObs)
        
                                    
        margVar1 = eSigma2 + thetaVar
        lik1 = 1e-300
        if margVar1 <= 1e100 and not np.isnan(margVar1):
            lik1 = (1 - pFlip) * gaussianPdf(deltaObs, mu, np.sqrt(margVar1)) + pFlip * gaussianPdf(deltaObs, -mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = gaussianPdf(deltaObs, 0.0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        prevStatePosterior = post
        
        w0 = post[0]
        if w0 > 0:
            alpha0 += 0.5 * w0
            beta0 += 0.5 * w0 * (deltaObs ** 2)
            if alpha0 > 1:
                var0 = beta0 / (alpha0 - 1)
        
        w = post[1]
        if w > 0:
            newKappa = kappa + w
            newMu = (kappa * mu + w * deltaObs) / newKappa
            newAlpha = alpha + 0.5 * w
            res = deltaObs - mu
            betaAdd = 0.5 * w * (res ** 2) * kappa / newKappa
            newBeta = beta + betaAdd
            mu = wrapAngle(newMu)
            kappa = newKappa
            alpha = newAlpha
            beta = newBeta
            if alpha > 1:
                sigma1 = np.sqrt(beta / (alpha - 1))
            
            inferredV = np.sign(deltaObs)
            varV = 1.0 / (eta * kappa) if kappa > 0 else 1e6
            varV = max(varV, 1e-6)
            
            if m > 0:
                l = 180.0 + kappa                                 
                kCurrent = np.zeros((m, m))
                for i in range(m):
                    for j in range(m):
                        dVal = np.abs(uniqueThetas[i] - uniqueThetas[j])
                        dVal = min(dVal, 360 - dVal)
                        kCurrent[i, j] = np.cos(np.pi * dVal / l)
                eig = np.linalg.eigvalsh(kCurrent)
                minEig = np.min(eig)
                if minEig < -1e-10:
                    nugget = -minEig + 1e-6
                    kCurrent += nugget * np.eye(m)
                sigmaV += kCurrent
                
                if currentTarget in thetaToIdx:
                    idxT = thetaToIdx[currentTarget]
                    h = np.zeros(m)
                    h[idxT] = 1.0
                    innovation = inferredV - np.dot(h, muV)
                    s = np.dot(h, np.dot(sigmaV, h)) + varV
                    if s < 1e-10:
                        s = 1e-10
                    kGain = np.dot(sigmaV, h) / s
                    muV += kGain * innovation
                    sigmaV -= np.outer(kGain, np.dot(h, sigmaV))
        
        trialCount += 1
    
    if not np.isfinite(logLikelihood):
        return 1e9
    return -logLikelihood

class BayesianStepper:
    def __init__(self, sigma0=0.1, hazard=0.01, rotation=30.0, eta=1.0, uniqueTargets=None):
        self.sigma0 = sigma0
        self.sigma1 = sigma0
        self.hazard = hazard
        self.rotation = rotation
        self.eta = eta
        self.alpha0 = 20.0
        self.beta0 = self.sigma0 ** 2 * (self.alpha0 - 1)
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.transMat = np.array([[1 - self.hazard, self.hazard],
                                  [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
        self.pFlip = 0
        self.var0 = self.sigma0 ** 2
      
                                              
        if uniqueTargets is None or len(uniqueTargets) == 0:
            self.m = 0
            self.muV = None
            self.sigmaV = None
            self.thetaToIdx = {}
        else:
            self.uniqueThetas = np.sort(np.unique(uniqueTargets))
            self.m = len(self.uniqueThetas)
            self.thetaToIdx = {th: i for i, th in enumerate(self.uniqueThetas)}
            self.kPrior = np.zeros((self.m, self.m))
            if self.m > 0:
                th1 = np.zeros((self.m, self.m))
                th2 = np.zeros((self.m, self.m))
                for i in range(self.m):
                    th1[i, :] = self.uniqueThetas[i]
                for j in range(self.m):
                    th2[:, j] = self.uniqueThetas[j]
                d = self.wrapAngle(th1 - th2)
                d = np.minimum(np.abs(d), 360 - np.abs(d))
                self.kPrior = np.cos(np.pi * d / 180)
            eig = np.linalg.eigvalsh(self.kPrior)
            minEig = np.min(eig)
            if minEig < -1e-10:
                nugget = -minEig + 1e-6
                self.kPrior += nugget * np.eye(self.m)
            self.muV = np.zeros(self.m)
            self.sigmaV = self.kPrior.copy()
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        deltaG = self.mu
        var0 = self.var0
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.m == 0 or currentTarget not in self.thetaToIdx:
            pFlip = 0.0
        else:
            idx = self.thetaToIdx[currentTarget]
            v = self.muV[idx]
            pFlip = (1 - v) / 2
            pFlip = np.clip(pFlip, 0, 1)
        self.pFlip = pFlip
        return {'pS0': pS0, 'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        var0 = self.var0
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * gaussianPdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * gaussianPdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = gaussianPdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w0 = post[0]
        if w0 > 0:
            newAlpha0 = self.alpha0 + 0.5 * w0
            betaAdd0 = 0.5 * w0 * (deltaObs ** 2)
            newBeta0 = self.beta0 + betaAdd0
            self.alpha0 = newAlpha0
            self.beta0 = newBeta0
            if self.alpha0 > 1:
                self.var0 = self.beta0 / (self.alpha0 - 1)
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
          
                                                                                                                                       
            inferredV = np.sign(deltaObs)
            varV = 1.0 / (self.eta * self.kappa) if self.kappa > 0 else 1e6
            varV = max(varV, 1e-6)
          
            if self.m > 0:
                l = 180.0 + self.kappa                                 
                kCurrent = np.zeros((self.m, self.m))
                for i in range(self.m):
                    for j in range(self.m):
                        dVal = np.abs(self.uniqueThetas[i] - self.uniqueThetas[j])
                        dVal = min(dVal, 360 - dVal)
                        kCurrent[i, j] = np.cos(np.pi * dVal / l)
                eig = np.linalg.eigvalsh(kCurrent)
                minEig = np.min(eig)
                if minEig < -1e-10:
                    nugget = -minEig + 1e-6
                    kCurrent += nugget * np.eye(self.m)
                self.sigmaV += kCurrent
                
                if currentTarget in self.thetaToIdx:
                    idx = self.thetaToIdx[currentTarget]
                    h = np.zeros(self.m)
                    h[idx] = 1.0
                    innovation = inferredV - np.dot(h, self.muV)
                    s = np.dot(h, np.dot(self.sigmaV, h)) + varV
                    if s < 1e-10:
                        s = 1e-10
                    kGain = np.dot(self.sigmaV, h) / s
                    self.muV += kGain * innovation
                    self.sigmaV -= np.outer(kGain, np.dot(h, self.sigmaV))
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS1 = predDict['pS1']
        expectedAim = pS1 * ((1 - pFlip) * (-deltaG) + pFlip * deltaG)
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, phases, rotation, targets, uniqueTargets):
        self.allAims = allAims
        self.mask = mask
        self.deltaObsArray = np.where(phases == 'rotation', rotation, 0.0).astype(np.float64)
        self.targets = targets
        self.uniqueTargets = np.sort(np.unique(uniqueTargets))
    def __call__(self, params):
        return computeNegLoglik(params, self.allAims, self.mask, self.deltaObsArray, self.targets, self.uniqueTargets)
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets = data
    objFunc = Objective(allAims, mask, phases, conVal, targets, uniqueTargets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 3,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1), np.log(1e2)),            
        (np.log(1e-9), np.log(1)),            
        (np.log(1e3), np.log(1e9)),         
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 20
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts:
        popSize = 6
        sigma = 6
        np.random.seed(0 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=0 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 50:                     
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [objFunc(x) for x in xSamples]
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                print(pp,restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                if currentBest[1] < bestValue * 0.9999:
                    globalSinceBest = 0
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                globalSinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.9999:
                    sinceBest = 0
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize(objFunc, bestX, bounds=boundsSingle, method='L-BFGS-B')
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 3
    logLikelihood = -bestFun
    logSigma0, logHazard, logEta = bestX
    sigma0 = np.exp(logSigma0)
    hazard = np.exp(logHazard * 10)
    eta = np.exp(logEta)
    xs = [sigma0, hazard, eta]
    stepperSingle = BayesianStepper(sigma0, hazard, conVal, eta, uniqueTargets)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(pp,logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, hazard, eta = fittedParams
    uniqueTargets = np.unique(targets)
    stepper = BayesianStepper(sigma0, hazard, rotation, eta, uniqueTargets)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        varG = predDict['varG']
        var0 = predDict['var0']
        if np.isnan(pFlip):
            samples = np.array([])
        else:
            states = np.random.choice([0, 1], size=numSamples, p=[pS0, pS1])
            mask0 = states == 0
            mask1 = states == 1
            num0 = mask0.sum()
            num1 = mask1.sum()
            samples0 = np.random.normal(0, np.sqrt(var0), size=num0)
            flips = np.random.binomial(1, pFlip, size=num1)
            means1 = np.where(flips == 1, deltaG, -deltaG)
            samples1 = np.random.normal(means1, np.sqrt(varG), size=num1)
            samples = np.empty(numSamples)
            samples[mask0] = samples0
            samples[mask1] = samples1
            samples = (samples + 180) % 360 - 180
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, hazard, eta = fittedParams
    uniqueTargets = np.unique(targets)
    stepper = BayesianStepper(sigma0, hazard, rotation, eta, uniqueTargets)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    trialsList = []
    aimsList = []
    componentsList = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pFlip = predDict['p_flip']
        varG = predDict['varG']
        var0 = predDict['var0']
        components = [
            {'type': 'S0', 'mean': 0, 'std': np.sqrt(var0), 'weight': pS0},
            {'type': 'No Flip', 'mean': -deltaG, 'std': np.sqrt(varG), 'weight': pS1 * (1 - pFlip)},
            {'type': 'Flip', 'mean': deltaG, 'std': np.sqrt(varG), 'weight': pS1 * pFlip},
        ]
        for comp in components:
            if comp['weight'] > 1e-6 and comp['std'] > 0:
                n = int(np.round(comp['weight'] * numSamples))
                if n > 0:
                    samples = np.random.normal(comp['mean'], comp['std'], n)
                    samples = (samples + 180) % 360 - 180
                    trialsList.extend([t] * n)
                    aimsList.extend(samples.tolist())
                    componentsList.extend([comp['type']] * n)
    df = pd.DataFrame({'trial': trialsList, 'aim': aimsList, 'component': componentsList})
    palette = {'S0': 'green', 'No Flip': 'blue', 'Flip': 'red'}
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=0.5, legend=False, ax=ax)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legendHandles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legendHandles)
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)]
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            uniqueTargets = np.unique(targetPositions)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions, uniqueTargets))
   
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//12) as pool:
            results = pool.map(fitSingle, dataList)
   
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues)
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0
    return 1 - (ssRes / ssTot)





    
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)
class BayesianStepper:
    def __init__(self, sigma0=0.1, hazard=0.01, rotation=30.0, betaDecay=0.01, eta=1.0, uniqueTargets=None):
        self.sigma0 = sigma0
        self.sigma1 = sigma0
        self.hazard = hazard
        self.rotation = rotation
        self.betaDecay = betaDecay
        self.eta = eta
        self.alpha0 = 20.0
        self.beta0 = self.sigma0 ** 2 * (self.alpha0 - 1)
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.transMat = np.array([[1 - self.hazard, self.hazard],
                                  [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
        self.pFlip = 0
        self.var0 = self.sigma0 ** 2
      
        # Discrete GP setup for unique targets
        if uniqueTargets is None or len(uniqueTargets) == 0:
            self.m = 0
            self.muV = None
            self.sigmaV = None
            self.thetaToIdx = {}
        else:
            self.uniqueThetas = np.sort(np.unique(uniqueTargets))
            self.m = len(self.uniqueThetas)
            self.thetaToIdx = {th: i for i, th in enumerate(self.uniqueThetas)}
            self.kPrior = np.zeros((self.m, self.m))
            for i, th1 in enumerate(self.uniqueThetas):
                for j, th2 in enumerate(self.uniqueThetas):
                    d = self.wrapAngle(th1 - th2)
                    d = min(abs(d), 360 - abs(d))
                    self.kPrior[i, j] = np.cos(np.pi * d / 180)
            eig = np.linalg.eigvals(self.kPrior)
            if np.min(eig) < -1e-10:
                nugget = -np.min(eig) + 1e-6
                self.kPrior += nugget * np.eye(self.m)
            self.muV = np.zeros(self.m)
            self.sigmaV = self.kPrior.copy()
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        deltaG = self.mu
        var0 = self.var0
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.m == 0 or currentTarget not in self.thetaToIdx:
            pFlip = 0.0
        else:
            idx = self.thetaToIdx[currentTarget]
            v = self.muV[idx]
            pFlip = (1 - v) / 2
            pFlip = np.clip(pFlip, 0, 1)
        self.pFlip = pFlip
        return {'pS0': pS0, 'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        var0 = self.var0
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w0 = post[0]
        if w0 > 0:
            newAlpha0 = self.alpha0 + 0.5 * w0
            betaAdd0 = 0.5 * w0 * (deltaObs ** 2)
            newBeta0 = self.beta0 + betaAdd0
            self.alpha0 = newAlpha0
            self.beta0 = newBeta0
            if self.alpha0 > 1:
                self.var0 = self.beta0 / (self.alpha0 - 1)
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
          
            # Update GP with inferred sign from observation (positive for rotational, propagates opposite for directional misinference)
            inferredV = np.sign(deltaObs)
            varV = 1.0 / (self.eta * self.kappa) if self.kappa > 0 else 1e6
            varV = max(varV, 1e-6)
          
            if self.m > 0:
                decay = np.exp(-self.betaDecay)
                self.muV *= decay
                self.sigmaV = decay ** 2 * self.sigmaV + (1 - decay ** 2) * self.kPrior
              
                if currentTarget in self.thetaToIdx:
                    idx = self.thetaToIdx[currentTarget]
                    h = np.zeros(self.m)
                    h[idx] = 1.0
                    innovation = inferredV - h @ self.muV
                    s = h @ self.sigmaV @ h.T + varV
                    if s < 1e-10:
                        s = 1e-10
                    kGain = (self.sigmaV @ h.T) / s
                    self.muV += kGain * innovation
                    self.sigmaV -= np.outer(kGain, h @ self.sigmaV)
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS1 = predDict['pS1']
        expectedAim = pS1 * ((1 - pFlip) * (-deltaG) + pFlip * deltaG)
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets, uniqueTargets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
        self.uniqueTargets = uniqueTargets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logHazard, logBeta, logEta = params
        sigma0 = np.exp(logSigma0)
        hazard = np.exp(logHazard * 10)
        betaDecay = np.exp(logBeta)
        eta = np.exp(logEta)
        stepper = BayesianStepper(sigma0, hazard, self.rotation, betaDecay, eta, self.uniqueTargets)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS0 = predDict['pS0']
            pS1 = predDict['pS1']
            deltaG = predDict['deltaG']
            varG = predDict['varG']
            var0 = predDict['var0']
            pFlip = predDict['p_flip']
            aim = self.allAims[trial]
            pdf0 = norm.pdf(aim, 0, np.sqrt(var0)) + 1e-300
            pdfNoFlip = norm.pdf(aim, -deltaG, np.sqrt(varG)) + 1e-300
            pdfFlip = norm.pdf(aim, deltaG, np.sqrt(varG)) + 1e-300
            pdf1 = (1 - pFlip) * pdfNoFlip + pFlip * pdfFlip
            pdf = pS0 * pdf0 + pS1 * pdf1
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets, uniqueTargets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 4,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1), np.log(1e2)), # logSigma0
        (np.log(1e-9), np.log(1)), # logHazard
        (np.log(1e-6), np.log(1e1)), # logBeta_decay
        (np.log(1e3), np.log(1e9)), # logEta
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 20
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts:
        popSize = 6
        sigma = 6
        np.random.seed(0 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=0 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 50:# and iteration < 40:
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [objFunc(x) for x in xSamples]
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                print(pp,restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                if currentBest[1] < bestValue * 0.9999:
                    globalSinceBest = 0
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                globalSinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.9999:
                    sinceBest = 0
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize(objFunc, bestX, bounds=boundsSingle, method='L-BFGS-B')
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 4
    logLikelihood = -bestFun
    logSigma0, logHazard, logBeta, logEta = bestX
    sigma0 = np.exp(logSigma0)
    hazard = np.exp(logHazard * 10)
    betaDecay = np.exp(logBeta)
    eta = np.exp(logEta)
    xs = [sigma0, hazard, betaDecay, eta]
    stepperSingle = BayesianStepper(sigma0, hazard, conVal, betaDecay, eta, uniqueTargets)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(pp,logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, hazard, betaDecay, eta = fittedParams
    uniqueTargets = np.unique(targets)
    stepper = BayesianStepper(sigma0, hazard, rotation, betaDecay, eta, uniqueTargets)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        varG = predDict['varG']
        var0 = predDict['var0']
        if not np.isnan(pFlip):
            states = np.random.choice([0, 1], size=numSamples, p=[pS0, pS1])
            samples = np.zeros(numSamples)
            for j in range(numSamples):
                s = states[j]
                if s == 0:
                    samples[j] = np.random.normal(0, np.sqrt(var0))
                elif s == 1:
                    flip = np.random.binomial(1, pFlip)
                    mean = deltaG if flip else -deltaG
                    samples[j] = np.random.normal(mean, np.sqrt(varG))
            samples = (samples + 180) % 360 - 180
        else:
            samples = np.array([])
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, hazard, betaDecay, eta = fittedParams
    uniqueTargets = np.unique(targets)
    stepper = BayesianStepper(sigma0, hazard, rotation, betaDecay, eta, uniqueTargets)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    data = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pFlip = predDict['p_flip']
        varG = predDict['varG']
        var0 = predDict['var0']
        components = [
            {'type': 'S0', 'mean': 0, 'std': np.sqrt(var0), 'weight': pS0},
            {'type': 'No Flip', 'mean': -deltaG, 'std': np.sqrt(varG), 'weight': pS1 * (1 - pFlip)},
            {'type': 'Flip', 'mean': deltaG, 'std': np.sqrt(varG), 'weight': pS1 * pFlip},
        ]
        for comp in components:
            if comp['weight'] > 1e-6 and comp['std'] > 0:
                n = int(np.round(comp['weight'] * numSamples))
                if n > 0:
                    samples = np.random.normal(comp['mean'], comp['std'], n)
                    samples = (samples + 180) % 360 - 180
                    for s in samples:
                        data.append({'trial': t, 'aim': s, 'component': comp['type']})
    df = pd.DataFrame(data)
    palette = {'S0': 'green', 'No Flip': 'blue', 'Flip': 'red'}
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=0.5, legend=False, ax=ax)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
"""     







"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)
class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, hazard=0.01, rotation=30.0, betaDecay=0.01, eta=1.0, uniqueTargets=None):
        self.sigma0 = sigma0
        self.sigma1 = sigma1Init + sigma0
        self.hazard = hazard
        self.rotation = rotation
        self.betaDecay = betaDecay
        self.eta = eta
        self.alpha0 = 20.0
        self.beta0 = self.sigma0 ** 2 * (self.alpha0 - 1)
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.transMat = np.array([[1 - self.hazard, self.hazard],
                                  [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
        self.pFlip = 0
        self.var0 = self.sigma0 ** 2
       
        # Discrete GP setup for unique targets
        if uniqueTargets is None or len(uniqueTargets) == 0:
            self.m = 0
            self.muV = None
            self.sigmaV = None
            self.thetaToIdx = {}
        else:
            self.uniqueThetas = np.sort(np.unique(uniqueTargets))
            self.m = len(self.uniqueThetas)
            self.thetaToIdx = {th: i for i, th in enumerate(self.uniqueThetas)}
            self.kPrior = np.zeros((self.m, self.m))
            for i, th1 in enumerate(self.uniqueThetas):
                for j, th2 in enumerate(self.uniqueThetas):
                    d = self.wrapAngle(th1 - th2)
                    d = min(abs(d), 360 - abs(d))
                    self.kPrior[i, j] = np.cos(np.pi * d / 180)
            eig = np.linalg.eigvals(self.kPrior)
            if np.min(eig) < -1e-10:
                nugget = -np.min(eig) + 1e-6
                self.kPrior += nugget * np.eye(self.m)
            self.muV = np.zeros(self.m)
            self.sigmaV = self.kPrior.copy()
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        deltaG = self.mu
        var0 = self.var0
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.m == 0 or currentTarget not in self.thetaToIdx:
            pFlip = 0.0
        else:
            idx = self.thetaToIdx[currentTarget]
            v = self.muV[idx]
            pFlip = (1 - v) / 2
            pFlip = np.clip(pFlip, 0, 1)
        self.pFlip = pFlip
        return {'pS0': pS0, 'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        var0 = self.var0
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w0 = post[0]
        if w0 > 0:
            newAlpha0 = self.alpha0 + 0.5 * w0
            betaAdd0 = 0.5 * w0 * (deltaObs ** 2)
            newBeta0 = self.beta0 + betaAdd0
            self.alpha0 = newAlpha0
            self.beta0 = newBeta0
            if self.alpha0 > 1:
                self.var0 = self.beta0 / (self.alpha0 - 1)
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
           
            # Update GP with inferred sign from observation (positive for rotational, propagates opposite for directional misinference)
            inferredV = np.sign(deltaObs)
            varV = 1.0 / (self.eta * self.kappa) if self.kappa > 0 else 1e6
            varV = max(varV, 1e-6)
           
            if self.m > 0:
                decay = np.exp(-self.betaDecay)
                self.muV *= decay
                self.sigmaV = decay ** 2 * self.sigmaV + (1 - decay ** 2) * self.kPrior
               
                if currentTarget in self.thetaToIdx:
                    idx = self.thetaToIdx[currentTarget]
                    h = np.zeros(self.m)
                    h[idx] = 1.0
                    innovation = inferredV - h @ self.muV
                    s = h @ self.sigmaV @ h.T + varV
                    if s < 1e-10:
                        s = 1e-10
                    kGain = (self.sigmaV @ h.T) / s
                    self.muV += kGain * innovation
                    self.sigmaV -= np.outer(kGain, h @ self.sigmaV)
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS1 = predDict['pS1']
        expectedAim = pS1 * ((1 - pFlip) * (-deltaG) + pFlip * deltaG)
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets, uniqueTargets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
        self.uniqueTargets = uniqueTargets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logBeta, logEta = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        hazard = np.exp(logHazard * 10)
        betaDecay = np.exp(logBeta)
        eta = np.exp(logEta)
        stepper = BayesianStepper(sigma0, sigma1Init, hazard, self.rotation, betaDecay, eta, self.uniqueTargets)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS0 = predDict['pS0']
            pS1 = predDict['pS1']
            deltaG = predDict['deltaG']
            varG = predDict['varG']
            var0 = predDict['var0']
            pFlip = predDict['p_flip']
            aim = self.allAims[trial]
            pdf0 = norm.pdf(aim, 0, np.sqrt(var0)) + 1e-300
            pdfNoFlip = norm.pdf(aim, -deltaG, np.sqrt(varG)) + 1e-300
            pdfFlip = norm.pdf(aim, deltaG, np.sqrt(varG)) + 1e-300
            pdf1 = (1 - pFlip) * pdfNoFlip + pFlip * pdfFlip
            pdf = pS0 * pdf0 + pS1 * pdf1
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets, uniqueTargets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 5,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1), np.log(1e2)), # logSigma0
        (-10, np.log(1e3)), # logSigma1_init
        (np.log(1e-9), np.log(1)), # logHazard
        (np.log(1e-6), np.log(1e1)), # logBeta_decay
        (np.log(1e3), np.log(1e9)), # logEta
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 20
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts:
        popSize = 6
        sigma = 6
        np.random.seed(0 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=0 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 50:# and iteration < 40:
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [objFunc(x) for x in xSamples]
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                print(pp,restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                if currentBest[1] < bestValue * 0.9999:
                    globalSinceBest = 0
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                globalSinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.9999:
                    sinceBest = 0
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize(objFunc, bestX, bounds=boundsSingle, method='L-BFGS-B')
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 5
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logBeta, logEta = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    betaDecay = np.exp(logBeta)
    eta = np.exp(logEta)
    xs = [sigma0, sigma1Init, hazard, betaDecay, eta]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, hazard, conVal, betaDecay, eta, uniqueTargets)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(pp,logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, betaDecay, eta = fittedParams
    uniqueTargets = np.unique(targets)
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, betaDecay, eta, uniqueTargets)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        varG = predDict['varG']
        var0 = predDict['var0']
        if not np.isnan(pFlip):
            states = np.random.choice([0, 1], size=numSamples, p=[pS0, pS1])
            samples = np.zeros(numSamples)
            for j in range(numSamples):
                s = states[j]
                if s == 0:
                    samples[j] = np.random.normal(0, np.sqrt(var0))
                elif s == 1:
                    flip = np.random.binomial(1, pFlip)
                    mean = deltaG if flip else -deltaG
                    samples[j] = np.random.normal(mean, np.sqrt(varG))
            samples = (samples + 180) % 360 - 180
        else:
            samples = np.array([])
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, betaDecay, eta = fittedParams
    uniqueTargets = np.unique(targets)
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, betaDecay, eta, uniqueTargets)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    data = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pFlip = predDict['p_flip']
        varG = predDict['varG']
        var0 = predDict['var0']
        components = [
            {'type': 'S0', 'mean': 0, 'std': np.sqrt(var0), 'weight': pS0},
            {'type': 'No Flip', 'mean': -deltaG, 'std': np.sqrt(varG), 'weight': pS1 * (1 - pFlip)},
            {'type': 'Flip', 'mean': deltaG, 'std': np.sqrt(varG), 'weight': pS1 * pFlip},
        ]
        for comp in components:
            if comp['weight'] > 1e-6 and comp['std'] > 0:
                n = int(np.round(comp['weight'] * numSamples))
                if n > 0:
                    samples = np.random.normal(comp['mean'], comp['std'], n)
                    samples = (samples + 180) % 360 - 180
                    for s in samples:
                        data.append({'trial': t, 'aim': s, 'component': comp['type']})
    df = pd.DataFrame(data)
    palette = {'S0': 'green', 'No Flip': 'blue', 'Flip': 'red'}
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=0.5, legend=False, ax=ax)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)]
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            uniqueTargets = np.unique(targetPositions)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions, uniqueTargets))
    
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//12) as pool:
            results = pool.map(fitSingle, dataList)
    
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues)
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0
    return 1 - (ssRes / ssTot)
"""

    





"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)
class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, sigmaExpl=0.1, hazard=0.01, rotation=30.0, alphaSim=1.0, shift=1.0, betaDecay=0.01, betaConfidence=0.01):
        self.sigma0 = sigma0
        self.sigma1 = sigma1Init
        self.sigmaExpl = sigmaExpl + sigma0
        self.hazard = hazard
        self.rotation = rotation
        self.alphaSim = alphaSim
        self.shift = shift
        self.betaDecay = betaDecay
        self.betaConfidence = betaConfidence
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.transMat = np.array([[1 - self.hazard, self.hazard / 2, self.hazard / 2],
                                  [self.hazard / 2, 1 - self.hazard, self.hazard / 2],
                                  [self.hazard / 2, self.hazard / 2, 1 - self.hazard]])
        self.prevStatePosterior = np.array([0.0, 0.0, 1.0])
        self.trialCount = 0
        self.uCos = 0.0
        self.uSin = 0.0
        self.w = 0.0
        self.pFlip = 0
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        deltaG = self.mu
        var0 = self.sigma0 ** 2
        varExpl = self.sigmaExpl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.w == 0:
            pFlip = 0.0
        else:
            radCurrent = np.deg2rad(currentTarget)
            cosC = np.cos(radCurrent)
            sinC = np.sin(radCurrent)
            meanCos = self.uCos / self.w
            meanSin = self.uSin / self.w
            weightedSim = cosC * meanCos + sinC * meanSin
            baseFlip = 1 / (1 + np.exp(self.alphaSim * (weightedSim - self.shift)))
            pFlip = baseFlip * np.exp(-self.betaConfidence * self.kappa)
            pFlip = np.clip(pFlip, 0, 1)
            self.pFlip = pFlip
        return {'pS0': pS0, 'pS1': pS1, 'pS2': pS2, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'var_expl': varExpl, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        var0 = self.sigma0 ** 2
        varExpl = self.sigmaExpl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        lik2 = norm.pdf(deltaObs, 0, np.sqrt(varExpl))
        lik2 = max(lik2, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1, lik2 * pS2])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
        radCurrent = np.deg2rad(currentTarget)
        cosCurrent = np.cos(radCurrent)
        sinCurrent = np.sin(radCurrent)
        expDecay = np.exp(-self.betaDecay)
        self.uCos = cosCurrent + expDecay * self.uCos
        self.uSin = sinCurrent + expDecay * self.uSin
        self.w = 1 + expDecay * self.w
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS1 = predDict['pS1']
        expectedAim = pS1 * ((1 - pFlip) * (-deltaG) + pFlip * deltaG)
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logBeta, logBetaConfidence, logSigmaExpl = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        sigmaExpl = np.exp(logSigmaExpl)
        hazard = np.exp(logHazard * 10)
        betaDecay = np.exp(logBeta)
        betaConfidence = np.exp(logBetaConfidence)
        stepper = BayesianStepper(sigma0, sigma1Init, sigmaExpl, hazard, self.rotation, 1.0, 1.0, betaDecay, betaConfidence)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS0 = predDict['pS0']
            pS1 = predDict['pS1']
            pS2 = predDict['pS2']
            deltaG = predDict['deltaG']
            varG = predDict['varG']
            var0 = predDict['var0']
            varExpl = predDict['var_expl']
            pFlip = predDict['p_flip']
            aim = self.allAims[trial]
            pdf0 = norm.pdf(aim, 0, np.sqrt(var0)) + 1e-300
            pdfExpl = norm.pdf(aim, 0, np.sqrt(varExpl)) + 1e-300
            pdfNoflip = norm.pdf(aim, -deltaG, np.sqrt(varG)) + 1e-300
            pdfFlip = norm.pdf(aim, deltaG, np.sqrt(varG)) + 1e-300
            pdf1 = (1 - pFlip) * pdfNoflip + pFlip * pdfFlip
            pdf = pS0 * pdf0 + pS1 * pdf1 + pS2 * pdfExpl
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 6,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1e-9), np.log(1e2)), # logSigma0
        (np.log(1e-9), np.log(1e3)), # logSigma1_init
        (np.log(1e-9), np.log(1)), # logHazard
        (np.log(1e-6), np.log(1e1)), # logBeta_decay
        (np.log(1e-4), np.log(1e3)), # logBetaConfidence
        (np.log(1), np.log(1e3)), # logSigmaExpl
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 1
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts:# and globalsinceBest < 5000 // popSize:
        popSize = 128
        sigma = 128#/= 2
        np.random.seed(999 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=999 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 50 and iteration < 40:#50000 // popSize
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [objFunc(x) for x in xSamples]  # Serial evaluation
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                print(pp,restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                if currentBest[1] < bestValue * 0.9995:
                    globalSinceBest = 0
                else:
                    pass#globalSinceBest += 1
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                globalSinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.9995:
                    sinceBest = 0
                else:
                    pass#sinceBest += 1
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize(objFunc, bestX, bounds=boundsSingle, method='L-BFGS-B')
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 6
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logBeta, logBetaConfidence, logSigmaExpl = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    betaDecay = np.exp(logBeta)
    betaConfidence = np.exp(logBetaConfidence)
    sigmaExpl = np.exp(logSigmaExpl)
    xs = [sigma0, sigma1Init, hazard, betaDecay, betaConfidence, sigmaExpl]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, sigmaExpl, hazard, conVal, 1.0, 1.0, betaDecay, betaConfidence)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(pp,logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, betaDecay, betaConfidence, sigmaExpl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigmaExpl, hazard, rotation, 1.0, 1.0, betaDecay, betaConfidence)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pS2 = predDict['pS2']
        varG = predDict['varG']
        var0 = predDict['var0']
        varExpl = predDict['var_expl']
        if not np.isnan(pFlip):
            states = np.random.choice([0, 1, 2], size=numSamples, p=[pS0, pS1, pS2])
            samples = np.zeros(numSamples)
            for j in range(numSamples):
                s = states[j]
                if s == 0:
                    samples[j] = np.random.normal(0, np.sqrt(var0))
                elif s == 2:
                    samples[j] = np.random.normal(0, np.sqrt(varExpl))
                elif s == 1:
                    flip = np.random.binomial(1, pFlip)
                    mean = deltaG if flip else -deltaG
                    samples[j] = np.random.normal(mean, np.sqrt(varG))
            samples = (samples + 180) % 360 - 180
        else:
            samples = np.array([])
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
    #plt.show()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, betaDecay, betaConfidence, sigmaExpl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigmaExpl, hazard, rotation, 1.0, 1.0, betaDecay, betaConfidence)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    data = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pS2 = predDict['pS2']
        pFlip = predDict['p_flip']
        varG = predDict['varG']
        var0 = predDict['var0']
        varExpl = predDict['var_expl']
        components = [
            {'type': 'S0', 'mean': 0, 'std': np.sqrt(var0), 'weight': pS0},
            {'type': 'No Flip', 'mean': -deltaG, 'std': np.sqrt(varG), 'weight': pS1 * (1 - pFlip)},
            {'type': 'Flip', 'mean': deltaG, 'std': np.sqrt(varG), 'weight': pS1 * pFlip},
            {'type': 'Exploratory', 'mean': 0, 'std': np.sqrt(varExpl), 'weight': pS2}
        ]
        for comp in components:
            if comp['weight'] > 1e-6 and comp['std'] > 0:
                n = int(np.round(comp['weight'] * numSamples))
                if n > 0:
                    samples = np.random.normal(comp['mean'], comp['std'], n)
                    samples = (samples + 180) % 360 - 180
                    for s in samples:
                        data.append({'trial': t, 'aim': s, 'component': comp['type']})
    df = pd.DataFrame(data)
    palette = {'S0': 'green', 'No Flip': 'blue', 'Flip': 'red', 'Exploratory': 'purple'}
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=0.5, legend=False, ax=ax)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
    #plt.show()
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)] # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
        
        # Parallelize across participants
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(fitSingle, dataList)
        
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues) # Ensure inputs are NumPy arrays
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0 # Handle constant true values
    return 1 - (ssRes / ssTot)
"""



"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)
class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, sigma_expl=0.1, hazard=0.01, rotation=30.0, alphaSim=1.0, shift=-0.565, betaDecay=0.01, betaConfidence=0.01):
        self.sigma0 = sigma0
        self.sigma1 = sigma1Init
        self.sigma_expl = sigma_expl #+ sigma0 #force order doesn't work for some reason
        self.hazard = hazard
        self.rotation = rotation
        self.alphaSim = alphaSim
        self.shift = shift
        self.betaDecay = betaDecay
        self.betaConfidence = betaConfidence
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.transMat = np.array([[1 - self.hazard, self.hazard / 2, self.hazard / 2],
                                  [self.hazard / 2, 1 - self.hazard, self.hazard / 2],
                                  [self.hazard / 2, self.hazard / 2, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0, 0.0])
        self.trialCount = 0
        self.uCos = 0.0
        self.uSin = 0.0
        self.w = 0.0
        self.pFlip = 0
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        deltaG = self.mu
        var0 = self.sigma0 ** 2
        var_expl = self.sigma_expl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.w == 0:
            pFlip = 0.0
        else:
            radCurrent = np.deg2rad(currentTarget)
            cosC = np.cos(radCurrent)
            sinC = np.sin(radCurrent)
            meanCos = self.uCos / self.w
            meanSin = self.uSin / self.w
            weightedSim = cosC * meanCos + sinC * meanSin
            baseFlip = 1 / (1 + np.exp(self.alphaSim * (weightedSim - self.shift)))
            pFlip = baseFlip * np.exp(-self.betaConfidence * self.kappa)
            pFlip = np.clip(pFlip, 0, 1)
            self.pFlip = pFlip
        return {'pS0': pS0, 'pS1': pS1, 'pS2': pS2, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'var_expl': var_expl, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        var0 = self.sigma0 ** 2
        var_expl = self.sigma_expl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        lik2 = norm.pdf(deltaObs, 0, np.sqrt(var_expl))
        lik2 = max(lik2, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1, lik2 * pS2])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
        radCurrent = np.deg2rad(currentTarget)
        cosCurrent = np.cos(radCurrent)
        sinCurrent = np.sin(radCurrent)
        expDecay = np.exp(-self.betaDecay)
        self.uCos = cosCurrent + expDecay * self.uCos
        self.uSin = sinCurrent + expDecay * self.uSin
        self.w = 1 + expDecay * self.w
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS1 = predDict['pS1']
        expectedAim = pS1 * ((1 - pFlip) * (-deltaG) + pFlip * deltaG)
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExpl = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        sigma_expl = np.exp(logSigmaExpl)
        hazard = np.exp(logHazard * 10)
        alphaSim = np.exp(logAlpha)
        betaDecay = np.exp(logBeta)
        betaConfidence = np.exp(logBetaConfidence)
        stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, self.rotation, alphaSim, shift, betaDecay, betaConfidence)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS0 = predDict['pS0']
            pS1 = predDict['pS1']
            pS2 = predDict['pS2']
            deltaG = predDict['deltaG']
            varG = predDict['varG']
            var0 = predDict['var0']
            var_expl = predDict['var_expl']
            pFlip = predDict['p_flip']
            aim = self.allAims[trial]
            pdf0 = norm.pdf(aim, 0, np.sqrt(var0)) + 1e-300
            pdf_expl = norm.pdf(aim, 0, np.sqrt(var_expl)) + 1e-300
            pdf_noflip = norm.pdf(aim, -deltaG, np.sqrt(varG)) + 1e-300
            pdf_flip = norm.pdf(aim, deltaG, np.sqrt(varG)) + 1e-300
            pdf1 = (1 - pFlip) * pdf_noflip + pFlip * pdf_flip
            pdf = pS0 * pdf0 + pS1 * pdf1 + pS2 * pdf_expl
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 8,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1e-9), np.log(1e2)), # logSigma0
        (np.log(1e-9), np.log(1e3)), # logSigma1_init
        (np.log(1e-9), np.log(1)), # logHazard
        (np.log(1e-4), np.log(1e5)), # logAlpha_sim
        (-3, 3), # shift (raw, bounded negative)
        (np.log(1e-6), np.log(1e1)), # logBeta_decay
        (np.log(1e-4), np.log(1e3)), # logBetaConfidence
        (np.log(1e-9), np.log(1e3)), # logSigmaExpl
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 1
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts:# and globalsinceBest < 5000 // popSize:
        popSize = 128
        sigma = 128#/= 2
        np.random.seed(999 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=999 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 50:#and iteration < 50000 // popSize
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [objFunc(x) for x in xSamples]  # Serial evaluation
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                print(pp,restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                if currentBest[1] < bestValue * 0.9995:
                    globalSinceBest = 0
                else:
                    pass#globalSinceBest += 1
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                globalSinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.9995:
                    sinceBest = 0
                else:
                    pass#sinceBest += 1
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize(objFunc, bestX, bounds=boundsSingle, method='L-BFGS-B')
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 8
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExpl = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    alphaSim = np.exp(logAlpha)
    betaDecay = np.exp(logBeta)
    betaConfidence = np.exp(logBetaConfidence)
    sigma_expl = np.exp(logSigmaExpl)
    xs = [sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, conVal, alphaSim, shift, betaDecay, betaConfidence)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(pp,logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pS2 = predDict['pS2']
        varG = predDict['varG']
        var0 = predDict['var0']
        var_expl = predDict['var_expl']
        if not np.isnan(pFlip):
            states = np.random.choice([0, 1, 2], size=numSamples, p=[pS0, pS1, pS2])
            samples = np.zeros(numSamples)
            for j in range(numSamples):
                s = states[j]
                if s == 0:
                    samples[j] = np.random.normal(0, np.sqrt(var0))
                elif s == 2:
                    samples[j] = np.random.normal(0, np.sqrt(var_expl))
                elif s == 1:
                    flip = np.random.binomial(1, pFlip)
                    mean = deltaG if flip else -deltaG
                    samples[j] = np.random.normal(mean, np.sqrt(varG))
            samples = (samples + 180) % 360 - 180
        else:
            samples = np.array([])
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
    #plt.show()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    data = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pS2 = predDict['pS2']
        pFlip = predDict['p_flip']
        varG = predDict['varG']
        var0 = predDict['var0']
        var_expl = predDict['var_expl']
        components = [
            {'type': 'S0', 'mean': 0, 'std': np.sqrt(var0), 'weight': pS0},
            {'type': 'No Flip', 'mean': -deltaG, 'std': np.sqrt(varG), 'weight': pS1 * (1 - pFlip)},
            {'type': 'Flip', 'mean': deltaG, 'std': np.sqrt(varG), 'weight': pS1 * pFlip},
            {'type': 'Exploratory', 'mean': 0, 'std': np.sqrt(var_expl), 'weight': pS2}
        ]
        for comp in components:
            if comp['weight'] > 1e-6 and comp['std'] > 0:
                n = int(np.round(comp['weight'] * numSamples))
                if n > 0:
                    samples = np.random.normal(comp['mean'], comp['std'], n)
                    samples = (samples + 180) % 360 - 180
                    for s in samples:
                        data.append({'trial': t, 'aim': s, 'component': comp['type']})
    df = pd.DataFrame(data)
    palette = {'S0': 'green', 'No Flip': 'blue', 'Flip': 'red', 'Exploratory': 'purple'}
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=0.5, legend=False, ax=ax)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
    #plt.show()
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)] # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
        
        # Parallelize across participants
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//3) as pool:
            results = pool.map(fitSingle, dataList)
        
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues) # Ensure inputs are NumPy arrays
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0 # Handle constant true values
    return 1 - (ssRes / ssTot)
"""




"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)
class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, sigma_expl=0.1, hazard=0.01, rotation=30.0, alphaSim=1.0, shift=-0.565, betaDecay=0.01, betaConfidence=0.01):
        self.sigma0 = sigma0
        self.sigma1 = sigma1Init
        self.sigma_expl = sigma_expl #+ sigma0 #force order doesn't work for some reason
        self.hazard = hazard
        self.rotation = rotation
        self.alphaSim = alphaSim
        self.shift = shift
        self.betaDecay = betaDecay
        self.betaConfidence = betaConfidence
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.transMat = np.array([[1 - self.hazard, self.hazard / 2, self.hazard / 2],
                                  [self.hazard / 2, 1 - self.hazard, self.hazard / 2],
                                  [self.hazard / 2, self.hazard / 2, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0, 0.0])
        self.trialCount = 0
        self.uCos = 0.0
        self.uSin = 0.0
        self.w = 0.0
        self.pFlip = 0
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        deltaG = self.mu
        var0 = self.sigma0 ** 2
        var_expl = self.sigma_expl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.w == 0:
            pFlip = 0.0
        else:
            radCurrent = np.deg2rad(currentTarget)
            cosC = np.cos(radCurrent)
            sinC = np.sin(radCurrent)
            meanCos = self.uCos / self.w
            meanSin = self.uSin / self.w
            weightedSim = cosC * meanCos + sinC * meanSin
            baseFlip = 1 / (1 + np.exp(self.alphaSim * (weightedSim - self.shift)))
            pFlip = baseFlip * np.exp(-self.betaConfidence * self.kappa)
            pFlip = np.clip(pFlip, 0, 1)
            self.pFlip = pFlip
        return {'pS0': pS0, 'pS1': pS1, 'pS2': pS2, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'var_expl': var_expl, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        var0 = self.sigma0 ** 2
        var_expl = self.sigma_expl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        lik2 = norm.pdf(deltaObs, 0, np.sqrt(var_expl))
        lik2 = max(lik2, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1, lik2 * pS2])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
        radCurrent = np.deg2rad(currentTarget)
        cosCurrent = np.cos(radCurrent)
        sinCurrent = np.sin(radCurrent)
        expDecay = np.exp(-self.betaDecay)
        self.uCos = cosCurrent + expDecay * self.uCos
        self.uSin = sinCurrent + expDecay * self.uSin
        self.w = 1 + expDecay * self.w
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS1 = predDict['pS1']
        expectedAim = pS1 * ((1 - pFlip) * (-deltaG) + pFlip * deltaG)
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExpl = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        sigma_expl = np.exp(logSigmaExpl)
        hazard = np.exp(logHazard * 10)
        alphaSim = np.exp(logAlpha)
        betaDecay = np.exp(logBeta)
        betaConfidence = np.exp(logBetaConfidence)
        stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, self.rotation, alphaSim, shift, betaDecay, betaConfidence)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS0 = predDict['pS0']
            pS1 = predDict['pS1']
            pS2 = predDict['pS2']
            deltaG = predDict['deltaG']
            varG = predDict['varG']
            var0 = predDict['var0']
            var_expl = predDict['var_expl']
            pFlip = predDict['p_flip']
            aim = self.allAims[trial]
            pdf0 = norm.pdf(aim, 0, np.sqrt(var0)) + 1e-300
            pdf_expl = norm.pdf(aim, 0, np.sqrt(var_expl)) + 1e-300
            pdf_noflip = norm.pdf(aim, -deltaG, np.sqrt(varG)) + 1e-300
            pdf_flip = norm.pdf(aim, deltaG, np.sqrt(varG)) + 1e-300
            pdf1 = (1 - pFlip) * pdf_noflip + pFlip * pdf_flip
            pdf = pS0 * pdf0 + pS1 * pdf1 + pS2 * pdf_expl
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 8,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1e-9), np.log(1e2)), # logSigma0
        (np.log(1e-9), np.log(1e3)), # logSigma1_init
        (np.log(1e-9), np.log(1)), # logHazard
        (np.log(1e-4), np.log(1e5)), # logAlpha_sim
        (-3, 3), # shift (raw, bounded negative)
        (np.log(1e-6), np.log(1e1)), # logBeta_decay
        (np.log(1e-4), np.log(1e3)), # logBetaConfidence
        (np.log(1e-9), np.log(1e3)), # logSigmaExpl
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 1
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts:# and globalsinceBest < 5000 // popSize:
        popSize = 128
        sigma = 128#/= 2
        np.random.seed(999 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=999 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as executor:
            while not es.should_stop() and sinceBest < 50:#and iteration < 50000 // popSize
                xSamples = [es.ask() for _ in range(es.population_size)]
                fValues = list(executor.map(objFunc, xSamples))
                solutions = list(zip(xSamples, fValues))
                es.tell(solutions)
                currentBest = min(solutions, key=lambda s: s[1])
                if currentBest[1] < bestValue:
                    print(restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                    if currentBest[1] < bestValue * 0.9995:
                        globalSinceBest = 0
                    else:
                        pass#globalSinceBest += 1
                    bestValue = currentBest[1]
                    bestX = currentBest[0]
                else:
                    globalSinceBest += 1
                if currentBest[1] < bestInRun:
                    if currentBest[1] < bestInRun * 0.9995:
                        sinceBest = 0
                    else:
                        pass#sinceBest += 1
                    bestInRun = currentBest[1]
                else:
                    sinceBest += 1
                iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize_parallel(objFunc, bestX, bounds=boundsArray)
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 8
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExpl = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    alphaSim = np.exp(logAlpha)
    betaDecay = np.exp(logBeta)
    betaConfidence = np.exp(logBetaConfidence)
    sigma_expl = np.exp(logSigmaExpl)
    xs = [sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, conVal, alphaSim, shift, betaDecay, betaConfidence)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pS2 = predDict['pS2']
        varG = predDict['varG']
        var0 = predDict['var0']
        var_expl = predDict['var_expl']
        if not np.isnan(pFlip):
            states = np.random.choice([0, 1, 2], size=numSamples, p=[pS0, pS1, pS2])
            samples = np.zeros(numSamples)
            for j in range(numSamples):
                s = states[j]
                if s == 0:
                    samples[j] = np.random.normal(0, np.sqrt(var0))
                elif s == 2:
                    samples[j] = np.random.normal(0, np.sqrt(var_expl))
                elif s == 1:
                    flip = np.random.binomial(1, pFlip)
                    mean = deltaG if flip else -deltaG
                    samples[j] = np.random.normal(mean, np.sqrt(varG))
            samples = (samples + 180) % 360 - 180
        else:
            samples = np.array([])
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
    #plt.show()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    palette = {'S0': 'green', 'No Flip': 'blue', 'Flip': 'red', 'Exploratory': 'purple'}
    y_grid = np.linspace(-180, 180, 1000)
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        pS0 = predDict['pS0']
        pS1 = predDict['pS1']
        pS2 = predDict['pS2']
        pFlip = predDict['p_flip']
        varG = predDict['varG']
        var0 = predDict['var0']
        var_expl = predDict['var_expl']
        components = [
            {'type': 'S0', 'mean': 0, 'std': np.sqrt(var0), 'weight': pS0, 'color': palette['S0']},
            {'type': 'No Flip', 'mean': -deltaG, 'std': np.sqrt(varG), 'weight': pS1 * (1 - pFlip), 'color': palette['No Flip']},
            {'type': 'Flip', 'mean': deltaG, 'std': np.sqrt(varG), 'weight': pS1 * pFlip, 'color': palette['Flip']},
            {'type': 'Exploratory', 'mean': 0, 'std': np.sqrt(var_expl), 'weight': pS2, 'color': palette['Exploratory']}
        ]
        for comp in components:
            if comp['weight'] > 1e-6 and comp['std'] > 0:
                density = comp['weight'] * norm.pdf(y_grid, comp['mean'], comp['std'])
                x_left = t - density
                x_right = t + density
                ax.fill_betweenx(y_grid, x_left, x_right, color=comp['color'], alpha=0.5, linewidth=0.5)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
    #plt.show()
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)] # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues) # Ensure inputs are NumPy arrays
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0 # Handle constant true values
    return 1 - (ssRes / ssTot)
"""





"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)
class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, sigma_expl=0.1, hazard=0.01, rotation=30.0, alphaSim=1.0, shift=-0.565, betaDecay=0.01, betaConfidence=0.01):
        self.sigma0 = sigma0
        self.sigma1 = sigma1Init
        self.sigma_expl = sigma_expl #+ sigma0 #force order doesn't work for some reason
        self.hazard = hazard
        self.rotation = rotation
        self.alphaSim = alphaSim
        self.shift = shift
        self.betaDecay = betaDecay
        self.betaConfidence = betaConfidence
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.state = 0
        self.transMat = np.array([[1 - self.hazard, self.hazard / 2, self.hazard / 2], 
                                  [self.hazard / 2, 1 - self.hazard, self.hazard / 2], 
                                  [self.hazard / 2, self.hazard / 2, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0, 0.0])
        self.trialCount = 0
        self.uCos = 0.0
        self.uSin = 0.0
        self.w = 0.0
        self.pFlip = 0
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        deltaG = self.mu
        var0 = self.sigma0 ** 2
        var_expl = self.sigma_expl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.w == 0:
            pFlip = 0.0
        else:
            radCurrent = np.deg2rad(currentTarget)
            cosC = np.cos(radCurrent)
            sinC = np.sin(radCurrent)
            meanCos = self.uCos / self.w
            meanSin = self.uSin / self.w
            weightedSim = cosC * meanCos + sinC * meanSin
            baseFlip = 1 / (1 + np.exp(self.alphaSim * (weightedSim - self.shift)))
            pFlip = baseFlip * np.exp(-self.betaConfidence * self.kappa)
            pFlip = np.clip(pFlip, 0, 1)
            self.pFlip = pFlip
        return {'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'var_expl': var_expl, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1, pS2 = predState
        var0 = self.sigma0 ** 2
        var_expl = self.sigma_expl ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        lik2 = norm.pdf(deltaObs, 0, np.sqrt(var_expl))
        lik2 = max(lik2, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1, lik2 * pS2])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
        self.state = np.argmax(post)
        radCurrent = np.deg2rad(currentTarget)
        cosCurrent = np.cos(radCurrent)
        sinCurrent = np.sin(radCurrent)
        expDecay = np.exp(-self.betaDecay)
        self.uCos = cosCurrent + expDecay * self.uCos
        self.uSin = sinCurrent + expDecay * self.uSin
        self.w = 1 + expDecay * self.w
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        if self.state == 1:
            expectedAim = (1 - pFlip) * (-deltaG) + pFlip * deltaG
        else:
            expectedAim = 0
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExpl = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        sigma_expl = np.exp(logSigmaExpl)
        hazard = np.exp(logHazard * 10)
        alphaSim = np.exp(logAlpha)
        betaDecay = np.exp(logBeta)
        betaConfidence = np.exp(logBetaConfidence)
        stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, self.rotation, alphaSim, shift, betaDecay, betaConfidence)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            if stepper.state == 1: # Mixture likelihood for flip
                pFlip = predDict['p_flip']
                varEff = predDict['varG'] + 1e-300
                pdfNoFlip = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(varEff)) + 1e-300
                pdfFlip = norm.pdf(self.allAims[trial], predDict['deltaG'], np.sqrt(varEff)) + 1e-300
                pdf = (1 - pFlip) * pdfNoFlip + pFlip * pdfFlip
            elif stepper.state == 0: # from S0 dist
                varEff = predDict['var0'] + 1e-300
                pdf = norm.pdf(self.allAims[trial], 0, np.sqrt(varEff)) + 1e-300
            else: # state 2, exploratory
                varEff = predDict['var_expl'] + 1e-300
                pdf = norm.pdf(self.allAims[trial], 0, np.sqrt(varEff)) + 1e-300
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 8,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1e-9), np.log(1e2)), # logSigma0
        (np.log(1e-9), np.log(1e3)), # logSigma1_init
        (np.log(1e-9), np.log(1)), # logHazard
        (np.log(1e-4), np.log(1e5)), # logAlpha_sim
        (-3, 3), # shift (raw, bounded negative)
        (np.log(1e-6), np.log(1e1)), # logBeta_decay
        (np.log(1e-4), np.log(1e3)), # logBetaConfidence
        (np.log(1e-9), np.log(1e4)), # logSigmaExpl
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 40
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    sigma = 128
    while restart < maxRestarts and globalsinceBest < 5000 // popSize:
        popSize = 12
        sigma = 6#/= 2
        np.random.seed(4321 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=1024 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-4) as executor:
            while not es.should_stop() and iteration < 5000 // popSize and sinceBest < 20:
                xSamples = [es.ask() for _ in range(es.population_size)]
                fValues = list(executor.map(objFunc, xSamples))
                solutions = list(zip(xSamples, fValues))
                es.tell(solutions)
                currentBest = min(solutions, key=lambda s: s[1])
                if currentBest[1] < bestValue:
                    print(restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                    if currentBest[1] < bestValue * .9995:
                        globalSinceBest = 0
                    bestValue = currentBest[1]
                    bestX = currentBest[0]
                 
                 
                else:
                    globalSinceBest += 1
                if currentBest[1] < bestInRun:
                    if currentBest[1] < bestInRun * .9995:
                        sinceBest = 0
                    bestInRun = currentBest[1]
                 
                else:
                    sinceBest += 1
                globalIt += 1
                iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize_parallel(objFunc, bestX, bounds=boundsArray)
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 8
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExpl = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    alphaSim = np.exp(logAlpha)
    betaDecay = np.exp(logBeta)
    betaConfidence = np.exp(logBetaConfidence)
    sigma_expl = np.exp(logSigmaExpl)
    xs = [sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, conVal, alphaSim, shift, betaDecay, betaConfidence)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    stepperStates = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        if stepperStates[i] == 1:
            if not np.isnan(pFlip):
                varEff = predDict['varG']
                flips = np.random.binomial(1, pFlip, numSamples)
                means = np.where(flips, deltaG, -deltaG)
                samples = np.random.normal(means, np.sqrt(varEff))
            else:
                samples = np.array([])
        elif stepperStates[i] == 0:
            varEff = predDict['var0']
            samples = np.random.normal(0, np.sqrt(varEff), numSamples)
        else:
            varEff = predDict['var_expl']
            samples = np.random.normal(0, np.sqrt(varEff), numSamples)
        samples = (samples + 180) % 360 - 180
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
    #plt.show()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigma_expl = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, sigma_expl, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    stepperStates = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
 
    dataList = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        deltaG = predDict['deltaG']
        if stepperStates[i] == 1:
            pFlip = predDict['p_flip']
            if not np.isnan(pFlip):
                numNoFlip = int(round((1 - pFlip) * numPlotSamples))
                numFlip = numPlotSamples - numNoFlip
                varEff = predDict['varG']
                if numNoFlip > 0:
                    samplesNoFlip = np.random.normal(-deltaG, np.sqrt(varEff), numNoFlip)
                    samplesNoFlip = (samplesNoFlip + 180) % 360 - 180
                    for sample in samplesNoFlip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'No Flip'})
                if numFlip > 0:
                    samplesFlip = np.random.normal(deltaG, np.sqrt(varEff), numFlip)
                    samplesFlip = (samplesFlip + 180) % 360 - 180
                    for sample in samplesFlip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'Flip'})
        elif stepperStates[i] == 0:
            varEff = predDict['var0']
            samples = np.random.normal(0, np.sqrt(varEff), numPlotSamples)
            samples = (samples + 180) % 360 - 180
            for sample in samples:
                dataList.append({'Trial': t, 'Aim': sample, 'Type': 'S0'})
        else:
            varEff = predDict['var_expl']
            samples = np.random.normal(0, np.sqrt(varEff), numPlotSamples)
            samples = (samples + 180) % 360 - 180
            for sample in samples:
                dataList.append({'Trial': t, 'Aim': sample, 'Type': 'Exploratory'})
    df = pd.DataFrame(dataList)
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='Trial', y='Aim', hue='Type', inner=None, palette={'S0': 'green', 'No Flip': 'blue', 'Flip': 'red', 'Exploratory': 'purple'}, ax=ax, density_norm='count', common_norm=False, linewidth=0.5, cut=0, bw_adjust=0.5, dodge=False, split=False)
    for artist in ax.collections:
        artist.set_alpha(0.5)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
    #plt.show()
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)] # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues) # Ensure inputs are NumPy arrays
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0 # Handle constant true values
    return 1 - (ssRes / ssTot)
"""

    

"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)
class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, hazard=0.01, rotation=30.0, alphaSim=1.0, shift=-0.565, betaDecay=0.01, betaConfidence=0.01):
        self.sigma0 = sigma0
        self.sigma1 = sigma1Init
        self.hazard = hazard
        self.rotation = rotation
        self.alphaSim = alphaSim
        self.shift = shift
        self.betaDecay = betaDecay
        self.betaConfidence = betaConfidence
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.state = 0
        self.transMat = np.array([[1 - self.hazard, self.hazard], [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
        self.uCos = 0.0
        self.uSin = 0.0
        self.w = 0.0
        self.pFlip = 0
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        deltaG = self.mu
        var0 = self.sigma0 ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.w == 0:
            pFlip = 0.0
        else:
            radCurrent = np.deg2rad(currentTarget)
            cosC = np.cos(radCurrent)
            sinC = np.sin(radCurrent)
            meanCos = self.uCos / self.w
            meanSin = self.uSin / self.w
            weightedSim = cosC * meanCos + sinC * meanSin
            baseFlip = 1 / (1 + np.exp(self.alphaSim * (weightedSim - self.shift)))
            pFlip = baseFlip * np.exp(-self.betaConfidence * self.kappa)
            pFlip = np.clip(pFlip, 0, 1)
            self.pFlip = pFlip
        return {'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'p_flip': pFlip}
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        var0 = self.sigma0 ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
        if pS1 > 0.5:
            self.state = 1
        else:
            self.state = 0
        radCurrent = np.deg2rad(currentTarget)
        cosCurrent = np.cos(radCurrent)
        sinCurrent = np.sin(radCurrent)
        expDecay = np.exp(-self.betaDecay)
        self.uCos = cosCurrent + expDecay * self.uCos
        self.uSin = sinCurrent + expDecay * self.uSin
        self.w = 1 + expDecay * self.w
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        if self.state == 1:
            expectedAim = (1 - pFlip) * (-deltaG) + pFlip * deltaG
        else:
            expectedAim = 0
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExtra = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        hazard = np.exp(logHazard * 10)
        alphaSim = np.exp(logAlpha)
        betaDecay = np.exp(logBeta)
        betaConfidence = np.exp(logBetaConfidence)
        sigmaExtra = np.exp(logSigmaExtra)
        stepper = BayesianStepper(sigma0, sigma1Init, hazard, self.rotation, alphaSim, shift, betaDecay, betaConfidence)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS1 = predDict['pS1']
            if stepper.state == 1: # Mixture likelihood for flip
                pFlip = predDict['p_flip']
                varEff = predDict['varG'] + sigmaExtra ** 2 + 1e-300
                pdfNoFlip = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(varEff)) + 1e-300
                pdfFlip = norm.pdf(self.allAims[trial], predDict['deltaG'], np.sqrt(varEff)) + 1e-300
                pdf = (1 - pFlip) * pdfNoFlip + pFlip * pdfFlip
            else: # from S0 dist
                varEff = predDict['var0'] +1e-300# + sigmaExtra ** 2 
                pdf = norm.pdf(self.allAims[trial], 0, np.sqrt(varEff)) + 1e-300
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 8,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1e-9), np.log(1e2)), # logSigma0
        (np.log(1), np.log(2e2)), # logSigma1_init
        (np.log(1e-8), np.log(.1)), # logHazard
        (np.log(1e-4), np.log(1e4)), # logAlpha_sim
        (-2, 2), # shift (raw, bounded negative)
        (np.log(1e-4), np.log(1e1)), # logBeta_decay
        (np.log(1e-4), np.log(1e1)), # logBetaConfidence
        (np.log(1e-4), np.log(1e2)), # logSigmaExtra
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 20
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    while restart < maxRestarts and globalsinceBest < 5000 // popSize:
        popSize = 48
        sigma = 24
        np.random.seed(4321 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=99 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as executor:
            while not es.should_stop() and iteration < 6000 // popSize and sinceBest < 20:
                xSamples = [es.ask() for _ in range(es.population_size)]
                fValues = list(executor.map(objFunc, xSamples))
                solutions = list(zip(xSamples, fValues))
                es.tell(solutions)
                currentBest = min(solutions, key=lambda s: s[1])
                if currentBest[1] < bestValue:
                    print(restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                    if currentBest[1] < bestValue * .999:
                        globalSinceBest = 0
                    bestValue = currentBest[1]
                    bestX = currentBest[0]
                  
                  
                else:
                    globalSinceBest += 1
                if currentBest[1] < bestInRun:
                    if currentBest[1] < bestInRun * .999:
                        sinceBest = 0
                    bestInRun = currentBest[1]
                  
                else:
                    sinceBest += 1
                globalIt += 1
                iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize_parallel(objFunc, bestX, bounds=boundsArray)
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 8
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence, logSigmaExtra = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    alphaSim = np.exp(logAlpha)
    betaDecay = np.exp(logBeta)
    betaConfidence = np.exp(logBetaConfidence)
    sigmaExtra = np.exp(logSigmaExtra)
    xs = [sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigmaExtra]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, hazard, conVal, alphaSim, shift, betaDecay, betaConfidence)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigmaExtra = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    stepperStates = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        if stepperStates[i] == 1:
            if not np.isnan(pFlip):
                varEff = predDict['varG']
                flips = np.random.binomial(1, pFlip, numSamples)
                means = np.where(flips, deltaG, -deltaG)
                samples = np.random.normal(means, np.sqrt(varEff))
            else:
                samples = np.array([])
        else:
            varEff = predDict['var0']
            samples = np.random.normal(0, np.sqrt(varEff), numSamples)
        samples = (samples + 180) % 360 - 180
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
    #plt.show()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence, sigmaExtra = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    stepperStates = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
  
    dataList = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        if stepperStates[i] == 1:
            pFlip = predDict['p_flip']
            if not np.isnan(pFlip):
                numNoFlip = int(round((1 - pFlip) * numPlotSamples))
                numFlip = numPlotSamples - numNoFlip
                varEff = predDict['varG']
                if numNoFlip > 0:
                    samplesNoFlip = np.random.normal(-deltaG, np.sqrt(varEff), numNoFlip)
                    samplesNoFlip = (samplesNoFlip + 180) % 360 - 180
                    for sample in samplesNoFlip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'No Flip'})
                if numFlip > 0:
                    samplesFlip = np.random.normal(deltaG, np.sqrt(varEff), numFlip)
                    samplesFlip = (samplesFlip + 180) % 360 - 180
                    for sample in samplesFlip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'Flip'})
        else:
            varEff = predDict['var0']
            samples = np.random.normal(0, np.sqrt(varEff), numPlotSamples)
            samples = (samples + 180) % 360 - 180
            for sample in samples:
                dataList.append({'Trial': t, 'Aim': sample, 'Type': 'S0'})
    df = pd.DataFrame(dataList)
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='Trial', y='Aim', hue='Type', inner=None, palette={'S0': 'green', 'No Flip': 'blue', 'Flip': 'red'}, ax=ax, scale='count', linewidth=0.5, cut=0, bw_adjust=0.5, dodge=False, split=False)
    for artist in ax.collections:
        artist.set_alpha(0.5)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
    #plt.show()
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)] # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues) # Ensure inputs are NumPy arrays
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0 # Handle constant true values
    return 1 - (ssRes / ssTot)
"""




"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace
np.random.seed(99)

class BayesianStepper:
    def __init__(self, sigma0=0.1, sigma1Init=0.1, hazard=0.01, rotation=30.0, alphaSim=1.0, shift=-0.565, betaDecay=0.01, betaConfidence=0.01):
        self.sigma0 = sigma0
        self.sigma1 = sigma1Init
        self.hazard = hazard
        self.rotation = rotation
        self.alphaSim = alphaSim
        self.shift = shift
        self.betaDecay = betaDecay
        self.betaConfidence = betaConfidence
        self.mu = 0.0
        self.kappa = self.sigma1 ** 2 / 1e12 if self.sigma1 > 0 else 1e-6
        self.alpha = 3.0
        self.beta = self.sigma1 ** 2 * (self.alpha - 1)
        self.state = 0
        self.transMat = np.array([[1 - self.hazard, self.hazard], [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
        self.uCos = 0.0
        self.uSin = 0.0
        self.w = 0.0
        self.pFlip = 0

    def wrapAngle(self, x):
        return (x + 180) % 360 - 180

    def getPredictive(self, trialNum, currentTarget):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        deltaG = self.mu
        var0 = self.sigma0 ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        varG = eSigma2 + thetaVar
        if self.w == 0:
            pFlip = 0.0
        else:
            radCurrent = np.deg2rad(currentTarget)
            cosC = np.cos(radCurrent)
            sinC = np.sin(radCurrent)
            meanCos = self.uCos / self.w
            meanSin = self.uSin / self.w
            weightedSim = cosC * meanCos + sinC * meanSin
            baseFlip = 1 / (1 + np.exp(self.alphaSim * (weightedSim - self.shift)))
            pFlip = baseFlip * np.exp(-self.betaConfidence * self.kappa)
            pFlip = np.clip(pFlip, 0, 1)
            self.pFlip = pFlip
        return {'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'p_flip': pFlip}

    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        var0 = self.sigma0 ** 2
        eSigma2 = self.beta / (self.alpha - 1) if self.alpha > 1 else self.sigma1 ** 2
        thetaVar = eSigma2 / self.kappa if self.kappa > 0 else 1e12
        margVar1 = eSigma2 + thetaVar
        if margVar1 > 1e100 or np.isnan(margVar1):
            lik1 = 1e-300
        else:
            lik1 = (1 - self.pFlip) * norm.pdf(deltaObs, self.mu, np.sqrt(margVar1)) + self.pFlip * norm.pdf(deltaObs, -self.mu, np.sqrt(margVar1))
            lik1 = max(lik1, 1e-300)
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        lik0 = max(lik0, 1e-300)
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
        w = post[1]
        if w > 0:
            newKappa = self.kappa + w
            newMu = (self.kappa * self.mu + w * deltaObs) / newKappa
            newAlpha = self.alpha + 0.5 * w
            res = deltaObs - self.mu
            betaAdd = 0.5 * w * (res ** 2) * self.kappa / newKappa
            newBeta = self.beta + betaAdd
            self.mu = self.wrapAngle(newMu)
            self.kappa = newKappa
            self.alpha = newAlpha
            self.beta = newBeta
            if self.alpha > 1:
                self.sigma1 = np.sqrt(self.beta / (self.alpha - 1))
        if pS1 > 0.5:
            self.state = 1
        else:
            self.state = 0
        radCurrent = np.deg2rad(currentTarget)
        cosCurrent = np.cos(radCurrent)
        sinCurrent = np.sin(radCurrent)
        expDecay = np.exp(-self.betaDecay)
        self.uCos = cosCurrent + expDecay * self.uCos
        self.uSin = sinCurrent + expDecay * self.uSin
        self.w = 1 + expDecay * self.w

    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        if self.state == 1:
            expectedAim = (1 - pFlip) * (-deltaG) + pFlip * deltaG
        else:
            expectedAim = 0
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim

class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets

    def __call__(self, params):
        numTrials = len(self.trials)
        logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence = params
        sigma0 = np.exp(logSigma0)
        sigma1Init = np.exp(logSigma1)
        hazard = np.exp(logHazard * 10)
        alphaSim = np.exp(logAlpha)
        betaDecay = np.exp(logBeta)
        betaConfidence = np.exp(logBetaConfidence)
        stepper = BayesianStepper(sigma0, sigma1Init, hazard, self.rotation, alphaSim, shift, betaDecay, betaConfidence)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            currentTarget = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, currentTarget)
            mOut = stepper.expectedMove(trial, currentTarget)
            mOuts[trial] = mOut
            pS1 = predDict['pS1']
            if stepper.state == 1:  # Mixture likelihood for flip
                pFlip = predDict['p_flip']
                varEff = predDict['varG'] + 1e-300
                pdfNoFlip = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(varEff)) + 1e-300
                pdfFlip = norm.pdf(self.allAims[trial], predDict['deltaG'], np.sqrt(varEff)) + 1e-300
                pdf = (1 - pFlip) * pdfNoFlip + pFlip * pdfFlip
            else:  # from S0 dist
                varEff = predDict['var0'] + 1e-300
                pdf = norm.pdf(self.allAims[trial], 0, np.sqrt(varEff)) + 1e-300
            if np.isinf(pdf) or np.isnan(pdf):
                pdf = 1e-300
            if self.mask[trial]:
                logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood

def fitSingle(data):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 7,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    boundsSingle = [
        (np.log(1e-9), np.log(1e2)),  # logSigma0
        (np.log(1), np.log(2e2)),  # logSigma1_init
        (np.log(1e-6), np.log(.1)),  # logHazard
        (np.log(1e-4), np.log(1e4)),  # logAlpha_sim
        (-2, 2),  # shift (raw, bounded negative)
        (np.log(1e-4), np.log(1e1)),  # logBeta_decay
        (np.log(1e-4), np.log(1e1)),  # logBetaConfidence
    ]
    boundsArray = np.array(boundsSingle)
    maxRestarts = 1
    defaultPopSize = 6
    largePopSize = defaultPopSize * 2
    popSize = defaultPopSize
    bestValue = np.inf
    bestX = None
    globalIt = 0
    restart = 0
    globalSinceBest = 0
    iteration = 0
    while restart < maxRestarts and globalSinceBest < 2400 // popSize:

        popSize = 64
        sigma = 24
        np.random.seed(4321 + restart)
        mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=9124 + restart)
        es.tolfun = 1e-4
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as executor:
            while not es.should_stop() and iteration < 8000 // popSize and sinceBest < 20:
                xSamples = [es.ask() for _ in range(es.population_size)]
                fValues = list(executor.map(objFunc, xSamples))
                solutions = list(zip(xSamples, fValues))
                es.tell(solutions)
                currentBest = min(solutions, key=lambda s: s[1])
                if currentBest[1] < bestValue:
                    print(restart, iteration, currentBest[1], currentBest[0], globalSinceBest)
                    if currentBest[1] < bestValue * .999:
                        globalSinceBest = 0
                    bestValue = currentBest[1]
                    bestX = currentBest[0]
                    
                    
                else:
                    globalSinceBest += 1
                if currentBest[1] < bestInRun:
                    if currentBest[1] < bestInRun * .999:
                        sinceBest = 0
                    bestInRun = currentBest[1]
                    
                else:
                    sinceBest += 1
                globalIt += 1
                iteration += 1
        print(restart, globalIt, popSize, globalSinceBest)
        restart += 1
    result = minimize_parallel(objFunc, bestX, bounds=boundsArray)
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    bestFun = bestValue
    paramCount = 7
    logLikelihood = -bestFun
    logSigma0, logSigma1, logHazard, logAlpha, shift, logBeta, logBetaConfidence = bestX
    sigma0 = np.exp(logSigma0)
    sigma1Init = np.exp(logSigma1)
    hazard = np.exp(logHazard * 10)
    alphaSim = np.exp(logAlpha)
    betaDecay = np.exp(logBeta)
    betaConfidence = np.exp(logBetaConfidence)
    xs = [sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence]
    stepperSingle = BayesianStepper(sigma0, sigma1Init, hazard, conVal, alphaSim, shift, betaDecay, betaConfidence)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        currentTarget = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, currentTarget)
        stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples) if numSamples > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, rmseVal, rSquared, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist(),
        'rSquared': rSquared
    }

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection

def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    stepperStates = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    modelSamples = []
    for i, predDict in enumerate(predDicts):
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        pFlip = predDict['p_flip']
        if stepperStates[i] == 1:
            if not np.isnan(pFlip):
                varEff = predDict['varG']
                flips = np.random.binomial(1, pFlip, numSamples)
                means = np.where(flips, deltaG, -deltaG)
                samples = np.random.normal(means, np.sqrt(varEff))
            else:
                samples = np.array([])
        else:
            varEff = predDict['var0']
            samples = np.random.normal(0, np.sqrt(varEff), numSamples)
        samples = (samples + 180) % 360 - 180
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number) + "testScatter.png", dpi=100)
    #plt.show()

def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    sigma0, sigma1Init, hazard, alphaSim, shift, betaDecay, betaConfidence = fittedParams
    stepper = BayesianStepper(sigma0, sigma1Init, hazard, rotation, alphaSim, shift, betaDecay, betaConfidence)
    predDicts = []
    stepperStates = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    
    dataList = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        if stepperStates[i] == 1:
            pFlip = predDict['p_flip']
            if not np.isnan(pFlip):
                numNoFlip = int(round((1 - pFlip) * numPlotSamples))
                numFlip = numPlotSamples - numNoFlip
                varEff = predDict['varG']
                if numNoFlip > 0:
                    samplesNoFlip = np.random.normal(-deltaG, np.sqrt(varEff), numNoFlip)
                    samplesNoFlip = (samplesNoFlip + 180) % 360 - 180
                    for sample in samplesNoFlip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'No Flip'})
                if numFlip > 0:
                    samplesFlip = np.random.normal(deltaG, np.sqrt(varEff), numFlip)
                    samplesFlip = (samplesFlip + 180) % 360 - 180
                    for sample in samplesFlip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'Flip'})
        else:
            varEff = predDict['var0']
            samples = np.random.normal(0, np.sqrt(varEff), numPlotSamples)
            samples = (samples + 180) % 360 - 180
            for sample in samples:
                dataList.append({'Trial': t, 'Aim': sample, 'Type': 'S0'})
    df = pd.DataFrame(dataList)
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='Trial', y='Aim', hue='Type', inner=None, palette={'S0': 'green', 'No Flip': 'blue', 'Flip': 'red'}, ax=ax, scale='count', linewidth=0.5, cut=0, bw_adjust=0.5, dodge=False, split=False)
    for artist in ax.collections:
        artist.set_alpha(0.5)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(number) + "testViolin.png", dpi=100)
    #plt.show()

class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []

    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.rSquareds = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)]  # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            self.rSquareds[i] = result['rSquared']

def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues)  # Ensure inputs are NumPy arrays
    predValues = np.array(predValues)
  
    if len(trueValues) != len(predValues):
        raise ValueError("Arrays must have the same length")
  
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
  
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0  # Handle constant true values
  
    return 1 - (ssRes / ssTot)
"""
    

    


"""

import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel

from scipy.optimize import minimize
from scipy.stats import qmc
import multiprocessing
from types import SimpleNamespace



np.random.seed(99)
class BayesianStepper:
    def __init__(self, fittingVariance=None, sigma0=0.1, sigma1_init=0.1, hazard=0.01, rotation=30.0, alpha_sim=1.0, beta_decay=0.01):
        self.fittingVariance = fittingVariance
        self.sigma0 = sigma0
        self.sigma1 = sigma1_init
        self.thetaMu = 0.0
        self.thetaVar = 1e12
        self.thetaPrec = 1/self.thetaVar
        self.state = 0
        self.hazard = hazard
        self.transMat = np.array([[1 - self.hazard, self.hazard], [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
        self.alpha_sim = alpha_sim
        self.beta_decay = beta_decay
        self.prev_targets = []
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum, current_target):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        deltaG = self.thetaMu
        var0 = self.sigma0 ** 2 # Fixed for S0
        varG = self.sigma1 ** 2 + self.thetaVar # Dynamic for S1
        if not self.prev_targets:
            p_flip = 0.0
        else:
            sims = []
            for prev in self.prev_targets[-1:]:
                ang_diff = min(abs(current_target - prev), 360 - abs(current_target - prev))
                rad_diff = np.deg2rad(ang_diff)
                sim = np.cos(rad_diff)
                sims.append(sim)
            max_sim = np.max(sims)
            base_flip = 1 / (1 + np.exp(self.alpha_sim * max_sim))
            decay = np.exp(-self.beta_decay * trialNum)
            p_flip = base_flip * decay
        return {'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0, 'p_flip': p_flip}
    def updatePosteriors(self, trialNum, deltaObs, current_target):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
    
        # Marginal variances for state inference
        var0 = self.sigma0 ** 2
        marg_var1 = self.sigma1 ** 2 + self.thetaVar
        if marg_var1 > 1e100 or np.isnan(marg_var1):
            lik1 = 1e-300
        else:
            lik1 = norm.pdf(deltaObs, self.thetaMu, np.sqrt(marg_var1))
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
    
        # Update theta posterior (and sigma1) only if pred pS1 > 0.5
        if pS1 > 0.5:
            noise_var = self.sigma1 ** 2
            lik_prec = post[1] / noise_var if noise_var > 0 and post[1] > 0 else 0.0
            new_theta_prec = self.thetaPrec + lik_prec
            if new_theta_prec > 1e-12:
                obs_contrib = post[1] * deltaObs / noise_var if noise_var > 0 else 0.0
                new_theta_mu = (self.thetaPrec * self.thetaMu + obs_contrib) / new_theta_prec
                self.thetaMu = self.wrapAngle(new_theta_mu)
                self.thetaPrec = new_theta_prec
                self.thetaVar = 1.0 / new_theta_prec
            else:
                self.thetaVar = 1e12
            self.state = 1
        else:
            self.thetaVar = 1.0 / self.thetaPrec if self.thetaPrec > 0 else 1e12
            self.state = 0
        self.prev_targets.append(current_target)
    def expectedMove(self, trialNum, current_target):
        predDict = self.getPredictive(trialNum, current_target)
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        p_flip = predDict['p_flip']
        if pS1 > 0.5 and self.state == 1:
            expected_aim = (1 - p_flip) * (-deltaG) + p_flip * deltaG
        else:
            expected_aim = 0
        expected_aim = self.wrapAngle(expected_aim)
        return expected_aim
class Objective:
    def __init__(self, isRmse, allAims, mask, trials, phases, rotation, targets):
        self.isRmse = isRmse
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
    def __call__(self, params):
        numTrials = len(self.trials)
        if self.isRmse:
            fittingVariance = None
            sigma0 = 0.1
            sigma1_init = 0.1
            hazard = 0.01
            alpha_sim = 1.0
            beta_decay = 0.01
        else:
            logFv, logSigma0, logSigma1, logHazard, logAlpha, logBeta = params
            fittingVariance = np.exp(logFv)
            sigma0 = np.exp(logSigma0)
            sigma1_init = np.exp(logSigma1)
            hazard = np.exp(logHazard*10)
            alpha_sim = np.exp(logAlpha)
            beta_decay = np.exp(logBeta)
        stepper = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, self.rotation, alpha_sim, beta_decay)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            current_target = self.targets[trial]
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial, current_target)
            mOut = stepper.expectedMove(trial, current_target)
            mOuts[trial] = mOut
            if not self.isRmse:
                fvAdd = fittingVariance if fittingVariance is not None else 0
                pS1 = predDict['pS1']
                if pS1 > 0.5 and stepper.state == 1: # Mixture likelihood for flip
                    p_flip = predDict['p_flip']
                    var_eff = predDict['varG'] + fvAdd + 1e-300
                    pdf_no_flip = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(var_eff)) + 1e-300
                    pdf_flip = norm.pdf(self.allAims[trial], predDict['deltaG'], np.sqrt(var_eff)) + 1e-300
                    pdf = (1 - p_flip) * pdf_no_flip + p_flip * pdf_flip
                else: # from S0 dist
                    var_eff = predDict['var0'] + fvAdd + 1e-300
                    pdf = norm.pdf(self.allAims[trial], 0, np.sqrt(var_eff)) + 1e-300
                if np.isinf(pdf) or np.isnan(pdf):
                    pdf = 1e-300
                if self.mask[trial]:
                    logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs, current_target)
        if self.isRmse:
            totErrs = self.allAims[self.mask] - mOuts[self.mask]
            if len(totErrs) == 0:
                return np.inf
            mu, std = norm.fit(totErrs)
            logLikelihood = np.sum(np.log(norm.pdf(totErrs, mu, std) + 1e-300))
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood



def fitSingle(data):
    allAims, mask, trials, isRmse, heightCap, compMags, pp, conVal, phases, targets = data
    objFunc = Objective(isRmse, allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 6,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    if isRmse:
        bestFun = objFunc(np.array([]))
        bestX = np.array([])
        paramCount = 0
    else:
        boundsSingle = [
            (np.log(1e-9), np.log(1e4)), # logFv
            (np.log(1e-9), np.log(5)), # logSigma0
            (np.log(1), np.log(30)), # logSigma1_init
            (np.log(1e-4), np.log(.01)), # logHazard
            (np.log(1e-9), np.log(1e4)), # logAlpha_sim
            (np.log(1e-9), np.log(1e4)), # logBeta_decay
        ]
        boundsArray = np.array(boundsSingle)
        maxRestarts = 15
        defaultPopSize = 6
        largePopSize = defaultPopSize * 2
        popSize = defaultPopSize
        bestValue = np.inf
        bestX = None
        globalIt = 0
        restart = 0
        globalSinceBest = 0
        iteration = 0
        while restart < maxRestarts and globalSinceBest < 1800//popSize:
            if globalSinceBest < iteration:
                largePopSize /= 1.5
                largePopSize = max(largePopSize, defaultPopSize*1.5)
            if restart % 2 == 1:
                popSize = defaultPopSize
                sigma = .5
            elif restart == 0:
                popSize = defaultPopSize*8
                sigma = 8
            else:
                popSize = int(largePopSize)
                sigma = 2
                largePopSize *= 1.5
            np.random.seed(4321 + restart)
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            #mean[3] = boundsArray[3, 0] + (boundsArray[3, 1] - boundsArray[3, 0]) * 0.4
            es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=999 + restart)
            es.tolfun = 1e-8
            sinceBest = 0
            bestInRun = 1e9
            iteration = 0
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as executor:
                while not es.should_stop() and iteration < 1000//popSize and sinceBest < 5:
                    xSamples = [es.ask() for _ in range(es.population_size)]
                    fValues = list(executor.map(objFunc, xSamples))
                    solutions = list(zip(xSamples, fValues))
                    es.tell(solutions)
                    currentBest = min(solutions, key=lambda s: s[1])
                    if currentBest[1] < bestValue:
                        bestValue = currentBest[1]
                        bestX = currentBest[0]
                        print(restart, iteration, bestValue, currentBest[0], globalSinceBest)
                        globalSinceBest = 0
                    else:
                        globalSinceBest += 1
                    if currentBest[1] < bestInRun:
                        bestInRun = currentBest[1]
                        sinceBest = 0
                    else:
                        sinceBest += 1
                    globalIt += 1
                    iteration += 1
            print(restart, globalIt, popSize, globalSinceBest)
            restart += 1
        result = minimize_parallel(objFunc, bestX, bounds=boundsArray)
        if result.fun < bestValue:
            bestValue = result.fun
            bestX = result.x
        bestFun = bestValue
        paramCount = 6
    logLikelihood = -bestFun
    if isRmse:
        fittingVariance = None
        sigma0 = 0.1
        sigma1_init = 0.1
        hazard = 0.01
        alpha_sim = 1.0
        beta_decay = 0.01
    else:
        logFv, logSigma0, logSigma1, logHazard, logAlpha, logBeta = bestX
        fittingVariance = np.exp(logFv)
        sigma0 = np.exp(logSigma0)
        sigma1_init = np.exp(logSigma1)
        hazard = np.exp(logHazard*10)
        alpha_sim = np.exp(logAlpha)
        beta_decay = np.exp(logBeta)
    xs = [fittingVariance, sigma0, sigma1_init, hazard, alpha_sim, beta_decay]
    stepperSingle = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, conVal, alpha_sim, beta_decay)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        current_target = targets[trial]
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial, current_target)
        stepperSingle.updatePosteriors(trial, deltaObs, current_target)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, targets, rotation=conVal, number=pp)
    return {
            'xs': xs,
            'mStates': mOutsSingle.tolist(),
            'rmse': rmseVal,
            'negLl': -logLikelihood,
            'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
            'allAims': allAims.tolist()
        }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, number=0):
    fittingVariance, sigma0, sigma1_init, hazard, alpha_sim, beta_decay = fittedParams
    stepper = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, rotation, alpha_sim, beta_decay)
    predDicts = []
    stepperStates = []
    for trial in trials:
        current_target = targets[trial]
        predDict = stepper.getPredictive(trial, current_target)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, current_target)
    modelSamples = []
    fvAdd = 0
    i = 0
    for predDict in predDicts:
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        p_flip = predDict['p_flip']
        if pS1 > 0.5 and stepperStates[i] == 1:
            if not np.isnan(p_flip):
                var_eff = predDict['varG'] + fvAdd
                flips = np.random.binomial(1, p_flip, numSamples)
                means = np.where(flips, deltaG, -deltaG)
                samples = np.random.normal(means, np.sqrt(var_eff))
            else:
                samples = np.array([])
        else:
            var_eff = predDict['var0'] + fvAdd
            samples = np.random.normal(0, np.sqrt(var_eff), numSamples)
        samples = (samples + 180) % 360 - 180
        modelSamples.append(samples)
        i+=1
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number)+"testScatter.png",dpi=100)
    plt.show()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=1000, numPlotSamples=1000, number=0):
    fittingVariance, sigma0, sigma1_init, hazard, alpha_sim, beta_decay = fittedParams
    stepper = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, rotation, alpha_sim, beta_decay)
    predDicts = []
    stepperStates = []
    for trial in trials:
        current_target = targets[trial]
        predDict = stepper.getPredictive(trial, current_target)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepperStates.append(stepper.state)
        stepper.updatePosteriors(trial, deltaObs, current_target)
        
    dataList = []
    fvAdd = 0
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        if pS1 > 0.5 and stepperStates[i] == 1:
            p_flip = predDict['p_flip']
            if not np.isnan(p_flip):
                num_no_flip = int(round((1 - p_flip) * numPlotSamples))
                num_flip = numPlotSamples - num_no_flip
                var_eff = predDict['varG'] + fvAdd
                if num_no_flip > 0:
                    samples_no_flip = np.random.normal(-deltaG, np.sqrt(var_eff), num_no_flip)
                    samples_no_flip = (samples_no_flip + 180) % 360 - 180
                    for sample in samples_no_flip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'No Flip'})
                if num_flip > 0:
                    samples_flip = np.random.normal(deltaG, np.sqrt(var_eff), num_flip)
                    samples_flip = (samples_flip + 180) % 360 - 180
                    for sample in samples_flip:
                        dataList.append({'Trial': t, 'Aim': sample, 'Type': 'Flip'})
        else:
            var_eff = predDict['var0'] + fvAdd
            samples = np.random.normal(0, np.sqrt(var_eff), numPlotSamples)
            samples = (samples + 180) % 360 - 180
            for sample in samples:
                dataList.append({'Trial': t, 'Aim': sample, 'Type': 'S0'})
    df = pd.DataFrame(dataList)
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='Trial', y='Aim', hue='Type', inner=None, palette={'S0': 'green', 'No Flip': 'blue', 'Flip': 'red'}, ax=ax, scale='count', linewidth=0.5, cut=0, bw_adjust=0.5, dodge=False, split=False)
    for artist in ax.collections:
        artist.set_alpha(0.5)
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(number)+"testViolin.png",dpi=100)
    plt.show()

    
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, isRmse=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.isRmse = isRmse
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)] # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.isRmse, self.heightCap, compMags, pp, self.conVal, phases, targetPositions))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
"""





"""
import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
from optimparallel import minimize_parallel
np.random.seed(99)
class BayesianStepper:
    def __init__(self, fittingVariance=None, sigma0=0.1, sigma1_init=0.1, hazard=0.01, rotation=30.0):
        self.fittingVariance = fittingVariance
        self.sigma0 = sigma0
        self.sigma1_init = sigma1_init
        self.thetaMu = 0.0
        self.thetaPrec = 0.0
        self.thetaVar = 1e12
        self.hazard = hazard
        self.transMat = np.array([[1 - self.hazard, self.hazard], [self.hazard, 1 - self.hazard]])
        self.prevStatePosterior = np.array([1.0, 0.0])
        self.trialCount = 0
    
        # Dynamic sigma1: Gamma prior on precision1 = 1 / sigma1² (shape alpha, rate beta)
        self.sigma1_alpha = 0.001 # Small for diffuseness; tune if needed
        self.sigma1_beta = self.sigma1_alpha * (self.sigma1_init ** 2) # Initial mean precision1 = 1 / sigma1_init²
        self.precision1 = self.sigma1_alpha / self.sigma1_beta if self.sigma1_beta > 0 else 0
        self.sigma1 = np.sqrt(1 / self.precision1) if self.precision1 > 0 else 1e6
    def wrapAngle(self, x):
        return (x + 180) % 360 - 180
    def getPredictive(self, trialNum):
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
        deltaG = self.thetaMu
        var0 = self.sigma0 ** 2 # Fixed for S0
        varG = self.sigma1 ** 2 + self.thetaVar # Dynamic for S1
        return {'pS1': pS1, 'deltaG': deltaG, 'varG': varG, 'var0': var0}
    def updatePosteriors(self, trialNum, deltaObs):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predState = self.transMat.T @ self.prevStatePosterior
        pS0, pS1 = predState
    
        # Marginal variances for state inference
        var0 = self.sigma0 ** 2
        marg_var1 = self.sigma1 ** 2 + self.thetaVar
        if marg_var1 > 1e100 or np.isnan(marg_var1):
            lik1 = 1e-300
        else:
            lik1 = norm.pdf(deltaObs, self.thetaMu, np.sqrt(marg_var1))
        lik0 = norm.pdf(deltaObs, 0, np.sqrt(var0))
        unnormPost = np.array([lik0 * pS0, lik1 * pS1])
        if np.sum(unnormPost) == 0:
            unnormPost = predState + 1e-300
        post = unnormPost / np.sum(unnormPost)
        self.prevStatePosterior = post
    
        # Update theta posterior (and sigma1) only if pred pS1 > 0.5
        if pS1 > 0.5:
            noise_var = self.sigma1 ** 2
            lik_prec = post[1] / noise_var if noise_var > 0 and post[1] > 0 else 0.0
            new_theta_prec = self.thetaPrec + lik_prec
            if new_theta_prec > 1e-12:
                obs_contrib = post[1] * deltaObs / noise_var if noise_var > 0 else 0.0
                new_theta_mu = (self.thetaPrec * self.thetaMu + obs_contrib) / new_theta_prec
                self.thetaMu = self.wrapAngle(new_theta_mu)
                self.thetaPrec = new_theta_prec
                self.thetaVar = 1.0 / new_theta_prec
            
                # Update sigma1 posterior: weighted residual update for Gamma on precision1
                residual = self.wrapAngle(deltaObs - self.thetaMu)
                weight = post[1]
                self.sigma1_alpha += 0.5 * weight
                self.sigma1_beta += 0.5 * weight * (residual ** 2)
                if self.sigma1_beta > 0:
                    self.precision1 = self.sigma1_alpha / self.sigma1_beta
                    self.sigma1 = np.sqrt(1 / self.precision1)
                else:
                    self.sigma1 = 1e6 # Fallback if degenerate
            else:
                self.thetaVar = 1e12
                # No sigma1 update
        else:
            self.thetaVar = 1.0 / self.thetaPrec if self.thetaPrec > 0 else 1e12
            
    def expectedMove(self, trialNum):
        predDict = self.getPredictive(trialNum)
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        if pS1 > 0.5:
            delta = deltaG
        else:
            delta = 0
        aim = -delta
        aim = self.wrapAngle(aim)
        return aim
class Objective:
    def __init__(self, isRmse, allAims, mask, trials, phases, rotation):
        self.isRmse = isRmse
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
    def __call__(self, params):
        numTrials = len(self.trials)
        if self.isRmse:
            fittingVariance = None
            sigma0 = 0.1
            sigma1_init = 0.1
            hazard = 0.01
        else:
            logFv, logSigma0, logSigma1, logHazard = params
            fittingVariance = np.exp(logFv)
            sigma0 = np.exp(logSigma0)
            sigma1_init = np.exp(logSigma1)
            hazard = np.exp(logHazard)
        stepper = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, self.rotation)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            deltaObs = self.rotation if self.phases[trial] == 'rotation' else 0.0
            predDict = stepper.getPredictive(trial)
            mOut = stepper.expectedMove(trial)
            mOuts[trial] = mOut
            if not self.isRmse:
                fvAdd = fittingVariance if fittingVariance is not None else 0
                pS1 = predDict['pS1']
                if pS1 > 0.5: # MAP-consistent likelihood: from S1 dist
                    var_eff = predDict['varG'] + fvAdd + 1e-300
                    pdf = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(var_eff)) + 1e-300
                else: # from S0 dist
                    var_eff = predDict['var0'] + fvAdd + 1e-300
                    pdf = norm.pdf(self.allAims[trial], 0, np.sqrt(var_eff)) + 1e-300
                if np.isinf(pdf) or np.isnan(pdf):
                    pdf = 1e-300
                if self.mask[trial]:
                    logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, deltaObs)
        if self.isRmse:
            totErrs = self.allAims[self.mask] - mOuts[self.mask]
            if len(totErrs) == 0:
                return np.inf
            mu, std = norm.fit(totErrs)
            logLikelihood = np.sum(np.log(norm.pdf(totErrs, mu, std) + 1e-300))
        totalLogLikelihood = logLikelihood
        if not np.isfinite(totalLogLikelihood):
            return 1e9
        return -totalLogLikelihood
def fitSingle(data):
    allAims, mask, trials, isRmse, heightCap, compMags, pp, conVal, phases = data
    objFunc = Objective(isRmse, allAims, mask, trials, phases, conVal)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 4,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    if isRmse:
        bestFun = objFunc(np.array([]))
        bestX = np.array([])
        paramCount = 0
    else:
        boundsSingle = [
            (np.log(1e-9), np.log(1e4)), # logFv
            (np.log(1e-9), np.log(10)), # logSigma0
            (np.log(1), np.log(1e4)), # logSigma1_init
            (np.log(1e-12), np.log(0.1)), # logHazard
        ]
        boundsArray = np.array(boundsSingle)
        maxRestarts = 12#000
        defaultPopSize = 6
        largePopSize = defaultPopSize * 2
        popSize = defaultPopSize
        bestValue = np.inf
        bestX = None
        globalIt = 0
        restart = 0
        globalSinceBest = 0
        iteration = 0
        while restart < maxRestarts and globalSinceBest < 1800//popSize:
            if globalSinceBest < iteration:
                largePopSize /= 1.5
                largePopSize = max(largePopSize,defaultPopSize*1.5)
            if restart % 2 == 1:
                popSize = defaultPopSize
                sigma = .5
            elif restart == 0:
                popSize = defaultPopSize*8
                sigma = 8
            else:
                popSize = int(largePopSize)
                sigma = 2
                largePopSize *= 1.5
            np.random.seed(4321 + restart)
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            #mean[0] = np.log(200)
            #mean[1] = np.log(10)
            #mean[3] = np.log(1e-9)
            es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=91 + restart)
            es.tolfun = 1e-8
            sinceBest = 0
            bestInRun = 1e9
            iteration = 0
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as executor:
                while not es.should_stop() and iteration < 1000//popSize and sinceBest < 10:
                    xSamples = [es.ask() for _ in range(es.population_size)]
                    fValues = list(executor.map(objFunc, xSamples))
                    solutions = list(zip(xSamples, fValues))
                    es.tell(solutions)
                    currentBest = min(solutions, key=lambda s: s[1])
                    if currentBest[1] < bestValue:
                        bestValue = currentBest[1]
                        bestX = currentBest[0]
                        print(restart,iteration,bestValue,currentBest[0],globalSinceBest)
                        globalSinceBest = 0
                    else:
                        globalSinceBest += 1
                    if currentBest[1] < bestInRun:
                        bestInRun = currentBest[1]
                        sinceBest = 0
                    else:
                        sinceBest += 1
                    globalIt += 1
                    iteration += 1
            print(restart,globalIt,popSize,globalSinceBest)
            restart += 1
        result = minimize_parallel(objFunc, bestX, bounds=boundsArray)
        if result.fun < bestValue:
            bestValue = result.fun
            bestX = result.x
        bestFun = bestValue
        paramCount = 4
    logLikelihood = -bestFun
    if isRmse:
        fittingVariance = None
        sigma0 = 0.1
        sigma1_init = 0.1
        hazard = 0.01
    else:
        logFv, logSigma0, logSigma1, logHazard = bestX
        fittingVariance = np.exp(logFv)
        sigma0 = np.exp(logSigma0)
        sigma1_init = np.exp(logSigma1)
        hazard = np.exp(logHazard)
    xs = [fittingVariance, sigma0, sigma1_init, hazard]
    stepperSingle = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, conVal)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        deltaObs = conVal if phases[trial] == 'rotation' else 0.0
        mOutsSingle[trial] = stepperSingle.expectedMove(trial)
        stepperSingle.updatePosteriors(trial, deltaObs)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, xs)
    violinPlotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, rotation=conVal, number=pp)
    plotModelVsHumanAims(xs, np.arange(len(allAims)), compMags, allAims, rotation=conVal, number=pp)
    return {
            'xs': xs,
            'mStates': mOutsSingle.tolist(),
            'rmse': rmseVal,
            'negLl': -logLikelihood,
            'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
            'allAims': allAims.tolist()
        }
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
def plotModelVsHumanAims(fittedParams, trials, compMags, humanAims, rotation=30.0, numSamples=100, number=0):
    fittingVariance, sigma0, sigma1_init, hazard = fittedParams
    stepper = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, rotation)
    predDicts = []
    for trial in trials:
        predDict = stepper.getPredictive(trial)
        predDicts.append(predDict)
        deltaObs = compMags[trial]#30.0 if trial in range(10, 50) else 0.0 # Dummy for plotting
        stepper.updatePosteriors(trial, deltaObs)
    modelSamples = []
    fvAdd = 0#fittingVariance if fittingVariance is not None else 0
    for predDict in predDicts:
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        if pS1 > 0.5: # MAP-consistent sampling: from S1 dist
            var_eff = predDict['varG'] + fvAdd
            samples = np.random.normal(-deltaG, np.sqrt(var_eff), numSamples)
        else: # from S0 dist
            var_eff = predDict['var0'] + fvAdd
            samples = np.random.normal(0, np.sqrt(var_eff), numSamples)
        samples = (samples + 180) % 360 - 180
        modelSamples.append(samples)
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.swarmplot(data=modelSamples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number)+"testScatter.png",dpi=100)
    plt.show()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, rotation=30.0, numSamples=1000, numPlotSamples=100, number=0):
    fittingVariance, sigma0, sigma1_init, hazard = fittedParams
    stepper = BayesianStepper(fittingVariance, sigma0, sigma1_init, hazard, rotation)
    predDicts = []
    for trial in trials:
        predDict = stepper.getPredictive(trial)
        predDicts.append(predDict)
        deltaObs = compMags[trial]#30.0 if trial in range(10, 50) else 0.0 # Dummy for plotting
        stepper.updatePosteriors(trial, deltaObs)
    modelSamples = []
    fvAdd = 0#fittingVariance if fittingVariance is not None else 0
    for predDict in predDicts:
        pS1 = predDict['pS1']
        deltaG = predDict['deltaG']
        if pS1 > 0.5: # MAP-consistent sampling: from S1 dist
            var_eff = predDict['varG'] + fvAdd
            samples = np.random.normal(-deltaG, np.sqrt(var_eff), numSamples)
        else: # from S0 dist
            var_eff = predDict['var0'] + fvAdd
            samples = np.random.normal(0, np.sqrt(var_eff), numSamples)
        samples = (samples + 180) % 360 - 180
        modelSamples.append(samples)
    dataList = []
    for i, t in enumerate(trials):
        subsamples = np.random.choice(modelSamples[i], size=numPlotSamples, replace=False)
        for sample in subsamples:
            dataList.append({'Trial': t, 'Aim': sample})
    df = pd.DataFrame(dataList)
    df['TrialViolin'] = df['Trial'] + 0.2
    df['TrialBox'] = df['Trial']
    df['TrialStrip'] = df['Trial'] - 0.2
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6))
    sns.violinplot(data=df, x='TrialViolin', y='Aim', inner=None, color='skyblue', alpha=0.5, ax=ax, scale='width', linewidth=0.5, cut=0, bw_adjust=0.5)
    violinCollections = [c for c in ax.collections if isinstance(c, PolyCollection)]
    for i, poly in enumerate(violinCollections):
        center = trials[i] + 0.2
        verts = poly.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], center, np.inf)
    ax.scatter(trials, humanAims, color='red', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Raincloud (Half-Violin + Box + Strip) vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(number)+"testViolin.png",dpi=100)
    plt.show()
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, isRmse=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.isRmse = isRmse
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)] # Use self.df to get full pDat
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, self.isRmse, self.heightCap, compMags, pp, self.conVal, phases))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
"""

            
            

"""
import numpy as np
from scipy.stats import norm, vonmises  # Added vonmises for potential future use
from scipy.special import expit
from scipy import linalg as la  # For stable eigh and pinvh
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
np.random.seed(99)

class BayesianStepper:
    def __init__(self, fittingVariance=None, sigmaObs=1.0, initialLogitPg=0.0, targets=None, thetaPrecision=1e9, priorPrec=1e-9, sigmaAim=1.0, kernelL=45.0, kernelAmp=1.0, hazard=0.01, logit_w=0.0, log_f=np.log(1/360)):
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.sigmaObs = sigmaObs
        self.sigmaAim = sigmaAim
        self.sigmaH0 = 10.0
        self.pG = expit(initialLogitPg)
        self.thetaMu = 0.0  # Changed to von Mises mu
        self.thetaKappa = thetaPrecision  # Changed to von Mises kappa
        self.thetaRx = self.thetaKappa * np.cos(np.deg2rad(self.thetaMu))
        self.thetaRy = self.thetaKappa * np.sin(np.deg2rad(self.thetaMu))
        self.targets = targets
        self.hazard = hazard
        self.trans_mat = np.array([[1 - self.hazard, self.hazard], [self.hazard, 1 - self.hazard]])
        self.prev_state_posterior = np.array([1.0, 0.0])  # Start unperturbed
        if targets is not None:
            unique_targets = np.unique(targets)
            self.targetPositions = np.sort(unique_targets) % 360
            self.nLocal = len(self.targetPositions)
        else:
            self.targetPositions = np.arange(0, 360, 45)
            self.nLocal = 8
        # Compute prior cov K for local GP with spectral mixture
        K = np.zeros((self.nLocal, self.nLocal))
        w = expit(logit_w)  # Weight for periodic component
        f = np.exp(log_f)  # Frequency for cos
        for i in range(self.nLocal):
            for j in range(self.nLocal):
                d = self.minAng(self.targetPositions[i], self.targetPositions[j])
                rbf = np.exp(- (d __ 2) / (2 * kernelL __ 2))
                cos_term = np.cos(2 * np.pi * f * d)
                # Spectral mixture: (1-w) * RBF + w * RBF * cos
                K[i, j] = kernelAmp __ 2 * ((1 - w) * rbf + w * rbf * cos_term)
        # Force symmetry and clip eigenvalues for PSD
        K = (K + K.T) / 2
        evals, evecs = la.eigh(K, check_finite=False)
        evals = np.maximum(evals, 0)
        K = evecs @ np.diag(evals) @ evecs.T
        # Precision matrix = inv(K + jitter) + diagonal prior prec
        jitter = 1e-6 * np.eye(self.nLocal)
        self.lPrecMat = np.linalg.inv(K + jitter) + np.eye(self.nLocal) * priorPrec
        # Force symmetry and clip for initial stability
        self.lPrecMat = (self.lPrecMat + self.lPrecMat.T) / 2
        evals, evecs = la.eigh(self.lPrecMat, check_finite=False)
        evals = np.maximum(evals, 1e-10)
        self.lPrecMat = evecs @ np.diag(evals) @ evecs.T
        self.lB = np.zeros(self.nLocal)
        self.lMean = np.zeros(self.nLocal)
        # Store initials for potential reset
        self.initialThetaKappa = thetaPrecision
        self.initialLogitPg = initialLogitPg
        self.initialLPrecMat = self.lPrecMat.copy()

    def minAng(self, a, b):
        d = np.abs(a - b) % 360
        return np.minimum(d, 360 - d)

    def wrap_angle(self, x):
        return (x + 180) % 360 - 180

    def getIdx(self, theta):
        theta_norm = theta % 360
        idx = np.where(np.isclose(self.targetPositions, theta_norm))[0]
        if len(idx) == 0:
            raise ValueError(f"No matching target angle for {theta}")
        return idx[0]

    def getPredictive(self, trialNum):
        idx = self.getIdx(self.targets[trialNum])
        # Prediction step for states
        pred_state = self.trans_mat.T @ self.prev_state_posterior
        pS0, pS1 = pred_state
        deltaG = self.thetaMu
        varG = self.sigmaAim ** 2 + (1 / self.thetaKappa if self.thetaKappa > 1e-10 else 1e10)
        muL = self.lMean[idx]
        lCov = np.linalg.pinv(self.lPrecMat, rcond=1e-15) if np.linalg.det(self.lPrecMat) > 1e-10 else np.eye(self.nLocal) * 1e10
        varL = self.sigmaAim ** 2 + lCov[idx, idx]
        return {'pS1': pS1, 'pG': self.pG, 'deltaG': deltaG, 'varG': varG, 'muL': muL, 'varL': varL}

    def updatePosteriors(self, trialNum, deltaObs):
        if self.targets is None or trialNum >= len(self.targets):
            return
        deltaObs = self.wrap_angle(deltaObs)  # Wrap input
        idx = self.getIdx(self.targets[trialNum])
        # Prediction step
        pred_state = self.trans_mat.T @ self.prev_state_posterior
        pS0, pS1 = pred_state
        # Compute state-dependent likelihoods for deltaObs
        deltaG = self.thetaMu
        predVarG = self.sigmaObs ** 2 + (1 / self.thetaKappa if self.thetaKappa > 1e-10 else 1e10)
        margLikG = norm.pdf(deltaObs, deltaG, np.sqrt(predVarG))
        lCov = np.linalg.pinv(self.lPrecMat, rcond=1e-15) if np.linalg.det(self.lPrecMat) > 1e-10 else np.eye(self.nLocal) * 1e10
        predVarL = self.sigmaObs ** 2 + lCov[idx, idx]
        margLikL = norm.pdf(deltaObs, self.lMean[idx], np.sqrt(predVarL))
        lik0 = norm.pdf(deltaObs, 0, self.sigmaObs)  # Changed to N(0, sigmaObs) for s=0
        lik1 = self.pG * margLikG + (1 - self.pG) * margLikL  # Perturbed: mixture
        # Update step
        unnorm_post = np.array([lik0 * pred_state[0], lik1 * pred_state[1]])
        post = unnorm_post / np.sum(unnorm_post)
        self.prev_state_posterior = post
        # Responsibilities for updates (only in s=1)
        resp_global = post[1] * self.pG
        resp_local = post[1] * (1 - self.pG)
        # Local update (weighted by resp_local)
        addInv = np.zeros((self.nLocal, self.nLocal))
        addInv[idx, idx] = resp_local / self.sigmaObs ** 2
        self.lPrecMat += addInv
        # Force symmetry
        self.lPrecMat = (self.lPrecMat + self.lPrecMat.T) / 2
        # Handle NaN/inf
        self.lPrecMat = np.nan_to_num(self.lPrecMat, posinf=1e10, neginf=-1e10)
        # Stability check
        if np.linalg.det(self.lPrecMat) < 1e-10:
            self.lPrecMat += np.eye(self.nLocal) * 1e-10
        # Always clip eigenvalues
        evals, evecs = la.eigh(self.lPrecMat, check_finite=False)
        evals = np.maximum(evals, 1e-10)
        self.lPrecMat = evecs @ np.diag(evals) @ evecs.T
        self.lB[idx] += resp_local * deltaObs / self.sigmaObs ** 2
        self.lMean = np.linalg.pinv(self.lPrecMat, rcond=1e-15).dot(self.lB)
        self.lMean = self.wrap_angle(self.lMean)  # Wrap local means
        # Global update (weighted by resp_global)
        addPrecision = resp_global / self.sigmaObs ** 2
        addRx = addPrecision * np.cos(np.deg2rad(deltaObs))
        addRy = addPrecision * np.sin(np.deg2rad(deltaObs))
        self.thetaRx += addRx
        self.thetaRy += addRy
        self.thetaKappa = np.sqrt(self.thetaRx ** 2 + self.thetaRy ** 2)
        if self.thetaKappa > 1e-10:
            self.thetaMu = np.rad2deg(np.arctan2(self.thetaRy, self.thetaRx))
            self.thetaMu = self.wrap_angle(self.thetaMu)  # Wrap mu
        # pG update (weighted by post[1])
        if margLikL > 0:
            bayesFactor = margLikG / margLikL
        else:
            bayesFactor = 1e10
        effectiveBF = bayesFactor ** post[1]  # Weight BF by post[1]
        self.pG = self.pG * effectiveBF / (self.pG * effectiveBF + (1 - self.pG)) if self.pG < 1 else self.pG

    def expectedMove(self, trialNum):
        predDict = self.getPredictive(trialNum)
        pS1 = predDict['pS1']
        pG = predDict['pG']
        deltaG = predDict['deltaG']
        muL = predDict['muL']
        delta = (1 - pS1) * 0 + pS1 * (pG * deltaG + (1 - pG) * muL)  # Delta 0 in s=0
        aim = -delta
        aim = self.wrap_angle(aim)  # Wrap aim
        return aim

class Objective:
    def __init__(self, isRmse, allAims, mask, trials, targets, compMags):
        self.isRmse = isRmse
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.targets = targets
        self.compMags = compMags

    def __call__(self, params):
        numTrials = len(self.trials)
        if self.isRmse:
            fittingVariance = None
            hazard = 0.01
            sigmaObs = 1.0
            initialLogitPg = 0.0
            thetaPrecision = 1e9
            priorPrec = 1e-9
            sigmaAim = 1.0
            kernelL = 45.0
            kernelAmp = 1.0
            logit_w = 0.0
            log_f = np.log(1/360)
        else:
            logFv, logHazard, logSigma, initialLogitPg, logThetaPrec, logPriorPrec, logSigmaAim, logKernelL, logKernelAmp, logit_w, log_f = params
            fittingVariance = np.exp(logFv)
            hazard = np.exp(logHazard)
            sigmaObs = np.exp(logSigma)
            initialLogitPg = initialLogitPg * 10
            thetaPrecision = np.exp(logThetaPrec)
            priorPrec = np.exp(logPriorPrec)
            sigmaAim = np.exp(logSigmaAim)
            kernelL = np.exp(logKernelL)
            kernelAmp = np.exp(logKernelAmp)
            logit_w *=1000
            # logit_w and log_f already unpacked
        stepper = BayesianStepper(fittingVariance, sigmaObs, initialLogitPg, self.targets, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, hazard, logit_w, log_f)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            predDict = stepper.getPredictive(trial)
            mOut = stepper.expectedMove(trial)
            mOuts[trial] = mOut
            if not self.isRmse:
                fvAdd = fittingVariance if fittingVariance is not None else 0
                pdfG = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(predDict['varG'] + fvAdd))
                pdfL = norm.pdf(self.allAims[trial], -predDict['muL'], np.sqrt(predDict['varL'] + fvAdd))
                pS1 = predDict['pS1']
                pG = predDict['pG']
                pdf_s1 = pG * pdfG + (1 - pG) * pdfL
                pdf = pS1 * pdf_s1 + (1 - pS1) * norm.pdf(self.allAims[trial], 0, np.sqrt(sigmaObs ** 2 + fvAdd)) + 1e-12  # Fixed sigmaObs
                if self.mask[trial]:
                    logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, self.compMags[trial])
        if self.isRmse:
            totErrs = self.allAims[self.mask] - mOuts[self.mask]
            if len(totErrs) == 0:
                return np.inf
            mu, std = norm.fit(totErrs)
            logLikelihood = np.sum(np.log(norm.pdf(totErrs, mu, std) + 1e-12))
        totalLogLikelihood = logLikelihood
        if np.isfinite(totalLogLikelihood) and not np.isnan(totalLogLikelihood):
            return -totalLogLikelihood
        return 1e9

def fitSingle(data):
    allAims, mask, trials, targets, isRmse, heightCap, compMags, pp = data
    objFunc = Objective(isRmse, allAims, mask, trials, targets, compMags)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 11,  # Updated for removed param
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    if isRmse:
        # No parameters to fit; compute directly with fixed values
        bestFun = objFunc(np.array([])) # Dummy call
        bestX = np.array([])
        paramCount = 0 # No fitted parameters (mu/std are auxiliary)
    else:
        boundsSingle = [
            (np.log(1e-9), np.log(1e5)), # logFv
            (np.log(1e-9), np.log(0.5)), # logHazard adjusted upper for slower transition
            (np.log(1e-9), np.log(2000)), # logSigma
            (-30, 2), # initialLogitPg
            (np.log(1e-9), np.log(1e9)), # logThetaPrec
            (np.log(1), np.log(1e9)), # logPriorPrec adjusted lower for low initial var
            (np.log(1e-9), np.log(1e3)), # logSigmaAim
            (np.log(30), np.log(360)), # logKernelL
            (np.log(0.1), np.log(1000)), # logKernelAmp
            (-.01, 10),  # logit_w limited to w <= 0.5 for PSD
            (np.log(1e-9), np.log(0.1))  # log_f (new)
        ]
        boundsArray = np.array(boundsSingle)
        # Bias initialization
        x0 = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        x0[0] = np.log(1) # low fittingVar
        x0[1] = np.log(0.001)  # low hazard
        x0[3] = -10  # low initial pG
        x0[4] = np.log(1)  # low thetaPrec, high varG early
        x0[5] = np.log(100)  # high priorPrec, low varL initial
        x0[6] = np.log(1)  # low sigmaAim
        x0[7] = (np.log(90) + np.log(360)) / 2  # Midpoint for logKernelL
        x0[8] = (np.log(0.1) + np.log(1000)) / 2  # Midpoint for logKernelAmp
        x0[9] = -5.0  # Bias logit_w negative for w<0.5
        x0[10] = np.log(1/360)  # Init f to 1/360
        maxRestarts = 1000 # Define your desired number of restarts here (e.g., 5-10 is common)
        defaultPopSize = 6 # Fixed small/default population size
        largePopSize = defaultPopSize * 2 # Starting large population size (will double in BIPOP large regime)
        popSize = defaultPopSize
        bestValue = np.inf
        bestX = None
        globalIt = 0
        restart = 0
        globalSinceBest = 0
        iteration = 0
        while restart < maxRestarts and globalSinceBest < 1800//popSize:
        
            if globalSinceBest < iteration:
                largePopSize /= 2#defaultPopSize * 2
                largePopSize = max(largePopSize,defaultPopSize*2)
            if restart % 2 == 1: # Small regime
                popSize = defaultPopSize
                sigma = .5 # Standard sigma for small
            elif restart == 0:
                popSize = defaultPopSize*4
                sigma = 4
            else: # Large regime
                popSize = int(largePopSize)
                sigma = 2 # Slightly larger sigma for more exploration in large regime
                largePopSize *= 2
        
            # Random init mean for each restart
            np.random.seed(4321 + restart)
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=124 + restart)
            es.tolfun = 1e-8
            sinceBest = 0
            bestInRun = 1e9
            iteration = 0
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()//2) as executor:
                while not es.should_stop() and sinceBest < 7:#30//popSize:# and iteration < 50:#900 // popSize:
                    xSamples = [es.ask() for _ in range(es.population_size)]
                    fValues = list(executor.map(objFunc, xSamples))
                    solutions = list(zip(xSamples, fValues))
                    es.tell(solutions)
                    # Update best (per generation)
                    currentBest = min(solutions, key=lambda s: s[1])
                    if currentBest[1] < bestValue:
                        bestValue = currentBest[1]
                        bestX = currentBest[0]
                        print(restart,iteration,bestValue,currentBest[0],globalSinceBest)
                        globalSinceBest = 0
                   
                    else:
                        globalSinceBest += 1
                    if currentBest[1] < bestInRun:
                        bestInRun = currentBest[1]
                        sinceBest = 0
                    else:
                        sinceBest += 1
               
                    globalIt += 1
                    iteration += 1
            print(restart,globalIt,popSize,globalSinceBest)
            restart += 1
        bestFun = bestValue
        paramCount = 11
    logLikelihood = -bestFun
    if isRmse:
        fittingVariance = None
        hazard = 0.01
        sigmaObs = 1.0
        initialLogitPg = 0.0
        thetaPrecision = 1e9
        priorPrec = 1e-9
        sigmaAim = 1.0
        kernelL = 45.0
        kernelAmp = 1.0
        logit_w = 0.0
        log_f = np.log(1/360)
    else:
        logFv, logHazard, logSigma, initialLogitPg, logThetaPrec, logPriorPrec, logSigmaAim, logKernelL, logKernelAmp, logit_w, log_f = bestX
        fittingVariance = np.exp(logFv)
        hazard = np.exp(logHazard)
        sigmaObs = np.exp(logSigma)
        initialLogitPg = initialLogitPg * 10
        thetaPrecision = np.exp(logThetaPrec)
        priorPrec = np.exp(logPriorPrec)
        sigmaAim = np.exp(logSigmaAim)
        kernelL = np.exp(logKernelL)
        kernelAmp = np.exp(logKernelAmp)
        logit_w *=1000
    xs = [fittingVariance, hazard, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, logit_w, log_f]
    stepperSingle = BayesianStepper(fittingVariance, sigmaObs, initialLogitPg, targets=targets, thetaPrecision=thetaPrecision, priorPrec=priorPrec, sigmaAim=sigmaAim, kernelL=kernelL, kernelAmp=kernelAmp, hazard=hazard, logit_w=logit_w, log_f=log_f)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        mOutsSingle[trial] = stepperSingle.expectedMove(trial)
        stepperSingle.updatePosteriors(trial, compMags[trial])
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, xs)
    violin_plot_model_vs_human_aims(xs, targets, np.arange(len(allAims)), compMags, allAims, number=pp)
    plot_model_vs_human_aims(xs, targets, np.arange(len(allAims)), compMags, allAims, number=pp)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist()
    }

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import PolyCollection
def plot_model_vs_human_aims(fitted_params, targets, trials, compMags, human_aims, num_samples=100,number=0):
    # Unpack fitted params (assume order from xs: fittingVariance, hazard, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, logit_w, log_f)
    fittingVariance, hazard, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, logit_w, log_f = fitted_params
    # Re-create the stepper with fitted params
    stepper = BayesianStepper(fittingVariance, sigmaObs, initialLogitPg, targets, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, hazard, logit_w, log_f)
    # Collect predDict for each trial by simulating the model
    pred_dicts = []
    for trial in trials:
        predDict = stepper.getPredictive(trial)
        pred_dicts.append(predDict)
        stepper.updatePosteriors(trial, compMags[trial])
    # Generate samples from model distributions for each trial
    model_samples = []
    fvAdd = fittingVariance if fittingVariance is not None else 0
    for predDict in pred_dicts:
        pS1 = predDict['pS1']
        pG = predDict['pG']
        deltaG = predDict['deltaG']
        varG = predDict['varG'] + fvAdd
        muL = predDict['muL']
        varL = predDict['varL'] + fvAdd
        # Sample from marginal mixture
        is_perturbed = np.random.rand(num_samples) < pS1
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            if is_perturbed[i]:
                is_global = np.random.rand() < pG
                if is_global:
                    samples[i] = np.random.normal(-deltaG, np.sqrt(varG))
                else:
                    samples[i] = np.random.normal(-muL, np.sqrt(varL))
            else:
                samples[i] = np.random.normal(0, np.sqrt(fvAdd + sigmaObs ** 2))  # Fixed to sigmaObs
        # Wrap to [-180, 180]
        samples = (samples + 180) % 360 - 180
        model_samples.append(samples)
    # Prepare data for plotting
    # Model: list of arrays, one per trial
    # Human: array matching trials
    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    # Swarm for model samples
    sns.swarmplot(data=model_samples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    # Overlay human aims as red points
    ax.scatter(trials, human_aims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig(str(number)+"testScatter.svg")
    plt.show()
# plot_model_vs_human_aims(self.xs[i], self.dat['targetPosition'].values, np.arange(len(self.mStates[i])), self.conVal, np.array(self.allAims[i]))
def violin_plot_model_vs_human_aims(fitted_params, targets, trials, compMags, human_aims, num_samples=1000, num_plot_samples=100,number=0):
    # Unpack fitted params (assume order from xs: fittingVariance, hazard, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, logit_w, log_f)
    fittingVariance, hazard, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, logit_w, log_f = fitted_params
    # Re-create the stepper with fitted params
    stepper = BayesianStepper(fittingVariance, sigmaObs, initialLogitPg, targets, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, hazard, logit_w, log_f)
    # Collect predDict for each trial by simulating the model
    pred_dicts = []
    for trial in trials:
        predDict = stepper.getPredictive(trial)
        pred_dicts.append(predDict)
        stepper.updatePosteriors(trial, compMags[trial])
    # Generate samples from model distributions for each trial
    model_samples = []
    fvAdd = fittingVariance if fittingVariance is not None else 0
    for predDict in pred_dicts:
        pS1 = predDict['pS1']
        pG = predDict['pG']
        deltaG = predDict['deltaG']
        varG = predDict['varG'] + fvAdd
        muL = predDict['muL']
        varL = predDict['varL'] + fvAdd
        # Sample from marginal mixture
        is_perturbed = np.random.rand(num_samples) < pS1
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            if is_perturbed[i]:
                is_global = np.random.rand() < pG
                if is_global:
                    samples[i] = np.random.normal(-deltaG, np.sqrt(varG))
                else:
                    samples[i] = np.random.normal(-muL, np.sqrt(varL))
            else:
                samples[i] = np.random.normal(0, np.sqrt(fvAdd + sigmaObs ** 2))  # Fixed to sigmaObs
        # Wrap to [-180, 180]
        samples = (samples + 180) % 360 - 180
        model_samples.append(samples)
    # Prepare long-form DataFrame for seaborn
    data_list = []
    for i, t in enumerate(trials):
        subsamples = np.random.choice(model_samples[i], size=num_plot_samples, replace=False)
        for sample in subsamples:
            data_list.append({'Trial': t, 'Aim': sample})
    df = pd.DataFrame(data_list)
    df['Trial_violin'] = df['Trial'] + 0.2
    df['Trial_box'] = df['Trial']
    df['Trial_strip'] = df['Trial'] - 0.2
    # Plot - make wider for spacing
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.3), 6)) # Increased multiplier for wider plot
    # Violin plot on right (half)
    sns.violinplot(data=df, x='Trial_violin', y='Aim', inner=None, color='skyblue', alpha=0.5, ax=ax, scale='width', linewidth=0.5, cut=0, bw_adjust=0.5)
    # Clip violins to right half
    violin_collections = [c for c in ax.collections if isinstance(c, PolyCollection)]
    for i, poly in enumerate(violin_collections):
        center = trials[i] + 0.2
        verts = poly.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], center, np.inf)
    # Boxplot for summary stats at center
    #sns.boxplot(data=df, x='Trial_box', y='Aim', color='black', width=0.05, showfliers=False, ax=ax)
    # Stripplot for 'rain' points on left
    #sns.stripplot(data=df, x='Trial_strip', y='Aim', jitter=0.15, alpha=0.3, color='blue', size=3, ax=ax)
    # Overlay human aims as red points
    ax.scatter(trials, human_aims, color='red', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Raincloud (Half-Violin + Box + Strip) vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(number)+"testViolin.svg")
    plt.show()
# plot_model_vs_human_aims(self.xs[i], self.dat['targetPosition'].values, np.arange(len(self.mStates[i])), self.conVal, np.array(self.allAims[i]))
class fitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, isRmse=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.isRmse = isRmse
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participants_in_condition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participants_in_condition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        targets = pDatFirst['targetPosition'].values
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.dat[(self.dat['participantNum'] == pp)]
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, targets, self.isRmse, self.heightCap, compMags,pp))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
            
"""
            
"""
import numpy as np
from scipy.stats import norm
from scipy.special import expit
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
np.random.seed(99)
class BayesianStepper:
    def __init__(self, fittingVariance=None, threshold=1.0, sigmaObs=1.0, initialLogitPg=0.0, targets=None, thetaPrecision=1e9, priorPrec=1e-9, sigmaAim=1.0, kernelL=45.0, kernelAmp=1.0, learningRate=1.0):
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.threshold = threshold
        self.sigmaObs = sigmaObs
        self.sigmaAim = sigmaAim
        self.learningRate = learningRate
        self.sigmaH0 = 10.0
        self.cumSurprise = 0.0
        self.perturbed = False
        self.pG = expit(initialLogitPg)
        self.thetaMean = 0
        self.thetaPrecision = thetaPrecision
        self.targets = targets
        if targets is not None:
            unique_targets = np.unique(targets)
            self.targetPositions = np.sort(unique_targets) % 360
            self.nLocal = len(self.targetPositions)
        else:
            self.targetPositions = np.arange(0, 360, 45)
            self.nLocal = 8
        # Compute prior cov K for local GP
        K = np.zeros((self.nLocal, self.nLocal))
        for i in range(self.nLocal):
            for j in range(self.nLocal):
                d = self.minAng(self.targetPositions[i], self.targetPositions[j])
                rbf = np.exp(- (d ** 2) / (2 * kernelL ** 2))
                cosTerm = np.cos(np.deg2rad(d))
                K[i, j] = kernelAmp ** 2 * rbf * cosTerm
        # Precision matrix = inv(K + jitter) + diagonal prior prec
        jitter = 1e-6 * np.eye(self.nLocal)
        self.lPrecMat = np.linalg.inv(K + jitter) + np.eye(self.nLocal) * priorPrec
        self.lB = np.zeros(self.nLocal)
        self.lMean = np.zeros(self.nLocal)
    def minAng(self, a, b):
        d = np.abs(a - b) % 360
        return np.minimum(d, 360 - d)
    def getIdx(self, theta):
        theta_norm = theta % 360
        idx = np.where(np.isclose(self.targetPositions, theta_norm))[0]
        if len(idx) == 0:
            raise ValueError(f"No matching target angle for {theta}")
        return idx[0]
    def getPredictive(self, trialNum):
        idx = self.getIdx(self.targets[trialNum])
        deltaG = self.thetaMean
        varG = self.sigmaAim ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
        muL = self.lMean[idx]
        lCov = np.linalg.pinv(self.lPrecMat) if np.linalg.det(self.lPrecMat) > 1e-10 else np.eye(self.nLocal) * 1e10
        varL = self.sigmaAim ** 2 + lCov[idx, idx]
        return {'pG': self.pG, 'deltaG': deltaG, 'varG': varG, 'muL': muL, 'varL': varL}
    def updatePosteriors(self, trialNum, deltaObs):
        if self.targets is None or trialNum >= len(self.targets):
            return
        idx = self.getIdx(self.targets[trialNum])
        # Surprise detection (change-point evidence)
        if not self.perturbed:
            surprise = (deltaObs ** 2) / (2 * self.sigmaH0 ** 2)
            self.cumSurprise += surprise
            if self.cumSurprise > self.threshold:
                self.perturbed = True
        if self.perturbed:
            # Global update
            predMeanG = self.thetaMean
            predVarG = self.sigmaObs ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
            margLikG = norm.pdf(deltaObs, predMeanG, np.sqrt(predVarG))
            addPrecision = (1 / self.sigmaObs ** 2) * self.learningRate
            oldSumWeighted = self.thetaMean * self.thetaPrecision
            self.thetaPrecision += addPrecision
            self.thetaMean = (oldSumWeighted + deltaObs * addPrecision) / self.thetaPrecision
            # Local update
            predMeanL = self.lMean[idx]
            lCov = np.linalg.pinv(self.lPrecMat) if np.linalg.det(self.lPrecMat) > 1e-10 else np.eye(self.nLocal) * 1e10
            predVarL = self.sigmaObs ** 2 + np.dot(lCov[idx], lCov[idx]) # Correction for varL
            margLikL = norm.pdf(deltaObs, predMeanL, np.sqrt(predVarL))
            # Bayes factor for pG update
            if margLikL > 0:
                bayesFactor = margLikG / margLikL
            else:
                bayesFactor = 1e10
            self.pG = self.pG * bayesFactor / (self.pG * bayesFactor + (1 - self.pG)) if self.pG < 1 else self.pG
            addInv = np.zeros((self.nLocal, self.nLocal))
            addInv[idx, idx] = (1 / self.sigmaObs ** 2) * self.learningRate
            self.lPrecMat += addInv
            if np.linalg.det(self.lPrecMat) < 1e-10:
                self.lPrecMat += np.eye(self.nLocal) * 1e-10
            self.lB[idx] += deltaObs / self.sigmaObs ** 2 * self.learningRate
            self.lMean = np.linalg.pinv(self.lPrecMat).dot(self.lB)
    def expectedMove(self, trialNum):
        predDict = self.getPredictive(trialNum)
        delta = predDict['pG'] * predDict['deltaG'] + (1 - predDict['pG']) * predDict['muL']
        aim = -delta
        aim = (aim + 180) % 360 - 180
        return aim
class Objective:
    def __init__(self, isRmse, allAims, mask, trials, targets, compMags):
        self.isRmse = isRmse
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.targets = targets
        self.compMags = compMags
    def __call__(self, params):
        numTrials = len(self.trials)
        if self.isRmse:
            fittingVariance = None
            threshold = 1.0
            sigmaObs = 1.0
            initialLogitPg = 0.0
            thetaPrecision = 1e9
            priorPrec = 1e-9
            sigmaAim = 1.0
            kernelL = 45.0
            kernelAmp = 1.0
            learningRate = 1.0
        else:
            logFv, logThresh, logSigma, initialLogitPg, logThetaPrec, logPriorPrec, logSigmaAim, logKernelL, logKernelAmp, logLearningRate = params
            fittingVariance = np.exp(logFv)
            threshold = np.exp(logThresh)
            sigmaObs = np.exp(logSigma)
            initialLogitPg = initialLogitPg * 10
            thetaPrecision = np.exp(logThetaPrec)
            priorPrec = np.exp(logPriorPrec)
            sigmaAim = np.exp(logSigmaAim)
            kernelL = np.exp(logKernelL)
            kernelAmp = np.exp(logKernelAmp)
            learningRate = expit(logLearningRate)
        stepper = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, self.targets, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            predDict = stepper.getPredictive(trial)
            mOut = stepper.expectedMove(trial)
            mOuts[trial] = mOut
           
            if not self.isRmse:
                fvAdd = fittingVariance if fittingVariance is not None else 0
                pdfG = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(predDict['varG'] + fvAdd))
                pdfL = norm.pdf(self.allAims[trial], -predDict['muL'], np.sqrt(predDict['varL'] + fvAdd))
                pdf = predDict['pG'] * pdfG + (1 - predDict['pG']) * pdfL + 1e-12
                if self.mask[trial]:
                    logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, self.compMags[trial])
        if self.isRmse:
            totErrs = self.allAims[self.mask] - mOuts[self.mask]
            if len(totErrs) == 0:
                return np.inf
            mu, std = norm.fit(totErrs)
            logLikelihood = np.sum(np.log(norm.pdf(totErrs, mu, std) + 1e-12))
        totalLogLikelihood = logLikelihood
        if np.isfinite(totalLogLikelihood) and not np.isnan(totalLogLikelihood):
            return -totalLogLikelihood
        return 1e9
def fitSingle(data):
    allAims, mask, trials, targets, isRmse, heightCap, compMags = data
    objFunc = Objective(isRmse, allAims, mask, trials, targets, compMags)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 10,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    if isRmse:
        # No parameters to fit; compute directly with fixed values
        bestFun = objFunc(np.array([])) # Dummy call
        bestX = np.array([])
        paramCount = 0 # No fitted parameters (mu/std are auxiliary)
    else:
        boundsSingle = [
            (np.log(1e-9), np.log(1e5)), # logFv
            (np.log(1), np.log(1e9)), # logThresh
            (np.log(1e-9), np.log(20)), # logSigma
            (-10, 2), # initialLogitPg
            (np.log(1e-9), np.log(1e9)), # logThetaPrec
            (np.log(1e-9), np.log(1e9)), # logPriorPrec
            (np.log(1e-9), np.log(1e1)), # logSigmaAim
            (np.log(30), np.log(360)), # logKernelL
            (np.log(0.1), np.log(100)), # logKernelAmp
            (-10, 0) # logLearningRate, expit to 0.1-0.5 for slow
        ]
        boundsArray = np.array(boundsSingle)
        x0 = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        maxRestarts = 1000 # Define your desired number of restarts here (e.g., 5-10 is common)
        defaultPopSize = 6 # Fixed small/default population size
        largePopSize = defaultPopSize * 2 # Starting large population size (will double in BIPOP large regime)
        popSize = defaultPopSize
        bestValue = np.inf
        bestX = None
        globalIt = 0
        restart = 0
        globalSinceBest = 0
        iteration = 0
        while restart < maxRestarts and globalSinceBest < 1800//popSize:
         
            if globalSinceBest < iteration:
                largePopSize /= 2#defaultPopSize * 2
                largePopSize = max(largePopSize,defaultPopSize*2)
            if restart % 2 == 1: # Small regime
                popSize = defaultPopSize
                sigma = .5 # Standard sigma for small
            elif restart == 0:
                popSize = defaultPopSize*8
                sigma = 4
            else: # Large regime
                popSize = int(largePopSize)
                sigma = 2 # Slightly larger sigma for more exploration in large regime
                largePopSize *= 2
         
            # Random init mean for each restart
            np.random.seed(4321 + restart)
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=99 + restart)
            es.tolfun = 1e-8
            sinceBest = 0
            bestInRun = 1e9
            iteration = 0
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as executor:
                while not es.should_stop() and sinceBest < 5:#30//popSize:# and iteration < 50:#900 // popSize:
                    xSamples = [es.ask() for _ in range(es.population_size)]
                    fValues = list(executor.map(objFunc, xSamples))
                    solutions = list(zip(xSamples, fValues))
                    es.tell(solutions)
                    # Update best (per generation)
                    currentBest = min(solutions, key=lambda s: s[1])
                    if currentBest[1] < bestValue:
                        bestValue = currentBest[1]
                        bestX = currentBest[0]
                        print(restart,iteration,bestValue,currentBest[0],globalSinceBest)
                        globalSinceBest = 0
                    
                    else:
                        globalSinceBest += 1
                    if currentBest[1] < bestInRun:
                        bestInRun = currentBest[1]
                        sinceBest = 0
                    else:
                        sinceBest += 1
                
                    globalIt += 1
                    iteration += 1
            print(restart,globalIt,popSize,globalSinceBest)
            restart += 1
        bestFun = bestValue
        paramCount = 10
    logLikelihood = -bestFun
    if isRmse:
        fittingVariance = None
        threshold = 1.0
        sigmaObs = 1.0
        initialLogitPg = 0.0
        thetaPrecision = 1e9
        priorPrec = 1e-9
        sigmaAim = 1.0
        kernelL = 45.0
        kernelAmp = 1.0
        learningRate = 1.0
    else:
        logFv, logThresh, logSigma, initialLogitPg, logThetaPrec, logPriorPrec, logSigmaAim, logKernelL, logKernelAmp, logLearningRate = bestX
        fittingVariance = np.exp(logFv)
        threshold = np.exp(logThresh)
        sigmaObs = np.exp(logSigma)
        initialLogitPg = initialLogitPg * 10
        thetaPrecision = np.exp(logThetaPrec)
        priorPrec = np.exp(logPriorPrec)
        sigmaAim = np.exp(logSigmaAim)
        kernelL = np.exp(logKernelL)
        kernelAmp = np.exp(logKernelAmp)
        learningRate = expit(logLearningRate)
    xs = [fittingVariance, threshold, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate]
    stepperSingle = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, targets=targets, thetaPrecision=thetaPrecision, priorPrec=priorPrec, sigmaAim=sigmaAim, kernelL=kernelL, kernelAmp=kernelAmp, learningRate=learningRate)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        mOutsSingle[trial] = stepperSingle.expectedMove(trial)
        stepperSingle.updatePosteriors(trial, compMags[trial])
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, xs)
    violin_plot_model_vs_human_aims(xs, targets, np.arange(len(allAims)), compMags, allAims)
    plot_model_vs_human_aims(xs, targets, np.arange(len(allAims)), compMags, allAims)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist()
    }
import matplotlib.pyplot as plt
import seaborn as sns
def plot_model_vs_human_aims(fitted_params, targets, trials, compMags, human_aims, num_samples=100):
    # Unpack fitted params (assume order from xs: fittingVariance, threshold, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate)
    fittingVariance, threshold, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate = fitted_params
    # Re-create the stepper with fitted params
    stepper = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, targets, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate)
    # Collect predDict for each trial by simulating the model
    pred_dicts = []
    for trial in trials:
        predDict = stepper.getPredictive(trial)
        pred_dicts.append(predDict)
        stepper.updatePosteriors(trial, compMags[trial])
    # Generate samples from model distributions for each trial
    model_samples = []
    fvAdd = fittingVariance if fittingVariance is not None else 0
    for predDict in pred_dicts:
        pG = predDict['pG']
        deltaG = predDict['deltaG']
        varG = predDict['varG'] + fvAdd
        muL = predDict['muL']
        varL = predDict['varL'] + fvAdd
        # Sample from mixture
        is_global = np.random.rand(num_samples) < pG
        samples = np.zeros(num_samples)
        samples[is_global] = np.random.normal(-deltaG, np.sqrt(varG), sum(is_global))
        samples[~is_global] = np.random.normal(-muL, np.sqrt(varL), sum(~is_global))
        # Wrap to [-180, 180]
        samples = (samples + 180) % 360 - 180
        model_samples.append(samples)
    # Prepare data for plotting
    # Model: list of arrays, one per trial
    # Human: array matching trials
    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    # Swarm for model samples
    sns.swarmplot(data=model_samples, ax=ax, color='blue', alpha=0.5, size=3, zorder=1)
    # Overlay human aims as red points
    ax.scatter(trials, human_aims, color='red', label='Human Aims', zorder=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.savefig("testScatter.svg")
    plt.show()
# plot_model_vs_human_aims(self.xs[i], self.dat['targetPosition'].values, np.arange(len(self.mStates[i])), self.conVal, np.array(self.allAims[i]))
def violin_plot_model_vs_human_aims(fitted_params, targets, trials, compMags, human_aims, num_samples=1000):
    # Unpack fitted params (assume order from xs: fittingVariance, threshold, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate)
    fittingVariance, threshold, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate = fitted_params
    # Re-create the stepper with fitted params
    stepper = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, targets, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp, learningRate)
    # Collect predDict for each trial by simulating the model
    pred_dicts = []
    for trial in trials:
        predDict = stepper.getPredictive(trial)
        pred_dicts.append(predDict)
        stepper.updatePosteriors(trial, compMags[trial])
    # Generate samples from model distributions for each trial
    model_samples = []
    fvAdd = fittingVariance if fittingVariance is not None else 0
    for predDict in pred_dicts:
        pG = predDict['pG']
        deltaG = predDict['deltaG']
        varG = predDict['varG'] + fvAdd
        muL = predDict['muL']
        varL = predDict['varL'] + fvAdd
        # Sample from mixture
        is_global = np.random.rand(num_samples) < pG
        samples = np.zeros(num_samples)
        samples[is_global] = np.random.normal(-deltaG, np.sqrt(varG), sum(is_global))
        samples[~is_global] = np.random.normal(-muL, np.sqrt(varL), sum(~is_global))
        # Wrap to [-180, 180]
        samples = (samples + 180) % 360 - 180
        model_samples.append(samples)
    # Prepare data for violin plot: dict of {trial: samples}
    violin_data = {t: model_samples[i] for i, t in enumerate(trials)}
    # Plot - make wide for many trials
    fig, ax = plt.subplots(figsize=(max(15, len(trials) * 0.1), 6)) # Scale width with num trials
    # Violin plot for model distributions (vertical KDE per trial)
    sns.violinplot(data=violin_data, inner=None, color='blue', alpha=0.5, ax=ax, scale='width', linewidth=0.5,cut=0)
    # Overlay human aims as red points
    ax.scatter(trials, human_aims, color='red', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Model Predicted Aim Distributions (Violin KDE) vs Human Aims')
    ax.set_ylim(-180, 180)
    ax.legend()
    plt.tight_layout()
    plt.savefig("testViolin.svg")
    plt.show()
# plot_model_vs_human_aims(self.xs[i], self.dat['targetPosition'].values, np.arange(len(self.mStates[i])), self.conVal, np.array(self.allAims[i]))
class fitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, isRmse=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.isRmse = isRmse
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            participants_in_condition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participants_in_condition)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        targets = pDatFirst['targetPosition'].values
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.dat[(self.dat['participantNum'] == pp)]
            allAims = pDat['aim'].values
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, targets, self.isRmse, self.heightCap, compMags))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']

"""

"""
import numpy as np
from scipy.stats import norm
from scipy.special import expit
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
np.random.seed(99)

class BayesianStepper:
    def __init__(self, fittingVariance=None, threshold=1.0, sigmaObs=1.0, initialLogitPg=0.0, targets=None, thetaPrecision=1e9, priorPrec=1e-9, sigmaAim=1.0, kernelL=45.0, kernelAmp=1.0):
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.threshold = threshold
        self.baseSigmaObs = sigmaObs  # Fitted base
        self.sigmaObs = sigmaObs  # Starts at base, updated per-trial
        self.sigmaAim = sigmaAim
        self.sigmaH0 = 10.0
        self.cumSurprise = 0.0
        self.perturbed = False
        self.pG = expit(initialLogitPg)
        self.thetaMean = 0
        self.thetaPrecision = thetaPrecision
        self.targets = targets
        if targets is not None:
            unique_targets = np.unique(targets)
            self.targetPositions = np.sort(unique_targets) % 360
            self.nLocal = len(self.targetPositions)
        else:
            self.targetPositions = np.arange(0, 360, 45)
            self.nLocal = 8
        # Compute prior cov K for local GP
        K = np.zeros((self.nLocal, self.nLocal))
        for i in range(self.nLocal):
            for j in range(self.nLocal):
                d = self.minAng(self.targetPositions[i], self.targetPositions[j])
                rbf = np.exp(- (d ** 2) / (2 * kernelL ** 2))
                cosTerm = np.cos(np.deg2rad(d))
                K[i, j] = kernelAmp ** 2 * rbf * cosTerm
        # Precision matrix = inv(K + jitter) + diagonal prior prec
        jitter = 1e-6 * np.eye(self.nLocal)
        self.lPrecMat = np.linalg.inv(K + jitter) + np.eye(self.nLocal) * priorPrec
        self.lB = np.zeros(self.nLocal)
        self.lMean = np.zeros(self.nLocal)

    def minAng(self, a, b):
        d = np.abs(a - b) % 360
        return np.minimum(d, 360 - d)

    def getIdx(self, theta):
        theta_norm = theta % 360
        idx = np.where(np.isclose(self.targetPositions, theta_norm))[0]
        if len(idx) == 0:
            raise ValueError(f"No matching target angle for {theta}")
        return idx[0]

    def getPredictive(self, trialNum):
        idx = self.getIdx(self.targets[trialNum])
        deltaG = self.thetaMean
        varG = self.sigmaAim ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
        muL = self.lMean[idx]
        lCov = np.linalg.pinv(self.lPrecMat) if np.linalg.det(self.lPrecMat) > 1e-10 else np.eye(self.nLocal) * 1e10
        varL = self.sigmaAim ** 2 + lCov[idx, idx]
        return {'pG': self.pG, 'deltaG': deltaG, 'varG': varG, 'muL': muL, 'varL': varL}

    def updatePosteriors(self, trialNum, deltaObs):
        if self.targets is None or trialNum >= len(self.targets):
            return
        idx = self.getIdx(self.targets[trialNum])
        # Surprise detection
        surprise = (deltaObs ** 2) / (2 * self.sigmaH0 ** 2)
        self.cumSurprise += surprise
        if self.cumSurprise > self.threshold:
            self.perturbed = True
        # Dynamic sigmaObs: base * (1 + surprise / (self.sigmaH0 ** 2))
        normSurprise = surprise / (self.sigmaH0 ** 2)
        self.sigmaObs = self.baseSigmaObs * (1 + normSurprise)
        if self.perturbed:
            # Global update
            predMeanG = self.thetaMean
            predVarG = self.sigmaObs ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
            margLikG = norm.pdf(deltaObs, predMeanG, np.sqrt(predVarG))
            addPrecision = 1 / self.sigmaObs ** 2
            oldSumWeighted = self.thetaMean * self.thetaPrecision
            self.thetaPrecision += addPrecision
            self.thetaMean = (oldSumWeighted + deltaObs * addPrecision) / self.thetaPrecision
            # Local update
            predMeanL = self.lMean[idx]
            lCov = np.linalg.pinv(self.lPrecMat) if np.linalg.det(self.lPrecMat) > 1e-10 else np.eye(self.nLocal) * 1e10
            predVarL = self.sigmaObs ** 2 + lCov[idx, idx]
            margLikL = norm.pdf(deltaObs, predMeanL, np.sqrt(predVarL))
            # Bayes factor
            if margLikL > 0:
                bayesFactor = margLikG / margLikL
            else:
                bayesFactor = 1e10
            self.pG = self.pG * bayesFactor / (self.pG * bayesFactor + (1 - self.pG)) if self.pG < 1 else self.pG
            # Update precision matrix and b for local
            addInv = np.zeros((self.nLocal, self.nLocal))
            addInv[idx, idx] = 1 / self.sigmaObs ** 2
            self.lPrecMat += addInv
            if np.linalg.det(self.lPrecMat) < 1e-10:
                self.lPrecMat += np.eye(self.nLocal) * 1e-10
            self.lB[idx] += deltaObs / self.sigmaObs ** 2
            self.lMean = np.linalg.pinv(self.lPrecMat).dot(self.lB)

    def expectedMove(self, trialNum):
        predDict = self.getPredictive(trialNum)
        delta = predDict['pG'] * predDict['deltaG'] + (1 - predDict['pG']) * predDict['muL']
        return -delta

class Objective:
    def __init__(self, isRmse, allAims, mask, trials, targets, conVal):
        self.isRmse = isRmse
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.targets = targets
        self.conVal = conVal
    def __call__(self, params):
        numTrials = len(self.trials)
        compMag = self.conVal
        if self.isRmse:
            fittingVariance = None
            threshold = 1.0
            sigmaObs = 1.0
            initialLogitPg = 0.0
            thetaPrecision = 1e9
            priorPrec = 1e-9
            sigmaAim = 1.0
            kernelL = 45.0
            kernelAmp = 1.0
        else:
            logFv, logThresh, logSigma, initialLogitPg, logThetaPrec, logPriorPrec, logSigmaAim, logKernelL, logKernelAmp = params
            fittingVariance = np.exp(logFv)
            threshold = np.exp(logThresh)
            sigmaObs = np.exp(logSigma)
            initialLogitPg = initialLogitPg * 10
            thetaPrecision = np.exp(logThetaPrec)
            priorPrec = np.exp(logPriorPrec)
            sigmaAim = np.exp(logSigmaAim)
            kernelL = np.exp(logKernelL)
            kernelAmp = np.exp(logKernelAmp)
        stepper = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, self.targets, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp)
        logLikelihood = 0.0
        mOuts = np.zeros(numTrials)
        for idx, trial in enumerate(self.trials):
            predDict = stepper.getPredictive(trial)
            mOut = stepper.expectedMove(trial)
            mOuts[trial] = mOut
            
            if not self.isRmse:
                fvAdd = fittingVariance if fittingVariance is not None else 0
                pdfG = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(predDict['varG'] + fvAdd))
                pdfL = norm.pdf(self.allAims[trial], -predDict['muL'], np.sqrt(predDict['varL'] + fvAdd))
                pdf = predDict['pG'] * pdfG + (1 - predDict['pG']) * pdfL + 1e-12
                if not np.isnan(self.allAims[trial]):
                    logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, compMag)
        if self.isRmse:
            totErrs = self.allAims[self.mask] - mOuts[self.mask]
            if len(totErrs) == 0:
                return np.inf
            mu, std = norm.fit(totErrs)
            logLikelihood = np.sum(np.log(norm.pdf(totErrs, mu, std) + 1e-12))
        totalLogLikelihood = logLikelihood
        if np.isfinite(totalLogLikelihood) and not np.isnan(totalLogLikelihood):
            return -totalLogLikelihood
        return 1e9

def fitSingle(data):
    allAims, mask, trials, targets, isRmse, heightCap, conVal = data
    objFunc = Objective(isRmse, allAims, mask, trials, targets, conVal)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 9,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    if isRmse:
        # No parameters to fit; compute directly with fixed values
        bestFun = objFunc(np.array([])) # Dummy call
        bestX = np.array([])
        paramCount = 0 # No fitted parameters (mu/std are auxiliary)
    else:
        boundsSingle = [
            (np.log(1e-9), np.log(1e5)),  # logFv
            (np.log(1), np.log(1e9)),  # logThresh
            (np.log(1e-9), np.log(1e4)),  # logSigma
            (-10, 1),  # initialLogitPg
            (np.log(1e-9), np.log(1e9)),  # logThetaPrec
            (np.log(1e-9), np.log(1e9)),  # logPriorPrec
            (np.log(1e-9), np.log(1e1)),  # logSigmaAim
            (np.log(30), np.log(360)),  # logKernelL
            (np.log(1), np.log(1e4))  # logKernelAmp
        ]
        boundsArray = np.array(boundsSingle)
        x0 = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        maxRestarts = 1000 # Define your desired number of restarts here (e.g., 5-10 is common)
        defaultPopSize = 6 # Fixed small/default population size
        largePopSize = defaultPopSize * 2 # Starting large population size (will double in BIPOP large regime)
        popSize = defaultPopSize
        bestValue = np.inf
        bestX = None
        globalIt = 0
        restart = 0
        globalSinceBest = 0
        iteration = 0
        while restart < maxRestarts and globalSinceBest < 1200//popSize:
          
            if globalSinceBest < iteration:
                largePopSize /= 2#defaultPopSize * 2
            if restart % 2 == 1: # Small regime
                popSize = defaultPopSize
                sigma = .5 # Standard sigma for small
            elif restart == 0:
                popSize = defaultPopSize*8
                sigma = 4
            else: # Large regime
                popSize = int(largePopSize)
                sigma = 2 # Slightly larger sigma for more exploration in large regime
                largePopSize *= 2
          
            # Random init mean for each restart
            np.random.seed(4321 + restart)
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=27 + restart)
            es.tolfun = 1e-8
            sinceBest = 0
            bestInRun = 1e9
            iteration = 0
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as executor:
                while not es.should_stop() and sinceBest < 5:#30//popSize:# and iteration < 50:#900 // popSize:
                    xSamples = [es.ask() for _ in range(es.population_size)]
                    fValues = list(executor.map(objFunc, xSamples))
                    solutions = list(zip(xSamples, fValues))
                    es.tell(solutions)
                    # Update best (per generation)
                    currentBest = min(solutions, key=lambda s: s[1])
                    if currentBest[1] < bestValue:
                        bestValue = currentBest[1]
                        bestX = currentBest[0]
                        print(restart,iteration,bestValue,currentBest[0],globalSinceBest)
                        globalSinceBest = 0
                     
                    else:
                        globalSinceBest += 1
                    if currentBest[1] < bestInRun:
                        bestInRun = currentBest[1]
                        sinceBest = 0
                    else:
                        sinceBest += 1
                 
                    globalIt += 1
                    iteration += 1
            print(restart,globalIt,popSize,globalSinceBest)
            restart += 1
        bestFun = bestValue
        paramCount = 9
    logLikelihood = -bestFun
    if isRmse:
        fittingVariance = None
        threshold = 1.0
        sigmaObs = 1.0
        initialLogitPg = 0.0
        thetaPrecision = 1e9
        priorPrec = 1e-9
        sigmaAim = 1.0
        kernelL = 45.0
        kernelAmp = 1.0
    else:
        logFv, logThresh, logSigma, initialLogitPg, logThetaPrec, logPriorPrec, logSigmaAim, logKernelL, logKernelAmp = bestX
        fittingVariance = np.exp(logFv)
        threshold = np.exp(logThresh)
        sigmaObs = np.exp(logSigma)
        initialLogitPg = initialLogitPg * 10
        thetaPrecision = np.exp(logThetaPrec)
        priorPrec = np.exp(logPriorPrec)
        sigmaAim = np.exp(logSigmaAim)
        kernelL = np.exp(logKernelL)
        kernelAmp = np.exp(logKernelAmp)
    compMag = conVal
    xs = [fittingVariance, threshold, sigmaObs, initialLogitPg, thetaPrecision, priorPrec, sigmaAim, kernelL, kernelAmp]
    stepperSingle = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, targets=targets, thetaPrecision=thetaPrecision, priorPrec=priorPrec, sigmaAim=sigmaAim, kernelL=kernelL, kernelAmp=kernelAmp)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        mOutsSingle[trial] = stepperSingle.expectedMove(trial)
        stepperSingle.updatePosteriors(trial, compMag)
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood, xs)
    print(allAims, mOutsSingle)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist()
    }

class fitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, isRmse=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.isRmse = isRmse
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp) & (self.dat['phase'] == self.fitPhase)]
        numTrials = len(pDatFirst)
        targets = pDatFirst['targetPosition'].values
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.dat[(self.dat['participantNum'] == pp) & (self.dat['phase'] == self.fitPhase)]
            allAims = pDat['aim'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, targets, self.isRmse, self.heightCap, self.conVal))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']
"""

"""
import numpy as np
from scipy.optimize import differential_evolution as evolution
from scipy.stats import norm
from scipy.special import expit
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from cmaes import CMA
np.random.seed(99)
class BayesianStepper:
    def __init__(self, fittingVariance=None, threshold=1.0, sigmaObs=1.0, initialLogitPg=0.0, targets=None, thetaPrecision=1e9, dInvPrec=1e-9, sigmaAim=1.0):
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.threshold = threshold
        self.sigmaObs = sigmaObs
        self.sigmaAim = sigmaAim
        self.sigmaH0 = 10.0
        self.cumSurprise = 0.0
        self.perturbed = False
        self.pG = expit(initialLogitPg)
        self.thetaMean = 0
        self.thetaPrecision = thetaPrecision
        self.dInvCov = np.eye(2) * dInvPrec
        self.dB = np.zeros(2)
        self.dMean = np.zeros(2)
        self.targets = targets
    def getFeatures(self, thetaRad):
        sinTheta = np.sin(thetaRad)
        cosTheta = np.cos(thetaRad)
        unitTheta = np.array([-sinTheta, cosTheta]) # Tangential unit vector
        return unitTheta
    def getPredictive(self, trialNum):
        thetaRad = np.deg2rad(self.targets[trialNum])
        unitTheta = self.getFeatures(thetaRad)
        deltaG = self.thetaMean
        varG = self.sigmaAim ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
        muL = np.dot(unitTheta, self.dMean)
        dCov = np.linalg.pinv(self.dInvCov) if np.linalg.det(self.dInvCov) > 1e-10 else np.eye(2) * 1e10
        varL = self.sigmaAim ** 2 + np.dot(unitTheta, dCov.dot(unitTheta))
        return {'pG': self.pG, 'deltaG': deltaG, 'varG': varG, 'muL': muL, 'varL': varL}
    def updatePosteriors(self, trialNum, deltaObs):
        if self.targets is None or trialNum >= len(self.targets):
            return
        thetaRad = np.deg2rad(self.targets[trialNum])
        unitTheta = self.getFeatures(thetaRad)
        # Surprise detection (change-point evidence)
        if not self.perturbed:
            surprise = (deltaObs ** 2) / (2 * self.sigmaH0 ** 2)
            self.cumSurprise += surprise
            if self.cumSurprise > self.threshold:
                self.perturbed = True
        if self.perturbed:
            # Global update
            predMeanG = self.thetaMean
            predVarG = self.sigmaObs ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
            margLikG = norm.pdf(deltaObs, predMeanG, np.sqrt(predVarG))
            addPrecision = 1 / self.sigmaObs ** 2
            oldSumWeighted = self.thetaMean * self.thetaPrecision
            self.thetaPrecision += addPrecision
            self.thetaMean = (oldSumWeighted + deltaObs * addPrecision) / self.thetaPrecision
            # Local update
            predMeanL = np.dot(unitTheta, self.dMean)
            dCov = np.linalg.pinv(self.dInvCov if np.linalg.det(self.dInvCov) > 1e-10 else np.eye(2) * 1e10)
            predVarL = self.sigmaObs ** 2 + np.dot(unitTheta, dCov.dot(unitTheta))
            margLikL = norm.pdf(deltaObs, predMeanL, np.sqrt(predVarL))
            # Bayes factor for pG update
            if margLikL > 0:
                bayesFactor = margLikG / margLikL
            else:
                bayesFactor = 1e10
            self.pG = self.pG * bayesFactor / (self.pG * bayesFactor + (1 - self.pG)) if self.pG < 1 else self.pG
            addInv = np.outer(unitTheta, unitTheta) / self.sigmaObs ** 2
            self.dInvCov += addInv
            if np.linalg.det(self.dInvCov) < 1e-10:
                self.dInvCov += np.eye(2) * 1e-10
            self.dB += unitTheta * deltaObs / self.sigmaObs ** 2
            self.dMean = np.linalg.pinv(self.dInvCov).dot(self.dB)
    def expectedMove(self, trialNum):
        predDict = self.getPredictive(trialNum)
        delta = predDict['pG'] * predDict['deltaG'] + (1 - predDict['pG']) * predDict['muL']
        return -delta
class Objective:
    def __init__(self, isRmse, allAims, mask, trials, targets, conVal):
        self.isRmse = isRmse
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.targets = targets
        self.conVal = conVal
    def __call__(self, params):
        numTrials = len(self.trials)
        compMag = self.conVal
        if self.isRmse:
            fittingVariance = None
            threshold = 1.0
            sigmaObs = 1.0
            initialLogitPg = 0.0
            thetaPrecision = 1e9
            dInvPrec = 1e-9
            sigmaAim = 1.0
        else:
            logFv, logThresh, logSigma, initialLogitPg, logThetaPrec, logDInvPrec, logSigmaAim = params
            fittingVariance = np.exp(logFv)
            threshold = np.exp(logThresh)
            sigmaObs = np.exp(logSigma)
            initialLogitPg = initialLogitPg * 10
            thetaPrecision = np.exp(logThetaPrec)
            dInvPrec = np.exp(logDInvPrec)
            sigmaAim = np.exp(logSigmaAim)
        stepper = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, self.targets, thetaPrecision, dInvPrec, sigmaAim)
        logLikelihood = 0.0
        mOuts = []
        pdfs = []
        for idx, trial in enumerate(self.trials):
            predDict = stepper.getPredictive(trial)
            mOut = stepper.expectedMove(trial)
            mOuts.append(mOut)
         
            if not self.isRmse:
                fvAdd = fittingVariance if fittingVariance is not None else 0
                pdfG = norm.pdf(self.allAims[trial], -predDict['deltaG'], np.sqrt(predDict['varG'] + fvAdd))
                pdfL = norm.pdf(self.allAims[trial], -predDict['muL'], np.sqrt(predDict['varL'] + fvAdd))
                pdf = predDict['pG'] * pdfG + (1 - predDict['pG']) * pdfL + 1e-12
                pdfs.append(pdf)
                if not np.isnan(self.allAims[trial]):
                    logLikelihood += np.log(pdf)
            stepper.updatePosteriors(trial, compMag)
        if self.isRmse:
            totErrs = self.allAims[self.mask] - mOuts[self.mask]
            if len(totErrs) == 0:
                return np.inf
            mu, std = norm.fit(totErrs)
            logLikelihood = np.sum(np.log(norm.pdf(totErrs, mu, std) + 1e-12))
        totalLogLikelihood = logLikelihood
        if np.isfinite(totalLogLikelihood) and not np.isnan(totalLogLikelihood):
            return -totalLogLikelihood
        return 1e9
def fitSingle(data):
    allAims, mask, trials, targets, isRmse, heightCap, conVal = data
    objFunc = Objective(isRmse, allAims, mask, trials, targets, conVal)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return {
            'xs': [None] * 7,
            'mStates': [0.0] * len(trials),
            'rmse': np.inf,
            'negLl': np.inf,
            'bic': np.inf,
            'allAims': allAims.tolist()
        }
    if isRmse:
        # No parameters to fit; compute directly with fixed values
        bestFun = objFunc(np.array([])) # Dummy call
        bestX = np.array([])
        paramCount = 0 # No fitted parameters (mu/std are auxiliary)
    else:
        boundsSingle = [(np.log(1e-9), np.log(1e5)), (np.log(1), np.log(1e9)), (np.log(1e-9), np.log(1e1)), (-10, 2), (np.log(1e-9), np.log(1e9)), (np.log(1e-9), np.log(1e9)), (np.log(1e-9), np.log(1e2))]
        boundsArray = np.array(boundsSingle)
        x0 = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
        maxRestarts = 1000 # Define your desired number of restarts here (e.g., 5-10 is common)
        defaultPopSize = 6 # Fixed small/default population size
        largePopSize = defaultPopSize * 2 # Starting large population size (will double in BIPOP large regime)
        popSize = defaultPopSize
        bestValue = np.inf
        bestX = None
        globalIt = 0
        restart = 0
        globalSinceBest = 0
        iteration = 0
        while restart < maxRestarts and globalSinceBest < 1200//popSize:
          
            if globalSinceBest < iteration:
                largePopSize /= 2#defaultPopSize * 2
            if restart % 2 == 1: # Small regime
                popSize = defaultPopSize
                sigma = .5 # Standard sigma for small
            else: # Large regime
                popSize = int(largePopSize)
                sigma = 2 # Slightly larger sigma for more exploration in large regime
                largePopSize *= 1.5
          
            # Random init mean for each restart
            np.random.seed(4321 + restart)
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=99 + restart)
            es.tolfun = 1e-8
            sinceBest = 0
            bestInRun = 1e9
            iteration = 0
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as executor:
                while not es.should_stop() and sinceBest < 5:#30//popSize:# and iteration < 50:#900 // popSize:
                    xSamples = [es.ask() for _ in range(es.population_size)]
                    fValues = list(executor.map(objFunc, xSamples))
                    solutions = list(zip(xSamples, fValues))
                    es.tell(solutions)
                    # Update best (per generation)
                    currentBest = min(solutions, key=lambda s: s[1])
                    if currentBest[1] < bestValue:
                        bestValue = currentBest[1]
                        bestX = currentBest[0]
                        print(restart,iteration,bestInRun,currentBest[0],globalSinceBest)
                        globalSinceBest = 0
                     
                    else:
                        globalSinceBest += 1
                    if currentBest[1] < bestInRun:
                        bestInRun = currentBest[1]
                        sinceBest = 0
                    else:
                        sinceBest += 1
                 
                    globalIt += 1
                    iteration += 1
            print(restart,globalIt,popSize,globalSinceBest)
            restart += 1
        bestFun = bestValue
        paramCount = 7
    logLikelihood = -bestFun
    # Extract parameters
    if isRmse:
        fittingVariance = None
        threshold = 1.0
        sigmaObs = 1.0
        initialLogitPg = 0.0
        thetaPrecision = 1e9
        dInvPrec = 1e-9
        sigmaAim = 1.0
    else:
        logFv, logThresh, logSigma, initialLogitPg, logThetaPrec, logDInvPrec, logSigmaAim = bestX
        fittingVariance = np.exp(logFv)
        threshold = np.exp(logThresh)
        sigmaObs = np.exp(logSigma)
        initialLogitPg = initialLogitPg * 10
        thetaPrecision = np.exp(logThetaPrec)
        dInvPrec = np.exp(logDInvPrec)
        sigmaAim = np.exp(logSigmaAim)
    compMag = conVal
    xs = [fittingVariance, threshold, sigmaObs, initialLogitPg, thetaPrecision, dInvPrec, sigmaAim]
    # Recompute mOuts
    stepperSingle = BayesianStepper(fittingVariance, threshold, sigmaObs, initialLogitPg, targets=targets, thetaPrecision=thetaPrecision, dInvPrec=dInvPrec, sigmaAim=sigmaAim)
    mOutsSingle = np.zeros(len(trials))
    for trial in trials:
        mOutsSingle[trial] = stepperSingle.expectedMove(trial)
        stepperSingle.updatePosteriors(trial, compMag)
    # Compute rmse
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    totErr = validAims - validMOuts
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamples)
    print(logLikelihood, paramCount * np.log(numSamples) - 2 * logLikelihood,xs)
    print(allAims, mOutsSingle)
    return {
        'xs': xs,
        'mStates': mOutsSingle.tolist(),
        'rmse': rmseVal,
        'negLl': -logLikelihood,
        'bic': paramCount * np.log(numSamples) - 2 * logLikelihood,
        'allAims': allAims.tolist()
    }
class fitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, isRmse=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.isRmse = isRmse
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        uniqP = self.dat['participantNum'].unique()
        numPpTotal = len(uniqP)
        if numPpTotal == 0:
            return
        self.bics = np.zeros(numPpTotal)
        self.rmses = np.zeros(numPpTotal)
        self.negLl = np.zeros(numPpTotal)
        self.mStates = [[] for _ in range(numPpTotal)]
        self.allAims = [[] for _ in range(numPpTotal)]
        self.xs = [[] for _ in range(numPpTotal)]
        # Assume all participants have the same number of trials and targets
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp) & (self.dat['phase'] == self.fitPhase)]
        numTrials = len(pDatFirst)
        targets = pDatFirst['targetPosition'].values
        trials = np.arange(numTrials)
        # Prepare data for each pp
        dataList = []
        for pp in uniqP:
            pDat = self.dat[(self.dat['participantNum'] == pp) & (self.dat['phase'] == self.fitPhase)]
            allAims = pDat['aim'].values
            mask = ~np.isnan(allAims)
            dataList.append((allAims, mask, trials, targets, self.isRmse, self.heightCap, self.conVal))
        results = [fitSingle(i) for i in dataList]
        for i, result in enumerate(results):
            self.xs[i] = result['xs']
            self.mStates[i] = result['mStates']
            self.rmses[i] = result['rmse']
            self.negLl[i] = result['negLl']
            self.bics[i] = result['bic']
            self.allAims[i] = result['allAims']

"""
            
"""

import numpy as np
from scipy.optimize import differential_evolution as evolution
from scipy.stats import norm
from scipy.special import expit
import multiprocessing
from functools import partial
import cma

class BayesianStepper:
    def __init__(self, fittingVariance=None, threshold=1.0, sigmaObs=1.0, tauG=100.0, tauL=100.0, initialLogitPg=0.0, targets=None):
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.threshold = threshold
        self.sigmaObs = sigmaObs
        self.tauG = tauG
        self.tauL = tauL
        self.sigmaH0 = 10.0
        self.cumSurprise = 0.0
        self.perturbed = False
        self.pG = expit(initialLogitPg)
        self.thetaMean = 0
        self.thetaPrecision = 1 / (self.tauG ** 2)
        self.dInvCov = np.eye(2) / (self.tauL ** 2)
        self.dB = np.zeros(2)
        self.dMean = np.zeros(2)
        self.targets = targets

    def getFeatures(self, tRad):
        sinT = np.sin(tRad)
        cosT = np.cos(tRad)
        uT = np.array([-sinT, cosT])  # Tangential unit vector
        return uT

    def getPredictive(self, trialNum):
        if not self.perturbed or trialNum < 0 or trialNum >= len(self.targets):
            return {'pG': self.pG, 'deltaG': 0.0, 'varG': 1e10, 'muL': 0.0, 'varL': 1e10}
        tRad = np.deg2rad(self.targets[trialNum])
        uT = self.getFeatures(tRad)
        deltaG = self.thetaMean
        varG = self.sigmaObs ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
        muL = np.dot(uT, self.dMean)
        dCov = np.linalg.pinv(self.dInvCov) if np.linalg.det(self.dInvCov) > 1e-10 else np.eye(2) * 1e10
        varL = np.dot(uT, dCov.dot(uT))
        return {'pG': self.pG, 'deltaG': deltaG, 'varG': varG, 'muL': muL, 'varL': varL}

    def updatePosteriors(self, trialNum, deltaObs):
        if self.targets is None or trialNum >= len(self.targets):
            return
        tRad = np.deg2rad(self.targets[trialNum])
        uT = self.getFeatures(tRad)
        # Surprise detection (change-point evidence)
        if not self.perturbed:
            surprise = (deltaObs ** 2) / (2 * self.sigmaH0 ** 2)
            self.cumSurprise += surprise
            if self.cumSurprise > self.threshold:
                self.perturbed = True
        if self.perturbed:
            # Global update
            predMeanG = self.thetaMean
            predVarG = self.sigmaObs ** 2 + (1 / self.thetaPrecision if self.thetaPrecision > 1e-10 else 1e10)
            margLikG = norm.pdf(deltaObs, predMeanG, np.sqrt(predVarG))
            addPrecision = 1 / self.sigmaObs ** 2
            oldSumWeighted = self.thetaMean * self.thetaPrecision
            self.thetaPrecision += addPrecision
            self.thetaMean = (oldSumWeighted + deltaObs * addPrecision) / self.thetaPrecision
            # Local update
            predMeanL = np.dot(uT, self.dMean)
            dCov = np.linalg.pinv(self.dInvCov) if np.linalg.det(self.dInvCov) > 1e-10 else np.eye(2) * 1e10
            predVarL = self.sigmaObs ** 2 + np.dot(uT, dCov.dot(uT))
            margLikL = norm.pdf(deltaObs, predMeanL, np.sqrt(predVarL))
            # Bayes factor for pG update
            if margLikL > 0:
                bayesFactor = margLikG / margLikL
            else:
                bayesFactor = 1e10
            self.pG = self.pG * bayesFactor / (self.pG * bayesFactor + (1 - self.pG)) if self.pG < 1 else self.pG
            addInv = np.outer(uT, uT) / self.sigmaObs ** 2
            self.dInvCov += addInv
            if np.linalg.det(self.dInvCov) < 1e-10:
                self.dInvCov += np.eye(2) * 1e-10
            self.dB += uT * deltaObs / self.sigmaObs ** 2
            self.dMean = np.linalg.pinv(self.dInvCov).dot(self.dB)

    def expectedMove(self, trialNum):
        predDict = self.getPredictive(trialNum)
        delta = predDict['pG'] * predDict['deltaG'] + (1 - predDict['pG']) * predDict['muL']
        return -delta

class fitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, rmse=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.dat = df
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.mStates = []
        self.allAims = []
        self.bics = []
        self.rmses = []
        self.negLl = []
        self.xs = []

    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        uniqP = self.dat['participantNum'].unique()
        self.bics = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLl = np.ones(len(uniqP)) * 100000
        self.mStates = [[]] * len(uniqP)
        self.allAims = [[]] * len(uniqP)
        self.xs = [[]] * len(uniqP)
        numProcesses = 6
        with multiprocessing.Pool(processes=numProcesses) as pool:
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, heightCap=self.heightCap, fitPhase=self.fitPhase)
            results = pool.starmap(worker, [(pp, i) for i, pp in enumerate(uniqP)])
        for res in results:
            if res is not None:
                it = res['it']
                self.bics[it] = res['bic']
                self.negLl[it] = res['negLl']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']

    def genDat(self, params, rots, trials=np.arange(-5, 35, 1)):
        if len(params) != 7:
            raise ValueError("Params: fittingVariance, threshold, compMagnitude, sigmaObs, tauG, tauL, initialLogitPg")
        fv, thresh, compMag, sigmaObs, tauG, tauL, initialLogitPg = params
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(fv, thresh, sigmaObs, tauG, tauL, initialLogitPg, targets=rots)
        states = []
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        for trialNum in trialsMod:
            mOut = stepper.expectedMove(trialNum)
            states.append(mOut)
            if trialNum >= 0:
                stepper.updatePosteriors(trialNum, compMag)
        return np.array(states)

def fitSinglePp(pp, it, dat, rmse, heightCap, fitPhase):
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    def localFitPp(params):
        if rmse:
            compMag = params[0]
            fittingVariance = None
            threshold = 1.0
            sigmaObs = 1.0
            tauG = 100.0
            tauL = 100.0
            initialLogitPg = 0.0
        else:
            logFv, logThresh, compFrac, logSigma, logTauG, logTauL, initialLogitPg = params
            fittingVariance = np.exp(logFv)
            threshold = np.exp(logThresh)
            compMag = compFrac * 180
            sigmaObs = np.exp(logSigma)
            tauG = np.exp(logTauG)
            tauL = np.exp(logTauL)
        stepper = BayesianStepper(fittingVariance, threshold, sigmaObs, tauG, tauL, initialLogitPg, targets=targets)
        mOuts = np.zeros_like(aims, dtype=float)
        predDicts = []
        for trial in trials:
            predDict = stepper.getPredictive(trial)
            predDicts.append(predDict)
            delta = predDict['pG'] * predDict['deltaG'] + (1 - predDict['pG']) * predDict['muL']
            mOuts[trial] = -delta
            stepper.updatePosteriors(trial, compMag)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validPredDicts = [predDicts[j] for j in np.where(mask)[0]]
        numSamp = len(validAims)
        if numSamp == 0:
            return np.inf
        if rmse:
            totErr = validAims - mOuts[mask]
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            mu, std = norm.fit(totErr)
            logLikelihood = np.sum(np.log(norm.pdf(totErr, mu, std) + 1e-12))
        else:
            logLikelihood = 0.0
            for validAim, predDict in zip(validAims, validPredDicts):
                pdfG = norm.pdf(validAim, -predDict['deltaG'], np.sqrt(predDict['varG'] + (fittingVariance or 0)))
                pdfL = norm.pdf(validAim, -predDict['muL'], np.sqrt(predDict['varL'] + (fittingVariance or 0)))
                pdf = predDict['pG'] * pdfG + (1 - predDict['pG']) * pdfL + 1e-12
                logLikelihood += np.log(pdf)
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        return 1e9
    if rmse:
        bounds = [(-heightCap, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bounds = [(np.log(1e-9), np.log(1e5)), (np.log(1e-3), np.log(1e6)), (-1, 1), (np.log(0.1), np.log(100)),
                  (np.log(1), np.log(1e4)), (np.log(1), np.log(1e4)), (-10, 10)]
        def getX0(bnds):
            return np.array([np.random.uniform(low, high) for low, high in bnds])
        bestX, bestFun = cma.fmin2(localFitPp, x0=getX0(bounds), sigma0=.5, restarts=3, bipop=False,
                                     options={'tolfun': 1e-8, 'tolfacupx': 1e6, 'popsize': 20, 'verbose': 0,
                                              'bounds': [[i[0] for i in bounds], [i[1] for i in bounds]]})
    if rmse:
        compMag = bestX[0]
        fittingVariance = None
        threshold = 1.0
        sigmaObs = 1.0
        tauG = 100.0
        tauL = 100.0
        initialLogitPg = 0.0
        k = 1
    else:
        logFv, logThresh, compFrac, logSigma, logTauG, logTauL, initialLogitPg = bestX
        fittingVariance = np.exp(logFv)
        threshold = np.exp(logThresh)
        compMag = compFrac * 180
        sigmaObs = np.exp(logSigma)
        tauG = np.exp(logTauG)
        tauL = np.exp(logTauL)
        k = len(bestX)
    stepper = BayesianStepper(fittingVariance, threshold, sigmaObs, tauG, tauL, initialLogitPg, targets=targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
        stepper.updatePosteriors(trial, compMag)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    sumSquares = np.sum(totErr ** 2)
    rmseVal = np.sqrt(sumSquares / numSamp)
    if rmse:
        mu, std = norm.fit(totErr)
        logLikelihood = np.sum(np.log(norm.pdf(totErr, mu, std) + 1e-12))
    else:
        logLikelihood = -bestFun
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLl = -logLikelihood
    return {
        'rmse': rmseVal,
        'it': it,
        'bic': bic,
        'negLl': negLl,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, threshold, compMag, sigmaObs, tauG, tauL, initialLogitPg]
    }
"""

                 


"""


import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma
class BayesianStepper:
    def __init__(self, stepHeight, fittingVariance=None, threshold=1.0, compMagnitude=None, flip_bias=0.0, targets=None, gamma=1.0, kappa_0=10.0, rate=0.0):
        self.stepHeight = stepHeight
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.targets = targets
        self.stateProb = np.array([1.0, 0.0])
        self.rHat = compMagnitude if compMagnitude is not None else stepHeight
        self.sigmaPert2 = 1.0 ** 2 # Fixed
        self.sigmaH02 = 10.0 # Fixed
        self.currentTrial = -1
        self.flip_bias = flip_bias
        self.threshold = threshold
        self.cum_surprise = 0.0
        self.perturbed = False
        self.trials_since_perturbed = 0
        self.gamma = gamma
        self.kappa_0 = kappa_0
        self.rate = rate
        self.target_memories = {}
  
    def get_features(self, tRad):
        cos_t = np.cos(tRad)
        sin_t = np.sin(tRad)
        sum_cos_weighted = 0.0
        sum_sin_weighted = 0.0
        sum_weight = 0.0
        for past_target, mem in self.target_memories.items():
            past_rad = np.deg2rad(past_target)
            delta = tRad - past_rad
            delta = (delta + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
            weight = ((1 + np.cos(delta)) / 2) ** self.kappa_0

            sum_cos_weighted += weight * mem[0]
            sum_sin_weighted += weight * mem[1]
            sum_weight += weight
        if sum_weight > 1e-6:
            mean_cos = sum_cos_weighted / sum_weight
            mean_sin = sum_sin_weighted / sum_weight
            #R_mean = np.sqrt(mean_cos**2 + mean_sin**2)
            #if R_mean > 1e-6:
            # mean_cos /= R_mean
            # mean_sin /= R_mean
        else:
            mean_cos = 0.0
            mean_sin = 0.0
        cos_delta = cos_t * mean_cos + sin_t * mean_sin
        sin_delta = sin_t * mean_cos - cos_t * mean_sin
        sim = (1 + cos_delta) / 2
        return mean_cos, mean_sin, cos_delta, sin_delta, sim
  
    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        if not self.perturbed:
            surprise = (pT ** 2) / (2 * self.sigmaH02)
            self.cum_surprise += surprise
            if self.cum_surprise > self.threshold:
                self.perturbed = True
        if self.perturbed:
            t = self.targets[trialNum]
            tRad = np.deg2rad(t)
            for past_target, mem in self.target_memories.items():
                
                #past_rad = np.deg2rad(past_target)
                #delta = tRad - past_rad
                #cos_delta_val = np.cos(delta)
                #sim = np.exp(self.kappa_0 * (cos_delta_val - 1))
                
                sim = 0
                retention = np.exp(-self.rate * (1 - sim))
                mem[0] *= retention
                mem[1] *= retention
            mean_cos, mean_sin, _, _, _ = self.get_features(tRad)
            observed_cos = np.cos(np.deg2rad(pT))
            observed_sin = np.sin(np.deg2rad(pT))
            if t not in self.target_memories:
                self.target_memories[t] = np.zeros(2)
            self.target_memories[t][0] = observed_cos #+= observed_cos - mean_cos
            self.target_memories[t][1] = observed_sin #+= observed_sin - mean_sin
            self.trials_since_perturbed += 1 # Increment counter
        self.stateProb = np.array([0.0, 1.0]) if self.perturbed else np.array([1.0, 0.0])
    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance # Simplified; add stochasticity if needed
    def expectedMove(self, trialNum):
        if trialNum < 0:
            return 0
        pT = self.stepHeight # Assume constant perturbation observation
        if trialNum >= len(self.targets):
            return 0
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        p_pert = self.stateProb[1]
        if p_pert == 0:
            expected = 0.0
        else:
            _, _, cos_delta, sin_delta, sim = self.get_features(tRad)
            kappa = self.kappa_0 # Finite and fittable
            # Bayesian logit with Beta-updated prior bias, scaled by gamma
            logit = self.flip_bias - self.gamma * np.log(self.trials_since_perturbed + 1) - 2 * kappa * sim
            prob_flip = expit(logit)
            expected_sign = 1 - 2 * prob_flip
            expectedUnderPert = self.rHat * expected_sign
            expected = p_pert * expectedUnderPert
        self.updatePosteriors(trialNum, pT)
        return -expected
     
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
        # Use multiprocessing Pool to parallelize
        numProcesses = multiprocessing.cpu_count()/2
        with multiprocessing.Pool(processes=numProcesses) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
      
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 6:
            ev, thresh, compMag, flip_bias, gamma, kappa_0 = params
            sh = 30 # Default or from context
            rate = 0.0
        elif len(params) == 7:
            ev, thresh, compMag, flip_bias, gamma, kappa_0, rate = params
            sh = 30 # Default or from context
        else:
            raise ValueError("Params should include fittingVariance, threshold, compMagnitude, flip_bias, gamma, kappa_0, [rate]")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, thresh, compMag, flip_bias, rots, gamma=gamma, kappa_0=kappa_0, rate=rate)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight = params[0] # Adjust as needed
            fittingVariance = None
            threshold = 1.0
            compMagnitude = params[0]
            flip_bias = 0.0
            gamma = 1.0
            kappa_0 = 10.0
            rate = 0.0
        else:
            log_fittingVariance, log_threshold, log_compMagnitude, scaled_flip_bias, log_gamma, log_kappa_0, log_rate = params
            fittingVariance = np.exp(log_fittingVariance)
            threshold = np.exp(log_threshold)
            compMagnitude = (log_compMagnitude) * 180
            flip_bias = scaled_flip_bias * 100
            gamma = np.exp(log_gamma)
            kappa_0 = np.exp(log_kappa_0)
            rate = np.exp(log_rate)
        stepper = BayesianStepper(rot, fittingVariance, threshold, compMagnitude, flip_bias, targets, gamma=gamma, kappa_0=kappa_0, rate=rate)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        else:
            return 1e9
    evolve = True
    if rmse:
        bounds = [(0, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(np.log(1e-9), np.log(1e5)), (np.log(1e-3), np.log(1e6)), (-1,10/180), (-2, 2), (np.log(0.001), np.log(1e3)), (np.log(1e-22), np.log(1e3)), (np.log(1e-10), np.log(1e3))]
        def getX0(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        for i in range(1):
            bestX,es = cma.fmin2(localFitPp,x0=getX0(bounds),sigma0=1e3,incpopsize=1.5,restarts=9,bipop=False,options={'tolfun':1e-11,'tolfacupx':1e6, 'popsize':3,'verbose': 0,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            #res = evolution(localFitPp, x0=getX0(bounds),bounds=bounds,popsize=90, workers=1)
            if False: # res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight = bestX[0]
        fittingVariance = None
        threshold = 1.0
        compMagnitude = bestX[0]
        flip_bias = 0.0
        gamma = 1.0
        kappa_0 = 10.0
        rate = 0.0
        k = 1
    else:
        log_fittingVariance, log_threshold, log_compMagnitude, scaled_flip_bias, log_gamma, log_kappa_0, log_rate = bestX
        fittingVariance = np.exp(log_fittingVariance)
        threshold = np.exp(log_threshold)
        compMagnitude = (log_compMagnitude) * 180
        flip_bias = scaled_flip_bias * 100
        gamma = np.exp(log_gamma)
        kappa_0 = np.exp(log_kappa_0)
        rate = np.exp(log_rate)
        k = len(bestX)
    stepper = BayesianStepper(rot, fittingVariance, threshold, compMagnitude, flip_bias, targets, gamma=gamma, kappa_0=kappa_0, rate=rate)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, threshold, compMagnitude, flip_bias, gamma, kappa_0, rate]
    }

"""

"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma
class BayesianStepper:
    def __init__(self, stepHeight, fittingVariance=None, threshold=1.0, compMagnitude=None, flip_bias=0.0, targets=None, gamma=1.0, kappa_0=10.0):
        self.stepHeight = stepHeight
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.targets = targets
        self.stateProb = np.array([1.0, 0.0])
        self.rHat = compMagnitude if compMagnitude is not None else stepHeight
        self.sigmaPert2 = 1.0 ** 2 # Fixed
        self.sigmaH02 = 10.0 # Fixed
        self.currentTrial = -1
        self.flip_bias = flip_bias
        self.threshold = threshold
        self.cum_surprise = 0.0
        self.perturbed = False
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.trials_since_perturbed = 0
        self.gamma = gamma
        self.kappa_0 = kappa_0
    
    def get_features(self, tRad):
        cos_t = np.cos(tRad)
        sin_t = np.sin(tRad)
        R = np.sqrt(self.sum_cos**2 + self.sum_sin**2)
        if R > 1e-6:
            mean_cos_norm = self.sum_cos / R
            mean_sin_norm = self.sum_sin / R
        else:
            mean_cos_norm = 0.0
            mean_sin_norm = 0.0
            R = 0.0
        cos_delta = cos_t * mean_cos_norm + sin_t * mean_sin_norm
        sin_delta = sin_t * mean_cos_norm - cos_t * mean_sin_norm
        x = np.array([R * cos_delta, R * sin_delta])
        return x, R, cos_delta, sin_delta
    
    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        if not self.perturbed:
            surprise = (pT ** 2) / (2 * self.sigmaH02)
            self.cum_surprise += surprise
            if self.cum_surprise > self.threshold:
                self.perturbed = True
        if self.perturbed:
            tRad = np.deg2rad(self.targets[trialNum])
            # Modified for only previous trial: reset to current (which becomes previous for next trial)
            self.sum_cos = np.cos(tRad)
            self.sum_sin = np.sin(tRad)
            self.trials_since_perturbed += 1  # Increment counter
        self.stateProb = np.array([0.0, 1.0]) if self.perturbed else np.array([1.0, 0.0])
  
    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance # Simplified; add stochasticity if needed
    def expectedMove(self, trialNum):
        if trialNum < 0:
            return 0
        pT = self.stepHeight # Assume constant perturbation observation
        if trialNum >= len(self.targets):
            return 0
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        prevTarget = self.targets[trialNum-1]
        prevTRad = np.deg2rad(prevTarget)
        p_pert = self.stateProb[1]
        if p_pert == 0:
            expected = 0.0
        else:
            x, R, cos_delta, sin_delta = self.get_features(tRad)
            kappa = self.kappa_0  # Finite and fittable; since single previous, R=1 always
            # Bayesian logit with Beta-updated prior bias, scaled by gamma
            logit = self.flip_bias - self.gamma * np.log(self.trials_since_perturbed + 1) - 2 * kappa * cos_delta
            prob_flip = expit(logit)
            expected_sign = 1 - 2 * prob_flip
            expectedUnderPert = self.rHat * expected_sign
            expected = p_pert * expectedUnderPert
        self.updatePosteriors(trialNum, pT)
        return -expected
       
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
 
        # Use multiprocessing Pool to parallelize
        numProcesses = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=numProcesses) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
 
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
        
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 6:
            ev, thresh, compMag, flip_bias, gamma, kappa_0 = params
            sh = 30 # Default or from context
        else:
            raise ValueError("Params should include fittingVariance, threshold, compMagnitude, flip_bias, gamma, kappa_0")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, thresh, compMag, flip_bias, rots, gamma=gamma, kappa_0=kappa_0)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight = params[0] # Adjust as needed
            fittingVariance = None
            threshold = 1.0
            compMagnitude = params[0]
            flip_bias = 0.0
            gamma = 1.0
            kappa_0 = 10.0
        else:
            log_fittingVariance, log_threshold, log_compMagnitude, scaled_flip_bias, log_gamma, log_kappa_0 = params
            fittingVariance = np.exp(log_fittingVariance)
            threshold = np.exp(log_threshold)
            compMagnitude = (log_compMagnitude) * 180
            flip_bias = scaled_flip_bias * 100
            gamma = np.exp(log_gamma)
            kappa_0 = np.exp(log_kappa_0)
        stepper = BayesianStepper(rot, fittingVariance, threshold, compMagnitude, flip_bias, targets, gamma=gamma, kappa_0=kappa_0)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
  
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        else:
            return 1e9
    evolve = True
    if rmse:
        bounds = [(0, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(np.log(1e-9), np.log(1e5)), (np.log(1e-3), np.log(1e6)), (-1,1), (-2, 2), (np.log(0.001), np.log(1e3)), (np.log(1e-22), np.log(1e3))]
        def getX0(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        for i in range(1):
            bestX,es = cma.fmin2(localFitPp,x0=getX0(bounds),sigma0=1e3,restarts=3,bipop=True,options={'tolfun':1e-11,'tolfacupx':1e6, 'popsize':20,'verbose': 0,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            #res = evolution(localFitPp, x0=getX0(bounds),bounds=bounds,popsize=90, workers=1)
            if False: # res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight = bestX[0]
        fittingVariance = None
        threshold = 1.0
        compMagnitude = bestX[0]
        flip_bias = 0.0
        gamma = 1.0
        kappa_0 = 10.0
        k = 1
    else:
        log_fittingVariance, log_threshold, log_compMagnitude, scaled_flip_bias, log_gamma, log_kappa_0 = bestX
        fittingVariance = np.exp(log_fittingVariance)
        threshold = np.exp(log_threshold)
        compMagnitude = (log_compMagnitude) * 180
        flip_bias = scaled_flip_bias * 100
        gamma = np.exp(log_gamma)
        kappa_0 = np.exp(log_kappa_0)
        k = len(bestX)
    stepper = BayesianStepper(rot, fittingVariance, threshold, compMagnitude, flip_bias, targets, gamma=gamma, kappa_0=kappa_0)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, threshold, compMagnitude, flip_bias, gamma, kappa_0]
    }
"""

"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma
class BayesianStepper:
    def **init**(self, stepHeight, fittingVariance=None, threshold=1.0, compMagnitude=None, flip_bias=0.0, targets=None):
        self.stepHeight = stepHeight
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.targets = targets
        self.stateProb = np.array([1.0, 0.0])
        self.rHat = compMagnitude if compMagnitude is not None else stepHeight
        self.sigmaPert2 = 1.0 ** 2 # Fixed
        self.sigmaH02 = 10.0 # Fixed
        self.currentTrial = -1
        self.flip_bias = flip_bias
        self.threshold = threshold
        self.cum_surprise = 0.0
        self.perturbed = False
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.n_hist = 0
   
    def estimate_kappa(self, r):
        if r == 0:
            return 0
        if r < 0.53:
            r2 = r * r
            r3 = r2 * r
            r5 = r2 * r3
            return 2 * r + r3 + 5 * r5 / 6
        elif r < 0.85:
            return -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            return 1 / (2 * (1 - r))
   
    def get_features(self, tRad):
        cos_t = np.cos(tRad)
        sin_t = np.sin(tRad)
        if self.n_hist > 0:
            mean_cos = self.sum_cos / self.n_hist
            mean_sin = self.sum_sin / self.n_hist
            r = np.sqrt(mean_cos**2 + mean_sin**2)
            if r > 1e-6:
                mean_cos_norm = mean_cos / r
                mean_sin_norm = mean_sin / r
            else:
                mean_cos_norm = 0.0
                mean_sin_norm = 0.0
                r = 0.0
        else:
            r = 0.0
            mean_cos_norm = 0.0
            mean_sin_norm = 0.0
        cos_delta = cos_t * mean_cos_norm + sin_t * mean_sin_norm
        print(cos_delta)
        sin_delta = sin_t * mean_cos_norm - cos_t * mean_sin_norm
        x = np.array([r * cos_delta, r * sin_delta])
        return x, r, cos_delta, sin_delta
   
    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        if not self.perturbed:
            surprise = (pT ** 2) / (2 * self.sigmaH02)
            self.cum_surprise += surprise
            if self.cum_surprise > self.threshold:
                self.perturbed = True
        if self.perturbed:
            tRad = np.deg2rad(self.targets[trialNum])
            self.sum_cos += np.cos(tRad) #+= if all hstory
            self.sum_sin += np.sin(tRad)
            self.n_hist += 1
        self.stateProb = np.array([0.0, 1.0]) if self.perturbed else np.array([1.0, 0.0])
   
    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance # Simplified; add stochasticity if needed
    def expectedMove(self, trialNum):
        if trialNum < 0:
            return 0
        pT = self.stepHeight # Assume constant perturbation observation
        if trialNum >= len(self.targets):
            return 0
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        p_pert = self.stateProb[1]
        if p_pert == 0:
            expected = 0.0
        else:
            x, r, cos_delta, sin_delta = self.get_features(tRad)
            kappa = self.estimate_kappa(r)
            logit = self.flip_bias - 2 * kappa * cos_delta
            prob_flip = expit(logit)
            expected_sign = 1 - 2 * prob_flip
            expectedUnderPert = self.rHat * expected_sign
            expected = p_pert * expectedUnderPert
        self.updatePosteriors(trialNum, pT)
        return -expected
       
class fitShell:
    def **init**(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
 
        # Use multiprocessing Pool to parallelize
   
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
 
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
        
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 4:
            ev, thresh, compMag, flip_bias = params
            sh = 30 # Default or from context
        else:
            raise ValueError("Params should include fittingVariance, threshold, compMagnitude, flip_bias")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '**len**'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, thresh, compMag, flip_bias, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight = params[0] # Adjust as needed
            fittingVariance = None
            threshold = 1.0
            compMagnitude = params[0]
            flip_bias = 0.0
        else:
            log_fittingVariance, log_threshold, log_compMagnitude, scaled_flip_bias = params
            fittingVariance = np.exp(log_fittingVariance)
            threshold = np.exp(log_threshold)
            compMagnitude = (log_compMagnitude) * 180
            flip_bias = scaled_flip_bias * 100
        stepper = BayesianStepper(rot, fittingVariance, threshold, compMagnitude, flip_bias, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
  
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        else:
            return 1e9
    evolve = True
    if rmse:
        bounds = [(0, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(np.log(1), np.log(1e5)), (np.log(1), np.log(1e5)), (-1, .1), (-2, 2)]
        def getX0(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        for i in range(1):
            bestX,es = cma.fmin2(localFitPp,x0=getX0(bounds),sigma0=1,restarts=6,bipop=True,options={'tolfun':1e-9,'tolfacupx':1e6,'verbose': 0,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            #res = evolution(localFitPp, x0=getX0(bounds),bounds=bounds,popsize=90, workers=1)
            if False: # res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight = bestX[0]
        fittingVariance = None
        threshold = 1.0
        compMagnitude = bestX[0]
        flip_bias = 0.0
        k = 1
    else:
        log_fittingVariance, log_threshold, log_compMagnitude, scaled_flip_bias = bestX
        fittingVariance = np.exp(log_fittingVariance)
        threshold = np.exp(log_threshold)
        compMagnitude = (log_compMagnitude) * 180
        flip_bias = scaled_flip_bias * 100
        k = len(bestX)
    stepper = BayesianStepper(rot, fittingVariance, threshold, compMagnitude, flip_bias, targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, threshold, compMagnitude, flip_bias]
    }
"""

"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma
class BayesianStepper:
    def __init__(self, stepHeight, fittingVariance=None, threshold=1.0, sinDirection=1.0, cosDirection=1.0, flip_bias=0.0, targets=None):
        self.stepHeight = stepHeight
        self.fittingVariance = np.abs(fittingVariance) if fittingVariance is not None else None
        self.targets = targets
        self.stateProb = np.array([1.0, 0.0])
        self.rHat = self.stepHeight
        self.sigmaPert2 = 1.0 ** 2 # Fixed
        self.sigmaH02 = 10.0 # Fixed, removed as parameter
        self.currentTrial = -1
        self.sinDirection = sinDirection
        self.cosDirection = cosDirection
        self.flip_bias = flip_bias
        self.threshold = threshold
        self.cum_surprise = 0.0
        self.perturbed = False
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.n_hist = 0
     
    def get_features(self, tRad):
        cos_t = np.cos(tRad)
        sin_t = np.sin(tRad)
        if self.n_hist > 0:
            mean_cos = self.sum_cos / self.n_hist
            mean_sin = self.sum_sin / self.n_hist
            r = np.sqrt(mean_cos**2 + mean_sin**2)
            if r > 1e-6:
                mean_cos_norm = mean_cos / r
                mean_sin_norm = mean_sin / r
            else:
                mean_cos_norm = 0.0
                mean_sin_norm = 0.0
                r = 0.0
        else:
            r = 0.0
            mean_cos_norm = 0.0
            mean_sin_norm = 0.0
        cos_delta = cos_t * mean_cos_norm + sin_t * mean_sin_norm
        sin_delta = sin_t * mean_cos_norm - cos_t * mean_sin_norm
        x = np.array([r * cos_delta, r * sin_delta])
        return x, r, cos_delta, sin_delta
     
    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        if not self.perturbed:
            surprise = (pT ** 2) / (2 * self.sigmaH02)
            self.cum_surprise += surprise
            if self.cum_surprise > self.threshold:
                self.perturbed = True
        if self.perturbed:
            tRad = np.deg2rad(self.targets[trialNum])
            self.sum_cos += np.cos(tRad)
            self.sum_sin += np.sin(tRad)
            self.n_hist += 1
        self.stateProb = np.array([0.0, 1.0]) if self.perturbed else np.array([1.0, 0.0])
     
    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance # Simplified; add stochasticity if needed
 
    def expectedMove(self, trialNum):
        if trialNum < 0:
            return 0
        pT = self.stepHeight # Assume constant perturbation observation
        if trialNum >= len(self.targets):
            return 0
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        p_pert = self.stateProb[1]
        if p_pert == 0:
            expected = 0.0
        else:
            x, r, cos_delta, sin_delta = self.get_features(tRad)
            logit = - self.cosDirection * cos_delta * r - self.sinDirection * sin_delta * r + self.flip_bias
            prob_flip = expit(logit)
            expected_sign = 1 - 2 * prob_flip
            expectedUnderPert = self.rHat * expected_sign
            expected = p_pert * expectedUnderPert
        self.updatePosteriors(trialNum, pT)
        return -expected
 
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
 
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
   
        # Use multiprocessing Pool to parallelize
     
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
   
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
          
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 5:
            ev, thresh, sinDirection, cosDirection, flip_bias = params
            sh = 30 # Default or from context
        else:
            raise ValueError("Params should include fittingVariance, threshold, sinDirection, cosDirection, flip_bias")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, thresh, sinDirection, cosDirection, flip_bias, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
 
# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight = params[0] # Adjust as needed
            fittingVariance = None
            threshold = 1.0
            sinDirection = 1.0
            cosDirection = 1.0
            flip_bias = 0.0
        else:
            log_fittingVariance, log_threshold, scaled_sinDirection, scaled_cosDirection, scaled_flip_bias = params
            fittingVariance = np.exp(log_fittingVariance)
            threshold = np.exp(log_threshold)
            sinDirection = scaled_sinDirection * 100
            cosDirection = scaled_cosDirection * 100
            flip_bias = scaled_flip_bias * 100
        stepper = BayesianStepper(rot, fittingVariance, threshold, sinDirection, cosDirection, flip_bias, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
    
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        else:
            return 1e9
    evolve = True
    if rmse:
        bounds = [(0, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(np.log(2), np.log(1e5)), (np.log(2), np.log(1e5)), (-1, 1), (-1, 1), (-1, 1)]
        def getX0(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        for i in range(1):
            bestX,es = cma.fmin2(localFitPp,x0=getX0(bounds),sigma0=.1,restarts=9,bipop=True,options={'tolfun':1e-9,'tolfacupx':1e6,'maxfeval':1e9,'verbose': 0,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            if False: # res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight = bestX[0]
        fittingVariance = None
        threshold = 1.0
        sinDirection = 1.0
        cosDirection = 1.0
        flip_bias = 0.0
        k = 1
    else:
        log_fittingVariance, log_threshold, scaled_sinDirection, scaled_cosDirection, scaled_flip_bias = bestX
        fittingVariance = np.exp(log_fittingVariance)
        threshold = np.exp(log_threshold)
        sinDirection = scaled_sinDirection * 100
        cosDirection = scaled_cosDirection * 100
        flip_bias = scaled_flip_bias * 100
        k = len(bestX)
    stepper = BayesianStepper(rot, fittingVariance, threshold, sinDirection, cosDirection, flip_bias, targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, threshold, sinDirection, cosDirection, flip_bias]
    }

"""


    
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma
class BayesianStepper:
    def __init__(self, stepHeight, fittingVariance=None, priorLogodds0=0.0, tau=1.0, priorLogoddsRt=0.0, executionVariance=10, sinDirection=1.0, cosDirection=1.0, processNoiseVar=0.0, decay=1.0, targets=None):
        self.stepHeight = stepHeight
        self.fittingVariance = np.abs(fittingVariance)
        self.targets = targets
        self.pChange = expit(-priorLogodds0)
        self.stateProb = np.array([1.0, 0.0])
        # Log posteriors for H_rot, H_trans
        self.logPost = np.array([0.0, -priorLogoddsRt], dtype=float)
        self.logPost -= scipy.special.logsumexp(self.logPost)
        # H_rot state
        self.rHat = 0.0
        self.rVar = tau ** 2 if tau > 0 else 1e-6
        # H_trans state
        self.wHat = np.zeros(2)
        self.wCov = (tau ** 2 if tau > 0 else 1e-6) * np.eye(2)
        self.sigmaPert2 = 1.0 ** 2 # Fixed
        self.sigmaH02 = executionVariance # Fixed
        self.postTargets = []
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.n_hist = 0
        self.currentTrial = -1
        self.sinDirection = sinDirection
        self.cosDirection = cosDirection
        self.processNoiseVar = processNoiseVar
        self.decay = decay
      
    def get_features(self, tRad):
        cos_t = np.cos(tRad)
        sin_t = np.sin(tRad)
        if self.n_hist > 0:
            mean_cos = self.sum_cos / self.n_hist
            mean_sin = self.sum_sin / self.n_hist
            r = np.sqrt(mean_cos**2 + mean_sin**2)
            if r > 1e-6:
                mean_cos_norm = mean_cos / r
                mean_sin_norm = mean_sin / r
            else:
                mean_cos_norm = 0.0
                mean_sin_norm = 0.0
                r = 0.0
        else:
            r = 0.0
            mean_cos_norm = 0.0
            mean_sin_norm = 0.0
        cos_delta = cos_t * mean_cos_norm + sin_t * mean_sin_norm
        sin_delta = sin_t * mean_cos_norm - cos_t * mean_sin_norm
        x = np.array([r * cos_delta, r * sin_delta])
        return x
      
    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        x = self.get_features(tRad)
        # Lik no_pert
        likNo = norm.pdf(pT, 0, np.sqrt(self.sigmaH02)) + 1e-12
        # Lik H_rot
        predMuRot = self.rHat
        predVarRot = self.rVar + self.sigmaPert2
        likRot = norm.pdf(pT, predMuRot, np.sqrt(predVarRot)) + 1e-12
        # Lik H_trans
        predMuTrans = np.dot(x, self.wHat)
        predVarTrans = np.dot(x, np.dot(self.wCov, x)) + self.sigmaPert2
        likTrans = norm.pdf(pT, predMuTrans, np.sqrt(predVarTrans)) + 1e-12
        # Compute sub priors
        subPost = np.exp(self.logPost) / np.sum(np.exp(self.logPost) + 1e-12)
        # Marginal lik for pert
        likPert = subPost[0] * likRot + subPost[1] * likTrans + 1e-12
        # HMM transition
        trans = np.array([[1 - self.pChange, self.pChange], [0, 1]])
        predictState = np.dot(trans.T, self.stateProb)
        likStates = np.array([likNo, likPert])
        updateState = predictState * likStates
        updateState /= np.sum(updateState) + 1e-12
        self.stateProb = updateState
        # Compute conditional post under pert
        condLiks = np.array([likRot, likTrans])
        conditional = subPost * condLiks
        conditionalPost = conditional / likPert if likPert > 0 else subPost
        # Responsibilities (overall posterior)
        p = self.stateProb[1]
        respRot = p * conditionalPost[0]
        respTrans = p * conditionalPost[1]
        # Update H_rot, weighted by responsibility
        if predVarRot > 0:
            k = self.rVar / predVarRot
            k *= respRot # Gate by responsibility
            self.rHat += (pT - predMuRot) *k
            self.rVar *= (1 - k)
        self.rVar += self.processNoiseVar
        self.rVar = max(self.rVar, 1e-6) # Prevent underflow/negatives
        # Update H_trans, weighted by responsibility
        if predVarTrans > 0:
            gain = np.dot(self.wCov, x) / predVarTrans
            gain *= respTrans # Gate by responsibility
            gain[0] *= self.cosDirection # Modify cos update direction
            gain[1] *= self.sinDirection # Modify sin update direction
            self.wHat += (pT - predMuTrans) * gain
            self.wCov -= np.outer(gain, np.dot(x, self.wCov))
        self.wCov += self.processNoiseVar * np.eye(2) # Add isotropic noise to covariance
        # Ensure PSD via eigenvalue clipping
        eigenvalues, eigenvectors = np.linalg.eigh(self.wCov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        self.wCov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Update sub post to marginal
        newSubPost = p * conditionalPost + (1 - p) * subPost
        self.logPost = np.log(newSubPost + 1e-12)
        self.logPost -= scipy.special.logsumexp(self.logPost)
        self.postTargets.append(target)
        self.sum_cos = self.decay * self.sum_cos + np.cos(tRad) * p
        self.sum_sin = self.decay * self.sum_sin + np.sin(tRad) * p
        self.n_hist = self.decay * self.n_hist + 1
      
    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance # Simplified; add stochasticity if needed
  
    def expectedMove(self, trialNum):
        #need to ad process noise if verying effective rot over time
        if trialNum < 0:
            return 0
        # Advance updates if necessary (assumes calls in sequential order)
        #while self.currentTrial < trialNum:
        #self.currentTrial += 1
        pT = self.stepHeight # Assume constant perturbation observation
        if trialNum >= len(self.targets):
            return 0
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        x = self.get_features(tRad)
        subPost = np.exp(self.logPost)
        subPost /= np.sum(subPost) + 1e-12
        predTrans = np.dot(x, self.wHat)
        expectedPertUnderPert = subPost[0] * self.rHat + subPost[1] * predTrans
        expectedPert = self.stateProb[1] * expectedPertUnderPert
        self.updatePosteriors(trialNum, pT)
        return -expectedPert
  
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
  
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
    
        # Use multiprocessing Pool to parallelize
      
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
    
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
           
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 9:
            ev, plo0, t, plrt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay = params
            sh = 30 # Default or from context
        else:
            raise ValueError("Params should include fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, plo0, t, plrt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
  
# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight = params[0] # Adjust as needed
            fittingVariance = None
            priorLogodds0 = 0.0
            tau = 1.0
            priorLogoddsRt = 0.0
            sinDirection = 1.0
            cosDirection = 1.0
            processNoiseVar = 0.0
            decay = 1.0
        else:
            fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay = params
        stepper = BayesianStepper(rot, fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
     
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            #sumSquares = np.sum(totErr ** 2)
            #rmseVal = np.sqrt(sumSquares / numSamp)
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        else:
            return 1e9
    evolve = True
    if rmse:
        bounds = [(0, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(1e-3, 1e5), (-8000, 10000), (0.001, 1000), (-1e4, 9000), (1e-3, 1e3), (-1, 1), (-1, 1), (0, 1e3), (0, 1)]
        def getX0(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        for i in range(1):
            #res = evolution(localFitPp, bounds=bounds, workers=1, popsize=30)
            bestX,es = cma.fmin2(localFitPp,x0=getX0(bounds),sigma0=1e4,restarts=6,bipop=True,options={'tolfun':1e-7,'tolfacupx':1e6,'maxfeval':60000,'verbose': 0,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            if False:# res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight = bestX[0]
        fittingVariance = None
        priorLogodds0 = 0.0
        tau = 1.0
        priorLogoddsRt = 0.0
        sinDirection = 1.0
        cosDirection = 1.0
        processNoiseVar = 0.0
        decay = 1.0
        executionVariance = 10
        k = 1
    else:
        fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay = bestX
        k = len(bestX)
    stepper = BayesianStepper(rot,fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay, targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, decay]
    }

"""
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma
class BayesianStepper:
    def __init__(self, stepHeight, fittingVariance=None, priorLogodds0=0.0, tau=1.0, priorLogoddsRt=0.0, executionVariance=10, sinDirection=1.0, cosDirection=1.0, processNoiseVar=0.0, targets=None):
        self.stepHeight = stepHeight
        self.fittingVariance = np.abs(fittingVariance)
        self.targets = targets
        self.pChange = expit(-priorLogodds0)
        self.stateProb = np.array([1.0, 0.0])
        # Log posteriors for H_rot, H_trans
        self.logPost = np.array([0.0, -priorLogoddsRt], dtype=float)
        self.logPost -= scipy.special.logsumexp(self.logPost)
        # H_rot state
        self.rHat = 0.0
        self.rVar = tau ** 2 if tau > 0 else 1e-6
        # H_trans state
        self.wHat = np.zeros(3)
        self.wCov = (tau ** 2 if tau > 0 else 1e-6) * np.eye(3)
        self.sigmaPert2 = 1.0 ** 2 # Fixed
        self.sigmaH02 = executionVariance # Fixed
        self.postTargets = []
        self.currentTrial = -1
        self.sinDirection = sinDirection
        self.cosDirection = cosDirection
        self.processNoiseVar = processNoiseVar
      
    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        prevTarget = self.targets[trialNum-1]
        tRadPrev = np.deg2rad(prevTarget)
        x = np.array([1,np.cos(tRad-tRadPrev), np.sin(tRad-tRadPrev)])
        # Lik no_pert
        likNo = norm.pdf(pT, 0, np.sqrt(self.sigmaH02)) + 1e-12
        # Lik H_rot
        predMuRot = self.rHat
        predVarRot = self.rVar + self.sigmaPert2
        likRot = norm.pdf(pT, predMuRot, np.sqrt(predVarRot)) + 1e-12
        # Lik H_trans
        predMuTrans = np.dot(x, self.wHat)
        predVarTrans = np.dot(x, np.dot(self.wCov, x)) + self.sigmaPert2
        likTrans = norm.pdf(pT, predMuTrans, np.sqrt(predVarTrans)) + 1e-12
        # Compute sub priors
        subPost = np.exp(self.logPost) / np.sum(np.exp(self.logPost) + 1e-12)
        # Marginal lik for pert
        likPert = subPost[0] * likRot + subPost[1] * likTrans + 1e-12
        # HMM transition
        trans = np.array([[1 - self.pChange, self.pChange], [0, 1]])
        predictState = np.dot(trans.T, self.stateProb)
        likStates = np.array([likNo, likPert])
        updateState = predictState * likStates
        updateState /= np.sum(updateState) + 1e-12
        self.stateProb = updateState
        # Compute conditional post under pert
        condLiks = np.array([likRot, likTrans])
        conditional = subPost * condLiks
        conditionalPost = conditional / likPert if likPert > 0 else subPost
        # Responsibilities (overall posterior)
        p = self.stateProb[1]
        respRot = conditionalPost[0]
        respTrans = conditionalPost[1]
        # Update H_rot, weighted by responsibility
        if predVarRot > 0:
            k = p * self.rVar / predVarRot
            #k *= respRot # Gate by responsibility
            self.rHat += (pT - predMuRot) *k
            self.rVar *= (1 - k)
        self.rVar += self.processNoiseVar
        self.rVar = max(self.rVar, 1e-6) # Prevent underflow/negatives
        # Update H_trans, weighted by responsibility
        if predVarTrans > 0:
            gain = p * np.dot(self.wCov, x) / predVarTrans
            #gain *= respTrans # Gate by responsibility
            gain[1] *= self.cosDirection # Modify cos update direction
            gain[2] *= self.sinDirection # Modify sin update direction
            self.wHat += (pT - predMuTrans) * gain
            self.wCov -= np.outer(gain, np.dot(x, self.wCov))
        self.wCov += self.processNoiseVar * np.eye(3) # Add isotropic noise to covariance
        # Ensure PSD via eigenvalue clipping
        eigenvalues, eigenvectors = np.linalg.eigh(self.wCov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        self.wCov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Update sub post to marginal
        newSubPost = p * conditionalPost + (1 - p) * subPost
        self.logPost = np.log(newSubPost + 1e-12)
        self.logPost -= scipy.special.logsumexp(self.logPost)
        self.postTargets.append(target)
      
    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance # Simplified; add stochasticity if needed
  
    def expectedMove(self, trialNum):
        #need to ad process noise if verying effective rot over time
        if trialNum < 0:
            return 0
        # Advance updates if necessary (assumes calls in sequential order)
        #while self.currentTrial < trialNum:
        #self.currentTrial += 1
        pT = self.stepHeight # Assume constant perturbation observation
        if trialNum >= len(self.targets):
            return 0
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        prevTarget = self.targets[trialNum-1]
        tRadPrev = np.deg2rad(prevTarget)
        x = np.array([1,np.cos(tRad-tRadPrev), np.sin(tRad-tRadPrev)])
        subPost = np.exp(self.logPost)
        subPost /= np.sum(subPost) + 1e-12
        predTrans = np.dot(x, self.wHat)
        expectedPertUnderPert = subPost[0] * self.rHat + subPost[1] * predTrans
        expectedPert = self.stateProb[1] * expectedPertUnderPert
        self.updatePosteriors(trialNum, pT)
        return -expectedPert
  
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
  
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
    
        # Use multiprocessing Pool to parallelize
      
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
    
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
           
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 8:
            ev, plo0, t, plrt, executionVariance, sinDirection, cosDirection, processNoiseVar = params
            sh = 30 # Default or from context
        else:
            raise ValueError("Params should include fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, plo0, t, plrt, executionVariance, sinDirection, cosDirection, processNoiseVar, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
  
# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight = params[0] # Adjust as needed
            fittingVariance = None
            priorLogodds0 = 0.0
            tau = 1.0
            priorLogoddsRt = 0.0
            sinDirection = 1.0
            cosDirection = 1.0
            processNoiseVar = 0.0
        else:
            fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar = params
        stepper = BayesianStepper(rot, fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
     
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            #sumSquares = np.sum(totErr ** 2)
            #rmseVal = np.sqrt(sumSquares / numSamp)
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        else:
            return 1e9
    evolve = True
    if rmse:
        bounds = [(0, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(1e-3, 1e5), (-8000, 10000), (0.001, 1000), (-1e4, 9000), (1e-3, 1e3), (-1, 1), (-1, 1), (0, 1e3)]
        def getX0(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        for i in range(1):
            #res = evolution(localFitPp, bounds=bounds, workers=1, popsize=30)
            bestX,es = cma.fmin2(localFitPp,x0=getX0(bounds),sigma0=1e3,restarts=6,bipop=True,options={'tolfun':1e-8,'tolfacupx':1e6,'maxfeval':30000,'verbose': 0,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            if False:# res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight = bestX[0]
        fittingVariance = None
        priorLogodds0 = 0.0
        tau = 1.0
        priorLogoddsRt = 0.0
        sinDirection = 1.0
        cosDirection = 1.0
        processNoiseVar = 0.0
        executionVariance = 10
        k = 1
    else:
        fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar = bestX
        k = len(bestX)
    stepper = BayesianStepper(rot,fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar]
    }

"""
    
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma
class BayesianStepper:
    def __init__(self, stepHeight, fittingVariance=None, priorLogodds0=0.0, tau=1.0, priorLogoddsRt=0.0, executionVariance=10, sinDirection=1.0, cosDirection=1.0, processNoiseVar=0.0, targets=None):
        self.stepHeight = stepHeight
        self.fittingVariance = np.abs(fittingVariance)
        self.targets = targets
        # Log posteriors for H0, H_rot, H_trans
        self.logPost = np.array([priorLogodds0, 0.0, -priorLogoddsRt], dtype=float)
        self.logPost -= scipy.special.logsumexp(self.logPost)
        # H_rot state
        self.rHat = 0.0
        self.rVar = tau ** 2 if tau > 0 else 1e-6
        # H_trans state
        self.wHat = np.zeros(3)
        self.wCov = (tau ** 2 if tau > 0 else 1e-6) * np.eye(3)
        self.sigmaPert2 = 1.0 ** 2 # Fixed
        self.sigmaH02 = executionVariance # Fixed
        self.postTargets = []
        self.currentTrial = -1
        self.sinDirection = sinDirection
        self.cosDirection = cosDirection
        self.processNoiseVar = processNoiseVar
       
    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        prevTarget = self.targets[trialNum-1]
        tRadPrev = np.deg2rad(prevTarget)
        x = np.array([1,np.cos(tRad-tRadPrev), np.sin(tRad-tRadPrev)])
        # Lik H0
        likH0 = norm.pdf(pT, 0, np.sqrt(self.sigmaH02)) + 1e-12
        # Lik H_rot
        predMuRot = self.rHat
        predVarRot = self.rVar + self.sigmaPert2
        likRot = norm.pdf(pT, predMuRot, np.sqrt(predVarRot)) + 1e-12
        # Lik H_trans
        predMuTrans = np.dot(x, self.wHat)
        predVarTrans = np.dot(x, np.dot(self.wCov, x)) + self.sigmaPert2
        likTrans = norm.pdf(pT, predMuTrans, np.sqrt(predVarTrans)) + 1e-12
        # Compute responsibilities (predictive posteriors)
        prior_post = np.exp(self.logPost) / np.sum(np.exp(self.logPost) + 1e-12)
        liks = np.array([likH0, likRot, likTrans])
        predicted_post = prior_post * liks
        predicted_post /= np.sum(predicted_post) + 1e-12
        # Update H_rot, weighted by responsibility
        if predVarRot > 0:
            k = self.rVar / predVarRot
            k *= predicted_post[1] # Gate by responsibility
            self.rHat += (pT - predMuRot) *k
            self.rVar *= (1 - k)
        self.rVar += self.processNoiseVar
        self.rVar = max(self.rVar, 1e-6)  # Prevent underflow/negatives
        # Update H_trans, weighted by responsibility
        if predVarTrans > 0:
            gain = np.dot(self.wCov, x) / predVarTrans
            gain *= predicted_post[2] # Gate by responsibility
            gain[1] *= self.cosDirection  # Modify cos update direction
            gain[2] *= self.sinDirection  # Modify sin update direction
            self.wHat += (pT - predMuTrans) * gain
            self.wCov -= np.outer(gain, np.dot(x, self.wCov))
        self.wCov += self.processNoiseVar * np.eye(3)  # Add isotropic noise to covariance
        # Ensure PSD via eigenvalue clipping
        eigenvalues, eigenvectors = np.linalg.eigh(self.wCov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        self.wCov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Update log posts
        logLiks = np.log(liks)
        self.logPost += logLiks
        self.logPost -= scipy.special.logsumexp(self.logPost)
        self.postTargets.append(target)
       
    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance # Simplified; add stochasticity if needed
   
    def expectedMove(self, trialNum):
        #need to ad process noise if verying effective rot over time
        if trialNum < 0:
            return 0
        # Advance updates if necessary (assumes calls in sequential order)
        #while self.currentTrial < trialNum:
        #self.currentTrial += 1
        pT = self.stepHeight # Assume constant perturbation observation
        post = np.exp(self.logPost)
        post /= np.sum(post) + 1e-12
        if trialNum >= len(self.targets):
            return 0
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        prevTarget = self.targets[trialNum-1]
        tRadPrev = np.deg2rad(prevTarget)
        x = np.array([1,np.cos(tRad-tRadPrev), np.sin(tRad-tRadPrev)])
        predTrans = np.dot(x, self.wHat)
        expected_pert = post[0] * 0 + post[1] * self.rHat + post[2] * predTrans
        self.updatePosteriors(trialNum, pT)
        return -expected_pert
   
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
   
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
     
        # Use multiprocessing Pool to parallelize
       
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
     
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
            
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 8:
            ev, plo0, t, plrt, executionVariance, sinDirection, cosDirection, processNoiseVar = params
            sh = 30 # Default or from context
        else:
            raise ValueError("Params should include fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, plo0, t, plrt, executionVariance, sinDirection, cosDirection, processNoiseVar, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
   
# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight = params[0] # Adjust as needed
            fittingVariance = None
            priorLogodds0 = 0.0
            tau = 1.0
            priorLogoddsRt = 0.0
            sinDirection = 1.0
            cosDirection = 1.0
            processNoiseVar = 0.0
        else:
            fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar = params
        stepper = BayesianStepper(rot, fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
      
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            #sumSquares = np.sum(totErr ** 2)
            #rmseVal = np.sqrt(sumSquares / numSamp)
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        if np.isfinite(logLikelihood) and not np.isnan(logLikelihood):
            return -logLikelihood
        else:
            return 1e9
    evolve = True
    if rmse:
        bounds = [(0, heightCap)]
        res = evolution(localFitPp, bounds=bounds, workers=1)
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(1e-3, 1e5), (-8000, 10000), (0.001, 1000), (-1e4, 9000), (1e-3, 1e3), (-1, 1), (-1, 1), (0, 1e3)]
        def getX0(bounds):
            return np.array([np.random.uniform(low, high) for low, high in bounds])
        for i in range(1):
            #res = evolution(localFitPp, bounds=bounds, workers=1, popsize=30)
            bestX,es = cma.fmin2(localFitPp,x0=getX0(bounds),sigma0=1e3,restarts=6,bipop=True,options={'tolfun':1e-8,'tolfacupx':1e6,'maxfeval':30000,'verbose': 0,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            if False:# res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight = bestX[0]
        fittingVariance = None
        priorLogodds0 = 0.0
        tau = 1.0
        priorLogoddsRt = 0.0
        sinDirection = 1.0
        cosDirection = 1.0
        processNoiseVar = 0.0
        executionVariance = 10
        k = 1
    else:
        fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar = bestX
        k = len(bestX)
    stepper = BayesianStepper(rot,fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar, targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, priorLogodds0, tau, priorLogoddsRt, executionVariance, sinDirection, cosDirection, processNoiseVar]
    }
"""
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing # Added for parallelism
from functools import partial # For passing args to worker function
import scipy.special
import cma

class BayesianStepper:
    def __init__(self, stepHeight, fittingVariance=None, priorLogodds0=0.0, tau=1.0, wSin=0, wCos=0, gamma=1, priorLogoddsRt=0.0, executionVariance=10, targets=None):
        self.stepHeight = stepHeight
        self.fittingVariance = fittingVariance
        self.targets = targets
        # Log posteriors for H0, H_rot, H_trans
        self.logPost = np.array([priorLogodds0, 0.0, -priorLogoddsRt], dtype=float)
        self.logPost -= scipy.special.logsumexp(self.logPost)
        # H_rot state
        self.rHat = 0.0
        self.rVar = tau ** 2 if tau > 0 else 1e-6
        # H_trans state
        self.wHat = np.zeros(2)
        self.wCov = (tau ** 2 if tau > 0 else 1e-6) * np.eye(2)
        self.sigmaPert2 = 1.0 ** 2  # Fixed
        self.sigmaH02 = executionVariance # Fixed
        self.wSin = wSin
        self.wCos = wCos
        self.gamma = gamma
        self.postTargets = []
        self.currentTrial = -1

    def updatePosteriors(self, trialNum, pT):
        if self.targets is None or trialNum >= len(self.targets):
            return
        target = self.targets[trialNum]
        tRad = np.deg2rad(target)
        x = np.array([np.cos(tRad), np.sin(tRad)])
        # Lik H0
        likH0 = norm.pdf(pT, 0, np.sqrt(self.sigmaH02)) + 1e-12
        # Lik H_rot
        predMuRot = self.rHat
        predVarRot = self.rVar + self.sigmaPert2
        likRot = norm.pdf(pT, predMuRot, np.sqrt(predVarRot)) + 1e-12
        # Update H_rot
        if predVarRot > 0:
            k = self.rVar / predVarRot
            self.rHat += k * (pT - predMuRot)
            self.rVar *= (1 - k)
        # Lik H_trans
        predMuTrans = np.dot(x, self.wHat)
        predVarTrans = np.dot(x, np.dot(self.wCov, x)) + self.sigmaPert2
        likTrans = norm.pdf(pT, predMuTrans, np.sqrt(predVarTrans)) + 1e-12
        # Update H_trans
        if predVarTrans > 0:
            gain = np.dot(self.wCov, x) / predVarTrans
            self.wHat += gain * (pT - predMuTrans)
            self.wCov -= np.outer(gain, np.dot(x, self.wCov))
        # Update log posts
        logLiks = np.log([likH0, likRot, likTrans])
        self.logPost += logLiks
        self.logPost -= scipy.special.logsumexp(self.logPost)
        self.postTargets.append(target)

    def modelMove(self, trialNum):
        move = self.expectedMove(trialNum)
        return move, self.fittingVariance  # Simplified; add stochasticity if needed

    def expectedMove(self, trialNum):
        #need to ad process noise if verying effective rot over time
        if trialNum < 0:
            return 0
        # Advance updates if necessary (assumes calls in sequential order)
        while self.currentTrial < trialNum:
            self.currentTrial += 1
            pT = self.stepHeight  # Assume constant perturbation observation
            self.updatePosteriors(self.currentTrial, pT)
        post = np.exp(self.logPost)
        post /= np.sum(post) + 1e-12
        pPert = post[1] + post[2]
        if pPert < 1e-6:
            return 0
        pTransGivenPert = post[2] / pPert
        if trialNum >= len(self.targets):
            return pPert * self.stepHeight * (1 - 2 * pTransGivenPert * expit(self.gamma))
        target = self.targets[trialNum]
        prevTarget = self.targets[trialNum-1]
        sinVal = np.sin(np.deg2rad(target-prevTarget))
        cosVal = np.cos(np.deg2rad(target-prevTarget))
        logit = self.gamma + self.wSin * sinVal + self.wCos * cosVal

        pFlip = pTransGivenPert * expit(logit)
        expectedS = 1 - 2 * pFlip
        return pPert * self.stepHeight * expectedS

class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
     
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.rmses = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
       
        # Use multiprocessing Pool to parallelize

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: # Use all available cores
            worker = partial(fitSinglePp, dat=self.dat, rmse=self.rmse, startCap=self.startCap,
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
       
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['bic']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
                self.rmses[it] = res['rmse']
              
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 8:
            sh, ev, plo0, t, ws, wc, g, plrt = params
        else:
            raise ValueError("Params should include stepHeight, fittingVariance, priorLogodds0, tau, wSin, wCos, gamma, priorLogoddsRt")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = BayesianStepper(sh, ev, plo0, t, ws, wc, g, plrt, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)

# Worker function to fit a single participant (must be top-level or picklable)
def fitSinglePp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    rot = -conVal
    def localFitPp(params):
        if rmse:
            
            stepHeight, wSin, wCos = params[:3]  # Adjust as needed
            fittingVariance = None
            priorLogodds0 = 0.0
            tau = 1.0
            gamma = 1.0
            priorLogoddsRt = 0.0
        else:
            fittingVariance, priorLogodds0, tau, wSin, wCos, gamma, priorLogoddsRt, executionVariance = params
        stepper = BayesianStepper(rot, fittingVariance, priorLogodds0, tau, wSin, wCos, gamma, priorLogoddsRt, executionVariance, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
        
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            #sumSquares = np.sum(totErr ** 2)
            #rmseVal = np.sqrt(sumSquares / numSamp)
            modelStd = np.sqrt(fittingVariance)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        return -logLikelihood
    evolve = True
    if rmse:
        bounds = [(0, heightCap), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit)]
        res = evolution(localFitPp, bounds=bounds, workers=1) 
        bestX = res.x
        bestFun = res.fun
    else:
        bestX = None
        bestFun = 1e9
        bounds = [(1e-3, 1e4), (-20, 100), (0.1, 100), (-100, 90), (-100, 90), (-90, 100), (-100, 20), (0, 1e4)]
        def getX0():
            return np.random.rand(1,len(bounds)).squeeze() * [(i[0]+i[1])/2 for i in bounds]
        for i in range(1):
            #res = evolution(localFitPp, bounds=bounds, workers=1, popsize=15) 
            #res = minimize(localFitPp,x0=[0,10,0,0,0,0,0,0], method=method, bounds=bounds) 
            bestX,es = cma.fmin2(localFitPp,x0=getX0,sigma0=1,restarts=4,bipop=True,options={'tolfun':1e-7,'tolfacupx':1e6,'maxfeval':6000,'verbose': 0})#,'bounds':[[i[0] for i in bounds],[i[1] for i in bounds]]})
            if False:# res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    if rmse:
        stepHeight, wSin, wCos = bestX
        fittingVariance = None
        priorLogodds0 = 0.0
        tau = 1.0
        gamma = 1.0
        priorLogoddsRt = 0.0
        k = 3
    else:
        fittingVariance, priorLogodds0, tau, wSin, wCos, gamma, priorLogoddsRt, executionVariance = bestX
        k = len(bestX)
    stepper = BayesianStepper(rot,fittingVariance, priorLogodds0, tau, wSin, wCos, gamma, priorLogoddsRt, executionVariance, targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmseVal = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
    else:
        sumSquares = np.sum(totErr ** 2)
        rmse = np.sqrt(sumSquares / numSamp)
        modelStd = np.sqrt(fittingVariance)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
    bic = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'rmse':rmse,
        'it': it,
        'bic': bic,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        'xs': [fittingVariance, priorLogodds0, tau, wSin, wCos, gamma, priorLogoddsRt, executionVariance]
    }


"""







"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.special import expit
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing  # Added for parallelism
from functools import partial  # For passing args to worker function

class Stepper:
    def __init__(self, stepStart, stepHeight, fittingVariance=None, wSin=0, wCos=0, beta=0, gamma=1, priorLogOdds=.5, targets=None):
        self.stepStart = stepStart
        self.stepHeight = stepHeight
        self.ev = fittingVariance
        self.wSin = wSin
        self.wCos = wCos
        self.beta = beta
        self.gamma = gamma
        self.priorLogOdds = priorLogOdds
        self.targets = targets
        self.targetCounts = {}
        self.evidence = 1.0
        self.postTargets = [] # Track post-stepStart targets for diversity

    def computePFlip(self, trialNum):
        if self.targets is None or trialNum < 0:
            return 0
        
        target = self.targets[trialNum]
        
        if trialNum < (self.stepStart - 1):
            return 0
        
        elif trialNum == (self.stepStart - 1):
            self.postTargets.append(target)
            return 0
        
        else:
            n = len(self.postTargets)
            
            # Compute mean vector and dissimilarity
            if n == 0:
                meanLength = 0
                similarity = 0
                prevCurrSim = 0
                logit = 0
            else:
                anglesPrev = np.deg2rad(np.array(self.postTargets))
                complexAnglesPrev = np.exp(1j * anglesPrev)
                meanVector = np.mean(complexAnglesPrev)
                meanLength = np.abs(meanVector)
                #currentVector = np.exp(1j * np.deg2rad(target))
                #similarity = np.real(np.vdot(currentVector, meanVector) / (meanLength * 1.0)) if meanLength > 0 else 0
                #prevAngle = self.postTargets[-1]
                #prevVector = np.exp(1j * np.deg2rad(prevAngle))
                #prevCurrSim = np.real(np.conj(currentVector) * prevVector)
                
            #diss = -similarity
            #diss = np.clip(diss,0,1.0)
            if n == 0:
                pTrans = expit(-self.priorLogOdds)
            else:
                circVar = 1 - meanLength ** 2
                logLikT = -circVar * n * self.beta
                logLikR = 0.0
                logPosteriorOdds = self.priorLogOdds + logLikR - logLikT
                pTrans = expit(-logPosteriorOdds)
                prevTarget = self.postTargets[-1]
                sinVal = np.sin(np.deg2rad(target))
                cosVal = np.cos(np.deg2rad(target))
                logit = self.wSin * sinVal + self.wCos * cosVal + self.gamma
                
                
            #pFlip = self.gamma * diss * pTrans
            pFlip = pTrans * expit(logit)# (1 / (1 + np.exp(-logit)))
            self.postTargets.append(target)
        return min(1, max(0, pFlip))

    def modelMove(self, trialNum):
        if trialNum < self.stepStart:
            return 0, self.ev
        pFlip = self.computePFlip(trialNum)
        s = -1 if np.random.rand() < pFlip else 1
        return s * self.stepHeight, self.ev
        
    def expectedMove(self, trialNum):
        if trialNum < self.stepStart:
            return 0
        pFlip = self.computePFlip(trialNum)
        expectedS = 1 - pFlip * 2 # 
        return self.stepHeight * expectedS
       
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
      
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
        
        # Use multiprocessing Pool to parallelize
        #t = fit_single_pp(uniqP[0],0,dat=self.dat, rmse=self.rmse, startCap=self.startCap, heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:  # Use all available cores
            worker = partial(fit_single_pp, dat=self.dat, rmse=self.rmse, startCap=self.startCap, 
                             heightCap=self.heightCap, method=self.method, fitPhase=self.fitPhase, conVal=self.conVal)
            results = pool.starmap(worker, [(pp, i, ) for i, pp in enumerate(uniqP)])
        
        # Collect results back into self
        for res in results:
            if res is not None:
                it = res['it']
                self.BICs[it] = res['BIC']
                self.negLL[it] = res['negLL']
                self.mStates[it] = res['mStates']
                self.allAims[it] = res['allAims']
                self.xs[it] = res['xs']
               
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 5:
            ss, sh, wSin, wCos, beta = params
            ev = None
        elif len(params) == 6:
            ss, sh, ev, wSin, wCos, beta = params
        else:
            raise ValueError("Params should include stepStart, stepHeight, [executionVar], wSin, wCos, beta")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = Stepper(ss, sh, ev, wSin, wCos, beta, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)

# Worker function to fit a single participant (must be top-level or picklable)
def fit_single_pp(pp, it, dat, rmse, startCap, heightCap, method, fitPhase, conVal):
    # Replicate self.pp and self.it locally
    executionVarBaseline = 1000
    circularWeightLimit = 1000
    pDat = dat[(dat['participantNum'] == pp) & (dat['phase'] == fitPhase)]
    blockNums = pDat['blockNum'].unique()
    pDat = pDat[pDat['blockNum'] == blockNums[0]]
    aims = pDat['aim'].values
    targets = pDat['targetPosition'].values
    trials = np.arange(len(aims))
    
    def local_fitPP(params):
        if rmse:
            stepStart, stepHeight, wSin, wCos, beta = params
            executionVar = None
        else:
            stepStart, stepHeight, executionVar, wSin, wCos, beta, gamma, priorLogOdds = params
            #stepStart, stepHeight, executionVar, wSin, wCos, beta = params
            #gamma = 1
            #beta = 0
        stepStart = int(np.ceil(stepStart))
        stepper = Stepper(stepStart, stepHeight, executionVar, wSin, wCos, beta, gamma, priorLogOdds, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if rmse:
            sumSquares = np.sum(totErr ** 2)
            rmse_val = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            modelStd = np.sqrt(executionVar)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        return -logLikelihood
    evolve = True
    if rmse:
        bounds = [(0, startCap), (0, heightCap), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit), (0, 1)]
        res = evolution(local_fitPP, bounds=bounds, workers=1)  # Set workers=1 to avoid nested parallelism issues
        bestX = res.x
        bestFun = res.fun
    elif evolve:
        bestX = None
        bestFun = 1e9
        #bounds = [(0, startCap), (0, heightCap), (1e-3, 1e4), (0, 1), (0, 1e4), (0, 10)]
        #bounds = [(0, startCap), (0, heightCap), (1e-3, 1e4), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit), (0, 1)]
        bounds = [(0, startCap), (0, heightCap), (1e-3, 1e4), (-100, 100), (-100, 100), (0,10), (-100, 100), (-1000,10)]
        for i in range(1):
            res = evolution(local_fitPP, bounds=bounds, workers=1, popsize=15)  # Set workers=1 to avoid nested parallelism issues
            if res.fun < bestFun:
                bestX = res.x
                bestFun = res.fun
    #brute force bfgs search
    #skipping for now as de should suffice
    else:
        bestX = None
        bestFun = 1e9
        numChunks = 40# startCap // 4
        chunkLen = startCap // numChunks
        points = np.linspace(0, startCap, numChunks, dtype=int)
        coarseResults = []
        subX0 = [-conVal, 5, 0,0,0]
        for fixedStepStart in points:
            def subFitPP(subParams):
                fullParams = [fixedStepStart] + list(subParams)
                return local_fitPP(fullParams)
            #subBounds = [(-heightCap, heightCap), (1e-3, 1e4), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit), (0, 1)]
            subBounds = [(0, heightCap), (1e-3, 1e4), (0, 1), (0, 1e4), (0, 10)]
            res = minimize(subFitPP, subX0, method=method, bounds=subBounds)
            coarseResults.append((fixedStepStart, res.fun, res.x))
            if res.fun < bestFun:
                bestFun = res.fun
                bestX = [fixedStepStart] + list(res.x)
        
        sortedCoarse = sorted(coarseResults, key=lambda x: x[1])
        topTwo = sortedCoarse[:2]
        s1, s2 = sorted([topTwo[0][0], topTwo[1][0]])
        for fixedStepStart in range(np.maximum(0, s1 - chunkLen * 5), np.minimum(startCap, s2 + chunkLen * 5)):
            if fixedStepStart in points:
                continue
            def subFitPP(subParams):
                fullParams = [fixedStepStart] + list(subParams)
                return local_fitPP(fullParams)
            #subBounds = [(-heightCap, heightCap), (1e-3, 1e4), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit), (0, 1)]
            #subBounds = [(-heightCap, heightCap), (1e-3, 1e4),(0, 1)]
            res = minimize(subFitPP, subX0, method=method, bounds=subBounds)
            if res.fun < bestFun:
                bestFun = res.fun
                bestX = [fixedStepStart] + list(res.x)

    if rmse:
        stepStart, stepHeight, wSin, wCos, beta = bestX
        executionVar = None
    else:
        #stepStart, stepHeight, executionVar, wSin, wCos, beta = bestX
        #gamma = 1
        #stepStart, stepHeight, executionVar, beta, gamma, eta= bestX
        stepStart, stepHeight, executionVar, wSin, wCos, beta, gamma, priorLogOdds = bestX
        #beta = 0
    stepStart = int(np.ceil(stepStart))
    stepper = Stepper(stepStart, stepHeight, executionVar, wSin, wCos, beta, gamma, priorLogOdds, targets)
    mOuts = np.zeros_like(aims, dtype=float)
    for trial in trials:
        mOuts[trial] = stepper.expectedMove(trial)
    mask = ~np.isnan(aims)
    validAims = aims[mask]
    validMOuts = mOuts[mask]
    totErr = validAims - validMOuts
    numSamp = len(totErr)
    if numSamp == 0:
        return None  
    if rmse:
        sumSquares = np.sum(totErr ** 2)
        rmse_val = np.sqrt(sumSquares / numSamp)
        sortedErr = np.sort(totErr)
        mu, std = norm.fit(sortedErr)
        logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        k = 3
    else:
        modelStd = np.sqrt(executionVar)
        liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
        logLikelihood = np.sum(np.log(liks))
        k = 8
    BIC = k * np.log(numSamp) - 2 * logLikelihood
    negLL = -logLikelihood
    return {
        'it': it,
        'BIC': BIC,
        'negLL': negLL,
        'mStates': mOuts.tolist(),
        'allAims': aims.tolist(),
        #'xs': bestX.tolist() if rmse else [stepStart, stepHeight, executionVar, wSin, wCos, beta]
        'xs': bestX.tolist() if rmse else [stepStart, stepHeight, executionVar, wSin, wCos, beta, gamma, priorLogOdds]
    }


"""

"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.stats import norm
import matplotlib.pyplot as plt

class Stepper:
    def __init__(self, stepStart, stepHeight, fittingVariance=None, wSin=0, wCos=0, beta=0, targets=None):
        self.stepStart = stepStart
        self.stepHeight = stepHeight
        self.ev = fittingVariance
        self.wSin = wSin
        self.wCos = wCos
        self.beta = beta
        self.targets = targets
        self.targetCounts = {}
        self.evidence = 0.0
        self.postTargets = [] # Track post-stepStart targets for diversity
        
    def computePFlip(self, trialNum):
        if self.targets is None or trialNum < 0:
            return 0
        target = self.targets[trialNum]
        sinVal = np.sin(np.deg2rad(target))
        cosVal = np.cos(np.deg2rad(target))
        logit = self.wSin * sinVal + self.wCos * cosVal
        dissimilarity = 0
        if trialNum < (self.stepStart - 1):
            pTrans = 0 # No modulation before step
        elif trialNum == (self.stepStart - 1):
            self.postTargets.append(target)
        else:
           
            if len(self.postTargets) > 0:
                angles = np.deg2rad(np.array(self.postTargets))
                complexAngles = np.exp(1j * angles)
                meanVector = np.mean(complexAngles)
                meanLength = np.abs(meanVector)
                diversity = 1 - meanLength
                currentVector = np.exp(1j * np.deg2rad(target))
                similarity = np.real(np.vdot(currentVector, meanVector) / (np.abs(meanVector) * 1.0))
                dissimilarity = (1 - similarity) / 2
                #clip otherwise occasional floating point below 0 due to finite precision
                dissimilarity = np.clip(dissimilarity, 0, 1.0)
            else:
                diversity = 0
           
            if target not in self.targetCounts:
                self.targetCounts[target] = 0
            self.targetCounts[target] += 1
            increment = 1.0 / self.targetCounts[target]
            #update evidence of rotation over trans
            self.evidence += increment * diversity #* dissimilarity
            pTrans = self.beta ** self.evidence
            self.postTargets.append(target)
       
        pFlip = pTrans * (1 / (1 + np.exp(-logit)))
        return min(1, max(0, pFlip))
       
    def modelMove(self, trialNum):
        if trialNum < self.stepStart:
            return 0, self.ev
        pFlip = self.computePFlip(trialNum)
        s = -1 if np.random.rand() < pFlip else 1
        return s * self.stepHeight, self.ev
    def expectedMove(self, trialNum):
        if trialNum < self.stepStart:
            return 0
        pFlip = self.computePFlip(trialNum)
        expectedS = 1 - 2 * pFlip
        return self.stepHeight * expectedS
        
class fitShell:
    def __init__(self, df, conVal='none', condition='none', startCap=320, fitLen=320, fitPhase='rotation', heightCap=180, rmse=False,
                 method='L-BFGS-B'):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitLen = fitLen
        self.startCap = startCap
        self.fitPhase = fitPhase
        self.heightCap = heightCap
        self.rmse = rmse
        self.allAims = []
        self.method = method
       
    def fitRot(self):
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.allAims = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
        i = 0
        circularWeightLimit = 100
        for pp in uniqP:
            executionVarBaseline = 1000
            self.pp = pp
            self.it = i
            if self.rmse:
                bounds = [(0, self.startCap), (0, self.heightCap), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit), (0, 1)]
                res = evolution(self.fitPP, bounds=bounds, workers=-1)
                self.updateStates(res.x)
            else:
                #bounds = [(0, self.startCap), (0, self.heightCap), (1e-3, 1e4), (-1e3, 1e3), (-1e3, 1e3), (0, 1)]
                #res = evolution(self.fitPP, bounds=bounds, workers=-1,popsize=60)
                bestX = None#res.x
                bestFun = 1e9#1eres.fun
                numChunks = self.startCap // 4
                chunkLen = self.startCap // numChunks
                points = np.linspace(0, self.startCap, numChunks, dtype=int)
                coarseResults = []
                subX0 = [-self.conVal, 50, 0, 0, 0.2]
                for fixedStepStart in points:
                    def subFitPP(subParams):
                        fullParams = [fixedStepStart] + list(subParams)
                        return self.fitPP(fullParams)
                    subBounds = [(-self.heightCap, self.heightCap), (1e-3, 1e4), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit), (0, 1)]
                    
                    res = minimize(subFitPP, subX0, method=self.method, bounds=subBounds)
                    #subX0 = res.x
                    coarseResults.append((fixedStepStart, res.fun, res.x))
                    print(f"Coarse: {fixedStepStart}/{self.startCap}, fun: {res.fun}", end='\r')
                    if res.fun < bestFun:
                        bestFun = res.fun
                        bestX = [fixedStepStart] + list(res.x)
                # Find the two best from coarse
                sortedCoarse = sorted(coarseResults, key=lambda x: x[1])
                topTwo = sortedCoarse[:2]
                s1, s2 = sorted([topTwo[0][0], topTwo[1][0]])
                #subX0 = bestX[1:]
                # Fine search: every integer between s1 and s2 (inclusive)
                for fixedStepStart in range(np.maximum(0, s1 - chunkLen * 5), np.minimum(self.startCap, s2 + chunkLen * 5)):
                    # Skip if already in coarse points to avoid redundant computation
                    if fixedStepStart in points:
                        continue
                    def subFitPP(subParams):
                        fullParams = [fixedStepStart] + list(subParams)
                        return self.fitPP(fullParams)
                    subBounds = [(-self.heightCap, self.heightCap), (1e-3, 1e4), (-circularWeightLimit, circularWeightLimit), (-circularWeightLimit, circularWeightLimit), (0, 1)]
                    res = minimize(subFitPP, subX0, method=self.method, bounds=subBounds)
                    print(f"Fine: {fixedStepStart}/{self.startCap}, fun: {res.fun}", end='\r')
                    if res.fun < bestFun:
                        bestFun = res.fun
                        bestX = [fixedStepStart] + list(res.x)
            print(bestFun,bestX)
            self.updateStates(bestX)
            print(i, 'out of', len(uniqP), ' , BIC: ,', self.BIC)
            i += 1
        print() # Move to a new line after the loop completes
        
    def fitPP(self, params):
        if self.rmse:
            stepStart, stepHeight, wSin, wCos, beta = params
            executionVar = None
        else:
            stepStart, stepHeight, executionVar, wSin, wCos, beta = params
        stepStart = int(np.ceil(stepStart))
        pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        blockNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == blockNums[0]]
        aims = pDat['aim'].values
        targets = pDat['targetPosition'].values
        trials = np.arange(len(aims))
        stepper = Stepper(stepStart, stepHeight, executionVar, wSin, wCos, beta, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if self.rmse:
            sumSquares = np.sum(totErr ** 2)
            rmse = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            modelStd = np.sqrt(executionVar)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        return -logLikelihood
        
    def updateStates(self, params):
        if self.rmse:
            stepStart, stepHeight, wSin, wCos, beta = params
            executionVar = None
        else:
            stepStart, stepHeight, executionVar, wSin, wCos, beta = params
        stepStart = int(np.ceil(stepStart))
        pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        blockNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == blockNums[0]]
        aims = pDat['aim'].values
        targets = pDat['targetPosition'].values
        trials = np.arange(len(aims))
        stepper = Stepper(stepStart, stepHeight, executionVar, wSin, wCos, beta, targets)
        mOuts = np.zeros_like(aims, dtype=float)
        for trial in trials:
            mOuts[trial] = stepper.expectedMove(trial)
        mask = ~np.isnan(aims)
        validAims = aims[mask]
        validMOuts = mOuts[mask]
        totErr = validAims - validMOuts
        numSamp = len(totErr)
        if numSamp == 0:
            return # Shouldn't happen, but guard
        if self.rmse:
            sumSquares = np.sum(totErr ** 2)
            rmse = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
            k = 5
        else:
            modelStd = np.sqrt(executionVar)
            liks = norm.pdf(validAims, validMOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
            k = 6#
        BIC = k * np.log(numSamp) - 2 * logLikelihood
        negLL = -logLikelihood
        if negLL < self.negLL[self.it]:
            self.negLL[self.it] = negLL
            self.BICs[self.it] = BIC
            self.BIC = BIC
            self.mStates[self.it] = mOuts.tolist()
            self.allAims[self.it] = aims
            if self.rmse:
                self.xs[self.it] = stepStart, stepHeight, wSin, wCos, beta
            else:
                self.xs[self.it] = stepStart, stepHeight, executionVar, wSin, wCos, beta
                
    def genDat(self, params, rots, trials=np.arange(-5,35,1)):
        if len(params) == 5:
            ss, sh, wSin, wCos, beta = params
            ev = None
        elif len(params) == 6:
            ss, sh, ev, wSin, wCos, beta = params
        else:
            raise ValueError("Params should include stepStart, stepHeight, [executionVar], wSin, wCos, beta")
        # Assume rots is iterable with targets; if scalar, convert to array
        if not hasattr(rots, '__len__'):
            rots = np.full(len(trials), rots)
        stepper = Stepper(ss, sh, ev, wSin, wCos, beta, rots)
        noise = 0 # Could set to sqrt(ev) if desired, but keeping as 0 per original
        trialsMod = np.asarray(trials)
        trialsMod[trialsMod >= 30] = -100
        states = []
        for trialNum in trialsMod:
            if trialNum < 0:
                mOut = 0
            else:
                mOut, _ = stepper.modelMove(trialNum)
            states.append(mOut + np.random.normal(0, noise))
        return np.array(states)
"""
