import numpy as np
from scipy.stats import norm
from scipy import linalg as la
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from scipy.stats import qmc
from types import SimpleNamespace
import numba as nb
from numba import njit
from numba.typed import Dict, List
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
from cmaes import CMA
import matplotlib; matplotlib.use('Agg')
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy.special import logsumexp as sp_logsumexp
np.random.seed(99)

@njit(cache=True)
def wrapAngle(x):
    return (x + 180) % 360 - 180

@njit(cache=True)
def safeLog(x):
    if x <= 0.0:
        return -1e9
    return math.log(x)

@njit(cache=True)
def logAddExp(a, b):
    if a == b:
        return a + math.log(2.0)
    maxVal = max(a, b)
    minVal = min(a, b)
    if maxVal - minVal > 700.0:
        return maxVal
    return maxVal + math.log(1.0 + math.exp(minVal - maxVal))

@njit(cache=True)
def normCdf(x, mu, sigma):
    if sigma <= 0.0:
        return 0.0 if x < mu else 1.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

@njit(cache=True)
def logsumexp(arr):
    n = len(arr)
    if n == 0:
        return -1e9
    maxV = arr[0]
    for i in range(1, n):
        if arr[i] > maxV:
            maxV = arr[i]
    if not math.isfinite(maxV):
        return -1e9
    s = 0.0
    for i in range(n):
        s += math.exp(arr[i] - maxV)
    return maxV + math.log(s)

@njit(cache=True)
def logGaussianPdf(x, mu, sigma):
    if sigma <= 0.0:
        return -1e9
    var = sigma**2
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * ((x - mu)**2 / var)

@njit(cache=True)
def vectorizedLogGaussianPdf(x, muArr, sigmaArr):
    result = np.full(len(muArr), -1e9)
    valid = sigmaArr > 0.0
    var = sigmaArr[valid]**2
    result[valid] = -0.5 * np.log(2 * math.pi * var) - 0.5 * ((x - muArr[valid])**2 / var)
    return result

@njit(cache=True)
def forHumanComparLogGaussianPdf(x, muArr, sigma):
    result = np.full(len(muArr), -1e9)
    var = sigma**2
    result = -0.5 * np.log(2 * math.pi * var) - 0.5 * ((x - muArr)**2 / var)
    return result

from numba import njit
import numpy as np
import math
EPS = 1e-12
LOGZERO = -1e9
TWO_PI = 2.0 * math.pi

@njit(inline='always')
def safeLog(x):
    if x <= 0.0:
        return math.log(EPS)
    return math.log(x)

@njit(inline='always')
def wrapAngle(x):
    while x <= -180.0:
        x += 360.0
    while x > 180.0:
        x -= 360.0
    return x

@njit(inline='always')
def logGaussianPdf(x, mean, sd):
    if sd <= 0.0:
        sd = EPS
    dev = (x - mean) / sd
    return -0.5 * (dev * dev) - 0.5 * math.log(TWO_PI) - math.log(sd)

@njit
def logSumExp1D(arr, n):
    if n == 0:
        return LOGZERO
    m = arr[0]
    for i in range(1, n):
        if arr[i] > m:
            m = arr[i]
    if m == -1e9:
        return LOGZERO
    s = 0.0
    for i in range(n):
        s += math.exp(arr[i] - m)
    return m + math.log(s + EPS)

@njit(inline='always')
def logAddExp(a, b):
    if a < b:
        a, b = b, a
    if a == LOGZERO:
        return b
    return a + math.log(1.0 + math.exp(b - a))

@njit(cache=True)
def computeNegLl(params, allAims, mask, trials, isRotation, rotation, targets):
    logSigmaMotor, logHazard, logitPS0, logVisCoeff = params
    sigmaMotor = math.exp(logSigmaMotor)
    hazard = 1.0 / (1.0 + math.exp(-logHazard))
    pS0 = 1.0 / (1.0 + math.exp(-logitPS0))
    visCoeff = math.exp(logVisCoeff)
    priorVar = 180.0**2
    priorPrec = 1.0 / priorVar if priorVar > 0.0 else 1e-6
    muPrior = 0.0
    initState = np.array([pS0, 1.0 - pS0])
    prevStatePosterior = initState.copy()
    numTrials = len(trials)
    maxT = numTrials + 1
    maxRun = 10
    logR = np.full((maxT, maxRun, 2), LOGZERO)
    logR[0, 0, 0] = safeLog(initState[0])
    logR[0, 0, 1] = safeLog(initState[1])
    muRun = np.full(maxRun, muPrior)
    precRun = np.full(maxRun, priorPrec)
    logLikelihood = 0.0
    for idx in range(numTrials):
        t = idx + 1
        runLimit = t if t < maxRun else maxRun
        trial = trials[idx]
        aim = allAims[trial]
        deltaObs = rotation if isRotation[trial] else 0.0
        deltaObs = wrapAngle(deltaObs)
        predState0 = (1.0 - hazard) * prevStatePosterior[0] + hazard * initState[0]
        predStateP = (1.0 - hazard) * prevStatePosterior[1] + hazard * initState[1]
        pStateSum = predState0 + predStateP
        if pStateSum <= 0.0:
            pS0Norm = initState[0]
            pPNorm = initState[1]
        else:
            pS0Norm = predState0 / pStateSum
            pPNorm = predStateP / pStateSum
        varMuArrPrev = np.empty(runLimit)
        for k in range(runLimit):
            if precRun[k] > 0.0:
                varMuArrPrev[k] = 1.0 / precRun[k]
            else:
                varMuArrPrev[k] = priorVar
        muRunPrev = muRun[:runLimit].copy()
        sigmaVis = visCoeff * math.fabs(deltaObs)
        logCompPArr = np.empty(runLimit)
        for k in range(runLimit):
            compVar = varMuArrPrev[k] + sigmaVis**2 + sigmaMotor**2
            logCompPArr[k] = logGaussianPdf(deltaObs, muRunPrev[k], math.sqrt(compVar))
        logSumPrevP = LOGZERO
        for k in range(runLimit):
            logSumPrevP = logAddExp(logSumPrevP, logR[t-1, k, 1])
        logMargLikG = LOGZERO
        for k in range(runLimit):
            logMargLikG = logAddExp(logMargLikG, logR[t-1, k, 1] + logCompPArr[k])
        logMargLikG -= logSumPrevP
        logLik0 = logGaussianPdf(deltaObs, 0.0, math.sqrt(sigmaVis**2 + sigmaMotor**2))
        unnormS0 = np.full(runLimit, LOGZERO)
        unnormS0[0] = logLik0 + safeLog(hazard) + safeLog(initState[0])
        for k in range(1, runLimit):
            unnormS0[k] = logR[t-1, k-1, 0] + logLik0 + safeLog(1.0 - hazard)
        unnormP = np.full(runLimit, LOGZERO)
        priorPredLikP0 = logGaussianPdf(deltaObs, muPrior, math.sqrt(priorVar + sigmaVis**2 + sigmaMotor**2))
        unnormP[0] = priorPredLikP0 + safeLog(hazard) + safeLog(initState[1])
        for k in range(1, runLimit):
            unnormP[k] = logR[t-1, k-1, 1] + logCompPArr[k-1] + safeLog(1.0 - hazard)
        unnormAll = np.empty(runLimit*2)
        for k in range(runLimit):
            unnormAll[k] = unnormS0[k]
            unnormAll[k + runLimit] = unnormP[k]
        logSumAll = LOGZERO
        for k in range(runLimit*2):
            logSumAll = logAddExp(logSumAll, unnormAll[k])
        for k in range(runLimit):
            logR[t, k, 0] = unnormS0[k] - logSumAll
            logR[t, k, 1] = unnormP[k] - logSumAll
        logUnnorm0 = logLik0 + safeLog(pS0Norm)
        logUnnormP = logMargLikG + safeLog(pPNorm)
        logSumReg = logAddExp(logUnnorm0, logUnnormP)
        if not math.isfinite(logSumReg):
            post0 = pS0Norm
            postP = pPNorm
        else:
            post0 = math.exp(logUnnorm0 - logSumReg)
            postP = math.exp(logUnnormP - logSumReg)
        prevStatePosterior[0] = post0
        prevStatePosterior[1] = postP
        effectiveObsPrec = np.empty(runLimit)
        for k in range(runLimit):
            obsVar = varMuArrPrev[k] + sigmaVis**2 + sigmaMotor**2
            effectiveObsPrec[k] = 1.0 / max(obsVar, EPS)
        newMu = np.full(maxRun, muPrior)
        newPrec = np.full(maxRun, priorPrec)
        for k in range(runLimit):
            if k == 0:
                postPrec = priorPrec + effectiveObsPrec[0]
                postMu = (priorPrec * muPrior + effectiveObsPrec[0] * deltaObs) / postPrec
                newPrec[0] = postPrec
                newMu[0] = postMu
            else:
                oldK = k - 1
                postPrec = precRun[oldK] + effectiveObsPrec[oldK]
                postMu = (precRun[oldK] * muRun[oldK] + effectiveObsPrec[oldK] * deltaObs) / postPrec
                newPrec[k] = postPrec
                newMu[k] = postMu
        muRun = newMu
        precRun = newPrec
        logPdfS0 = logGaussianPdf(aim, 0.0, sigmaMotor)
        logPdfP = LOGZERO
        for k in range(runLimit):
            compVar = varMuArrPrev[k] + sigmaMotor**2
            logComp = logGaussianPdf(aim, -muRunPrev[k], math.sqrt(compVar))
            logPdfP = logAddExp(logPdfP, logR[t-1, k, 1] + logComp)
        logSumPrevP_forAim = LOGZERO
        for k in range(runLimit):
            logSumPrevP_forAim = logAddExp(logSumPrevP_forAim, logR[t-1, k, 1])
        logPdfP -= logSumPrevP_forAim
        logPdf = logAddExp(safeLog(pS0Norm) + logPdfS0, safeLog(pPNorm) + logPdfP)
        if mask[trial]:
            logLikelihood += logPdf
    if not math.isfinite(logLikelihood):
        return 1e9
    return -logLikelihood

class BayesianStepper:
    def __init__(self, sigmaMotor=0.1, rotation=30.0, hazard=0.01, pS0=0.99, visCoeff=0.1):
        self.sigmaMotor = sigmaMotor
        self.rotation = rotation
        self.hazard = hazard
        self.pS0 = pS0
        self.visCoeff = visCoeff
        self.initState = np.array([pS0, 1 - pS0])        
        self.sigmaMag2 = sigmaMotor**2
        self.priorVar = 180**2
        self.priorPrec = 1.0 / self.priorVar if self.priorVar > 0 else 1e-6
        self.muPrior = 0.0
        self.prevStatePosterior = self.initState.copy()
        self.trialCount = 0
        maxT = 401
        self.maxRun = 10
        self.logR = np.full((maxT, self.maxRun, 2), -np.inf)
        self.logR[0, 0, 0] = np.log(self.initState[0]) if self.initState[0] > 0 else -np.inf
        self.logR[0, 0, 1] = np.log(self.initState[1]) if self.initState[1] > 0 else -np.inf
        self.muRun = np.full(self.maxRun, 0.0)
        self.precRun = np.full(self.maxRun, self.priorPrec)
        self.epsilon = 1e-6

    def wrapAngle(self, x):
        return (x + 180) % 360 - 180

    def _safeLog(self, x):
        if x <= 0.0:
            return -1e9
        return np.log(x)

    def _logAddExp(self, a, b):
        if np.isinf(a) or np.isinf(b):
            return max(a, b)
        if a < b:
            a, b = b, a
        if a - b > 700.0:
            return a
        return a + np.log(1.0 + np.exp(b - a))

    def getPredictive(self, trialNum, currentTarget):
        predState0 = (1 - self.hazard) * self.prevStatePosterior[0] + self.hazard * self.initState[0]
        predStateP = (1 - self.hazard) * self.prevStatePosterior[1] + self.hazard * self.initState[1]
        pStateSum = predState0 + predStateP
        if pStateSum <= 0.0:
            pS0 = self.initState[0]
            pP = self.initState[1]
        else:
            pS0 = predState0 / pStateSum
            pP = predStateP / pStateSum
        var0 = self.sigmaMotor**2
        t = trialNum + 1
        runLimit = min(t, self.maxRun)
        logSumPrev_P = sp_logsumexp(self.logR[t-1, :runLimit, 1])
        pKArr_P = np.exp(self.logR[t-1, :runLimit, 1] - logSumPrev_P)
        varMuArr = np.where(self.precRun[:runLimit] > 0, 1.0 / self.precRun[:runLimit], self.priorVar)
        muAct = np.sum(pKArr_P * self.muRun[:runLimit])
        varActTemp = np.sum(pKArr_P * (varMuArr + self.muRun[:runLimit]**2))
        varAct = max(varActTemp - muAct**2, 0.0)
        expectedPert = pP * muAct
        plannedOffset = -expectedPert
        plannedAimAngle = self.wrapAngle(currentTarget + plannedOffset)
        return {
            'pS0': pS0,
            'pP': pP,
            'pG': pP,
            'pD': 0.0,
            'var0': var0,
            'plannedAimAngle': plannedAimAngle,
            'muAct': muAct,
            'varAct': varAct,
            'pKArr': pKArr_P.copy(),
            'muRun': self.muRun[:runLimit].copy(),
            'varMuArr': varMuArr.copy(),
            'runLimit': runLimit
        }

    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        sigmaVis = self.visCoeff * abs(deltaObs)
        effectiveSigma = math.sqrt(self.sigmaMotor**2 + sigmaVis**2)
        t = trialNum + 1
        runLimit = min(t, self.maxRun)
        varMuArr = np.where(self.precRun[:runLimit] > 0, 1.0 / self.precRun[:runLimit], self.priorVar)
        predVarArr = effectiveSigma**2 + varMuArr
        logPredLik = vectorizedLogGaussianPdf(deltaObs, self.muRun[:runLimit], np.sqrt(np.maximum(predVarArr, 0.0)))
        priorPredLik = logGaussianPdf(deltaObs, self.muPrior, math.sqrt(max(effectiveSigma**2 + self.priorVar, 0.0)))
                      
        unnorm_P = np.full(runLimit, -np.inf)
        unnorm_P[0] = priorPredLik + np.log(self.hazard) + np.log(self.initState[1])
        unnorm_P[1:runLimit] = self.logR[t-1, :runLimit-1, 1] + logPredLik[:runLimit-1] + np.log(1 - self.hazard)
                       
        logPredLik_S0 = logGaussianPdf(deltaObs, 0.0, effectiveSigma)
        unnorm_S0 = np.full(runLimit, -np.inf)
        unnorm_S0[0] = logPredLik_S0 + np.log(self.hazard) + np.log(self.initState[0])
        unnorm_S0[1:runLimit] = self.logR[t-1, :runLimit-1, 0] + logPredLik_S0 + np.log(1 - self.hazard)
                                
        unnorm_all = np.concatenate((unnorm_S0, unnorm_P))
        logSum = sp_logsumexp(unnorm_all)
        self.logR[t, :runLimit, 0] = unnorm_S0 - logSum
        self.logR[t, :runLimit, 1] = unnorm_P - logSum
        logSumPrev_P = sp_logsumexp(self.logR[t-1, :runLimit, 1])
        logLik0 = logGaussianPdf(deltaObs, 0.0, effectiveSigma)
        logLikGK = vectorizedLogGaussianPdf(deltaObs, self.muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikG = sp_logsumexp(self.logR[t-1, :runLimit, 1] + logLikGK) - logSumPrev_P
        logMargLikP = logMargLikG
        predDict = self.getPredictive(trialNum, currentTarget)
        pS0 = predDict['pS0']
        pP = predDict['pP']
        logUnnorm0 = logLik0 + self._safeLog(pS0)
        logUnnormP = logMargLikP + self._safeLog(pP)
        if np.isinf(logUnnorm0) and logUnnorm0 < 0:
            logUnnorm0 = -np.inf
        if np.isinf(logUnnormP) and logUnnormP < 0:
            logUnnormP = -np.inf
        logSumReg = sp_logsumexp([logUnnorm0, logUnnormP])
        if not math.isfinite(logSumReg):
            post0 = pS0
            postP = pP
        else:
            post0 = math.exp(logUnnorm0 - logSumReg)
            postP = math.exp(logUnnormP - logSumReg)
        self.prevStatePosterior = np.array([post0, postP])
        effectiveObsPrec = 1.0 / np.maximum(predVarArr, self.epsilon)
        newMu = np.full(self.maxRun, 0.0)
        newPrec = np.full(self.maxRun, self.priorPrec)
        for k in range(runLimit):
            if k == 0:
                postPrec = self.priorPrec + effectiveObsPrec[0]
                postMu = (self.priorPrec * self.muPrior + effectiveObsPrec[0] * deltaObs) / postPrec
                newPrec[0] = postPrec
                newMu[0] = postMu
            else:
                oldK = k - 1
                postPrec = self.precRun[oldK] + effectiveObsPrec[oldK]
                postMu = (self.precRun[oldK] * self.muRun[oldK] + effectiveObsPrec[oldK] * deltaObs) / postPrec
                newPrec[k] = postPrec
                newMu[k] = postMu
        self.muRun = newMu
        self.precRun = newPrec

    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        expectedPert = predDict['pG'] * predDict['muAct']
        expectedAim = predDict['pS0'] * 0 - expectedPert
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim

def plotCombined(modelExplicit, mOutsSingle, humanExplicit, allAims, trials, number, plotIdentifier, fittedParams, targets, compMags, numSamples=100):
    sigmaMotor, hazard, pS0, visCoeff = fittedParams
    stepper = BayesianStepper(sigmaMotor, compMags[0], hazard, pS0, visCoeff)
    aims = []
    componentsList = []
    trialsList = []
    for trial in trials:
        deltaObs = compMags[trial]
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        pS0 = predDict['pS0']
        pG = predDict['pG']
        pD = predDict['pD']
        pKArr = predDict['pKArr']
        muRun = predDict['muRun']
        varMuArr = predDict['varMuArr']
        runLimit = predDict['runLimit']
        numS0 = int(numSamples * pS0)
        numG = int(numSamples * pG)
        numD = numSamples - numS0 - numG
        if numS0 > 0:
            samp = norm.rvs(loc=0, scale=sigmaMotor, size=numS0)
            samp = np.array([stepper.wrapAngle(s) for s in samp])
            aims.extend(samp)
            componentsList.extend(['S0'] * numS0)
            trialsList.extend([trial] * numS0)
        if numG > 0:
            ks = np.random.choice(range(runLimit), size=numG, p=pKArr)
            locsG = -muRun[ks]
            scalesG = np.sqrt(sigmaMotor**2 + varMuArr[ks])
            sampG = norm.rvs(loc=locsG, scale=scalesG, size=numG)
            sampG = np.array([stepper.wrapAngle(s) for s in sampG])
            aims.extend(sampG)
            componentsList.extend(['G'] * numG)
            trialsList.extend([trial] * numG)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    df = pd.DataFrame({'trial': trialsList, 'aim': aims, 'component': componentsList})
    fig, ax = plt.subplots(figsize=(15, 6))
    trialBins = np.arange(min(trials)-0.5, max(trials)+1.5, 1)
    binWidth = 3
    aimBins = np.arange(-180, 180 + binWidth, binWidth)
    hist, xedges, yedges = np.histogram2d(df['trial'], df['aim'], bins=(trialBins, aimBins))
    hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-10)
    hist = np.ma.masked_where(hist == 0, hist)
    im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='Greys',
                   extent=[min(trials), max(trials), -180, 180], interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Normalized Density')
    ax.scatter(trials, humanExplicit, color='red', marker='x', s=30, alpha=0.4, label='Human Aim')
    ax.hlines(y=0, xmin=0, xmax=400, linewidth=2, color='green', alpha=1)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Degrees')
    ax.set_title(f'Model Predictive Density vs Human for Participant {number}')
    legendElements = [
        Line2D([0], [0], color='grey', label='Model Predictive Density', linewidth=5, alpha=0.5),
        Line2D([0], [0], marker='x', color='darkred', label='Human Aim', markersize=6, linestyle='None', alpha=0.8)
    ]
    ax.legend(handles=legendElements)
    plt.savefig(plotIdentifier + str(number) + "_combined.png", dpi=200)
    plt.clf()
    plt.close()

def plotPointEstimates(modelExplicit, mOutsSingle, humanExplicit, trials, number, plotIdentifier, fittedParams, targets, compMags):
    sigmaMotor, hazard, pS0, visCoeff = fittedParams
    stepper = BayesianStepper(sigmaMotor, compMags[0], hazard, pS0, visCoeff)
    modelAims = []
    for trial in trials:
        deltaObs = compMags[trial]
        currentTarget = targets[trial]
        expectedAim = stepper.expectedMove(trial, currentTarget)
        modelAims.append(expectedAim)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(trials, humanExplicit, color='red', marker='x', s=30, alpha=0.4, label='Human Aim')
    ax.scatter(trials, modelAims, color='blue', marker='o', s=30, alpha=0.4, label='Model Aim')
    ax.hlines(y=0, xmin=0, xmax=400, linewidth=2, color='green', alpha=1)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Degrees')
    ax.set_title(f'Model Point Estimates vs Human Aims for Participant {number}')
    legendElements = [
        Line2D([0], [0], marker='x', color='darkred', label='Human Aim', markersize=6, linestyle='None', alpha=0.8),
        Line2D([0], [0], marker='o', color='blue', label='Model Aim', markersize=6, linestyle='None', alpha=0.8)
    ]
    ax.legend(handles=legendElements)
    plt.savefig(plotIdentifier + str(number) + "_point_estimates.png", dpi=200)
    plt.clf()
    plt.close()

def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, numPlotSamples=100, number=0, plotIdentifier=''):
    sigmaMotor, hazard, pS0, visCoeff = fittedParams
    stepper = BayesianStepper(sigmaMotor, rotation, hazard, pS0, visCoeff)
    predDicts = []
    for trial in trials:
        currentTarget = targets[trial]
        predDict = stepper.getPredictive(trial, currentTarget)
        predDicts.append(predDict)
        deltaObs = compMags[trial]
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    aims = []
    componentsList = []
    trialsList = []
    for i, t in enumerate(trials):
        predDict = predDicts[i]
        pS0 = predDict['pS0']
        pG = predDict['pG']
        pD = predDict['pD']
        pKArr = predDict['pKArr']
        muRun = predDict['muRun']
        varMuArr = predDict['varMuArr']
        runLimit = predDict['runLimit']
        numS0 = int(numSamples * pS0)
        numG = int(numSamples * pG)
        numD = numSamples - numS0 - numG
        if numS0 > 0:
            samp = norm.rvs(loc=0, scale=sigmaMotor, size=numS0)
            samp = np.array([stepper.wrapAngle(s) for s in samp])
            aims.extend(samp)
            componentsList.extend(['S0'] * numS0)
            trialsList.extend([t] * numS0)
        if numG > 0:
            ks = np.random.choice(range(runLimit), size=numG, p=pKArr)
            locsG = -muRun[ks]
            scalesG = np.sqrt(sigmaMotor**2 + varMuArr[ks])
            sampG = norm.rvs(loc=locsG, scale=scalesG, size=numG)
            sampG = np.array([stepper.wrapAngle(s) for s in sampG])
            aims.extend(sampG)
            componentsList.extend(['G'] * numG)
            trialsList.extend([t] * numG)
    df = pd.DataFrame({'trial': trialsList, 'aim': aims, 'component': componentsList})
    palette = {'S0': 'green', 'G': 'blue'}
    fig, ax = plt.subplots(figsize=(60, 15))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=.5, legend=False, ax=ax)
    for collection in ax.collections:
        if isinstance(collection, mcoll.PolyCollection):
            for path in collection.get_paths():
                vertices = path.vertices
                center = vertices[:, 0].mean()
                vertices[vertices[:, 0] < center, 0] = center
    ax.scatter(trials, humanAims, color='black', label='Human Aims', zorder=2, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Aim (degrees)')
    ax.set_title('Separate Model Predicted Aim Distributions vs Human Aims')
    ax.set_xticks(trials)
    ax.set_xticklabels(trials)
    legendHandles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(handles=legendHandles)
    plt.tight_layout()
    plt.savefig(plotIdentifier + str(number) + "testViolin.png", dpi=100)
    plt.clf()
    plt.close()

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

def plotSignFlipErrorProbability(dataList, fittedParamsList, plotIdentifier):
    allDists = []
    allProbsModel = []
    allProbsHuman = []
    for i, data in enumerate(dataList):
        allAims, mask, trials, _, compMags, pp, conVal, phases, targets, uniqueTargets, _, _, humanExplicit = data
        rawX = fittedParamsList[i]
        logSigmaMotor, logHazard, logitPS0, logVisCoeff = rawX
        sigmaMotor = np.exp(logSigmaMotor)
        hazard = 1 / (1 + np.exp(-logHazard))
        pS0 = 1 / (1 + np.exp(-logitPS0))
        visCoeff = np.exp(logVisCoeff)
        stepper = BayesianStepper(sigmaMotor, conVal, hazard, pS0, visCoeff)
        postS0List = []
        observedSignList = []
        modelAims = []
        for trial in trials:
            currentTarget = targets[trial]
            deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
            effectiveObs = deltaObs
            observedSign = np.sign(effectiveObs) if abs(effectiveObs) > 0 else 0
            observedSignList.append(observedSign)
            modelAim = stepper.expectedMove(trial, currentTarget)
            modelAims.append(modelAim)
            stepper.updatePosteriors(trial, deltaObs, currentTarget)
            postS0List.append(stepper.prevStatePosterior[0])
        postS0Array = np.array(postS0List)
        rotationIndices = np.where(np.array([p.lower() == 'rotation' for p in phases]))[0]
        if len(rotationIndices) == 0:
            continue
        rotationStart = rotationIndices[0]
        cpIndices = np.where(postS0Array < 0.5)[0]
        cpIndicesAfter = cpIndices[cpIndices >= rotationStart]
        if len(cpIndicesAfter) == 0:
            continue
        cpTrial = cpIndicesAfter[0]
        for j in range(1, 6):
            tt = cpTrial + j
            if tt >= len(trials):
                break
            prevTarget = targets[tt - 1]
            currTarget = targets[tt]
            dist = np.abs(stepper.wrapAngle(currTarget - prevTarget))
            dist = min(dist, 360 - dist)
            obsSign = observedSignList[tt]
            deltaObsTt = conVal if phases[tt].lower() == 'rotation' else 0.0
            effective = deltaObsTt
            if deltaObsTt == 0 or obsSign == 0:
                continue
            absEffective = abs(effective)
            lowerBound = 0.3 * absEffective
            upperBound = 1.7 * absEffective
            modelAim = modelAims[tt]
            absModelAim = abs(modelAim)
            signModel = np.sign(modelAim)
            probErrorModel = (lowerBound <= absModelAim <= upperBound) and (signModel == obsSign)
            probErrorModel = 0 if signModel == 0 else probErrorModel
                                     
            humanComp = humanExplicit[tt]
            absHumanComp = abs(humanComp)
            signHuman = np.sign(humanComp)
            probErrorHuman = 1 if (absHumanComp >= lowerBound and
                                   absHumanComp <= upperBound and
                                   signHuman == obsSign) else 0
            probErrorHuman = 0 if signHuman == 0 else probErrorHuman
            allDists.append(dist)
            allProbsModel.append(probErrorModel)
            allProbsHuman.append(probErrorHuman)
    if allDists:
        dfModel = pd.DataFrame({'Distance': allDists, 'Probability': allProbsModel, 'Type': 'Model'})
        dfHuman = pd.DataFrame({'Distance': allDists, 'Probability': allProbsHuman, 'Type': 'Human'})
        df = pd.concat([dfModel, dfHuman])
        grouped = df.groupby(['Distance', 'Type'])['Probability'].mean().reset_index()
        grouped = grouped.sort_values('Distance')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=grouped, x='Distance', y='Probability', hue='Type', marker='o')
        plt.xlabel('Absolute Distance to Previous Target (degrees)')
        plt.ylabel('Average Probability/Frequency of Sign Flip Error')
        plt.title('Average Sign Flip Error Probability (Model) and Frequency (Human) vs Distance (Aggregated over First Five Trials)')
        plt.savefig(plotIdentifier + "signFlipError.png", dpi=100)
        plt.clf()
        plt.close()

class Objective:
    def __init__(self, allAims, mask, trials, phases, rotation, targets):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
        self.isRotation = np.array([p.lower() == 'rotation' for p in phases], dtype=bool)

    def __call__(self, params):
        return computeNegLl(params, self.allAims, self.mask, self.trials, self.isRotation, self.rotation, self.targets)

def fitSingle(data, boundsSingle, popSizeMultiplier):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets, plotIdentifier, popSizeMultiplier, humanExplicit = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return np.zeros(4), 0.0, 0.0
    boundsArray = np.array(boundsSingle)
    maxRestarts = 1
    bestValue = np.inf
    bestX = None
    restart = 0
    globalSinceBest = 0
    while restart < maxRestarts:
        popSize = int(512 * popSizeMultiplier)
        sigma = 8 * np.log(1 + popSizeMultiplier)
        np.random.seed(33 + restart)
        if maxRestarts == 1:
            mean = (boundsArray[:, 0] + boundsArray[:, 1]) / 2
            mean[2] = 10
            mean[1] = -10
            mean[3] = np.log(0.2)
        else:
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            mean[2] = 10
            mean[1] = -10
            mean[3] = np.log(0.2)
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=33 + restart)
        es.tolfun = 1e-2
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 25:
            xSamples = [es.ask() for _ in range(es.population_size)]
            fValues = [objFunc(x) for x in xSamples]
            solutions = list(zip(xSamples, fValues))
            es.tell(solutions)
            currentBest = min(solutions, key=lambda s: s[1])
            if currentBest[1] < bestValue:
                print(pp, restart, iteration, currentBest[1], currentBest[0], globalSinceBest, sinceBest)
                if currentBest[1] < bestValue * 0.9999:
                    globalSinceBest = 0
                else:
                    globalSinceBest += 1
                bestValue = currentBest[1]
                bestX = currentBest[0]
            else:
                globalSinceBest += 1
            if currentBest[1] < bestInRun:
                if currentBest[1] < bestInRun * 0.9999:
                    sinceBest = 0
                else:
                    sinceBest += 1
                bestInRun = currentBest[1]
            else:
                sinceBest += 1
            iteration += 1
        print(pp, restart, popSize, globalSinceBest, iteration)
        restart += 1
    result = minimize(objFunc, bestX, bounds=boundsSingle, method='L-BFGS-B')
    if result.fun < bestValue:
        bestValue = result.fun
        bestX = result.x
    negll = bestValue
    return bestX, bestValue, negll

class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, plotIdentifier='', numCores=multiprocessing.cpu_count()//2, popSizeMultiplier=1):
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
        self.numCores = numCores
        self.plotIdentifier = plotIdentifier
        self.popSizeMultiplier = popSizeMultiplier
        self.rmsesDist = []
        self.rSquaredsDist = []
        self.rSquareds = []

    def fitRot(self, numCores=multiprocessing.cpu_count()//2):
        if self.condition != 'none':
            participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            self.dat = self.df[self.df['participantNum'].isin(participantsInCondition)]
        uniqP = self.dat['participantNum'].unique()
        self.participantNums = uniqP
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
        self.rmsesDist = np.zeros(numPpTotal)
        self.rSquaredsDist = np.zeros(numPpTotal)
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)]
            humanExplicit = pDat['aim'].values
            allAims = humanExplicit
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            uniqueTargets = np.unique(targetPositions[~np.isnan(targetPositions)])
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions, uniqueTargets, self.plotIdentifier, self.popSizeMultiplier, humanExplicit))
        boundsSingle = [
            (np.log(1), np.log(30)),                
            (-700, 5),            
            (0, 700),           
            (np.log(0.01), np.log(1))              
        ]
        N = numPpTotal
        with multiprocessing.Pool(processes=self.numCores) as pool:
            results = pool.starmap(fitSingle, [(dataList[i], boundsSingle, self.popSizeMultiplier) for i in range(N)])
        indivParams = np.array([r[0] for r in results])
        currentNeglls = np.array([r[2] for r in results])
        numSamplesList = [np.sum(d[1]) for d in dataList]
        humanExplicitList = [d[12] for d in dataList]
        ppList = [d[5] for d in dataList]
        for i in range(N):
            rawX = indivParams[i]
            logSigmaMotor, logHazard, logitPS0, logVisCoeff = rawX
            sigmaMotor = np.exp(logSigmaMotor)
            hazard = 1 / (1 + np.exp(-logHazard))
            pS0 = 1 / (1 + np.exp(-logitPS0))
            visCoeff = np.exp(logVisCoeff)
            xs = [sigmaMotor, hazard, pS0, visCoeff]
            allAims, mask, trials, _, _, pp, conVal, phases, targets, uniqueTargets, _, _, humanExplicit = dataList[i]
            stepperSingle = BayesianStepper(sigmaMotor, conVal, hazard, pS0, visCoeff)
            mOutsSingle = np.zeros(len(trials))
            modelExplicit = np.zeros(len(trials))
            for trial in trials:
                currentTarget = targets[trial]
                predDict = stepperSingle.getPredictive(trial, currentTarget)
                expectedAim = stepperSingle.expectedMove(trial, currentTarget)
                mOutsSingle[trial] = expectedAim
                modelExplicit[trial] = expectedAim
                deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
                stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
            validAims = allAims[mask]
            validMOuts = mOutsSingle[mask]
            totErr = validAims - validMOuts
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamplesList[i]) if numSamplesList[i] > 0 else np.inf
            rSquared = computeRSquared(validAims, validMOuts)
            negLlI = currentNeglls[i]
            bicI = 4 * np.log(numSamplesList[i]) + 2 * negLlI
            plotCombined(modelExplicit, mOutsSingle, humanExplicitList[i], allAims, trials, pp, self.plotIdentifier, xs, targets, compMags)
                                                                                                                                         
            compMags = [conVal if phases[j].lower() == 'rotation' else 0.0 for j in range(len(phases))]
                                                                                                                                                   
            self.xs[i] = xs
            self.mStates[i] = mOutsSingle.tolist()
            self.rmses[i] = rmseVal
            self.negLl[i] = negLlI
            self.bics[i] = bicI
            self.allAims[i] = allAims.tolist()
            self.rSquareds[i] = rSquared
            ese = np.zeros(len(trials))
            stepperSingle = BayesianStepper(sigmaMotor, conVal, hazard, pS0, visCoeff)
            for trial in trials:
                if not mask[trial]:
                    continue
                currentTarget = targets[trial]
                predDict = stepperSingle.getPredictive(trial, currentTarget)
                pS0, pG, pD = predDict['pS0'], predDict['pG'], predDict['pD']
                pKArr = predDict['pKArr']
                muRun = predDict['muRun']
                varMuArr = predDict['varMuArr']
                yT = allAims[trial]
                runLimit = predDict['runLimit']
                deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
                sigmaVis = visCoeff * math.fabs(deltaObs)
                sigmaMag2 = sigmaMotor**2
                effectiveSigmaForAim = sigmaMotor
                eseS0 = pS0 * (yT**2 + effectiveSigmaForAim**2)
                eseG = pG * np.sum(pKArr[:runLimit] * ((yT + muRun[:runLimit])**2 + sigmaMag2 + varMuArr[:runLimit]))
                ese[trial] = eseS0 + eseG
                stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
            numSamples = np.sum(mask)
            rmseDist = np.sqrt(np.sum(ese[mask]) / numSamples) if numSamples > 0 else np.inf
            ssTot = np.sum((validAims - np.mean(validAims))**2) if numSamples > 0 else 1.0
            r2Dist = 1 - np.sum(ese[mask]) / ssTot if ssTot > 0 else (1.0 if np.sum(ese[mask]) == 0 else 0.0)
            self.rmsesDist[i] = rmseDist
            self.rSquaredsDist[i] = r2Dist
        plotSignFlipErrorProbability(dataList, indivParams, self.plotIdentifier)
