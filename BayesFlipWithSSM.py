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
def wrapAngleArr(angles, center):
    return (angles - center + 180) % 360 - 180
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
EPS = 1e-12
LOGZERO = -1e9
TWO_PI = 2.0 * math.pi
@njit(cache=True)
def computeNegLl(params, allAims, mask, trials, isRotation, rotation, targets, uniqueThetas):
    logSigmaMotor, logHazard, logitPGlobal, logitPS0, logVisCoeff, logA, logB, logSigmaGen = params
    sigmaMotor = math.exp(logSigmaMotor)
    sigmaAim2 = sigmaMotor**2
    hazard = 1.0 / (1.0 + math.exp(-logHazard))
    pGlobal = 1.0 / (1.0 + math.exp(-logitPGlobal))
    pS0 = 1.0 / (1.0 + math.exp(-logitPS0))
    visCoeff = math.exp(logVisCoeff)
    a = math.exp(logA)
    b = math.exp(logB)
    sigmaGen = math.exp(logSigmaGen)
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
    pGlobalRun = np.full(maxRun, pGlobal)
    globalSign = 0.0
    logLikelihood = 0.0
    thetaToIdx = Dict.empty(key_type=nb.types.float64, value_type=nb.types.int64)
    m = len(uniqueThetas)
    if m > 0:
        for i in range(m):
            thetaToIdx[uniqueThetas[i]] = i
    numConfig = 1 << m if m <= 15 else 0
    if m > 15:
        return 1e9
    S = np.zeros((numConfig, m))
    for c in range(numConfig):
        for i in range(m):
            S[c, i] = 1.0 if (c & (1 << i)) else -1.0
    J = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            d = math.fabs(uniqueThetas[i] - uniqueThetas[j])
            d = min(d, 360 - d)
            J[i, j] = J[j, i] = math.cos(math.pi * d / 180.0)
    logPrior = np.zeros(numConfig)
    for c in range(numConfig):
        energy = 0.0
        for i in range(m):
            for j in range(i):
                energy += J[i, j] * S[c, i] * S[c, j]
        logPrior[c] = energy
    logZ = logsumexp(logPrior)
    logProbS = logPrior - logZ
    step = 1.0
    angles = np.arange(0.0, 360.0, step, dtype=np.float64)
    xVec = np.zeros(len(angles), dtype=np.float64)
    for idx in range(numTrials):
        t = idx + 1
        runLimit = t if t < maxRun else maxRun
        trial = trials[idx]
        aim = allAims[trial]
        deltaObs = rotation if isRotation[trial] else 0.0
        deltaObs = wrapAngle(deltaObs)
        currentTarget = targets[trial]
        hasTarget = currentTarget in thetaToIdx
        pPos = 0.5
        if hasTarget and m > 0:
            idxTheta = thetaToIdx[currentTarget]
            logSumPos = LOGZERO
            for c in range(numConfig):
                if S[c, idxTheta] > 0.0:
                    logSumPos = logAddExp(logSumPos, logProbS[c])
            pPos = math.exp(logSumPos) if logSumPos != LOGZERO else 0.0
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
        logSumPrevP = LOGZERO
        for k in range(runLimit):
            logSumPrevP = logAddExp(logSumPrevP, logR[t-1, k, 1])
        pKArr = np.empty(runLimit)
        for k in range(runLimit):
            pKArr[k] = math.exp(logR[t-1, k, 1] - logSumPrevP)
        muAct = 0.0
        for k in range(runLimit):
            muAct += pKArr[k] * muRunPrev[k]
        varActTemp = 0.0
        for k in range(runLimit):
            varActTemp += pKArr[k] * (varMuArrPrev[k] + muRunPrev[k]**2)
        varAct = max(varActTemp - muAct**2, 0.0)
        predictivePGglobal = 0.0
        for k in range(runLimit):
            predictivePGglobal += pKArr[k] * pGlobalRun[k]
        expectedPert = pPNorm * (predictivePGglobal * globalSign * muAct + (1.0 - predictivePGglobal) * (2.0 * pPos - 1.0) * muAct)
        plannedOffset = -expectedPert
        plannedAimAngle = wrapAngle(currentTarget + plannedOffset)
        plannedPos = (plannedAimAngle + 360.0) % 360.0
        idxMu = int(round(plannedPos / step)) % len(angles)
        implicitCompensation = xVec[int(idxMu)]
        effectiveDeltaObs = deltaObs + implicitCompensation
        absEffective = math.fabs(effectiveDeltaObs)
        signEffective = math.copysign(1.0, effectiveDeltaObs) if absEffective > 0 else 0.0
        sigmaVis = visCoeff * absEffective
        effectiveSigma = math.sqrt(sigmaVis**2 + sigmaAim2)
        logCompPArr = np.empty(runLimit)
        obsVarP = sigmaVis**2 + sigmaAim2
        for k in range(runLimit):
            compVar = varMuArrPrev[k] + obsVarP
            logCompPArr[k] = logGaussianPdf(absEffective, muRunPrev[k], math.sqrt(compVar))
        logMargLikG = LOGZERO
        for k in range(runLimit):
            obsVarG = varMuArrPrev[k] + obsVarP
            logMargLikG = logAddExp(logMargLikG, logR[t-1, k, 1] + logGaussianPdf(effectiveDeltaObs, globalSign * muRunPrev[k], math.sqrt(obsVarG)))
        logMargLikG -= logSumPrevP
        logMargLikDPos = LOGZERO
        for k in range(runLimit):
            obsVarD = varMuArrPrev[k] + obsVarP
            logMargLikDPos = logAddExp(logMargLikDPos, logR[t-1, k, 1] + logGaussianPdf(effectiveDeltaObs, muRunPrev[k], math.sqrt(obsVarD)))
        logMargLikDPos -= logSumPrevP
        logMargLikDNeg = LOGZERO
        for k in range(runLimit):
            obsVarDN = varMuArrPrev[k] + obsVarP
            logMargLikDNeg = logAddExp(logMargLikDNeg, logR[t-1, k, 1] + logGaussianPdf(effectiveDeltaObs, -muRunPrev[k], math.sqrt(obsVarDN)))
        logMargLikDNeg -= logSumPrevP
        logMargLikD = logAddExp(safeLog(pPos) + logMargLikDPos, safeLog(1.0 - pPos) + logMargLikDNeg)
        logMargLikP = logAddExp(safeLog(predictivePGglobal) + logMargLikG, safeLog(1.0 - predictivePGglobal) + logMargLikD)
        logLik0 = logGaussianPdf(effectiveDeltaObs, 0.0, effectiveSigma)
        unnormS0 = np.full(runLimit, LOGZERO)
        unnormS0[0] = logLik0 + safeLog(hazard) + safeLog(initState[0])
        for k in range(1, runLimit):
            unnormS0[k] = logR[t-1, k-1, 0] + logLik0 + safeLog(1.0 - hazard)
        priorPredLik = logGaussianPdf(absEffective, muPrior, math.sqrt(priorVar + obsVarP))
        unnormP = np.full(runLimit, LOGZERO)
        unnormP[0] = priorPredLik + safeLog(hazard) + safeLog(initState[1])
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
        logUnnormP = logMargLikP + safeLog(pPNorm)
        logSumReg = logAddExp(logUnnorm0, logUnnormP)
        if not math.isfinite(logSumReg):
            post0 = pS0Norm
            postP = pPNorm
        else:
            post0 = math.exp(logUnnorm0 - logSumReg)
            postP = math.exp(logUnnormP - logSumReg)
        prevStatePosterior[0] = post0
        prevStatePosterior[1] = postP
        pGlobalRun[0] = pGlobal
        for k in range(1, runLimit):
            oldK = k - 1
            logLikG = logGaussianPdf(effectiveDeltaObs, globalSign * muRunPrev[oldK], math.sqrt(varMuArrPrev[oldK] + obsVarP))
            logLikDPos = logGaussianPdf(effectiveDeltaObs, muRunPrev[oldK], math.sqrt(varMuArrPrev[oldK] + obsVarP))
            logLikDNeg = logGaussianPdf(effectiveDeltaObs, -muRunPrev[oldK], math.sqrt(varMuArrPrev[oldK] + obsVarP))
            logLikD = logAddExp(safeLog(pPos) + logLikDPos, safeLog(1.0 - pPos) + logLikDNeg)
            unnormG = logLikG + safeLog(pGlobalRun[oldK])
            unnormD = logLikD + safeLog(1.0 - pGlobalRun[oldK])
            logMarg = logAddExp(unnormG, unnormD)
            pGlobalRun[k] = math.exp(unnormG - logMarg) if logMarg != LOGZERO else 0.0
        effObsPrec = postP / obsVarP if obsVarP > EPS else 0.0
        newMu = np.full(maxRun, muPrior)
        newPrec = np.full(maxRun, priorPrec)
        for k in range(runLimit):
            if k == 0:
                postPrec = priorPrec + effObsPrec
                postMu = (priorPrec * muPrior + absEffective * effObsPrec) / postPrec if postPrec > 0.0 else muPrior
                newPrec[0] = postPrec
                newMu[0] = postMu
            else:
                oldK = k - 1
                postPrec = precRun[oldK] + effObsPrec
                postMu = (precRun[oldK] * muRun[oldK] + absEffective * effObsPrec) / postPrec if postPrec > 0.0 else muRun[oldK]
                newPrec[k] = postPrec
                newMu[k] = postMu
        muRun = newMu
        precRun = newPrec
        logPdfS0 = logGaussianPdf(aim, implicitCompensation, math.sqrt(sigmaAim2))
        logPdfMargG = LOGZERO
        for k in range(runLimit):
            compVar = varMuArrPrev[k] + sigmaAim2
            logComp = logGaussianPdf(aim, implicitCompensation - globalSign * muRunPrev[k], math.sqrt(compVar))
            logPdfMargG = logAddExp(logPdfMargG, logR[t-1, k, 1] + logComp)
        logPdfMargG -= logSumPrevP
        logPdfMargDPos = LOGZERO
        for k in range(runLimit):
            compVar = varMuArrPrev[k] + sigmaAim2
            logComp = logGaussianPdf(aim, implicitCompensation - muRunPrev[k], math.sqrt(compVar))
            logPdfMargDPos = logAddExp(logPdfMargDPos, logR[t-1, k, 1] + logComp)
        logPdfMargDPos -= logSumPrevP
        logPdfMargDNeg = LOGZERO
        for k in range(runLimit):
            compVar = varMuArrPrev[k] + sigmaAim2
            logComp = logGaussianPdf(aim, implicitCompensation + muRunPrev[k], math.sqrt(compVar))
            logPdfMargDNeg = logAddExp(logPdfMargDNeg, logR[t-1, k, 1] + logComp)
        logPdfMargDNeg -= logSumPrevP
        logPdfMargD = logAddExp(safeLog(pPos) + logPdfMargDPos, safeLog(1.0 - pPos) + logPdfMargDNeg)
        logPdfP = logAddExp(safeLog(predictivePGglobal) + logPdfMargG, safeLog(1.0 - predictivePGglobal) + logPdfMargD)
        logPdf = logAddExp(safeLog(pS0Norm) + logPdfS0, safeLog(pPNorm) + logPdfP)
        if mask[trial]:
            logLikelihood += logPdf
        if absEffective > 0.0:
            globalSign = signEffective
        postpGlobal = math.exp(safeLog(predictivePGglobal) + logMargLikG - logMargLikP) if safeLog(predictivePGglobal) + logMargLikG != LOGZERO else 0.0
        postD = postP * (1 - postpGlobal)
        if postD > 0 and hasTarget and m > 0:
            idxTheta = thetaToIdx[currentTarget]
            logLikS = np.full(numConfig, LOGZERO)
            for c in range(numConfig):
                s = S[c, idxTheta]
                logMargLikDS = logMargLikDPos if s > 0 else logMargLikDNeg
                logLikS[c] = logMargLikDS
            temp = logProbS + logLikS
            logZNew = logsumexp(temp)
            logProbS = temp - logZNew
        eT = effectiveDeltaObs
        d = wrapAngleArr(angles, plannedAimAngle)
        closeMask = np.abs(d) < (4.0 * sigmaGen)
        g = np.zeros(len(angles))
        for i in range(len(angles)):
            if closeMask[i]:
                g[i] = math.exp(-d[i]**2 / (2.0 * sigmaGen**2))
        for i in range(len(angles)):
            xVec[i] = a * xVec[i] - b * g[i] * eT
    if not math.isfinite(logLikelihood):
        return 1e9
    return -logLikelihood
class BayesianStepper:
    def __init__(self, sigmaMotor=0.1, rotation=30.0, hazard=0.01, pS0=0.99, visCoeff=0.1, pGlobal=0.5, a=0.99, b=0.01, sigmaGen=30.0, uniqueTargets=None):
        self.sigmaMotor = sigmaMotor
        self.sigmaAim2 = self.sigmaMotor**2
        self.rotation = rotation
        self.hazard = hazard
        self.pS0 = pS0
        self.visCoeff = visCoeff
        self.pGlobal = pGlobal
        self.a = a
        self.b = b
        self.sigmaGen = sigmaGen
        self.postpGlobal = pGlobal
        self.initState = np.array([pS0, 1 - pS0])        
        self.priorVar = 180**2
        self.priorPrec = 1.0 / self.priorVar if self.priorVar > 0 else 1e-6
        self.muPrior = 0.0
        self.prevStatePosterior = self.initState.copy()
        self.trialCount = 0
        maxT = 401
        self.maxRun = 10
        self.LOGZERO = -1e9
        self.logR = np.full((maxT, self.maxRun, 2), self.LOGZERO)
        self.logR[0, 0, 0] = np.log(self.initState[0]) if self.initState[0] > 0 else self.LOGZERO
        self.logR[0, 0, 1] = np.log(self.initState[1]) if self.initState[1] > 0 else self.LOGZERO
        self.muRun = np.full(self.maxRun, 0.0)
        self.precRun = np.full(self.maxRun, self.priorPrec)
        self.pGlobalRun = np.full(self.maxRun, pGlobal)
        self.globalSign = 0.0
        self.epsilon = 1e-6
        self.step = 1
        self.angles = np.arange(0, 360, self.step)
        self.xVec = np.zeros(len(self.angles))
        self.implicitCompensation = 0.0
        if uniqueTargets is None:
            uniqueTargets = np.array([])
        self.uniqueThetas = np.sort(np.unique(uniqueTargets))
        self.m = len(self.uniqueThetas)
        self.thetaToIdx = {th: i for i, th in enumerate(self.uniqueThetas)}
        if self.m > 15:
            raise ValueError("Too many unique targets")
        self.numConfig = 1 << self.m
        self.S = np.zeros((self.numConfig, self.m))
        for c in range(self.numConfig):
            for i in range(self.m):
                self.S[c, i] = 1 if (c & (1 << i)) else -1
        J = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(i):
                d = min(abs(self.uniqueThetas[i] - self.uniqueThetas[j]), 360 - abs(self.uniqueThetas[i] - self.uniqueThetas[j]))
                J[i, j] = J[j, i] = np.cos(np.pi * d / 180)
        logPrior = np.zeros(self.numConfig)
        for c in range(self.numConfig):
            energy = 0.0
            for i in range(self.m):
                for j in range(i):
                    energy += J[i, j] * self.S[c, i] * self.S[c, j]
            logPrior[c] = energy
        logZ = sp_logsumexp(logPrior)
        self.logProbS = logPrior - logZ
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
        var0 = self.sigmaAim2
        pPos = 0.5
        if self.m > 0 and currentTarget in self.thetaToIdx:
            idxTheta = self.thetaToIdx[currentTarget]
            logSumPos = self.LOGZERO
            for c in range(self.numConfig):
                if self.S[c, idxTheta] > 0:
                    logSumPos = self._logAddExp(logSumPos, self.logProbS[c])
            pPos = np.exp(logSumPos) if logSumPos > self.LOGZERO else 0.0
        t = trialNum + 1
        runLimit = min(t, self.maxRun)
        logSumPrevP = sp_logsumexp(self.logR[t-1, :runLimit, 1])
        pKArrP = np.exp(self.logR[t-1, :runLimit, 1] - logSumPrevP)
        pKArrP = np.nan_to_num(pKArrP, nan=0.0)
        varMuArr = np.where(self.precRun[:runLimit] > 0, 1.0 / self.precRun[:runLimit], self.priorVar)
        muAct = np.sum(pKArrP * self.muRun[:runLimit])
        varActTemp = np.sum(pKArrP * (varMuArr + self.muRun[:runLimit]**2))
        varAct = max(varActTemp - muAct**2, 0.0)
        predictivePGglobal = np.sum(pKArrP * self.pGlobalRun[:runLimit])
        expectedPert = pP * (predictivePGglobal * self.globalSign * muAct + (1 - predictivePGglobal) * (2 * pPos - 1) * muAct)
        plannedOffset = -expectedPert
        plannedAimAngle = self.wrapAngle(currentTarget + plannedOffset)
        plannedPos = (plannedAimAngle + 360) % 360
        idxMu = int(round(plannedPos / self.step)) % len(self.angles)
        self.implicitCompensation = self.xVec[idxMu]
        return {
            'pS0': pS0,
            'pP': pP,
            'pG': pP * predictivePGglobal,
            'pD': pP * (1 - predictivePGglobal),
            'var0': var0,
            'plannedAimAngle': plannedAimAngle,
            'muAct': muAct,
            'varAct': varAct,
            'pKArr': pKArrP.copy(),
            'muRun': self.muRun[:runLimit].copy(),
            'varMuArr': varMuArr.copy(),
            'runLimit': runLimit,
            'predictive_pGlobal': predictivePGglobal,
            'pPos': pPos,
            'globalSign': self.globalSign,
            'implicitCompensation': self.implicitCompensation
        }
    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        predDict = self.getPredictive(trialNum, currentTarget)
        effectiveDeltaObs = deltaObs + predDict['implicitCompensation']
        absEffective = abs(effectiveDeltaObs)
        signEffective = np.sign(effectiveDeltaObs) if absEffective > 0 else 0.0
        sigmaVis = self.visCoeff * absEffective
        effectiveSigma = np.sqrt(sigmaVis**2 + self.sigmaAim2)
        obsVarP = sigmaVis**2 + self.sigmaAim2
        t = trialNum + 1
        runLimit = min(t, self.maxRun)
        varMuArr = np.where(self.precRun[:runLimit] > 0, 1.0 / self.precRun[:runLimit], self.priorVar)
        predVarArr = obsVarP + varMuArr
        logPredLik = vectorizedLogGaussianPdf(absEffective, self.muRun[:runLimit], np.sqrt(np.maximum(predVarArr, 0.0)))
        priorPredLik = logGaussianPdf(absEffective, self.muPrior, np.sqrt(max(self.priorVar + obsVarP, 0.0)))
                      
        unnormP = np.full(runLimit, self.LOGZERO)
        unnormP[0] = priorPredLik + np.log(self.hazard) + np.log(self.initState[1])
        unnormP[1:runLimit] = self.logR[t-1, :runLimit-1, 1] + logPredLik[:runLimit-1] + np.log(1 - self.hazard)
                       
        logPredLikS0 = logGaussianPdf(effectiveDeltaObs, 0.0, effectiveSigma)
        unnormS0 = np.full(runLimit, self.LOGZERO)
        unnormS0[0] = logPredLikS0 + np.log(self.hazard) + np.log(self.initState[0])
        unnormS0[1:runLimit] = self.logR[t-1, :runLimit-1, 0] + logPredLikS0 + np.log(1 - self.hazard)
                                
        unnormAll = np.concatenate((unnormS0, unnormP))
        logSum = sp_logsumexp(unnormAll)
        self.logR[t, :runLimit, 0] = unnormS0 - logSum
        self.logR[t, :runLimit, 1] = unnormP - logSum
        logSumPrevP = sp_logsumexp(self.logR[t-1, :runLimit, 1])
        pS0 = predDict['pS0']
        pP = predDict['pP']
        predictivePGglobal = predDict['predictive_pGlobal']
        pPos = predDict['pPos']
        logLik0 = logGaussianPdf(effectiveDeltaObs, 0.0, effectiveSigma)
        logLikGK = vectorizedLogGaussianPdf(effectiveDeltaObs, self.globalSign * self.muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikG = sp_logsumexp(self.logR[t-1, :runLimit, 1] + logLikGK) - logSumPrevP
        logLikDPosK = vectorizedLogGaussianPdf(effectiveDeltaObs, self.muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikDPos = sp_logsumexp(self.logR[t-1, :runLimit, 1] + logLikDPosK) - logSumPrevP
        logLikDNegK = vectorizedLogGaussianPdf(effectiveDeltaObs, -self.muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikDNeg = sp_logsumexp(self.logR[t-1, :runLimit, 1] + logLikDNegK) - logSumPrevP
        logMargLikD = self._logAddExp(np.log(pPos) + logMargLikDPos, np.log(1 - pPos) + logMargLikDNeg)
        logMargLikP = self._logAddExp(np.log(predictivePGglobal) + logMargLikG, np.log(1 - predictivePGglobal) + logMargLikD)
        logUnnorm0 = logLik0 + self._safeLog(pS0)
        logUnnormP = logMargLikP + self._safeLog(pP)
        if np.isinf(logUnnorm0) and logUnnorm0 < 0:
            logUnnorm0 = self.LOGZERO
        if np.isinf(logUnnormP) and logUnnormP < 0:
            logUnnormP = self.LOGZERO
        logSumReg = self._logAddExp(logUnnorm0, logUnnormP)
        if not math.isfinite(logSumReg):
            post0 = pS0
            postP = pP
        else:
            post0 = math.exp(logUnnorm0 - logSumReg)
            postP = math.exp(logUnnormP - logSumReg)
        self.prevStatePosterior = np.array([post0, postP])
        self.postpGlobal = np.exp(np.log(predictivePGglobal) + logMargLikG - logMargLikP) if not np.isinf(np.log(predictivePGglobal) + logMargLikG) else 0.0
        self.pGlobalRun[0] = self.pGlobal
        for kk in range(1, runLimit):
            oldK = kk - 1
            logLikG = logGaussianPdf(effectiveDeltaObs, self.globalSign * self.muRun[oldK], np.sqrt(varMuArr[oldK] + obsVarP))
            logLikDPos = logGaussianPdf(effectiveDeltaObs, self.muRun[oldK], np.sqrt(varMuArr[oldK] + obsVarP))
            logLikDNeg = logGaussianPdf(effectiveDeltaObs, -self.muRun[oldK], np.sqrt(varMuArr[oldK] + obsVarP))
            logLikD = self._logAddExp(np.log(pPos) + logLikDPos, np.log(1 - pPos) + logLikDNeg)
            unnormG = logLikG + self._safeLog(self.pGlobalRun[oldK])
            unnormD = logLikD + self._safeLog(1 - self.pGlobalRun[oldK])
            logMarg = self._logAddExp(unnormG, unnormD)
            self.pGlobalRun[kk] = np.exp(unnormG - logMarg) if not np.isinf(unnormG - logMarg) else 0.0
        effObsPrec = postP / obsVarP if obsVarP > self.epsilon else 0.0
        newMu = np.full(self.maxRun, 0.0)
        newPrec = np.full(self.maxRun, self.priorPrec)
        for k in range(runLimit):
            if k == 0:
                postPrec = self.priorPrec + effObsPrec
                postMu = (self.priorPrec * self.muPrior + absEffective * effObsPrec) / postPrec if postPrec > 0 else self.muPrior
                newPrec[0] = postPrec
                newMu[0] = postMu
            else:
                oldK = k - 1
                postPrec = self.precRun[oldK] + effObsPrec
                postMu = (self.precRun[oldK] * self.muRun[oldK] + absEffective * effObsPrec) / postPrec if postPrec > 0 else self.muRun[oldK]
                newPrec[k] = postPrec
                newMu[k] = postMu
        self.muRun = newMu
        self.precRun = newPrec
        if absEffective > 0.0:
            self.globalSign = signEffective
        plannedAimAngle = predDict['plannedAimAngle']
        eT = effectiveDeltaObs
        d = self.wrapAngle(self.angles - plannedAimAngle)
        closeMask = np.abs(d) < (4 * self.sigmaGen)
        g = np.zeros(len(self.angles))
        g[closeMask] = np.exp( -d[closeMask]**2 / (2 * self.sigmaGen**2 ))
        self.xVec = self.a * self.xVec - self.b * g * eT
        postD = postP * (1 - self.postpGlobal)
        if postD > 0 and self.m > 0 and currentTarget in self.thetaToIdx:
            idxTheta = self.thetaToIdx[currentTarget]
            logLikS = np.full(self.numConfig, self.LOGZERO)
            for c in range(self.numConfig):
                s = self.S[c, idxTheta]
                logMargLikDS = logMargLikDPos if s > 0 else logMargLikDNeg
                logLikS[c] = logMargLikDS
            temp = self.logProbS + logLikS
            logZNew = sp_logsumexp(temp)
            self.logProbS = temp - logZNew
    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        expectedPert = predDict['pP'] * (predDict['predictive_pGlobal'] * predDict['globalSign'] * predDict['muAct'] + (1 - predDict['predictive_pGlobal']) * (2 * predDict['pPos'] - 1) * predDict['muAct'])
        expectedAim = -expectedPert
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim
def plotCombined(modelExplicit, modelImps, mOutsSingle, humanExplicit, humanImps, allAims, trials, number, plotIdentifier, fittedParams, targets, compMags, numSamples=100):
    sigmaMotor, hazard, pGlobal, pS0, visCoeff, a, b, sigmaGen = fittedParams
    stepper = BayesianStepper(sigmaMotor, compMags[0], hazard, pS0, visCoeff, pGlobal, a, b, sigmaGen, np.unique(targets))
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
        globalSign = predDict['globalSign']
        pPos = predDict['pPos']
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
            locsG = -globalSign * muRun[ks]
            scalesG = np.sqrt(sigmaMotor**2 + varMuArr[ks])
            sampG = norm.rvs(loc=locsG, scale=scalesG, size=numG)
            sampG = np.array([stepper.wrapAngle(s) for s in sampG])
            aims.extend(sampG)
            componentsList.extend(['G'] * numG)
            trialsList.extend([trial] * numG)
        if numD > 0:
            ks = np.random.choice(range(runLimit), size=numD, p=pKArr)
            signs = np.random.binomial(1, pPos, size=numD) * 2 - 1
            locsD = -signs * muRun[ks]
            scalesD = np.sqrt(sigmaMotor**2 + varMuArr[ks])
            sampD = norm.rvs(loc=locsD, scale=scalesD, size=numD)
            sampD = np.array([stepper.wrapAngle(s) for s in sampD])
            aims.extend(sampD)
            componentsList.extend(['D'] * numD)
            trialsList.extend([trial] * numD)
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
    ax.scatter(trials, humanExplicit, color='red', marker='x', s=30, alpha=0.4, label='Human Explicit')
    ax.scatter(trials, humanImps, color='blue', marker='x', s=20, alpha=0.5, label='Human Implicit')
    ax.scatter(trials, modelImps, color='dodgerblue', marker='o', s=20, alpha=0.4, label='Model Implicit', facecolors='none', linewidths=0.8)
    ax.hlines(y=0, xmin=0, xmax=400, linewidth=2, color='green', alpha=1)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Degrees')
    ax.set_title(f'Model Predictive Density vs Human for Participant {number}')
    legendElements = [
        Line2D([0], [0], color='grey', label='Model Predictive Density', linewidth=5, alpha=0.5),
        Line2D([0], [0], marker='x', color='darkred', label='Human Explicit', markersize=6, linestyle='None', alpha=0.8),
        Line2D([0], [0], marker='x', color='blue', label='Human Implicit', markersize=6, linestyle='None', alpha=0.8),
        Line2D([0], [0], marker='o', color='dodgerblue', label='Model Implicit', markersize=6, linestyle='None', alpha=0.4)
    ]
    ax.legend(handles=legendElements)
    plt.savefig(plotIdentifier + str(number) + "_combined.png", dpi=200)
    plt.clf()
    plt.close()
def plotPointEstimates(modelExplicit, mOutsSingle, modelImps, humanExplicit, humanImps, trials, number, plotIdentifier, fittedParams, targets, compMags):
    sigmaMotor, hazard, pGlobal, pS0, visCoeff, a, b, sigmaGen = fittedParams
    stepper = BayesianStepper(sigmaMotor, compMags[0], hazard, pS0, visCoeff, pGlobal, a, b, sigmaGen, np.unique(targets))
    modelAims = []
    modelImplicit = []
    for trial in trials:
        deltaObs = compMags[trial]
        currentTarget = targets[trial]
        expectedAim = stepper.expectedMove(trial, currentTarget)
        modelAims.append(expectedAim)
        modelImplicit.append(stepper.implicitCompensation)
        stepper.updatePosteriors(trial, deltaObs, currentTarget)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.scatter(trials, humanExplicit, color='red', marker='x', s=30, alpha=0.4, label='Human Explicit')
    ax.scatter(trials, modelAims, color='blue', marker='o', s=30, alpha=0.4, label='Model Explicit')
    ax.scatter(trials, humanImps, color='orange', marker='s', s=20, alpha=0.5, label='Human Implicit')
    ax.scatter(trials, modelImplicit, color='cyan', marker='^', s=20, alpha=0.5, label='Model Implicit')
    ax.hlines(y=0, xmin=0, xmax=400, linewidth=2, color='green', alpha=1)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Degrees')
    ax.set_title(f'Model Point Estimates vs Human Aims for Participant {number}')
    legendElements = [
        Line2D([0], [0], marker='x', color='darkred', label='Human Explicit', markersize=6, linestyle='None', alpha=0.8),
        Line2D([0], [0], marker='o', color='blue', label='Model Explicit', markersize=6, linestyle='None', alpha=0.8),
        Line2D([0], [0], marker='s', color='orange', label='Human Implicit', markersize=6, linestyle='None', alpha=0.8),
        Line2D([0], [0], marker='^', color='cyan', label='Model Implicit', markersize=6, linestyle='None', alpha=0.8)
    ]
    ax.legend(handles=legendElements)
    plt.savefig(plotIdentifier + str(number) + "_point_estimates.png", dpi=200)
    plt.clf()
    plt.close()
def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanExplicit, targets, rotation=30.0, numSamples=100, numPlotSamples=100, number=0, plotIdentifier='', humanImps=None, modelImps=None):
    sigmaMotor, hazard, pGlobal, pS0, visCoeff, a, b, sigmaGen = fittedParams
    stepper = BayesianStepper(sigmaMotor, rotation, hazard, pS0, visCoeff, pGlobal, a, b, sigmaGen, np.unique(targets))
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
        globalSign = predDict['globalSign']
        pPos = predDict['pPos']
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
            locsG = -globalSign * muRun[ks]
            scalesG = np.sqrt(sigmaMotor**2 + varMuArr[ks])
            sampG = norm.rvs(loc=locsG, scale=scalesG, size=numG)
            sampG = np.array([stepper.wrapAngle(s) for s in sampG])
            aims.extend(sampG)
            componentsList.extend(['G'] * numG)
            trialsList.extend([t] * numG)
        if numD > 0:
            ks = np.random.choice(range(runLimit), size=numD, p=pKArr)
            signs = np.random.binomial(1, pPos, size=numD) * 2 - 1
            locsD = -signs * muRun[ks]
            scalesD = np.sqrt(sigmaMotor**2 + varMuArr[ks])
            sampD = norm.rvs(loc=locsD, scale=scalesD, size=numD)
            sampD = np.array([stepper.wrapAngle(s) for s in sampD])
            aims.extend(sampD)
            signLabel = 'D+' if (2 * pPos - 1) > 0 else 'D-'
            componentsList.extend([signLabel] * numD)
            trialsList.extend([t] * numD)
    df = pd.DataFrame({'trial': trialsList, 'aim': aims, 'component': componentsList})
    palette = {'S0': 'green', 'G': 'blue', 'D+': 'red', 'D-': 'orange'}
    fig, ax = plt.subplots(figsize=(60, 15))
    sns.violinplot(data=df, x='trial', y='aim', hue='component', palette=palette, dodge=False, density_norm='count', inner=None, alpha=.5, legend=False, ax=ax)
    for collection in ax.collections:
        if isinstance(collection, mcoll.PolyCollection):
            for path in collection.get_paths():
                vertices = path.vertices
                center = vertices[:, 0].mean()
                vertices[vertices[:, 0] < center, 0] = center
    ax.scatter(trials, humanExplicit, color='black', label='Human Explicit', zorder=2, s=10)
    if humanImps is not None:
        ax.scatter(trials, humanImps, color='lightblue', label='Human Implicit', zorder=1, s=5, alpha=0.3)
    if modelImps is not None:
        ax.scatter(trials, modelImps, color='magenta', label='Model Implicit', zorder=1, s=5, alpha=0.3)
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
        allAims, mask, trials, _, compMags, pp, conVal, phases, targets, uniqueTargets, _, popSizeMultiplier, humanExplicit, humanImps = data
        rawX = fittedParamsList[i]
        logSigmaMotor, logHazard, logitPGlobal, logitPS0, logVisCoeff, logA, logB, logSigmaGen = rawX
        sigmaMotor = np.exp(logSigmaMotor)
        hazard = 1 / (1 + np.exp(-logHazard))
        pGlobal = 1 / (1 + np.exp(-logitPGlobal))
        pS0 = 1 / (1 + np.exp(-logitPS0))
        visCoeff = np.exp(logVisCoeff)
        a = np.exp(logA)
        b = np.exp(logB)
        sigmaGen = np.exp(logSigmaGen)
        stepper = BayesianStepper(sigmaMotor, conVal, hazard, pS0, visCoeff, pGlobal, a, b, sigmaGen, uniqueTargets)
        postS0List = []
        predPDList = []
        predPPosList = []
        observedSignList = []
        modelAims = []
        for trial in trials:
            currentTarget = targets[trial]
            predDict = stepper.getPredictive(trial, currentTarget)
            pD = predDict['pP'] * (1 - predDict['predictive_pGlobal'])
            predPDList.append(pD)
            pPos = predDict['pPos']
            predPPosList.append(pPos)
            deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
            humanImp = humanImps[trial]
            effective = deltaObs + humanImp
            observedSign = np.sign(effective) if abs(effective) > 0 else 0
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
            pD = predPDList[tt]
            pPos = predPPosList[tt]
            obsSign = observedSignList[tt]
            deltaObsTt = conVal if phases[tt].lower() == 'rotation' else 0.0
            humanImpTt = humanImps[tt]
            effective = deltaObsTt + humanImpTt
            if deltaObsTt == 0 or obsSign == 0:
                continue
            absEffective = abs(effective)
            lowerBound = 0.3 * absEffective
            upperBound = 1.7 * absEffective
            condProbError = pPos if obsSign < 0 else (1 - pPos)
            probErrorModel = pD * condProbError
                                     
            humanComp = humanExplicit[tt]
            absHumanComp = abs(humanComp)
            signHuman = np.sign(humanComp)
            probErrorHuman = 1 if (lowerBound <= absHumanComp <= upperBound and signHuman == obsSign) else 0
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
        self.uniqueThetas = np.unique(targets)
        self.isRotation = np.array([p.lower() == 'rotation' for p in phases], dtype=bool)
    def __call__(self, params):
        return computeNegLl(params, self.allAims, self.mask, self.trials, self.isRotation, self.rotation, self.targets, self.uniqueThetas)
def fitSingle(data, boundsSingle, popSizeMultiplier):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets, plotIdentifier, popSizeMultiplier, humanExplicit, humanImps = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return np.zeros(8), 0.0, 0.0
    boundsArray = np.array(boundsSingle)
    maxRestarts = 1
    bestValue = np.inf
    bestX = None
    restart = 0
    globalSinceBest = 0
    while restart < maxRestarts:
        popSize = int(512 * popSizeMultiplier)
        sigma = 1 * np.log(1 + popSizeMultiplier)
        np.random.seed(33 + restart)
        if maxRestarts == 1:
            mean = (boundsArray[:, 0] + boundsArray[:, 1]) / 2
            mean[3] = 10
            mean[1] = -10
            mean[2] = 0
            mean[4] = np.log(0.2)
            mean[5] = np.log(0.99)
            mean[6] = np.log(0.01)
            mean[7] = np.log(30)
        else:
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            mean[3] = 10
            mean[1] = -10
            mean[2] = 0
            mean[4] = np.log(0.2)
            mean[5] = np.log(0.99)
            mean[6] = np.log(0.01)
            mean[7] = np.log(30)
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
        self.implicitComps = []
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
        self.implicitComps = [[] for _ in range(numPpTotal)]
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
            humanImps = pDat['imp'].values
            allAims = humanExplicit + humanImps
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            uniqueTargets = np.unique(targetPositions[~np.isnan(targetPositions)])
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, self.conVal, phases, targetPositions, uniqueTargets, self.plotIdentifier, self.popSizeMultiplier, humanExplicit, humanImps))
        boundsSingle = [
            (np.log(1), np.log(30)),                
            (-700, 5),            
            (-700, 700),               
            (0, 700),           
            (np.log(0.01), np.log(1)),              
            (np.log(0.95), np.log(0.995)),       
            (np.log(0.004), np.log(0.2)),       
            (np.log(20), np.log(45))              
        ]
        N = numPpTotal
        with multiprocessing.Pool(processes=self.numCores) as pool:
            results = pool.starmap(fitSingle, [(dataList[i], boundsSingle, self.popSizeMultiplier) for i in range(N)])
        indivParams = np.array([r[0] for r in results])
        currentNeglls = np.array([r[2] for r in results])
        numSamplesList = [np.sum(d[1]) for d in dataList]
        humanExplicitList = [d[12] for d in dataList]
        humanImpsList = [d[13] for d in dataList]
        ppList = [d[5] for d in dataList]
        for i in range(N):
            rawX = indivParams[i]
            logSigmaMotor, logHazard, logitPGlobal, logitPS0, logVisCoeff, logA, logB, logSigmaGen = rawX
            sigmaMotor = np.exp(logSigmaMotor)
            hazard = 1 / (1 + np.exp(-logHazard))
            pGlobal = 1 / (1 + np.exp(-logitPGlobal))
            pS0 = 1 / (1 + np.exp(-logitPS0))
            visCoeff = np.exp(logVisCoeff)
            a = np.exp(logA)
            b = np.exp(logB)
            sigmaGen = np.exp(logSigmaGen)
            xs = [sigmaMotor, hazard, pGlobal, pS0, visCoeff, a, b, sigmaGen]
            allAims, mask, trials, _, _, pp, conVal, phases, targets, uniqueTargets, _, _, humanExplicit, humanImps = dataList[i]
            stepperSingle = BayesianStepper(sigmaMotor, conVal, hazard, pS0, visCoeff, pGlobal, a, b, sigmaGen, uniqueTargets)
            mOutsSingle = np.zeros(len(trials))
            modelExplicit = np.zeros(len(trials))
            modelImps = np.zeros(len(trials))
            for trial in trials:
                currentTarget = targets[trial]
                predDict = stepperSingle.getPredictive(trial, currentTarget)
                expectedExplicit = stepperSingle.expectedMove(trial, currentTarget)
                implicit = predDict['implicitCompensation']
                mOutsSingle[trial] = expectedExplicit + implicit
                modelExplicit[trial] = expectedExplicit
                modelImps[trial] = implicit
                deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
                stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
            validAims = allAims[mask]
            validMOuts = mOutsSingle[mask]
            totErr = validAims - validMOuts
            sumSquares = np.sum(totErr ** 2)
            rmseVal = np.sqrt(sumSquares / numSamplesList[i]) if numSamplesList[i] > 0 else np.inf
            rSquared = computeRSquared(validAims, validMOuts)
            negLlI = currentNeglls[i]
            bicI = 8 * np.log(numSamplesList[i]) + 2 * negLlI
            plotCombined(modelExplicit, modelImps, mOutsSingle, humanExplicit, humanImps, allAims, trials, pp, self.plotIdentifier, xs, targets, compMags)
                                                                                                                                                        
            compMags = [conVal if phases[j].lower() == 'rotation' else 0.0 for j in range(len(phases))]
                                                                                                                                                                                      
            self.xs[i] = xs
            self.mStates[i] = mOutsSingle.tolist()
            self.implicitComps[i] = modelImps.tolist()
            self.rmses[i] = rmseVal
            self.negLl[i] = negLlI
            self.bics[i] = bicI
            self.allAims[i] = allAims.tolist()
            self.rSquareds[i] = rSquared
            ese = np.zeros(len(trials))
            stepperSingle = BayesianStepper(sigmaMotor, conVal, hazard, pS0, visCoeff, pGlobal, a, b, sigmaGen, uniqueTargets)
            for trial in trials:
                if not mask[trial]:
                    continue
                currentTarget = targets[trial]
                predDict = stepperSingle.getPredictive(trial, currentTarget)
                pS0 = predDict['pS0']
                pG = predDict['pG']
                pD = predDict['pD']
                pKArr = predDict['pKArr']
                muRun = predDict['muRun']
                varMuArr = predDict['varMuArr']
                runLimit = predDict['runLimit']
                globalSign = predDict['globalSign']
                pPos = predDict['pPos']
                yT = allAims[trial]
                implicit = predDict['implicitCompensation']
                sigmaAim2 = sigmaMotor**2
                eseS0 = pS0 * ((yT - implicit)**2 + sigmaAim2)
                eseG = pG * np.sum(pKArr[:runLimit] * ((yT - (implicit - globalSign * muRun[:runLimit]))**2 + sigmaAim2 + varMuArr[:runLimit]))
                eseD = pD * np.sum(pKArr[:runLimit] * (pPos * ((yT - (implicit - muRun[:runLimit]))**2 + sigmaAim2 + varMuArr[:runLimit]) + (1 - pPos) * ((yT - (implicit + muRun[:runLimit]))**2 + sigmaAim2 + varMuArr[:runLimit])))
                ese[trial] = eseS0 + eseG + eseD
                deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
                stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
            numSamples = np.sum(mask)
            rmseDist = np.sqrt(np.sum(ese[mask]) / numSamples) if numSamples > 0 else np.inf
            ssTot = np.sum((validAims - np.mean(validAims))**2) if numSamples > 0 else 1.0
            r2Dist = 1 - np.sum(ese[mask]) / ssTot if ssTot > 0 else (1.0 if np.sum(ese[mask]) == 0 else 0.0)
            self.rmsesDist[i] = rmseDist
            self.rSquaredsDist[i] = r2Dist
        plotSignFlipErrorProbability(dataList, indivParams, self.plotIdentifier)