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
def computeNegll(params, allAims, mask, trials, isRotation, rotation, targets, uniqueThetas):
    logHazard, logSigma1, logitPGlobal, logitPS0, logVisCoeff, logSigma0 = params
    hazard = 1 / (1 + math.exp(-logHazard))
    sigma1 = math.exp(logSigma1)
    pGlobal = 1 / (1 + math.exp(-logitPGlobal))
    pS0 = 1 / (1 + math.exp(-logitPS0))
    visCoeff = math.exp(logVisCoeff)
    sigma0 = math.exp(logSigma0)
    sigmaMag2 = sigma1**2
    priorVar = 2e4
    priorPrec = 1.0 / priorVar if priorVar > 0 else 1e-6
    muPrior = 0.0
    globalSign = 0.0
    epsilon = 1e-6
    initState = np.array([pS0, (1 - pS0) * pGlobal, (1 - pS0) * (1 - pGlobal)])
    prevStatePosterior = initState.copy()
    logLikelihood = 0.0
    numTrials = len(trials)
    maxT = numTrials + 1
    maxRun = 200
    logR = np.full((maxT, maxRun), -1e9)
    logR[0, 0] = 0.0
    muRun = np.full(maxRun, 0.0)
    precRun = np.full(maxRun, priorPrec)
    m = len(uniqueThetas)
    thetaToIdx = Dict.empty(key_type=nb.float64, value_type=nb.int64)
    for i in range(m):
        thetaToIdx[uniqueThetas[i]] = i
    th1 = np.zeros((m, m))
    th2 = np.zeros((m, m))
    for i in range(m):
        th1[i, :] = uniqueThetas[i] * np.ones(m)
        th2[:, i] = uniqueThetas[i] * np.ones(m)
    d = ((th1 - th2) + 180) % 360 - 180
    d = np.minimum(np.abs(d), 360 - np.abs(d))
    kPriorD = np.cos(math.pi * d / 180)
    eig = np.linalg.eigvalsh(kPriorD)
    minEig = np.min(eig)
    if minEig < -1e-10:
        nugget = -minEig + 1e-6
        kPriorD += nugget * np.eye(m)
    muVd = np.zeros(m)
    sigmaVd = kPriorD.copy()
    for idx in range(numTrials):
        t = idx + 1
        runLimit = min(t, maxRun)
        trial = trials[idx]
        currentTarget = targets[trial]
        hasTarget = currentTarget in thetaToIdx
        aim = allAims[trial]
        deltaObs = rotation if isRotation[trial] else 0.0
        deltaObs = wrapAngle(deltaObs)
        predState0 = (1 - hazard) * prevStatePosterior[0] + hazard * initState[0]
        predStateG = (1 - hazard) * prevStatePosterior[1] + hazard * initState[1]
        predStateD = (1 - hazard) * prevStatePosterior[2] + hazard * initState[2]
        pStateSum = predState0 + predStateG + predStateD
        pS0Norm = predState0 / pStateSum if pStateSum > 0 else 1.0
        pGNorm = predStateG / pStateSum if pStateSum > 0 else 0.0
        pDNorm = predStateD / pStateSum if pStateSum > 0 else 0.0
        meanD = 0.0
        varD = 0.0
        if hasTarget:
            idxTheta = thetaToIdx[currentTarget]
            meanD = muVd[idxTheta]
            varD = sigmaVd[idxTheta, idxTheta]
        if varD > 1e-6:
            z = meanD / math.sqrt(varD)
            pPos = normCdf(z, 0.0, 1.0)
        else:
            pPos = 1.0 if meanD > 0 else 0.0
        logSumPrev = logsumexp(logR[t-1, :runLimit])
        pKArr = np.exp(logR[t-1, :runLimit] - logSumPrev)
        varMuArr = np.where(precRun[:runLimit] > 0, 1.0 / precRun[:runLimit], priorVar)
        muAct = np.sum(pKArr * muRun[:runLimit])
        varActTemp = np.sum(pKArr * (varMuArr + muRun[:runLimit]**2))
        varAct = max(varActTemp - muAct**2, 0.0)
        expectedPert = pGNorm * globalSign * muAct + pDNorm * (2 * pPos - 1) * muAct
        plannedOffset = -expectedPert
        plannedAimAngle = wrapAngle(currentTarget + plannedOffset)
        effectiveDeltaObs = deltaObs
        sigmaVis = visCoeff * math.fabs(effectiveDeltaObs)
        effectiveSigma0 = math.sqrt(sigma0**2 + sigmaVis**2)
        effectiveSigma0ForAim = sigma0
        effectiveSigmaMag2 = sigmaMag2 + sigmaVis**2
        absEffective = math.fabs(effectiveDeltaObs)
        signEffective = math.copysign(1.0, effectiveDeltaObs)
        predVarArr = effectiveSigmaMag2 + varMuArr
        logPredLik = vectorizedLogGaussianPdf(absEffective, muRun[:runLimit], np.sqrt(np.maximum(predVarArr, 0.0)))
        priorPredLik = logGaussianPdf(absEffective, muPrior, math.sqrt(max(effectiveSigmaMag2 + priorVar, 0.0)))
        logChange = priorPredLik + safeLog(hazard)
        logGrowth = logR[t-1, :runLimit-1] + logPredLik[:runLimit-1] + safeLog(1 - hazard)
        logR[t, 0] = logChange
        logR[t, 1:runLimit] = logGrowth
        logSum = logsumexp(logR[t, :runLimit])
        logR[t, :runLimit] -= logSum
        logLik0 = logGaussianPdf(effectiveDeltaObs, 0.0, effectiveSigma0)
        logPdfM0 = logGaussianPdf(aim, 0.0, effectiveSigma0ForAim)
        logLikGK = vectorizedLogGaussianPdf(effectiveDeltaObs, globalSign * muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikG = logsumexp(logR[t-1, :runLimit] + logLikGK) - logSumPrev
        logPdfGK = vectorizedLogGaussianPdf(aim, -globalSign * muRun[:runLimit], np.sqrt(sigmaMag2 + varMuArr))
        logPdfMargG = logsumexp(logR[t-1, :runLimit] + logPdfGK) - logSumPrev
        logLikDPosK = vectorizedLogGaussianPdf(effectiveDeltaObs, muRun[:runLimit], np.sqrt(predVarArr))
        logLikDNegK = vectorizedLogGaussianPdf(effectiveDeltaObs, -muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikDPos = logsumexp(logR[t-1, :runLimit] + logLikDPosK) - logSumPrev
        logMargLikDNeg = logsumexp(logR[t-1, :runLimit] + logLikDNegK) - logSumPrev
        logMargLikD = logAddExp(safeLog(pPos) + logMargLikDPos, safeLog(1 - pPos) + logMargLikDNeg)
        logPdfDPosK = vectorizedLogGaussianPdf(aim, -muRun[:runLimit], np.sqrt(sigmaMag2 + varMuArr))
        logPdfDNegK = vectorizedLogGaussianPdf(aim, muRun[:runLimit], np.sqrt(sigmaMag2 + varMuArr))
        logPdfMargDPos = logsumexp(logR[t-1, :runLimit] + logPdfDPosK) - logSumPrev
        logPdfMargDNeg = logsumexp(logR[t-1, :runLimit] + logPdfDNegK) - logSumPrev
        logPdfMargD = logAddExp(safeLog(pPos) + logPdfMargDPos, safeLog(1 - pPos) + logPdfMargDNeg)
        logUnnorm0 = logLik0 + safeLog(predState0)
        logUnnormG = logMargLikG + safeLog(predStateG)
        logUnnormD = logMargLikD + safeLog(predStateD)
        if math.isinf(logUnnorm0) and logUnnorm0 < 0:
            logUnnorm0 = -1e9
        if math.isinf(logUnnormG) and logUnnormG < 0:
            logUnnormG = -1e9
        if math.isinf(logUnnormD) and logUnnormD < 0:
            logUnnormD = -1e9
        logSum = logAddExp(logUnnorm0, logAddExp(logUnnormG, logUnnormD))
        post0 = math.exp(logUnnorm0 - logSum) if logUnnorm0 > -1e9 else 0.0
        postG = math.exp(logUnnormG - logSum) if logUnnormG > -1e9 else 0.0
        postD = math.exp(logUnnormD - logSum) if logUnnormD > -1e9 else 0.0
        postSum = post0 + postG + postD
        if postSum == 0.0:
            post = np.array([predState0, predStateG, predStateD])
            postSum = post[0] + post[1] + post[2]
            if postSum > 0:
                post /= postSum
            else:
                post = np.array([1/3, 1/3, 1/3])
        else:
            post = np.array([post0, postG, postD]) / postSum
        prevStatePosterior = post
        postPert = postG + postD
        newMu = np.full(maxRun, 0.0)
        newPrec = np.full(maxRun, priorPrec)
        for k in range(runLimit):
            effectiveObsPrec = postPert * (1.0 / effectiveSigmaMag2)
            if k == 0:
                newMu[0] = (priorPrec * muPrior + absEffective * effectiveObsPrec) / (priorPrec + effectiveObsPrec)
                newPrec[0] = priorPrec + effectiveObsPrec
            else:
                oldK = k - 1
                newMu[k] = (precRun[oldK] * muRun[oldK] + absEffective * effectiveObsPrec) / (precRun[oldK] + effectiveObsPrec)
                newPrec[k] = precRun[oldK] + effectiveObsPrec
        muRun = newMu
        precRun = newPrec
        logPdf = logAddExp(safeLog(pS0Norm) + logPdfM0, logAddExp(safeLog(pGNorm) + logPdfMargG, safeLog(pDNorm) + logPdfMargD))
        if mask[trial]:
            logLikelihood += logPdf
        if absEffective > 0.0:
            globalSign = signEffective
        if postD > 0 and hasTarget:
            idxTheta = thetaToIdx[currentTarget]
            logPostPos = safeLog(pPos) + logMargLikDPos - logMargLikD
            postPos = math.exp(logPostPos) if logPostPos > -1e9 else 0.0
            hM = np.zeros(m)
            hM[idxTheta] = 1.0
            weightH = postD
            if weightH > 0:
                observedSign = math.copysign(1.0, effectiveDeltaObs)
                muV = muVd
                sigmaV = sigmaVd
                innovation = observedSign - np.dot(hM, muV)
                effectiveWeight = weightH
                effectiveWeight = max(effectiveWeight, 1e-300)
                sigmaDir2 = 4 * postPos * (1 - postPos) + 1e-6
                s = np.dot(np.dot(hM, sigmaV), hM) + sigmaDir2
                if s < 1e-10:
                    s = 1e-10
                kGain = effectiveWeight * np.dot(sigmaV, hM) / s
                muVd = muV + kGain * innovation
                sigmaVd = sigmaV - np.outer(kGain, np.dot(hM, sigmaV))
    if not math.isfinite(logLikelihood):
        return 1e9
    return -logLikelihood

class BayesianStepper:
    def __init__(self, sigma1=0.1, sigma0=0.1, rotation=30.0, hazard=0.01, uniqueTargets=None, pGlobal=0.5, pS0=0.99, visCoeff=0.1):
        self.sigma1 = sigma1
        self.sigma0 = sigma0
        self.rotation = rotation
        self.hazard = hazard
        self.pGlobal = pGlobal
        self.pS0 = pS0
        self.visCoeff = visCoeff
        self.initState = np.array([pS0, (1 - pS0) * pGlobal, (1 - pS0) * (1 - pGlobal)])
        self.sigmaMag2 = self.sigma1**2
        self.priorVar = 2e4
        self.priorPrec = 1.0 / self.priorVar if self.priorVar > 0 else 1e-6
        self.muPrior = 0.0
        self.globalSign = 0.0
        self.prevStatePosterior = self.initState.copy()
        self.trialCount = 0
        maxT = 401
        self.maxRun = 200
        self.logR = np.full((maxT, self.maxRun), -1e9)
        self.logR[0, 0] = 0.0
        self.muRun = np.full(self.maxRun, 0.0)
        self.precRun = np.full(self.maxRun, self.priorPrec)
        self.epsilon = 1e-6
        if uniqueTargets is None or len(uniqueTargets) == 0:
            self.m = 0
            self.muVd = None
            self.sigmaVd = None
            self.thetaToIdx = {}
        else:
            self.uniqueThetas = np.sort(np.unique(uniqueTargets))
            self.m = len(self.uniqueThetas)
            self.thetaToIdx = {th: i for i, th in enumerate(self.uniqueThetas)}
            th1 = np.zeros((self.m, self.m))
            th2 = np.zeros((self.m, self.m))
            for i in range(self.m):
                th1[i, :] = self.uniqueThetas[i] * np.ones(self.m)
                th2[:, i] = self.uniqueThetas[i] * np.ones(self.m)
            d = ((th1 - th2) + 180) % 360 - 180
            d = np.minimum(np.abs(d), 360 - np.abs(d))
            self.kPriorD = np.cos(np.pi * d / 180)
            eig = np.linalg.eigvalsh(self.kPriorD)
            minEig = np.min(eig)
            if minEig < -1e-10:
                nugget = -minEig + 1e-6
                self.kPriorD += nugget * np.eye(self.m)
            self.muVd = np.zeros(self.m)
            self.sigmaVd = self.kPriorD.copy()

    def wrapAngle(self, x):
        return (x + 180) % 360 - 180

    def getPredictive(self, trialNum, currentTarget):
        predState0 = (1 - self.hazard) * self.prevStatePosterior[0] + self.hazard * self.initState[0]
        predStateG = (1 - self.hazard) * self.prevStatePosterior[1] + self.hazard * self.initState[1]
        predStateD = (1 - self.hazard) * self.prevStatePosterior[2] + self.hazard * self.initState[2]
        pStateSum = predState0 + predStateG + predStateD
        pS0 = predState0 / pStateSum if pStateSum > 0 else 1.0
        pG = predStateG / pStateSum if pStateSum > 0 else 0.0
        pD = predStateD / pStateSum if pStateSum > 0 else 0.0
        var0 = self.sigma0**2
        meanD = 0.0
        varD = 0.0
        pPos = 0.5
        if self.m > 0 and currentTarget in self.thetaToIdx:
            idx = self.thetaToIdx[currentTarget]
            meanD = self.muVd[idx]
            varD = self.sigmaVd[idx, idx]
            if varD > 1e-6:
                z = meanD / math.sqrt(varD)
                pPos = normCdf(z, 0.0, 1.0)
            else:
                pPos = 1.0 if meanD > 0 else 0.0
        muAct = 0.0
        varAct = 0.0
        t = trialNum + 1
        runLimit = min(t, self.maxRun)
        logSum = logsumexp(self.logR[t-1, :runLimit])
        pKArr = np.exp(self.logR[t-1, :runLimit] - logSum)
        varMuArr = np.where(self.precRun[:runLimit] > 0, 1.0 / self.precRun[:runLimit], self.priorVar)
        muAct = np.sum(pKArr * self.muRun[:runLimit])
        varActTemp = np.sum(pKArr * (varMuArr + self.muRun[:runLimit]**2))
        varAct = max(varActTemp - muAct**2, 0.0)
        expectedPert = pG * self.globalSign * muAct + pD * (2 * pPos - 1) * muAct
        plannedOffset = -expectedPert
        plannedAimAngle = self.wrapAngle(currentTarget + plannedOffset)
        return {
            'pS0': pS0,
            'pG': pG,
            'pD': pD,
            'var0': var0,
            'plannedAimAngle': plannedAimAngle,
            'muAct': muAct,
            'varAct': varAct,
            'pPos': pPos,
            'meanD': meanD,
            'varD': varD,
            'globalSign': self.globalSign,
            'pKArr': pKArr.copy(),
            'muRun': self.muRun[:runLimit].copy(),
            'varMuArr': varMuArr.copy(),
            'runLimit': runLimit
        }

    def updatePosteriors(self, trialNum, deltaObs, currentTarget):
        self.trialCount += 1
        deltaObs = self.wrapAngle(deltaObs)
        absDeltaObs = math.fabs(deltaObs)
        signDeltaObs = math.copysign(1.0, deltaObs) if absDeltaObs > 0 else 1.0
        predDict = self.getPredictive(trialNum, currentTarget)
        pS0 = predDict['pS0']
        pG = predDict['pG']
        pD = predDict['pD']
        plannedAimAngle = predDict['plannedAimAngle']
        effectiveDeltaObs = deltaObs
        sigmaVis = self.visCoeff * math.fabs(effectiveDeltaObs)
        effectiveSigma0 = math.sqrt(self.sigma0**2 + sigmaVis**2)
        effectiveSigma0ForAim = self.sigma0
        effectiveSigmaMag2 = self.sigmaMag2 + sigmaVis**2
        absEffective = math.fabs(effectiveDeltaObs)
        signEffective = math.copysign(1.0, effectiveDeltaObs) if absEffective > 0 else 1.0
        t = trialNum + 1
        runLimit = min(t, self.maxRun)
        varMuArr = np.where(self.precRun[:runLimit] > 0, 1.0 / self.precRun[:runLimit], self.priorVar)
        predVarArr = effectiveSigmaMag2 + varMuArr
        logPredLik = vectorizedLogGaussianPdf(absEffective, self.muRun[:runLimit], np.sqrt(np.maximum(predVarArr, 0.0)))
        priorPredLik = logGaussianPdf(absEffective, self.muPrior, math.sqrt(max(effectiveSigmaMag2 + self.priorVar, 0.0)))
        logChange = priorPredLik + safeLog(self.hazard)
        logGrowth = self.logR[t-1, :runLimit-1] + logPredLik[:runLimit-1] + safeLog(1 - self.hazard)
        self.logR[t, 0] = logChange
        self.logR[t, 1:runLimit] = logGrowth
        logSum = logsumexp(self.logR[t, :runLimit])
        self.logR[t, :runLimit] -= logSum
        logSumPrev = logsumexp(self.logR[t-1, :runLimit])
        logLik0 = logGaussianPdf(effectiveDeltaObs, 0.0, effectiveSigma0)
        logLikGK = vectorizedLogGaussianPdf(effectiveDeltaObs, self.globalSign * self.muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikG = logsumexp(self.logR[t-1, :runLimit] + logLikGK) - logSumPrev
        meanD = predDict['meanD']
        varD = predDict['varD']
        if varD > 1e-6:
            z = meanD / math.sqrt(varD)
            pPos = normCdf(z, 0.0, 1.0)
        else:
            pPos = 1.0 if meanD > 0 else 0.0
        logLikDPosK = vectorizedLogGaussianPdf(effectiveDeltaObs, self.muRun[:runLimit], np.sqrt(predVarArr))
        logLikDNegK = vectorizedLogGaussianPdf(effectiveDeltaObs, -self.muRun[:runLimit], np.sqrt(predVarArr))
        logMargLikDPos = logsumexp(logR[t-1, :runLimit] + logLikDPosK) - logSumPrev
        logMargLikDNeg = logsumexp(logR[t-1, :runLimit] + logLikDNegK) - logSumPrev
        logMargLikD = logAddExp(safeLog(pPos) + logMargLikDPos, safeLog(1 - pPos) + logMargLikDNeg)
        logUnnorm0 = logLik0 + safeLog(predDict['pS0'])
        logUnnormG = logMargLikG + safeLog(predDict['pG'])
        logUnnormD = logMargLikD + safeLog(predDict['pD'])
        if math.isinf(logUnnorm0) and logUnnorm0 < 0:
            logUnnorm0 = -1e9
        if math.isinf(logUnnormG) and logUnnormG < 0:
            logUnnormG = -1e9
        if math.isinf(logUnnormD) and logUnnormD < 0:
            logUnnormD = -1e9
        logSum = logAddExp(logUnnorm0, logAddExp(logUnnormG, logUnnormD))
        post0 = math.exp(logUnnorm0 - logSum) if logUnnorm0 > -1e9 else 0.0
        postG = math.exp(logUnnormG - logSum) if logUnnormG > -1e9 else 0.0
        postD = math.exp(logUnnormD - logSum) if logUnnormD > -1e9 else 0.0
        postSum = post0 + postG + postD
        if postSum == 0.0:
            post = np.array([predDict['pS0'], predDict['pG'], predDict['pD']])
            postSum = post[0] + post[1] + post[2]
            if postSum > 0:
                post /= postSum
            else:
                post = np.array([1/3, 1/3, 1/3])
        else:
            post = np.array([post0, postG, postD]) / postSum
        self.prevStatePosterior = post
        postPert = postG + postD
        newMu = np.full(self.maxRun, 0.0)
        newPrec = np.full(self.maxRun, self.priorPrec)
        effectiveObsPrec = postPert * (1.0 / effectiveSigmaMag2)
        for k in range(runLimit):
            if k == 0:
                newMu[0] = (self.priorPrec * self.muPrior + absEffective * effectiveObsPrec) / (self.priorPrec + effectiveObsPrec)
                newPrec[0] = self.priorPrec + effectiveObsPrec
            else:
                oldK = k - 1
                newMu[k] = (self.precRun[oldK] * self.muRun[oldK] + absEffective * effectiveObsPrec) / (self.precRun[oldK] + effectiveObsPrec)
                newPrec[k] = self.precRun[oldK] + effectiveObsPrec
        self.muRun = newMu
        self.precRun = newPrec
        if absEffective > 0:
            self.globalSign = signEffective
        if postD > 0 and self.m > 0 and currentTarget in self.thetaToIdx:
            idx = self.thetaToIdx[currentTarget]
            logPostPos = safeLog(pPos) + logMargLikDPos - logMargLikD
            postPos = math.exp(logPostPos) if logPostPos > -1e9 else 0.0
            h = np.zeros(self.m)
            h[idx] = 1.0
            weightH = postD
            if weightH > 0:
                observedSign = math.copysign(1.0, effectiveDeltaObs)
                muV = self.muVd.copy()
                sigmaV = self.sigmaVd.copy()
                innovation = observedSign - np.dot(h, muV)
                effectiveWeight = weightH
                effectiveWeight = max(effectiveWeight, 1e-300)
                self.sigmaDir2 = 4 * postPos * (1 - postPos) + 1e-6
                s = np.dot(np.dot(h, sigmaV), h) + self.sigmaDir2
                if s < 1e-10:
                    s = 1e-10
                kGain = effectiveWeight * np.dot(sigmaV, h) / s
                self.muVd = muV + kGain * innovation
                self.sigmaVd = sigmaV - np.outer(kGain, np.dot(h, sigmaV))

    def expectedMove(self, trialNum, currentTarget):
        predDict = self.getPredictive(trialNum, currentTarget)
        pS0 = predDict['pS0']
        pG = predDict['pG']
        pD = predDict['pD']
        meanD = predDict['meanD']
        muAct = predDict['muAct']
        pPos = predDict['pPos']
        expectedPert = predDict['pG'] * self.globalSign * predDict['muAct'] + predDict['pD'] * (2 * pPos - 1) * predDict['muAct']
        expectedAim = predDict['pS0'] * 0 + -expectedPert
        expectedAim = self.wrapAngle(expectedAim)
        return expectedAim

def plotCombined(modelExplicit, mOutsSingle, humanExplicit, allAims, trials, number, plotIdentifier, fittedParams, targets, compMags, numSamples=100):
    hazard, sigma1, pGlobal, pS0, visCoeff, sigma0 = fittedParams
    uniqueTargets = np.unique(targets[~np.isnan(targets)])
    stepper = BayesianStepper(sigma1, sigma0, compMags[0], hazard, uniqueTargets, pGlobal, pS0, visCoeff)
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
        globalSign = predDict['globalSign']
        meanD = predDict['meanD']
        varD = predDict['varD']
        pKArr = predDict['pKArr']
        muRun = predDict['muRun']
        varMuArr = predDict['varMuArr']
        runLimit = predDict['runLimit']
        numS0 = int(numSamples * pS0)
        numG = int(numSamples * pG)
        numD = numSamples - numS0 - numG
        if numS0 > 0:
            samp = norm.rvs(loc=0, scale=sigma0, size=numS0)
            samp = np.array([wrapAngle(s) for s in samp])
            aims.extend(samp)
            componentsList.extend(['S0'] * numS0)
            trialsList.extend([trial] * numS0)
        if numG > 0:
            ks = np.random.choice(range(runLimit), size=numG, p=pKArr)
            locsG = -globalSign * muRun[ks]
            scalesG = np.sqrt(sigma1**2 + varMuArr[ks])
            sampG = norm.rvs(loc=locsG, scale=scalesG, size=numG)
            sampG = np.array([wrapAngle(s) for s in sampG])
            aims.extend(sampG)
            componentsList.extend(['G'] * numG)
            trialsList.extend([trial] * numG)
        if numD > 0:
            ks = np.random.choice(range(runLimit), size=numD, p=pKArr)
            pPos = predDict['pPos']
            signs = np.random.binomial(1, pPos, size=numD) * 2 - 1
            meansD = signs * muRun[ks]
            varsD = sigma1**2 + varMuArr[ks]
            locsD = -meansD
            scalesD = np.sqrt(np.maximum(varsD, 0.0))
            sampD = norm.rvs(loc=locsD, scale=scalesD, size=numD)
            sampD = np.array([wrapAngle(s) for s in sampD])
            aims.extend(sampD)
            signLabel = 'D'
            componentsList.extend([signLabel] * numD)
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

def violinPlotModelVsHumanAims(fittedParams, trials, compMags, humanAims, targets, rotation=30.0, numSamples=100, numPlotSamples=100, number=0, plotIdentifier=''):
    hazard, sigma1, pGlobal, pS0, visCoeff, sigma0 = fittedParams
    uniqueTargets = np.unique(targets[~np.isnan(targets)])
    stepper = BayesianStepper(sigma1, sigma0, rotation, hazard, uniqueTargets, pGlobal, pS0, visCoeff)
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
        globalSign = predDict['globalSign']
        meanD = predDict['meanD']
        varD = predDict['varD']
        pKArr = predDict['pKArr']
        muRun = predDict['muRun']
        varMuArr = predDict['varMuArr']
        runLimit = predDict['runLimit']
        numS0 = int(numSamples * pS0)
        numG = int(numSamples * pG)
        numD = numSamples - numS0 - numG
        if numS0 > 0:
            samp = norm.rvs(loc=0, scale=sigma0, size=numS0)
            samp = np.array([wrapAngle(s) for s in samp])
            aims.extend(samp)
            componentsList.extend(['S0'] * numS0)
            trialsList.extend([t] * numS0)
        if numG > 0:
            ks = np.random.choice(range(runLimit), size=numG, p=pKArr)
            locsG = -globalSign * muRun[ks]
            scalesG = np.sqrt(sigma1**2 + varMuArr[ks])
            sampG = norm.rvs(loc=locsG, scale=scalesG, size=numG)
            sampG = np.array([wrapAngle(s) for s in sampG])
            aims.extend(sampG)
            componentsList.extend(['G'] * numG)
            trialsList.extend([t] * numG)
        if numD > 0:
            ks = np.random.choice(range(runLimit), size=numD, p=pKArr)
            if varD > 1e-6:
                z = meanD / math.sqrt(varD)
                pPos = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            else:
                pPos = 1.0 if meanD > 0 else 0.0
            signs = np.random.binomial(1, pPos, size=numD) * 2 - 1
            meansD = signs * muRun[ks]
            varsD = sigma1**2 + varMuArr[ks]
            locsD = -meansD
            scalesD = np.sqrt(np.maximum(varsD, 0.0))
            sampD = norm.rvs(loc=locsD, scale=scalesD, size=numD)
            sampD = np.array([wrapAngle(s) for s in sampD])
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
        logHazard, logSigma1, logitPGlobal, logitPS0, logVisCoeff, logSigma0 = rawX
        hazard = 1 / (1 + np.exp(-logHazard))
        sigma1 = np.exp(logSigma1)
        pGlobal = 1 / (1 + np.exp(-logitPGlobal))
        pS0 = 1 / (1 + np.exp(-logitPS0))
        visCoeff = np.exp(logVisCoeff)
        sigma0 = np.exp(logSigma0)
        stepper = BayesianStepper(sigma1, sigma0, conVal, hazard, uniqueTargets, pGlobal, pS0, visCoeff)
        postS0List = []
        predPDList = []
        predPPosList = []
        observedSignList = []
        for trial in trials:
            currentTarget = targets[trial]
            predDict = stepper.getPredictive(trial, currentTarget)
            predPDList.append(predDict['pD'])
            pPos = predDict['pPos']
            predPPosList.append(pPos)
            deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
            effectiveObs = deltaObs
            observedSign = np.sign(effectiveObs) if abs(effectiveObs) > 0 else 0
            observedSignList.append(observedSign)
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
            dist = np.abs(wrapAngle(currTarget - prevTarget))
            dist = min(dist, 360 - dist)
            pD = predPDList[tt]
            pPos = predPPosList[tt]
            obsSign = observedSignList[tt]
            deltaObsTt = conVal if phases[tt].lower() == 'rotation' else 0.0
            effectiveObs = deltaObsTt
            if deltaObsTt == 0 or obsSign == 0:
                continue
            condProbError = pPos if obsSign < 0 else (1 - pPos)
            probError = pD * condProbError
            humanComp = humanExplicit[tt]
            absEffective = abs(effectiveObs)
            lowerBound = 0.3 * absEffective
            upperBound = 1.7 * absEffective
            absHumanComp = abs(humanComp)
            probErrorH = 0 if (absHumanComp >= lowerBound and
                               absHumanComp <= upperBound and
                               np.sign(humanComp) != np.sign(effectiveObs)) else 1
            probErrorH = 0 if np.sign(humanComp) == 0 else probErrorH
            allDists.append(dist)
            allProbsModel.append(probError)
            allProbsHuman.append(probErrorH)
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
    def __init__(self, allAims, mask, trials, phases, rotation, targets, uniqueThetas):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotation = rotation
        self.targets = targets
        self.uniqueThetas = np.sort(np.unique(uniqueThetas))
        self.isRotation = np.array([p.lower() == 'rotation' for p in phases], dtype=bool)

    def __call__(self, params):
        return computeNegll(params, self.allAims, self.mask, self.trials, self.isRotation, self.rotation, self.targets, self.uniqueThetas)

def fitSingle(data, boundsSingle, popSizeMultiplier):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets, plotIdentifier, popSizeMultiplier, humanExplicit = data
    objFunc = Objective(allAims, mask, trials, phases, conVal, targets, uniqueTargets)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return np.zeros(6), 0.0, 0.0
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
            mean[2] = 0
            mean[0] = -10
            mean[3] = 10
            mean[4] = np.log(0.1)
            mean[5] = np.log(0.1)
        else:
            mean = np.random.uniform(boundsArray[:, 0], boundsArray[:, 1])
            mean[2] = 0
            mean[0] = -10
            mean[3] = 10
            mean[4] = np.log(0.1)
            mean[5] = np.log(0.1)
        es = CMA(mean=mean, sigma=sigma, bounds=boundsArray, population_size=popSize, seed=33 + restart)
        es.tolfun = 1e-2
        sinceBest = 0
        bestInRun = 1e9
        iteration = 0
        while not es.should_stop() and sinceBest < 15:
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
            (-700, 0),            
            (np.log(1), np.log(2e2)),            
            (-700, 700),               
            (1, 700),           
            (np.log(0.01), np.log(1)),             
            (np.log(0.1), np.log(5))            
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
            logHazard, logSigma1, logitPGlobal, logitPS0, logVisCoeff, logSigma0 = rawX
            hazard = 1 / (1 + np.exp(-logHazard))
            sigma1 = np.exp(logSigma1)
            pGlobal = 1 / (1 + np.exp(-logitPGlobal))
            pS0 = 1 / (1 + np.exp(-logitPS0))
            visCoeff = np.exp(logVisCoeff)
            sigma0 = np.exp(logSigma0)
            xs = [hazard, sigma1, pGlobal, pS0, visCoeff, sigma0]
            allAims, mask, trials, _, _, pp, conVal, phases, targets, uniqueTargets, _, _, humanExplicit = dataList[i]
            stepperSingle = BayesianStepper(sigma1, sigma0, conVal, hazard, uniqueTargets, pGlobal, pS0, visCoeff)
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
            bicI = 6 * np.log(numSamplesList[i]) + 2 * negLlI
            plotCombined(modelExplicit, mOutsSingle, humanExplicitList[i], allAims, trials, pp, self.plotIdentifier, xs, targets, compMags)
            compMags = [conVal if phases[j].lower() == 'rotation' else 0.0 for j in range(len(phases))]
            violinPlotModelVsHumanAims(xs, trials, compMags, humanExplicitList[i], targets, conVal, number=pp, plotIdentifier=self.plotIdentifier)
            self.xs[i] = xs
            self.mStates[i] = mOutsSingle.tolist()
            self.rmses[i] = rmseVal
            self.negLl[i] = negLlI
            self.bics[i] = bicI
            self.allAims[i] = allAims.tolist()
            self.rSquareds[i] = rSquared
            ese = np.zeros(len(trials))
            stepperSingle = BayesianStepper(sigma1, sigma0, conVal, hazard, uniqueTargets, pGlobal, pS0, visCoeff)
            for trial in trials:
                if not mask[trial]:
                    continue
                currentTarget = targets[trial]
                predDict = stepperSingle.getPredictive(trial, currentTarget)
                pS0, pG, pD = predDict['pS0'], predDict['pG'], predDict['pD']
                globalSign = predDict['globalSign']
                pKArr = predDict['pKArr']
                muRun = predDict['muRun']
                varMuArr = predDict['varMuArr']
                pPos = predDict['pPos']
                yT = allAims[trial]
                runLimit = predDict['runLimit']
                deltaObs = conVal if phases[trial].lower() == 'rotation' else 0.0
                sigmaVis = visCoeff * math.fabs(deltaObs)
                sigmaMag2 = sigma1**2
                effectiveSigma0ForAim = sigma0
                eseS0 = pS0 * (yT**2 + effectiveSigma0ForAim**2)
                eseG = pG * np.sum(pKArr[:runLimit] * ((yT + globalSign * muRun[:runLimit])**2 + sigmaMag2 + varMuArr[:runLimit]))
                eseD = pD * np.sum(pKArr[:runLimit] * (
                    pPos * ((yT + muRun[:runLimit])**2 + sigmaMag2 + varMuArr[:runLimit]) +
                    (1 - pPos) * ((yT - muRun[:runLimit])**2 + sigmaMag2 + varMuArr[:runLimit])
                ))
                ese[trial] = eseS0 + eseG + eseD
                stepperSingle.updatePosteriors(trial, deltaObs, currentTarget)
            numSamples = np.sum(mask)
            rmseDist = np.sqrt(np.sum(ese[mask]) / numSamples) if numSamples > 0 else np.inf
            ssTot = np.sum((validAims - np.mean(validAims))**2) if numSamples > 0 else 1.0
            r2Dist = 1 - np.sum(ese[mask]) / ssTot if ssTot > 0 else (1.0 if np.sum(ese[mask]) == 0 else 0.0)
            self.rmsesDist[i] = rmseDist
            self.rSquaredsDist[i] = r2Dist
        plotSignFlipErrorProbability(dataList, indivParams, self.plotIdentifier)