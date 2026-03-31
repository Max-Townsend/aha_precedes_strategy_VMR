
import numba as nb
from numba import njit, float64, int64, types
from numba.typed import Dict
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import numpy as np
from scipy.stats import t
from scipy.special import logsumexp as sp_logsumexp
import multiprocessing
np.random.seed(42)
SEED = 42
EPS = 1e-12
LOGZERO = -1e9
TWO_PI = 2.0 * math.pi
ALPHA_MIN = 1.0
ALPHA_MAX = 130.0
@njit(cache=False, fastmath=False)
def wrapAngle(x):
    return (x + 180) % 360 - 180
@njit(cache=False, fastmath=False)
def wrapAngleArr(angles, ref):
    return wrapAngle(angles - ref)
@njit(cache=False, fastmath=False)
def safeExp(x):
    return np.exp(np.minimum(np.maximum(x, -700.0),700.0))
@njit(cache=False, fastmath=False)
def safeLog(x):
    if x <= 0.0:
        return -1e9
    return math.log(x)
@njit(cache=False, fastmath=False)
def logAddExp(a, b):
    if a == b:
        return a + math.log(2.0)
    maxVal = max(a, b)
    minVal = min(a, b)
    if maxVal - minVal > 700.0:
        return maxVal
    return maxVal + math.log(1.0 + math.exp(minVal - maxVal))
@njit(cache=False, fastmath=False)
def logSumExp(arr):
    n = len(arr)
    if n == 0:
        return -1e9
    maxV = -np.inf
    hasInf = False
    hasNan = False
    for x in arr:
        if np.isnan(x):
            hasNan = True
        elif np.isinf(x) and x > 0:
            hasInf = True
        elif x > maxV:
            maxV = x
    if hasInf:
        return np.inf
    if hasNan:
        return np.nan
    if not math.isfinite(maxV):
        return -1e9
    if maxV < -700.0:
        return maxV
    s = 0.0
    for x in arr:
        if np.isfinite(x):
            s += safeExp(x - maxV)
    if s == 0.0:
        return maxV
    return maxV + safeLog(s)
@njit(cache=False, fastmath=False)
def logGaussianPdf(x, mu, sigma):
    if sigma <= 0.0:
        return -1e9
    var = sigma**2
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * ((x - mu)**2 / var)
@njit(cache=False, fastmath=False)
def logStudentTPdf(x, mu, scale, df):
    if scale <= 0.0 or df <= 0.0:
        return -1e9
    halfDfPlusOne = (df + 1.0) / 2.0
    halfDf = df / 2.0
    logConst = safeLog(math.gamma(halfDfPlusOne)) - safeLog(math.gamma(halfDf))\
                - 0.5 * math.log(math.pi * df) - math.log(scale)
    z = ((x - mu) / scale)**2 / df
    logTail = -halfDfPlusOne * math.log(1.0 + z)
    return logConst + logTail
@njit(cache=False, fastmath=False, error_model='numpy')
def getMatchedScale(alpha, beta, varMu, obsVarP):
    df = 2.0 * alpha
    eEta = beta / max(alpha - 1.0, 1e-10)
    varTotal = varMu + eEta + obsVarP
    if df > 2.0:
        return math.sqrt(varTotal * (df - 2.0) / df)
    else:
        return math.sqrt(varTotal) * 10.0
@njit(cache=False, fastmath=False)
def computeNegLl(params, allAims, mask, trials, isRotation, rotations, targets, uniqueThetas,
                 A=0.99, B=0.01, sigmaGen=30.0):
    logSigmaMotor, logitPGlobal, logitPLocal, logPriorStrength, visCoeff, logBetaTauShared, logHazardBeta, logAlphaTauSharedPrior = params
    sigmaMotor = math.exp(logSigmaMotor)
    sigmaAim2 = sigmaMotor**2
    priorStrength = math.exp(logPriorStrength)
    betaTauShared = math.exp(logBetaTauShared)
    baseDf =130.0
    alphaTauSharedPrior = math.exp(logAlphaTauSharedPrior)
    betaTauSharedPrior = betaTauShared
    logPG = logitPGlobal
    logPL = logitPLocal
    logPD = 0.0
    logZP = logSumExp(np.array([logPG, logPL, logPD]))
    pGlobal = math.exp(logPG - logZP)
    pLocal = math.exp(logPL - logZP)
    pDir = math.exp(logPD - logZP)
    alphaChange = 1.0
    betaNoChange = math.exp(logHazardBeta)
    alphaPert = np.array([EPS, pGlobal, pDir, pLocal])
    priorKappa = 1e-6
    priorMuG = 0.0
    priorKappaG = priorKappa
    priorMuD = 0.0
    priorKappaD = priorKappa
    priorMuL = 0.0
    priorKappaL = priorKappa
    priorAlpha = alphaTauSharedPrior
    priorBeta = betaTauSharedPrior
    initState = np.array([1.0,0.0,0.0,0.0])
    prevStatePosterior = initState.copy()
    numTrials = len(trials)
    maxT = numTrials + 1
    maxRun = 5
    logR = np.full((maxT, maxRun, 4), LOGZERO)
    logR[0, 0, :] = np.log(initState + EPS)
    muRunG = np.full(maxRun, priorMuG)
    kappaRunG = np.full(maxRun, priorKappaG)
    muRunD = np.full(maxRun, priorMuD)
    kappaRunD = np.full(maxRun, priorKappaD)
    m = len(uniqueThetas)
    muRunL = np.full((maxRun, m), priorMuL)
    kappaRunL = np.full((maxRun, m), priorKappaL)
    alphaRunG = np.full(maxRun, priorAlpha)
    betaRunG = np.full(maxRun, priorBeta)
    alphaRunD = np.full(maxRun, priorAlpha)
    betaRunD = np.full(maxRun, priorBeta)
    alphaRunL = np.full(maxRun, priorAlpha)
    betaRunL = np.full(maxRun, priorBeta)
    thetaToIdx = Dict.empty(key_type=types.float64, value_type=types.int64)
    for i in range(m):
        thetaToIdx[uniqueThetas[i]] = i
    numConfig = 1 << m if m <= 15 else 0
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
    logZ = logSumExp(logPrior)
    logProbSPrior = logPrior - logZ
    logProbSRun = np.full((maxRun, numConfig), LOGZERO)
    for k in range(maxRun):
        logProbSRun[k] = logProbSPrior
    angles = np.arange(360.0, dtype=np.float64)
    xVec = np.zeros(360, dtype=np.float64)
    plannedOffsets = np.full(numTrials, 0.0)
    implicitComps = np.full(numTrials, 0.0)
    pGs = np.full(numTrials, 0.0)
    pDs = np.full(numTrials, 0.0)
    pLs = np.full(numTrials, 0.0)
    pPoss = np.full(numTrials, 0.0)
    runLimits = np.zeros(numTrials, dtype=types.int64)
    pKArrGs = np.full((numTrials, maxRun), 0.0)
    pKArrDs = np.full((numTrials, maxRun), 0.0)
    pKArrLs = np.full((numTrials, maxRun), 0.0)
    muRunGPrevs = np.full((numTrials, maxRun), 0.0)
    muRunDPrevs = np.full((numTrials, maxRun), 0.0)
    muRunLPrevs = np.full((numTrials, maxRun), 0.0)
    alphaRunGPrevs = np.full((numTrials, maxRun), 0.0)
    alphaRunDPrevs = np.full((numTrials, maxRun), 0.0)
    alphaRunLPrevs = np.full((numTrials, maxRun), 0.0)
    betaRunGPrevs = np.full((numTrials, maxRun), 0.0)
    betaRunDPrevs = np.full((numTrials, maxRun), 0.0)
    betaRunLPrevs = np.full((numTrials, maxRun), 0.0)
    kappaRunGPrevs = np.full((numTrials, maxRun), 0.0)
    kappaRunDPrevs = np.full((numTrials, maxRun), 0.0)
    kappaRunLPrevs = np.full((numTrials, maxRun), 0.0)
    post0s = np.full(numTrials, 0.0)
    logLikelihood = 0.0
    logHalf = math.log(0.5)
    logPSignPosG = np.full(maxRun, logHalf)
    logPSignNegG = np.full(maxRun, logHalf)
    logPSignPosL = np.full((maxRun, m), logHalf)
    logPSignNegL = np.full((maxRun, m), logHalf)
    logPSignPosGPredAll = np.full((numTrials, maxRun), logHalf)
    logPSignNegGPredAll = np.full((numTrials, maxRun), logHalf)
    logPSignPosLPredAll = np.full((numTrials, maxRun, m), logHalf)
    logPSignNegLPredAll = np.full((numTrials, maxRun, m), logHalf)
    pPosKPrevs = np.full((numTrials, maxRun), 0.5)
    for idx in range(numTrials):
        t = idx + 1
        runLimit = min(t, maxRun)
        trial = trials[idx]
        aim = allAims[trial]
        deltaObs = rotations[idx] if isRotation[trial] else 0.0
        deltaObs = wrapAngle(deltaObs)
        currentTarget = targets[trial]
        idxTheta = thetaToIdx[currentTarget]
        pertPriors = alphaPert / (alphaPert.sum() + EPS)
        pChange = alphaChange / (alphaChange + betaNoChange + EPS)
        pNoChange = 1.0 - pChange
        prevStatePosterior[0] = math.exp(logSumExp(logR[t-1, :runLimit, 0]))
        prevStatePosterior[1] = math.exp(logSumExp(logR[t-1, :runLimit, 1]))
        prevStatePosterior[2] = math.exp(logSumExp(logR[t-1, :runLimit, 2]))
        prevStatePosterior[3] = math.exp(logSumExp(logR[t-1, :runLimit, 3]))
        prevStatePosterior /= prevStatePosterior.sum() + EPS
        predState = pNoChange * prevStatePosterior + pChange * pertPriors
        pS0Norm = predState[0]
        pGNorm = predState[1]
        pDNorm = predState[2]
        pLNorm = predState[3]
        logSumPrev0 = logSumExp(logR[t-1, :runLimit, 0])
        logSumPrevG = logSumExp(logR[t-1, :runLimit, 1])
        logSumPrevD = logSumExp(logR[t-1, :runLimit, 2])
        logSumPrevL = logSumExp(logR[t-1, :runLimit, 3])
        pKArrG = np.exp(logR[t-1, :runLimit, 1] - logSumPrevG)
        pKArrD = np.exp(logR[t-1, :runLimit, 2] - logSumPrevD)
        pKArrL = np.exp(logR[t-1, :runLimit, 3] - logSumPrevL)
        muGPred = np.full(runLimit, priorMuG)
        kappaGPred = np.full(runLimit, priorKappaG)
        muDPred = np.full(runLimit, priorMuD)
        kappaDPred = np.full(runLimit, priorKappaD)
        muLPred = np.full((runLimit, m), priorMuL)
        kappaLPred = np.full((runLimit, m), priorKappaL)
        alphaPredG = np.full(runLimit, priorAlpha)
        betaPredG = np.full(runLimit, priorBeta)
        alphaPredD = np.full(runLimit, priorAlpha)
        betaPredD = np.full(runLimit, priorBeta)
        alphaPredL = np.full(runLimit, priorAlpha)
        betaPredL = np.full(runLimit, priorBeta)
        logPSignPosGPred = np.full(runLimit, logHalf)
        logPSignNegGPred = np.full(runLimit, logHalf)
        logPSignPosLPred = np.full((runLimit, m), logHalf)
        logPSignNegLPred = np.full((runLimit, m), logHalf)
        logProbSPred = np.full((runLimit, numConfig), LOGZERO)
        for k in range(1, runLimit):
            if runLimit < maxRun or k < runLimit - 1:
                muGPred[k] = muRunG[k-1]
                kappaGPred[k] = kappaRunG[k-1]
                muDPred[k] = muRunD[k-1]
                kappaDPred[k] = kappaRunD[k-1]
                muLPred[k, :] = muRunL[k-1, :]
                kappaLPred[k, :] = kappaRunL[k-1, :]
                alphaPredG[k] = alphaRunG[k-1]
                betaPredG[k] = betaRunG[k-1]
                alphaPredD[k] = alphaRunD[k-1]
                betaPredD[k] = betaRunD[k-1]
                alphaPredL[k] = alphaRunL[k-1]
                betaPredL[k] = betaRunL[k-1]
                logPSignPosGPred[k] = logPSignPosG[k-1]
                logPSignNegGPred[k] = logPSignNegG[k-1]
                logPSignPosLPred[k, :] = logPSignPosL[k-1, :]
                logPSignNegLPred[k, :] = logPSignNegL[k-1, :]
                logProbSPred[k] = logProbSRun[k-1]
                eEtaG = betaRunG[k-1] / max(alphaRunG[k-1] - 1.0, 1e-10)
                varPredG = 1.0 / kappaRunG[k-1] + eEtaG
                kappaGPred[k] = 1.0 / max(varPredG, EPS)
                eEtaD = betaRunD[k-1] / max(alphaRunD[k-1] - 1.0, 1e-10)
                varPredD = 1.0 / kappaRunD[k-1] + eEtaD
                kappaDPred[k] = 1.0 / max(varPredD, EPS)
                eEtaL = betaRunL[k-1] / max(alphaRunL[k-1] - 1.0, 1e-10)
                for i in range(m):
                    varPredL = 1.0 / kappaRunL[k-1, i] + eEtaL
                    kappaLPred[k, i] = 1.0 / max(varPredL, EPS)
            else:
                pkShortG = pKArrG[runLimit-2]
                pkTailG = pKArrG[runLimit-1]
                sumPkG = pkShortG + pkTailG
                wShortG = pkShortG / sumPkG if sumPkG > EPS else 0.5
                wTailG = 1.0 - wShortG
                muShortG = muRunG[runLimit-2]
                kappaShortG = kappaRunG[runLimit-2]
                alphaShortG = alphaRunG[runLimit-2]
                betaShortG = betaRunG[runLimit-2]
                logPSignPosShortG = logPSignPosG[runLimit-2]
                logPSignNegShortG = logPSignNegG[runLimit-2]
                eEtaShortG = betaShortG / max(alphaShortG - 1.0, 1e-10)
                varShortPredG = 1.0 / kappaShortG + eEtaShortG
                muTailG = muRunG[runLimit-1]
                kappaTailG = kappaRunG[runLimit-1]
                alphaTailG = alphaRunG[runLimit-1]
                betaTailG = betaRunG[runLimit-1]
                logPSignPosTailG = logPSignPosG[runLimit-1]
                logPSignNegTailG = logPSignNegG[runLimit-1]
                eEtaTailG = betaTailG / max(alphaTailG - 1.0, 1e-10)
                varTailPredG = 1.0 / kappaTailG + eEtaTailG
                mixedMuG = wShortG * muShortG + wTailG * muTailG
                mixedVarG = wShortG * varShortPredG + wTailG * varTailPredG + wShortG * wTailG * (muShortG - muTailG)**2
                mixedVarG = max(mixedVarG, EPS)
                muGPred[k] = mixedMuG
                kappaGPred[k] = 1.0 / mixedVarG
                meanShortG = betaShortG / max(alphaShortG - 1.0, EPS) if alphaShortG > 1.0 else 0.0
                varShortG = betaShortG**2 / ((alphaShortG - 1.0)**2 * (alphaShortG - 2.0)) if alphaShortG > 2.0 else 1e12
                meanTailG = betaTailG / max(alphaTailG - 1.0, EPS) if alphaTailG > 1.0 else 0.0
                varTailG = betaTailG**2 / ((alphaTailG - 1.0)**2 * (alphaTailG - 2.0)) if alphaTailG > 2.0 else 1e12
                mixedMeanG = wShortG * meanShortG + wTailG * meanTailG
                mixedVarG = wShortG * varShortG + wTailG * varTailG + wShortG * wTailG * (meanShortG - meanTailG)**2
                mixedVarG = max(mixedVarG, EPS)
                aMixedG = 2.0 + mixedMeanG**2 / mixedVarG
                aMixedG = max(min(aMixedG, ALPHA_MAX), ALPHA_MIN)
                bMixedG = mixedMeanG * (aMixedG - 1.0)
                alphaPredG[k] = aMixedG
                betaPredG[k] = bMixedG
                pPosShortG = math.exp(logPSignPosShortG)
                pPosTailG = math.exp(logPSignPosTailG)
                mixedPPosG = wShortG * pPosShortG + wTailG * pPosTailG
                logPSignPosGPred[k] = safeLog(mixedPPosG)
                logPSignNegGPred[k] = safeLog(1.0 - mixedPPosG)
                pkShortD = pKArrD[runLimit-2]
                pkTailD = pKArrD[runLimit-1]
                sumPkD = pkShortD + pkTailD
                wShortD = pkShortD / sumPkD if sumPkD > EPS else 0.5
                wTailD = 1.0 - wShortD
                muShortD = muRunD[runLimit-2]
                kappaShortD = kappaRunD[runLimit-2]
                alphaShortD = alphaRunD[runLimit-2]
                betaShortD = betaRunD[runLimit-2]
                logProbSShort = logProbSRun[runLimit-2]
                eEtaShortD = betaShortD / max(alphaShortD - 1.0, 1e-10)
                varShortPredD = 1.0 / kappaShortD + eEtaShortD
                muTailD = muRunD[runLimit-1]
                kappaTailD = kappaRunD[runLimit-1]
                alphaTailD = alphaRunD[runLimit-1]
                betaTailD = betaRunD[runLimit-1]
                logProbSTail = logProbSRun[runLimit-1]
                eEtaTailD = betaTailD / max(alphaTailD - 1.0, 1e-10)
                varTailPredD = 1.0 / kappaTailD + eEtaTailD
                mixedMuD = wShortD * muShortD + wTailD * muTailD
                mixedVarD = wShortD * varShortPredD + wTailD * varTailPredD + wShortD * wTailD * (muShortD - muTailD)**2
                mixedVarD = max(mixedVarD, EPS)
                muDPred[k] = mixedMuD
                kappaDPred[k] = 1.0 / mixedVarD
                meanShortD = betaShortD / max(alphaShortD - 1.0, EPS) if alphaShortD > 1.0 else 0.0
                varShortD = betaShortD**2 / ((alphaShortD - 1.0)**2 * (alphaShortD - 2.0)) if alphaShortD > 2.0 else 1e12
                meanTailD = betaTailD / max(alphaTailD - 1.0, EPS) if alphaTailD > 1.0 else 0.0
                varTailD = betaTailD**2 / ((alphaTailD - 1.0)**2 * (alphaTailD - 2.0)) if alphaTailD > 2.0 else 1e12
                mixedMeanD = wShortD * meanShortD + wTailD * meanTailD
                mixedVarD = wShortD * varShortD + wTailD * varTailD + wShortD * wTailD * (meanShortD - meanTailD)**2
                mixedVarD = max(mixedVarD, EPS)
                aMixedD = 2.0 + mixedMeanD**2 / mixedVarD
                aMixedD = max(min(aMixedD, ALPHA_MAX), ALPHA_MIN)
                bMixedD = mixedMeanD * (aMixedD - 1.0)
                alphaPredD[k] = aMixedD
                betaPredD[k] = bMixedD
                psShort = safeExp(logProbSShort)
                psTail = safeExp(logProbSTail)
                mixedPs = wShortD * psShort + wTailD * psTail
                mixedPs /= np.sum(mixedPs) + EPS
                logProbSPred[k] = np.log(mixedPs + EPS)
                pkShortL = pKArrL[runLimit-2]
                pkTailL = pKArrL[runLimit-1]
                sumPkL = pkShortL + pkTailL
                wShortL = pkShortL / sumPkL if sumPkL > EPS else 0.5
                wTailL = 1.0 - wShortL
                alphaShortL = alphaRunL[runLimit-2]
                betaShortL = betaRunL[runLimit-2]
                meanShortL = betaShortL / max(alphaShortL - 1.0, EPS) if alphaShortL > 1.0 else 0.0
                varShortL = betaShortL**2 / ((alphaShortL - 1.0)**2 * (alphaShortL - 2.0)) if alphaShortL > 2.0 else 1e12
                alphaTailL = alphaRunL[runLimit-1]
                betaTailL = betaRunL[runLimit-1]
                meanTailL = betaTailL / max(alphaTailL - 1.0, EPS) if alphaTailL > 1.0 else 0.0
                varTailL = betaTailL**2 / ((alphaTailL - 1.0)**2 * (alphaTailL - 2.0)) if alphaTailL > 2.0 else 1e12
                mixedMeanL = wShortL * meanShortL + wTailL * meanTailL
                mixedVarL = wShortL * varShortL + wTailL * varTailL + wShortL * wTailL * (meanShortL - meanTailL)**2
                mixedVarL = max(mixedVarL, EPS)
                aMixedL = 2.0 + mixedMeanL**2 / mixedVarL
                aMixedL = max(min(aMixedL, ALPHA_MAX), ALPHA_MIN)
                bMixedL = mixedMeanL * (aMixedL - 1.0)
                alphaPredL[k] = aMixedL
                betaPredL[k] = bMixedL
                eEtaShortL = betaShortL / max(alphaShortL - 1.0, 1e-10)
                eEtaTailL = betaTailL / max(alphaTailL - 1.0, 1e-10)
                for ii in range(m):
                    muShortL = muRunL[runLimit-2, ii]
                    kappaShortL = kappaRunL[runLimit-2, ii]
                    logPSignPosShortL = logPSignPosL[runLimit-2, ii]
                    logPSignNegShortL = logPSignNegL[runLimit-2, ii]
                    varShortPredL = 1.0 / kappaShortL + eEtaShortL
                    muTailL = muRunL[runLimit-1, ii]
                    kappaTailL = kappaRunL[runLimit-1, ii]
                    logPSignPosTailL = logPSignPosL[runLimit-1, ii]
                    logPSignNegTailL = logPSignNegL[runLimit-1, ii]
                    varTailPredL = 1.0 / kappaTailL + eEtaTailL
                    mixedMuL = wShortL * muShortL + wTailL * muTailL
                    mixedVarL = wShortL * varShortPredL + wTailL * varTailPredL + wShortL * wTailL * (muShortL - muTailL)**2
                    mixedVarL = max(mixedVarL, EPS)
                    muLPred[k, ii] = mixedMuL
                    kappaLPred[k, ii] = 1.0 / mixedVarL
                    pPosShortL = math.exp(logPSignPosShortL)
                    pPosTailL = math.exp(logPSignPosTailL)
                    mixedPPosL = wShortL * pPosShortL + wTailL * pPosTailL
                    logPSignPosLPred[k, ii] = safeLog(mixedPPosL)
                    logPSignNegLPred[k, ii] = safeLog(1.0 - mixedPPosL)
        logProbSPred[0] = logProbSPrior
        pPosK = np.full(runLimit, 0.5)
        for k in range(runLimit):
            logSumPos = LOGZERO
            for c in range(numConfig):
                if S[c, idxTheta] > 0.0:
                    logSumPos = logAddExp(logSumPos, logProbSPred[k][c])
            pPosK[k] = math.exp(logSumPos) if logSumPos != LOGZERO else 0.5
            if not np.isfinite(pPosK[k]):
                pPosK[k] = 0.5
            pPosK[k] = min(max(0.0, pPosK[k]), 1.0)
        logPSignPosGPredAll[idx, :runLimit] = logPSignPosGPred[:runLimit]
        logPSignNegGPredAll[idx, :runLimit] = logPSignNegGPred[:runLimit]
        logPSignPosLPredAll[idx, :runLimit, :] = logPSignPosLPred[:runLimit, :]
        logPSignNegLPredAll[idx, :runLimit, :] = logPSignNegLPred[:runLimit, :]
        expectedPertGExact = np.zeros(runLimit)
        expectedPertLExact = np.zeros(runLimit)
        for k in range(runLimit):
            pPosG = math.exp(logPSignPosGPred[k])
            expectedPertGExact[k] = (2.0 * pPosG - 1.0) * muGPred[k]
            pPosL = math.exp(logPSignPosLPred[k, idxTheta])
            expectedPertLExact[k] = (2.0 * pPosL - 1.0) * muLPred[k, idxTheta]
        expectedPertG = np.sum(pKArrG * expectedPertGExact)
        expectedPertL = np.sum(pKArrL * expectedPertLExact)
        expectedPertD = 0.0
        for k in range(runLimit):
            expectedPertD += pKArrD[k] * muDPred[k] * (2.0 * pPosK[k] - 1.0)
        expectedPert = pGNorm * expectedPertG +\
                       pDNorm * expectedPertD +\
                       pLNorm * expectedPertL
        plannedOffset = -expectedPert
        meanAimAngle = wrapAngle(currentTarget + plannedOffset)
        implicitCompensation = xVec[int(meanAimAngle % 360)]
        effectiveDeltaObs = deltaObs + implicitCompensation
        absEffective = math.fabs(effectiveDeltaObs)
        sigmaVis = visCoeff * absEffective
        obsVarP = sigmaVis**2 + sigmaAim2
        logMargGK = np.full(runLimit, LOGZERO)
        logMargDK = np.full(runLimit, LOGZERO)
        logMargDPosK = np.full(runLimit, LOGZERO)
        logMargDNegK = np.full(runLimit, LOGZERO)
        logMargLK = np.full(runLimit, LOGZERO)
        logAimGK = np.full(runLimit, LOGZERO)
        logAimDK = np.full(runLimit, LOGZERO)
        logAimDPosK = np.full(runLimit, LOGZERO)
        logAimDNegK = np.full(runLimit, LOGZERO)
        logAimLK = np.full(runLimit, LOGZERO)
        for k in range(runLimit):
            alpha = alphaPredG[k]
            beta = betaPredG[k]
            df = 2.0 * alpha
            varMu = 1.0 / kappaGPred[k] if kappaGPred[k] > 1e-10 else 1e6
            scaleApprox = getMatchedScale(alpha, beta, varMu, obsVarP)
            logLikPos = logStudentTPdf(effectiveDeltaObs, muGPred[k], scaleApprox, df)
            logLikNeg = logStudentTPdf(effectiveDeltaObs, -muGPred[k], scaleApprox, df)
            logMargGK[k] = logAddExp(logPSignPosGPred[k] + logLikPos, logPSignNegGPred[k] + logLikNeg)
            scaleApproxAim = getMatchedScale(alpha, beta, varMu, sigmaAim2)
            muAimPos = implicitCompensation - muGPred[k]
            muAimNeg = implicitCompensation + muGPred[k]
            logAimPos = logStudentTPdf(aim, muAimPos, scaleApproxAim, df)
            logAimNeg = logStudentTPdf(aim, muAimNeg, scaleApproxAim, df)
            logAimGK[k] = logAddExp(logPSignPosGPred[k] + logAimPos, logPSignNegGPred[k] + logAimNeg)
            alpha = alphaPredD[k]
            beta = betaPredD[k]
            df = 2.0 * alpha
            varMu = 1.0 / kappaDPred[k] if kappaDPred[k] > 1e-10 else 1e6
            scaleApprox = getMatchedScale(alpha, beta, varMu, obsVarP)
            logMargDPosK[k] = logStudentTPdf(effectiveDeltaObs, muDPred[k], scaleApprox, df)
            logMargDNegK[k] = logStudentTPdf(effectiveDeltaObs, -muDPred[k], scaleApprox, df)
            logMargDK[k] = logAddExp(safeLog(pPosK[k]) + logMargDPosK[k], safeLog(1.0 - pPosK[k]) + logMargDNegK[k])
            scaleApproxAim = getMatchedScale(alpha, beta, varMu, sigmaAim2)
            muAimPos = implicitCompensation - muDPred[k]
            muAimNeg = implicitCompensation + muDPred[k]
            logAimDPosK[k] = logStudentTPdf(aim, muAimPos, scaleApproxAim, df)
            logAimDNegK[k] = logStudentTPdf(aim, muAimNeg, scaleApproxAim, df)
            logAimDK[k] = logAddExp(safeLog(pPosK[k]) + logAimDPosK[k], safeLog(1.0 - pPosK[k]) + logAimDNegK[k])
            alpha = alphaPredL[k]
            beta = betaPredL[k]
            df = 2.0 * alpha
            varMu = 1.0 / kappaLPred[k, idxTheta] if kappaLPred[k, idxTheta] > 1e-10 else 1e6
            scaleApprox = getMatchedScale(alpha, beta, varMu, obsVarP)
            logLikPos = logStudentTPdf(effectiveDeltaObs, muLPred[k, idxTheta], scaleApprox, df)
            logLikNeg = logStudentTPdf(effectiveDeltaObs, -muLPred[k, idxTheta], scaleApprox, df)
            logMargLK[k] = logAddExp(logPSignPosLPred[k, idxTheta] + logLikPos, logPSignNegLPred[k, idxTheta] + logLikNeg)
            scaleApproxAim = getMatchedScale(alpha, beta, varMu, sigmaAim2)
            muAimPos = implicitCompensation - muLPred[k, idxTheta]
            muAimNeg = implicitCompensation + muLPred[k, idxTheta]
            logAimPos = logStudentTPdf(aim, muAimPos, scaleApproxAim, df)
            logAimNeg = logStudentTPdf(aim, muAimNeg, scaleApproxAim, df)
            logAimLK[k] = logAddExp(logPSignPosLPred[k, idxTheta] + logAimPos, logPSignNegLPred[k, idxTheta] + logAimNeg)
        logWeightedExact = logSumExp(logR[t-1, :runLimit, 1] + logMargGK)
        logMargLikG = logWeightedExact - logSumPrevG
        logWeightedExact = logSumExp(logR[t-1, :runLimit, 2] + logMargDK)
        logMargLikD = logWeightedExact - logSumPrevD
        logWeightedExact = logSumExp(logR[t-1, :runLimit, 3] + logMargLK)
        logMargLikL = logWeightedExact - logSumPrevL
        logLik0 = logStudentTPdf(absEffective, 0.0, math.sqrt(obsVarP),baseDf)
        unnormS0 = np.full(runLimit, LOGZERO)
        unnormS0[0] = logLik0 + safeLog(pChange) + safeLog(pertPriors[0])
        if runLimit > 1:
            unnormS0[1:] = logR[t-1, :runLimit-1, 0] + logLik0 + safeLog(pNoChange)
        unnormG = np.full(runLimit, LOGZERO)
        unnormG[0] = logMargGK[0] + safeLog(pChange) + safeLog(pertPriors[1])
        if runLimit > 1:
            unnormG[1:] = logR[t-1, :runLimit-1, 1] + logMargGK[1:] + safeLog(pNoChange)
        unnormD = np.full(runLimit, LOGZERO)
        unnormD[0] = logMargDK[0] + safeLog(pChange) + safeLog(pertPriors[2])
        if runLimit > 1:
            unnormD[1:] = logR[t-1, :runLimit-1, 2] + logMargDK[1:] + safeLog(pNoChange)
        unnormL = np.full(runLimit, LOGZERO)
        unnormL[0] = logMargLK[0] + safeLog(pChange) + safeLog(pertPriors[3])
        if runLimit > 1:
            unnormL[1:] = logR[t-1, :runLimit-1, 3] + logMargLK[1:] + safeLog(pNoChange)
        if runLimit == maxRun and runLimit > 1:
            logMassLumpedG = logAddExp(logR[t-1, runLimit-2, 1], logR[t-1, runLimit-1, 1])
            unnormG[runLimit-1] = logMassLumpedG + logMargGK[runLimit-1] + safeLog(pNoChange)
            logMassLumpedD = logAddExp(logR[t-1, runLimit-2, 2], logR[t-1, runLimit-1, 2])
            unnormD[runLimit-1] = logMassLumpedD + logMargDK[runLimit-1] + safeLog(pNoChange)
            logMassLumpedL = logAddExp(logR[t-1, runLimit-2, 3], logR[t-1, runLimit-1, 3])
            unnormL[runLimit-1] = logMassLumpedL + logMargLK[runLimit-1] + safeLog(pNoChange)
        unnormAll = np.concatenate((unnormS0, unnormG, unnormD, unnormL))
        logSumAll = logSumExp(unnormAll)
        logR[t, :runLimit, 0] = unnormS0 - logSumAll
        logR[t, :runLimit, 1] = unnormG - logSumAll
        logR[t, :runLimit, 2] = unnormD - logSumAll
        logR[t, :runLimit, 3] = unnormL - logSumAll
        logUnnorm0 = logLik0 + safeLog(pS0Norm)
        logUnnormG = logMargLikG + safeLog(pGNorm)
        logUnnormD = logMargLikD + safeLog(pDNorm)
        logUnnormL = logMargLikL + safeLog(pLNorm)
        logSumReg = logSumExp(np.array([logUnnorm0, logUnnormG, logUnnormD, logUnnormL]))
        post0 = math.exp(logUnnorm0 - logSumReg)
        postG = math.exp(logUnnormG - logSumReg)
        postD = math.exp(logUnnormD - logSumReg)
        postL = math.exp(logUnnormL - logSumReg)
        prevStatePosterior = np.array([post0, postG, postD, postL])
        obsPrec = 1.0 / max(obsVarP, EPS)
        for k in range(runLimit):
            alpha = alphaPredG[k]
            beta = betaPredG[k]
            df = 2.0 * alpha
            varMuG = 1.0 / kappaGPred[k] if kappaGPred[k] > 1e-10 else 1e6
            eEta = beta / max(alpha - 1.0, 1e-10)
            predVarEta = varMuG + eEta
            predPrecEta = 1.0 / max(predVarEta, EPS)
            postPrecEta = predPrecEta + obsPrec
            postVarEta = 1.0 / max(postPrecEta, EPS)
            postMuEta = (muGPred[k] * predPrecEta + absEffective * obsPrec) / postPrecEta if postPrecEta > EPS else muGPred[k]
            scaleApprox = getMatchedScale(alpha, beta, varMuG, obsVarP)
            muPred = muGPred[k]
            dPosSq = ((effectiveDeltaObs - muPred)**2) / (scaleApprox ** 2)
            dNegSq = ((effectiveDeltaObs + muPred)**2) / (scaleApprox ** 2)
            wPos = (df + 1.0) / (df + dPosSq) if (df + dPosSq) > 0 else 0.0
            wNeg = (df + 1.0) / (df + dNegSq) if (df + dNegSq) > 0 else 0.0
            logLikPos = logStudentTPdf(effectiveDeltaObs, muPred, scaleApprox, df)
            logLikNeg = logStudentTPdf(effectiveDeltaObs, -muPred, scaleApprox, df)
            logPriorPos = logPSignPosGPred[k]
            logPriorNeg = logPSignNegGPred[k]
            logMarg = logAddExp(logPriorPos + logLikPos, logPriorNeg + logLikNeg)
            pPosPost = safeExp(logPriorPos + logLikPos - logMarg)
            w = pPosPost * wPos + (1.0 - pPosPost) * wNeg
            postMeanMuG = postMuEta
            postVarMuG = postVarEta
            if postVarMuG < EPS:
                postVarMuG = EPS
            muRunG[k] = postMeanMuG
            kappaRunG[k] = 1.0 / postVarMuG
            residual = (absEffective - postMuEta)**2 + postVarEta
            effectiveResidual = max(residual - obsVarP, 0.0)
            alphaNew = alpha + 0.5 * w
            betaNew = beta + 0.5 * w * effectiveResidual
            alphaNew = min(max(alphaNew, ALPHA_MIN), ALPHA_MAX)
            betaNew = max(betaNew, 1e-9)
            alphaRunG[k] = alphaNew
            betaRunG[k] = betaNew
            varProcess = beta / max(alpha, 1e-10)
            varTotalG = varMuG + varProcess + obsVarP
            scaleG = math.sqrt(varTotalG)
            logLikPos = logStudentTPdf(effectiveDeltaObs, muGPred[k], scaleG, df)
            logLikNeg = logStudentTPdf(effectiveDeltaObs, -muGPred[k], scaleG, df)
            logLikPos /= priorStrength
            logLikNeg /= priorStrength
            logPriorPos = logPSignPosGPred[k]
            logPriorNeg = logPSignNegGPred[k]
            logPostPos = logLikPos + logPriorPos
            logPostNeg = logLikNeg + logPriorNeg
            logZ = logAddExp(logPostPos, logPostNeg)
            logPSignPosG[k] = logPostPos - logZ
            logPSignNegG[k] = logPostNeg - logZ
            alpha = alphaPredD[k]
            beta = betaPredD[k]
            df = 2.0 * alpha
            varMuD = 1.0 / kappaDPred[k] if kappaDPred[k] > 1e-10 else 1e6
            eEta = beta / max(alpha - 1.0, 1e-10)
            predVarEta = varMuD + eEta
            predPrecEta = 1.0 / max(predVarEta, EPS)
            postPrecEta = predPrecEta + obsPrec
            postVarEta = 1.0 / max(postPrecEta, EPS)
            postMuEta = (muDPred[k] * predPrecEta + absEffective * obsPrec) / postPrecEta if postPrecEta > EPS else muDPred[k]
            scaleApprox = getMatchedScale(alpha, beta, varMuD, obsVarP)
            muPred = muDPred[k]
            dPosSq = ((effectiveDeltaObs - muPred)**2) / (scaleApprox ** 2)
            dNegSq = ((effectiveDeltaObs + muPred)**2) / (scaleApprox ** 2)
            wPos = (df + 1.0) / (df + dPosSq) if (df + dPosSq) > 0 else 0.0
            wNeg = (df + 1.0) / (df + dNegSq) if (df + dNegSq) > 0 else 0.0
            logLikPos = logStudentTPdf(effectiveDeltaObs, muPred, scaleApprox, df)
            logLikNeg = logStudentTPdf(effectiveDeltaObs, -muPred, scaleApprox, df)
            logPriorPos = safeLog(pPosK[k])
            logPriorNeg = safeLog(1.0 - pPosK[k])
            logMarg = logAddExp(logPriorPos + logLikPos, logPriorNeg + logLikNeg)
            pPosPost = safeExp(logPriorPos + logLikPos - logMarg)
            w = pPosPost * wPos + (1.0 - pPosPost) * wNeg
            postMeanMuD = postMuEta
            postVarMuD = postVarEta
            if postVarMuD < EPS:
                postVarMuD = EPS
            muRunD[k] = postMeanMuD
            kappaRunD[k] = 1.0 / postVarMuD
            residual = (absEffective - postMuEta)**2 + postVarEta
            effectiveResidual = max(residual - obsVarP, 0.0)
            alphaNew = alpha + 0.5 * w
            betaNew = beta + 0.5 * w * effectiveResidual
            alphaNew = min(max(alphaNew, ALPHA_MIN), ALPHA_MAX)
            betaNew = max(betaNew, 1e-9)
            alphaRunD[k] = alphaNew
            betaRunD[k] = betaNew
            alpha = alphaPredL[k]
            beta = betaPredL[k]
            df = 2.0 * alpha
            varMuL = 1.0 / kappaLPred[k, idxTheta] if kappaLPred[k, idxTheta] > 1e-10 else 1e6
            eEta = beta / max(alpha - 1.0, 1e-10)
            predVarEta = varMuL + eEta
            predPrecEta = 1.0 / max(predVarEta, EPS)
            postPrecEta = predPrecEta + obsPrec
            postVarEta = 1.0 / max(postPrecEta, EPS)
            postMuEta = (muLPred[k, idxTheta] * predPrecEta + absEffective * obsPrec) / postPrecEta if postPrecEta > EPS else muLPred[k, idxTheta]
            scaleApprox = getMatchedScale(alpha, beta, varMuL, obsVarP)
            muPred = muLPred[k, idxTheta]
            dPosSq = ((effectiveDeltaObs - muPred)**2) / (scaleApprox ** 2)
            dNegSq = ((effectiveDeltaObs + muPred)**2) / (scaleApprox ** 2)
            wPos = (df + 1.0) / (df + dPosSq) if (df + dPosSq) > 0 else 0.0
            wNeg = (df + 1.0) / (df + dNegSq) if (df + dNegSq) > 0 else 0.0
            logLikPos = logStudentTPdf(effectiveDeltaObs, muPred, scaleApprox, df)
            logLikNeg = logStudentTPdf(effectiveDeltaObs, -muPred, scaleApprox, df)
            logPriorPos = logPSignPosLPred[k, idxTheta]
            logPriorNeg = logPSignNegLPred[k, idxTheta]
            logMarg = logAddExp(logPriorPos + logLikPos, logPriorNeg + logLikNeg)
            pPosPost = safeExp(logPriorPos + logLikPos - logMarg)
            w = pPosPost * wPos + (1.0 - pPosPost) * wNeg
            postMeanMuL = postMuEta
            postVarMuL = postVarEta
            if postVarMuL < EPS:
                postVarMuL = EPS
            muRunL[k, idxTheta] = postMeanMuL
            kappaRunL[k, idxTheta] = 1.0 / postVarMuL
            residual = (absEffective - postMuEta)**2 + postVarEta
            effectiveResidual = max(residual - obsVarP, 0.0)
            alphaNew = alpha + 0.5 * w
            betaNew = beta + 0.5 * w * effectiveResidual
            alphaNew = min(max(alphaNew, ALPHA_MIN), ALPHA_MAX)
            betaNew = max(betaNew, 1e-9)
            alphaRunL[k] = alphaNew
            betaRunL[k] = betaNew
            varProcess = beta / max(alpha, 1e-10)
            varTotalL = varMuL + varProcess + obsVarP
            scaleL = math.sqrt(varTotalL)
            logLikPos = logStudentTPdf(effectiveDeltaObs, muLPred[k, idxTheta], scaleL, df)
            logLikNeg = logStudentTPdf(effectiveDeltaObs, -muLPred[k, idxTheta], scaleL, df)
            logLikPos /= priorStrength
            logLikNeg /= priorStrength
            logPriorPos = logPSignPosLPred[k, idxTheta]
            logPriorNeg = logPSignNegLPred[k, idxTheta]
            logPostPos = logLikPos + logPriorPos
            logPostNeg = logLikNeg + logPriorNeg
            logZ = logAddExp(logPostPos, logPostNeg)
            logPSignPosL[k, idxTheta] = logPostPos - logZ
            logPSignNegL[k, idxTheta] = logPostNeg - logZ
        for k in range(runLimit):
            logLikSK = np.full(numConfig, LOGZERO)
            for c in range(numConfig):
                s = S[c, idxTheta]
                logMargLikDSK = logMargDPosK[k] if s > 0.0 else logMargDNegK[k]
                logLikSK[c] = logMargLikDSK / priorStrength
            temp = logProbSPred[k] + logLikSK
            logZNew = logSumExp(temp)
            logProbSRun[k] = temp - logZNew
        pPosKPrevs[idx, :runLimit] = pPosK
        logWeightedExact = logSumExp(logR[t-1, :runLimit, 1] + logAimGK)
        logAimLikG = logWeightedExact - logSumPrevG
        logWeightedExact = logSumExp(logR[t-1, :runLimit, 2] + logAimDK)
        logAimLikD = logWeightedExact - logSumPrevD
        logWeightedExact = logSumExp(logR[t-1, :runLimit, 3] + logAimLK)
        logAimLikL = logWeightedExact - logSumPrevL
        logAimLikS0 = logStudentTPdf(aim, implicitCompensation, math.sqrt(sigmaAim2),baseDf)
        logAimPdf = logSumExp(np.array([
            safeLog(pS0Norm) + logAimLikS0,
            safeLog(pGNorm) + logAimLikG,
            safeLog(pDNorm) + logAimLikD,
            safeLog(pLNorm) + logAimLikL
        ]))
        if mask[trial]:
            logLikelihood += logAimPdf
        eT = effectiveDeltaObs
        d = wrapAngleArr(angles, meanAimAngle)
        closeMask = np.abs(d) < (4.0 * sigmaGen)
        g = np.zeros(len(angles))
        g[closeMask] = safeExp(-d[closeMask]**2 / (2.0 * sigmaGen**2))
        xVec = A * xVec - B * g * eT
        plannedOffsets[idx] = plannedOffset
        implicitComps[idx] = implicitCompensation
        pGs[idx] = pGNorm
        pDs[idx] = pDNorm
        pLs[idx] = pLNorm
        pPoss[idx] = np.sum(pKArrD * pPosK)
        runLimits[idx] = runLimit
        pKArrGs[idx, :runLimit] = pKArrG
        pKArrDs[idx, :runLimit] = pKArrD
        pKArrLs[idx, :runLimit] = pKArrL
        muRunGPrevs[idx, :runLimit] = muGPred
        muRunDPrevs[idx, :runLimit] = muDPred
        muRunLPrevs[idx, :runLimit] = muLPred[:, idxTheta]
        alphaRunGPrevs[idx, :runLimit] = alphaPredG[:runLimit]
        alphaRunDPrevs[idx, :runLimit] = alphaPredD[:runLimit]
        alphaRunLPrevs[idx, :runLimit] = alphaPredL[:runLimit]
        betaRunGPrevs[idx, :runLimit] = betaPredG[:runLimit]
        betaRunDPrevs[idx, :runLimit] = betaPredD[:runLimit]
        betaRunLPrevs[idx, :runLimit] = betaPredL[:runLimit]
        kappaRunGPrevs[idx, :runLimit] = kappaGPred
        kappaRunDPrevs[idx, :runLimit] = kappaDPred
        kappaRunLPrevs[idx, :runLimit] = kappaLPred[:, idxTheta]
        post0s[idx] = post0
    negll = -logLikelihood if math.isfinite(logLikelihood) else 1e12
    return (negll, post0s, plannedOffsets, implicitComps, pGs, pDs, pLs, pPoss, runLimits,
            pKArrGs, pKArrDs, pKArrLs, muRunGPrevs, muRunDPrevs, muRunLPrevs,
            alphaRunGPrevs, alphaRunDPrevs, alphaRunLPrevs,
            betaRunGPrevs, betaRunDPrevs, betaRunLPrevs,
            kappaRunGPrevs, kappaRunDPrevs, kappaRunLPrevs,
            logPSignPosGPredAll, logPSignNegGPredAll, logPSignPosLPredAll, logPSignNegLPredAll, pPosKPrevs)
def plotCombined(modelExplicit, modelImps, mOutsSingle, humanExplicit, humanImps, allAims, trials, number, plotIdentifier, fittedParams, targets, compMags, phases,
                 logPSignPosGPredAll, logPSignNegGPredAll, logPSignPosLPredAll, logPSignNegLPredAll,
                 numSamples=200):
    sigmaMotor, pGlobal, pLocal, priorStrength, visCoeff, betaEta, hazardBeta, pS0, A, B, sigmaGen, alphaTauSharedPrior, alphaChange = fittedParams
    pDir = max(1 - pGlobal - pLocal, 1e-10)
    logitPGlobal = np.log(pGlobal / pDir) if pGlobal > 0 else -700
    logitPLocal = np.log(pLocal / pDir) if pLocal > 0 else -700
    logSigmaMotor = np.log(sigmaMotor)
    logPriorStrength = np.log(priorStrength)
    logBetaEta = np.log(betaEta)
    logHazardBeta = np.log(hazardBeta)
    logAlphaTauSharedPrior = np.log(alphaTauSharedPrior)
    params = np.array([logSigmaMotor, logitPGlobal, logitPLocal, logPriorStrength, visCoeff, logBetaEta, logHazardBeta, logAlphaTauSharedPrior])
    numTrials = len(trials)
    isRotation = np.array([phases[trial].lower() == 'rotation' for trial in trials], dtype=bool)
    rotations = compMags
    uniqueThetas = np.unique(targets)
    m = len(uniqueThetas)
    thetaToIdx = {uniqueThetas[i]: i for i in range(m)}
    _, post0s, plannedOffsets, implicitComps, pGs, pDs, pLs, pPoss, runLimits,\
    pKArrGs, pKArrDs, pKArrLs, muRunGPrevs, muRunDPrevs, muRunLPrevs,\
    alphaRunGPrevs, alphaRunDPrevs, alphaRunLPrevs,\
    betaRunGPrevs, betaRunDPrevs, betaRunLPrevs,\
    kappaRunGPrevs, kappaRunDPrevs, kappaRunLPrevs,\
    logPSignPosGPredAll, logPSignNegGPredAll,\
    logPSignPosLPredAll, logPSignNegLPredAll, pPosKPrevs = computeNegLl(
        params, np.zeros(numTrials), np.zeros(numTrials, dtype=bool), trials, isRotation, rotations, targets, uniqueThetas, A, B, sigmaGen
    )
    aims = []
    componentsList = []
    trialsList = []
    for idx, trial in enumerate(trials):
        pG = pGs[idx]
        pD = pDs[idx]
        pL = pLs[idx]
        p0 = post0s[idx]
        runLimit = runLimits[idx]
        pKArrG = pKArrGs[idx, :runLimit]
        pKArrD = pKArrDs[idx, :runLimit]
        pKArrL = pKArrLs[idx, :runLimit]
        muRunG = muRunGPrevs[idx, :runLimit]
        muRunD = muRunDPrevs[idx, :runLimit]
        muRunL = muRunLPrevs[idx, :runLimit]
        alphaG = alphaRunGPrevs[idx, :runLimit]
        alphaD = alphaRunDPrevs[idx, :runLimit]
        alphaL = alphaRunLPrevs[idx, :runLimit]
        betaG = betaRunGPrevs[idx, :runLimit]
        betaD = betaRunDPrevs[idx, :runLimit]
        betaL = betaRunLPrevs[idx, :runLimit]
        kappaG = kappaRunGPrevs[idx, :runLimit]
        kappaD = kappaRunDPrevs[idx, :runLimit]
        kappaL = kappaRunLPrevs[idx, :runLimit]
        compProbs = np.array([p0, pG, pD, pL])
        compProbs = np.maximum(compProbs, 1e-10)
        s = compProbs.sum()
        if s <= 0 or not np.isfinite(s):
            aims.extend([np.nan] * numSamples)
            componentsList.extend(['Error'] * numSamples)
            trialsList.extend([trial] * numSamples)
            continue
        compProbs /= s
        numS0, numG, numD, numL = np.random.multinomial(numSamples, compProbs)
        def safeRunChoice(pKArr, size):
            if size == 0:
                return np.array([], dtype=int)
            p = pKArr.copy()
            s = p.sum()
            if s > 1e-12:
                p = p / s
            else:
                p = np.ones(len(p)) / len(p)
            return np.random.choice(range(runLimit), size=size, p=p)
        if numS0 > 0:
            loc = implicitComps[idx]
            scale = sigmaMotor
            df = 130.0
            samp = t.rvs(df, loc=loc, scale=scale, size=numS0)
            aims.extend([wrapAngle(s) for s in samp])
            componentsList.extend(['S0'] * numS0)
            trialsList.extend([trial] * numS0)
        if numG > 0:
            ks = safeRunChoice(pKArrG, numG)
            signsG = np.zeros(numG, dtype=int)
            for i, k in enumerate(ks):
                pPos = math.exp(logPSignPosGPredAll[idx, k])
                signsG[i] = np.random.choice([1, -1], p=[pPos, 1 - pPos])
            locsG = signsG * (-muRunG[ks])
            varMusG = np.zeros(numG)
            varProcessesG = np.zeros(numG)
            for i, k in enumerate(ks):
                varMusG[i] = 1.0 / kappaG[k] if kappaG[k] > 1e-10 else 1e6
                varProcessesG[i] = betaG[k] / (alphaG[k] - 1.0) if alphaG[k] > 1e-10 else 1e6
            varTotalsG = np.array([getMatchedScale(alphaG[ks[j]], betaG[ks[j]], varMusG[j], sigmaMotor**2) for j in range(numG)])
            scalesG = varTotalsG
            dfsG = 2 * alphaG[ks]
            samp = np.array([t.rvs(dfsG[i], locsG[i], scalesG[i]) for i in range(numG)])
            aims.extend([wrapAngle(s) for s in samp])
            componentsList.extend(['G'] * numG)
            trialsList.extend([trial] * numG)
        if numD > 0:
            ks = safeRunChoice(pKArrD, numD)
            signs = np.zeros(numD)
            for i, k in enumerate(ks):
                pPos = pPosKPrevs[idx, k]
                signs[i] = np.random.binomial(1, pPos) * 2 - 1
            locsD = -signs * muRunD[ks]
            varMusD = np.zeros(numD)
            varProcessesD = np.zeros(numD)
            for i, k in enumerate(ks):
                varMusD[i] = 1.0 / kappaD[k] if kappaD[k] > 1e-10 else 1e6
                varProcessesD[i] = betaD[k] / (alphaD[k] - 1.0) if alphaD[k] > 1e-10 else 1e6
            varTotalsD = np.array([getMatchedScale(alphaD[ks[j]], betaD[ks[j]], varMusD[j], sigmaMotor**2) for j in range(numD)])
            scalesD = varTotalsD
            dfsD = 2 * alphaD[ks]
            samp = np.array([t.rvs(dfsD[i], locsD[i], scalesD[i]) for i in range(numD)])
            aims.extend([wrapAngle(s) for s in samp])
            componentsList.extend(['D'] * numD)
            trialsList.extend([trial] * numD)
        if numL > 0:
            ks = safeRunChoice(pKArrL, numL)
            signsL = np.zeros(numL, dtype=int)
            idxTheta = thetaToIdx[targets[trial]] if m > 0 else 0
            for i, k in enumerate(ks):
                pPos = math.exp(logPSignPosLPredAll[idx, k, idxTheta])
                signsL[i] = np.random.choice([1, -1], p=[pPos, 1 - pPos])
            locsL = signsL * (-muRunL[ks])
            varMusL = np.zeros(numL)
            varProcessesL = np.zeros(numL)
            for i, k in enumerate(ks):
                varMusL[i] = 1.0 / kappaL[k] if kappaL[k] > 1e-10 else 1e6
                varProcessesL[i] = betaL[k] / (alphaL[k] - 1.0) if alphaL[k] > 1e-10 else 1e6
            varTotalsL = np.array([getMatchedScale(alphaL[ks[j]], betaL[ks[j]], varMusL[j], sigmaMotor**2) for j in range(numL)])
            scalesL = varTotalsL
            dfsL = 2 * alphaL[ks]
            samp = np.array([t.rvs(dfsL[i], locsL[i], scalesL[i]) for i in range(numL)])
            aims.extend([wrapAngle(s) for s in samp])
            componentsList.extend(['L'] * numL)
            trialsList.extend([trial] * numL)
    df = pd.DataFrame({'trial': trialsList, 'aim': aims, 'component': componentsList})
    fig, ax = plt.subplots(figsize=(15, 6))
    trialBins = np.arange(min(trials)-0.5, max(trials)+1.5, 1)
    binWidth = 3
    aimBins = np.arange(-180, 180 + binWidth, binWidth)
    hist, xedges, yedges = np.histogram2d(df['trial'], df['aim'], bins=(trialBins, aimBins))
    hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-10)
    hist = np.ma.masked_where(hist == 0, hist)
    im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='viridis',
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
    plt.close()
def processAndPlotSingle(i, dataItem, rawX, negll, numSamples, plotIdentifier, A, B, sigmaGen):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets, _, popSizeMultiplier, humanExplicit, humanImps = dataItem
    logSigmaMotor, logitPGlobal, logitPLocal, logPriorStrength, visCoeff, logBetaEta, logHazardBeta, logAlphaTauSharedPrior = rawX
    sigmaMotor = np.exp(logSigmaMotor)
    betaEta = np.exp(logBetaEta)
    hazardBeta = np.exp(logHazardBeta)
    alphaTauSharedPrior = np.exp(logAlphaTauSharedPrior)
    priorStrength = np.exp(logPriorStrength)
    logPG = logitPGlobal
    logPL = logitPLocal
    logPD = 0.0
    logZP = sp_logsumexp(np.array([logPG, logPL, logPD]))
    pGlobal = np.exp(logPG - logZP)
    pLocal = np.exp(logPL - logZP)
    pDir = np.exp(logPD - logZP)
    xs = [sigmaMotor, pGlobal, pLocal, priorStrength, visCoeff, betaEta, hazardBeta, 0.0, A, B, sigmaGen, alphaTauSharedPrior, 1.0]
    params = np.array([logSigmaMotor, logitPGlobal, logitPLocal, logPriorStrength, visCoeff, logBetaEta, logHazardBeta, logAlphaTauSharedPrior])
    numTrials = len(trials)
    isRotation = np.array([phases[trial].lower() == 'rotation' for trial in trials], dtype=bool)
    rotations = compMags
    uniqueThetas = np.unique(targets)
    m = len(uniqueThetas)
    thetaToIdx = {uniqueThetas[i]: i for i in range(m)}
    _, post0s, plannedOffsets, implicitComps, pGs, pDs, pLs, pPoss, runLimits,\
    pKArrGs, pKArrDs, pKArrLs, muRunGPrevs, muRunDPrevs, muRunLPrevs,\
    alphaRunGPrevs, alphaRunDPrevs, alphaRunLPrevs,\
    betaRunGPrevs, betaRunDPrevs, betaRunLPrevs,\
    kappaRunGPrevs, kappaRunDPrevs, kappaRunLPrevs,\
    logPSignPosGPredAll, logPSignNegGPredAll,\
    logPSignPosLPredAll, logPSignNegLPredAll, pPosKPrevs = computeNegLl(
        params, np.zeros(numTrials), np.zeros(numTrials, dtype=bool), trials, isRotation, rotations, targets, uniqueThetas, A, B, sigmaGen
    )
    modelExplicit = plannedOffsets
    modelImps = implicitComps
    mOutsSingle = modelExplicit + modelImps
    plotCombined(modelExplicit, modelImps, mOutsSingle, humanExplicit, humanImps, allAims, trials, pp, plotIdentifier, xs, targets, compMags, phases,
                 logPSignPosGPredAll, logPSignNegGPredAll, logPSignPosLPredAll, logPSignNegLPredAll,
                 numSamples=200)
    ese = np.zeros(len(trials))
    for idx, trial in enumerate(trials):
        if not mask[trial]:
            continue
        pG = pGs[idx]
        pD = pDs[idx]
        pL = pLs[idx]
        pKArrG = pKArrGs[idx, :runLimits[idx]]
        pKArrD = pKArrDs[idx, :runLimits[idx]]
        pKArrL = pKArrLs[idx, :runLimits[idx]]
        muRunG = muRunGPrevs[idx, :runLimits[idx]]
        muRunD = muRunDPrevs[idx, :runLimits[idx]]
        muRunL = muRunLPrevs[idx, :runLimits[idx]]
        alphaG = alphaRunGPrevs[idx, :runLimits[idx]]
        alphaD = alphaRunDPrevs[idx, :runLimits[idx]]
        alphaL = alphaRunLPrevs[idx, :runLimits[idx]]
        betaG = betaRunGPrevs[idx, :runLimits[idx]]
        betaD = betaRunDPrevs[idx, :runLimits[idx]]
        betaL = betaRunLPrevs[idx, :runLimits[idx]]
        eEtaKsG = betaG / np.maximum(alphaG - 1, 1e-10)
        eEtaKsG[0] = 0.0
        eEtaKsD = betaD / np.maximum(alphaD - 1, 1e-10)
        eEtaKsD[0] = 0.0
        eEtaKsL = betaL / np.maximum(alphaL - 1, 1e-10)
        eEtaKsL[0] = 0.0
        pPosK = pPosKPrevs[idx, :runLimits[idx]]
        pPosGK = np.exp(logPSignPosGPredAll[idx, :runLimits[idx]])
        idxTheta = thetaToIdx[targets[trial]] if len(uniqueThetas) > 0 else 0
        pPosLK = np.exp(logPSignPosLPredAll[idx, :runLimits[idx], idxTheta])
        yT = allAims[trial]
        implicit = implicitComps[idx]
        sigmaAim2 = sigmaMotor**2
        eseS0 = post0s[idx] * ((yT - implicit)**2 + sigmaAim2)
        eseG = pG * np.sum(pKArrG * ((yT - implicit)**2 + sigmaAim2 + eEtaKsG + muRunG**2 + 2 * (yT - implicit) * muRunG * (2 * pPosGK - 1)))
        eseD = pD * np.sum(pKArrD * ((yT - implicit)**2 + sigmaAim2 + eEtaKsD + muRunD**2 + 2 * (yT - implicit) * muRunD * (2 * pPosK - 1)))
        eseL = pL * np.sum(pKArrL * ((yT - implicit)**2 + sigmaAim2 + eEtaKsL + muRunL**2 + 2 * (yT - implicit) * muRunL * (2 * pPosLK - 1)))
        ese[trial] = eseS0 + eseG + eseD + eseL
    validAims = allAims[mask]
    validMOuts = mOutsSingle[mask]
    rmseVal = np.sqrt(np.sum((validAims - validMOuts)**2) / len(validAims)) if len(validAims) > 0 else np.inf
    rSquared = computeRSquared(validAims, validMOuts)
    rmseDist = np.sqrt(np.sum(ese[mask]) / len(validAims)) if len(validAims) > 0 else np.inf
    ssTot = np.sum((validAims - np.mean(validAims))**2) if len(validAims) > 0 else 1.0
    r2Dist = 1 - np.sum(ese[mask]) / ssTot if ssTot > 0 else 0.0
    bicI = 8 * np.log(len(validAims)) + 2 * negll
    return rmseVal, rSquared, bicI, rmseDist, r2Dist, modelImps.tolist(), mOutsSingle.tolist(),\
           post0s, pGs, pDs, pLs, pPoss, runLimits,\
           pKArrGs, pKArrDs, pKArrLs, muRunGPrevs, muRunDPrevs, muRunLPrevs,\
           alphaRunGPrevs, alphaRunDPrevs, alphaRunLPrevs,\
           betaRunGPrevs, betaRunDPrevs, betaRunLPrevs,\
           kappaRunGPrevs, kappaRunDPrevs, kappaRunLPrevs,\
           logPSignPosGPredAll, logPSignNegGPredAll, logPSignPosLPredAll, logPSignNegLPredAll, pPosKPrevs
def computeRSquared(trueValues, predValues):
    trueValues = np.array(trueValues)
    predValues = np.array(predValues)
    if len(trueValues) != len(predValues):
        return 0.0
    ssRes = np.sum((trueValues - predValues) ** 2)
    ssTot = np.sum((trueValues - np.mean(trueValues)) ** 2)
    if ssTot == 0:
        return 1.0 if ssRes == 0 else 0.0
    return 1 - (ssRes / ssTot)
class Objective:
    def __init__(self, allAims, mask, trials, phases, rotations, targets, bounds, A=0.99, B=0.01, sigmaGen=30.0):
        self.allAims = allAims
        self.mask = mask
        self.trials = trials
        self.phases = phases
        self.rotations = rotations
        self.targets = targets
        self.uniqueThetas = np.unique(targets)
        self.isRotation = np.array([p.lower() == 'rotation' for p in phases], dtype=bool)
        self.A = A
        self.B = B
        self.sigmaGen = sigmaGen
        self.bounds = np.array(bounds)
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]
    def denormalize(self, paramsNormalized):
        params = self.lower + paramsNormalized * (self.upper - self.lower)
        return np.clip(params, self.lower, self.upper)
    def __call__(self, paramsNormalized):
        paramsOriginal = self.denormalize(paramsNormalized)
        return computeNegLl(paramsOriginal, self.allAims, self.mask, self.trials, self.isRotation, self.rotations, self.targets, self.uniqueThetas, self.A, self.B, self.sigmaGen)[0]
def fitSingle(data, boundsSingle, popSizeMultiplier, A=0.99, B=0.01, sigmaGen=30.0):
    allAims, mask, trials, heightCap, compMags, pp, conVal, phases, targets, uniqueTargets, plotIdentifier, popSizeMultiplier, humanExplicit, humanImps = data
    objFunc = Objective(allAims, mask, trials, phases, compMags, targets, boundsSingle, A, B, sigmaGen)
    numSamples = np.sum(mask)
    if numSamples == 0:
        return np.zeros(8), 0.0, 0.0
    nParams = len(boundsSingle)
    normalizedBounds = [(0.0, 1.0) for _ in range(nParams)]
    maxRestarts = 100    
    shallowResults = []
    sampler = LatinHypercube(d=nParams, seed=SEED)
    initSamples = sampler.random(n=maxRestarts)
    for restart in range(maxRestarts):
        x0Normalized = initSamples[restart]
        res = minimize(objFunc, x0Normalized, method='L-BFGS-B', bounds=normalizedBounds, options={'maxiter': 50, 'ftol': 3e-5})
        shallowResults.append((res.fun, res.x, restart))
        if res.fun < min([r[0] for r in shallowResults[:-1]] or [np.inf]):
            print(pp, restart, res.fun, res.x)
    shallowResults.sort(key=lambda x: x[0])
    top = shallowResults[:1+maxRestarts//10]
    bestValue = np.inf
    bestXNormalized = None
    for fun, xNormalized, originalRestart in top:
        resDeep = minimize(objFunc, xNormalized, method='L-BFGS-B', bounds=normalizedBounds, options={'maxiter': 300, 'ftol': 1e-8})
        if resDeep.fun < bestValue:
            print(pp, f"deep_{originalRestart}", resDeep.fun, resDeep.x)
            bestValue = resDeep.fun
            bestXNormalized = resDeep.x
    if bestXNormalized is not None:
        bestX = objFunc.denormalize(bestXNormalized)
    else:
        bestX = np.zeros(8)
    negll = bestValue
    return bestX, bestValue, negll
class FitShell:
    def __init__(self, df, conVal='none', condition='none', fitPhase='rotation', heightCap=180, plotIdentifier='', numCores=multiprocessing.cpu_count()//2, popSizeMultiplier=1, A=0.99, B=0.01, sigmaGen=30.0):
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
        self.A = A
        self.B = B
        self.sigmaGen = sigmaGen
        self.uniqueThetas = []
        self.targets = []
        self.post0s = []
        self.pGs = []
        self.pDs = []
        self.pLs = []
        self.pPoss = []
        self.runLimits = []
        self.pKArrGs = []
        self.pKArrDs = []
        self.pKArrLs = []
        self.muRunGPrevs = []
        self.muRunDPrevs = []
        self.muRunLPrevs = []
        self.alphaRunGPrevs = []
        self.alphaRunDPrevs = []
        self.alphaRunLPrevs = []
        self.betaRunGPrevs = []
        self.betaRunDPrevs = []
        self.betaRunLPrevs = []
        self.kappaRunGPrevs = []
        self.kappaRunDPrevs = []
        self.kappaRunLPrevs = []
        self.logPSignPosGPredAll = []
        self.logPSignNegGPredAll = []
        self.logPSignPosLPredAll = []
        self.logPSignNegLPredAll = []
        self.pPosKPrevs = []
    def fitRot(self, numCores=multiprocessing.cpu_count()//2):
        if self.condition != 'none':
            if isinstance(self.conVal, (int, float)):
                participantsInCondition = self.df[self.df[self.condition] == self.conVal]['participantNum'].unique()
            else:
                participantsInCondition = self.df[self.df[self.condition].isin(self.conVal)]['participantNum'].unique()
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
        self.uniqueThetas = [[] for _ in range(numPpTotal)]
        self.targets = [[] for _ in range(numPpTotal)]
        self.post0s = [[] for _ in range(numPpTotal)]
        self.pGs = [[] for _ in range(numPpTotal)]
        self.pDs = [[] for _ in range(numPpTotal)]
        self.pLs = [[] for _ in range(numPpTotal)]
        self.pPoss = [[] for _ in range(numPpTotal)]
        self.runLimits = [[] for _ in range(numPpTotal)]
        self.pKArrGs = [[] for _ in range(numPpTotal)]
        self.pKArrDs = [[] for _ in range(numPpTotal)]
        self.pKArrLs = [[] for _ in range(numPpTotal)]
        self.muRunGPrevs = [[] for _ in range(numPpTotal)]
        self.muRunDPrevs = [[] for _ in range(numPpTotal)]
        self.muRunLPrevs = [[] for _ in range(numPpTotal)]
        self.alphaRunGPrevs = [[] for _ in range(numPpTotal)]
        self.alphaRunDPrevs = [[] for _ in range(numPpTotal)]
        self.alphaRunLPrevs = [[] for _ in range(numPpTotal)]
        self.betaRunGPrevs = [[] for _ in range(numPpTotal)]
        self.betaRunDPrevs = [[] for _ in range(numPpTotal)]
        self.betaRunLPrevs = [[] for _ in range(numPpTotal)]
        self.kappaRunGPrevs = [[] for _ in range(numPpTotal)]
        self.kappaRunDPrevs = [[] for _ in range(numPpTotal)]
        self.kappaRunLPrevs = [[] for _ in range(numPpTotal)]
        self.logPSignPosGPredAll = [[] for _ in range(numPpTotal)]
        self.logPSignNegGPredAll = [[] for _ in range(numPpTotal)]
        self.logPSignPosLPredAll = [[] for _ in range(numPpTotal)]
        self.logPSignNegLPredAll = [[] for _ in range(numPpTotal)]
        self.pPosKPrevs = [[] for _ in range(numPpTotal)]
        firstPp = uniqP[0]
        pDatFirst = self.dat[(self.dat['participantNum'] == firstPp)]
        numTrials = len(pDatFirst)
        trials = np.arange(numTrials)
        dataList = []
        aPps = []
        bPps = []
        for pp in uniqP:
            pDat = self.df[(self.df['participantNum'] == pp)]
            humanExplicit = pDat['aim'].values
            humanImps = pDat['imp'].values
            allAims = humanExplicit + humanImps
            phases = pDat['phase'].values
            compMags = pDat[self.condition].values
            validCompMags = compMags[~np.isnan(compMags)]
            nonZeroCompMags = validCompMags[validCompMags != 0]
            if isinstance(self.conVal, (int, float)):
                rotPp = self.conVal
                aPp = self.A
                bPp = self.B
            else:
                uniqueNonZero = np.unique(nonZeroCompMags)
                if len(uniqueNonZero) != 1:
                    raise ValueError(f"Could not infer a single non-zero rotation for participant {pp}")
                rotPp = uniqueNonZero[0]
                if rotPp not in self.conVal:
                    raise ValueError(f"rotPp {rotPp} not in conVal {self.conVal} for participant {pp}")
                idx = self.conVal.index(rotPp)
                if not isinstance(self.A, list) or len(self.A) != len(self.conVal):
                    raise ValueError("A must be a list matching the length of conVal when conVal is a list")
                if not isinstance(self.B, list) or len(self.B) != len(self.conVal):
                    raise ValueError("B must be a list matching the length of conVal when conVal is a list")
                aPp = self.A[idx]
                bPp = self.B[idx]
            aPps.append(aPp)
            bPps.append(bPp)
            targetPositions = pDat['targetPosition'].values
            mask = ~np.isnan(allAims)
            uniqueTargets = np.unique(targetPositions[~np.isnan(targetPositions)])
            dataList.append((allAims, mask, trials, self.heightCap, compMags, pp, rotPp, phases, targetPositions, uniqueTargets, self.plotIdentifier, self.popSizeMultiplier, humanExplicit, humanImps))
        boundsSingle = [
            (np.log(1e-4), np.log(30)),
            (-35, 8),
            (-30, 10),
            (np.log(1e-1), np.log(1e4)),
            (0.0, 1.0),
            (np.log(1e-2), np.log(1e6)),
            (np.log(1e-4), np.log(1e8)),
            (np.log(1.0), np.log(130.0)),
        ]
        N = numPpTotal
        with multiprocessing.Pool(processes=self.numCores) as pool:
            results = pool.starmap(fitSingle, [(dataList[i], boundsSingle, self.popSizeMultiplier, aPps[i], bPps[i], self.sigmaGen) for i in range(N)])
        indivParams = np.array([r[0] for r in results])
        currentNeglls = np.array([r[2] for r in results])
        numSamplesList = [np.sum(d[1]) for d in dataList]
        self.allAims = [d[0].tolist() for d in dataList]
        self.xs = []
        for i in range(N):
            rawX = indivParams[i]
            logSigmaMotor, logitPGlobal, logitPLocal, logPriorStrength, visCoeff, logBetaEta, logHazardBeta, logAlphaTauSharedPrior = rawX
            sigmaMotor = np.exp(logSigmaMotor)
            betaEta = np.exp(logBetaEta)
            hazardBeta = np.exp(logHazardBeta)
            alphaTauSharedPrior = np.exp(logAlphaTauSharedPrior)
            priorStrength = np.exp(logPriorStrength)
            logPG = logitPGlobal
            logPL = logitPLocal
            logPD = 0.0
            logZP = sp_logsumexp(np.array([logPG, logPL, logPD]))
            pGlobal = np.exp(logPG - logZP)
            pLocal = np.exp(logPL - logZP)
            pDir = np.exp(logPD - logZP)
            xs = [sigmaMotor, pGlobal, pLocal, priorStrength, visCoeff, betaEta, hazardBeta, 0.0, aPps[i], bPps[i], self.sigmaGen, alphaTauSharedPrior, 1.0]
            self.xs.append(xs)
        self.negLl = currentNeglls
        with multiprocessing.Pool(processes=self.numCores) as pool:
            plotResults = pool.starmap(processAndPlotSingle, [(i, dataList[i], indivParams[i], currentNeglls[i], numSamplesList[i], self.plotIdentifier, aPps[i], bPps[i], self.sigmaGen) for i in range(N)])
        for i, (rmseVal, rSquared, bicI, rmseDist, r2Dist, modelImpsList, mOutsSingleList, post0s, pGs, pDs, pLs, pPoss, runLimits,
                pKArrGs, pKArrDs, pKArrLs, muRunGPrevs, muRunDPrevs, muRunLPrevs,
                alphaRunGPrevs, alphaRunDPrevs, alphaRunLPrevs,
                betaRunGPrevs, betaRunDPrevs, betaRunLPrevs,
                kappaRunGPrevs, kappaRunDPrevs, kappaRunLPrevs,
                logPSignPosGPredAll, logPSignNegGPredAll, logPSignPosLPredAll, logPSignNegLPredAll, pPosKPrevs) in enumerate(plotResults):
            self.rmses[i] = rmseVal
            self.bics[i] = bicI
            self.rSquareds[i] = rSquared
            self.rmsesDist[i] = rmseDist
            self.rSquaredsDist[i] = r2Dist
            self.implicitComps[i] = modelImpsList
            self.mStates[i] = mOutsSingleList
            self.uniqueThetas[i] = dataList[i][9].tolist()
            self.targets[i] = dataList[i][8].tolist()
            self.post0s[i] = post0s
            self.pGs[i] = pGs
            self.pDs[i] = pDs
            self.pLs[i] = pLs
            self.pPoss[i] = pPoss
            self.runLimits[i] = runLimits
            self.pKArrGs[i] = pKArrGs
            self.pKArrDs[i] = pKArrDs
            self.pKArrLs[i] = pKArrLs
            self.muRunGPrevs[i] = muRunGPrevs
            self.muRunDPrevs[i] = muRunDPrevs
            self.muRunLPrevs[i] = muRunLPrevs
            self.alphaRunGPrevs[i] = alphaRunGPrevs
            self.alphaRunDPrevs[i] = alphaRunDPrevs
            self.alphaRunLPrevs[i] = alphaRunLPrevs
            self.betaRunGPrevs[i] = betaRunGPrevs
            self.betaRunDPrevs[i] = betaRunDPrevs
            self.betaRunLPrevs[i] = betaRunLPrevs
            self.kappaRunGPrevs[i] = kappaRunGPrevs
            self.kappaRunDPrevs[i] = kappaRunDPrevs
            self.kappaRunLPrevs[i] = kappaRunLPrevs
            self.logPSignPosGPredAll[i] = logPSignPosGPredAll
            self.logPSignNegGPredAll[i] = logPSignNegGPredAll
            self.logPSignPosLPredAll[i] = logPSignPosLPredAll
            self.logPSignNegLPredAll[i] = logPSignNegLPredAll
            self.pPosKPrevs[i] = pPosKPrevs

