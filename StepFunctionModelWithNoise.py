"""
Created on Sat Mar 19 12:51:10 2022

@author: 44796
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
    def __init__(self,stepStart,stepHeight,executionVariance=None):
        self.stepStart = stepStart
        self.stepHeight = stepHeight
        self.ev = executionVariance
        
    def modelMove(self,trialNum):
        if trialNum < self.stepStart:
            return 0, self.ev
        else:
            return self.stepHeight, self.ev
        

class fitShell:
    def __init__(self,df, conVal='none',condition='none',startCap=320,fitLen=320,fitPhase='rotation',heightCap=150,rmse=False,
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
        for pp in uniqP:
            executionVarBaseline = 1000
            self.pp = pp
            self.it = i
            if self.rmse:
                bounds = [(0, self.startCap), (0, self.heightCap)]
                res = evolution(self.fitPP, bounds=bounds, workers=-1)
            else:
                bounds = [(0, self.startCap), (0, self.heightCap), (1, 10000)]
                res = evolution(self.fitPP, bounds=bounds, workers=-1)
                                                            
            self.update_states(res.x)
            print(i, 'out of', len(uniqP), ' , BIC: ,', self.BIC, end='\r')  
            
            i += 1
        print()                                               

    def fitPP(self,params):
        if self.rmse:
            stepStart,stepHeight = params
            executionVar = None
        else:
            stepStart,stepHeight,executionVar = params
        stepStart = int(np.ceil(stepStart))
        pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        blockNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == blockNums[0]]
        aims = pDat['aim'].values
        trials = np.arange(len(aims))
        mOuts = np.zeros_like(aims, dtype=float)
        mOuts[trials >= stepStart] = stepHeight
        mask = ~np.isnan(aims)
        valid_aims = aims[mask]
        valid_mOuts = mOuts[mask]
        totErr = valid_aims - valid_mOuts
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
            liks = norm.pdf(valid_aims, valid_mOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        return -logLikelihood

    def update_states(self, params):
        if self.rmse:
            stepStart,stepHeight = params
            executionVar = None
        else:
            stepStart,stepHeight,executionVar = params
        stepStart = int(np.ceil(stepStart))
        pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        blockNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == blockNums[0]]
        aims = pDat['aim'].values
        trials = np.arange(len(aims))
        mOuts = np.zeros_like(aims, dtype=float)
        mOuts[trials >= stepStart] = stepHeight
        mask = ~np.isnan(aims)
        valid_aims = aims[mask]
        valid_mOuts = mOuts[mask]
        totErr = valid_aims - valid_mOuts
        numSamp = len(totErr)
        if numSamp == 0:
            return                               
        if self.rmse:
            sumSquares = np.sum(totErr ** 2)
            rmse = np.sqrt(sumSquares / numSamp)
            sortedErr = np.sort(totErr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            modelStd = np.sqrt(executionVar)
            liks = norm.pdf(valid_aims, valid_mOuts, modelStd) + 1e-12
            logLikelihood = np.sum(np.log(liks))
        k = 3
        
        BIC = k * np.log(numSamp) - 2 * logLikelihood
        if BIC < 0:                                                            
            print(logLikelihood)
            print(stepStart,stepHeight,executionVar)
            print(valid_aims,valid_mOuts)
            print(np.log(liks))
            print(liks)
        negLL = -logLikelihood
        if negLL < self.negLL[self.it]:
            self.negLL[self.it] = negLL
            self.BICs[self.it] = BIC
            self.BIC = BIC
            self.mStates[self.it] = mOuts.tolist()
            self.allAims[self.it] = aims
            if self.rmse:
                self.xs[self.it] = stepStart,stepHeight
            else:
                self.xs[self.it] = stepStart,stepHeight,executionVar

    def genDat(self,params,rots,trials=np.arange(-5,35,1)):
        ss,sh = params
        noise = 0
        trials_mod = np.asarray(trials)
        trials_mod[trials_mod >= 30] = -100
        mOuts = np.where(trials_mod < ss, 0, sh)
        states = mOuts + np.random.normal(0, noise, len(trials_mod))
        return states

