"""
Created on Sat Mar 19 12:51:10 2022

@author: 44796
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution as evolution
from optimparallel import minimize_parallel
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import math 
import multiprocessing as mp
    
class ActionSelectionModelThreeParam:
    def __init__(self, alpha, sigmaE, sigmaM):
        self.alpha = alpha
        self.sigmaM = sigmaM                 
        self.sigmaE = sigmaE
        self.desiredCurr = 0
        self.sigmaN = sigmaM + sigmaE

    def updateModel(self, observed, desiredPrev, outcome):
        if outcome == 0:
            self.desiredCurr = desiredPrev
            self.sigmaN = self.sigmaM + self.sigmaE
        else:
            self.desiredCurr = desiredPrev + self.alpha * (observed - desiredPrev)
            self.sigmaN = self.sigmaM
        while self.desiredCurr > 180:
            self.desiredCurr -= 360
        while self.desiredCurr <= -180:
            self.desiredCurr += 360
            
    def getModelParams(self):
        return self.desiredCurr, self.sigmaN   
        
def simulate_logll(args):
    alpha, sigmaE, sigmaM, aims, rot = args
    ASM = ActionSelectionModelThreeParam(alpha, sigmaE, sigmaM)
    totErr = []
    sts = []
    liks = []
    logliks = []
    states = []
    for t in range(len(aims)):
        aim = aims[t]
        mErr = 0
        modelMean, modelVar = ASM.getModelParams()
        states.append(modelMean)
        kappa = 1000*np.pi*(1/modelVar)
        if not np.isnan(aim):
            sts.append((modelMean,modelVar))
            aim = int(aim)
            mErr = modelMean - aim
            sig = np.sqrt(modelVar)
            lik = norm.pdf(aim,modelMean,sig) + 1e-12
            liks.append(lik)
            sample = np.random.normal(modelMean,sig)
            if np.abs(sample - rot) <= 3:
                outcome = 1
            else:
                outcome = 0
            ASM.updateModel(sample, modelMean, outcome)
            totErr.append(mErr)
        else:
            sts.append((np.nan,np.nan))
                          
        
    logLL = np.sum(np.log(liks))
    return logLL, states
        
class fitShell:
    def __init__(self,df, conVal='none',condition='none',fitPhase='rotation',fitExecutionNoise=False,hasOutcomeMeasure=False,flipRot=False):
        self.conVal = conVal
        self.condition = condition
        self.df = df
        self.pp = 0
        self.mStates = [[]]
        self.dat = df
        self.BICs = []
        self.fitPhase = fitPhase
        self.fee = fitExecutionNoise
        self.hasOutcomeMeasure = hasOutcomeMeasure
        self.flipRot = flipRot
        
    def fitRot(self,plotting=False):
        lrUB = 1
        sigmaEUB = 10000
        sigmaMUB = 10000
        if self.condition != 'none':
            self.dat = self.df[(self.df[self.condition] == self.conVal)]
        else:
            self.dat = self.df
        uniqP = self.dat['participantNum'].unique()
        self.BICs = np.zeros(len(uniqP))
        self.negLL = np.ones(len(uniqP))*100000
        self.mStates = [[]]*len(uniqP)
        self.xs = [[]]*len(uniqP)
        i = 0
        for pp in uniqP:
            self.pp = pp
            self.it = i
            if not self.fee:
                print("functionality removed, please set fitExecutionNoise to True")
            else:
                res = minimize(self.fitPP,[0.9,500,500],method='Powell',bounds=[(0,lrUB),(0.1,sigmaEUB),(0,sigmaMUB)]) 
                                                                                              
                                                                                                                   
            self.xs.append(res.x)
            self.negLL[self.it] = res.fun
            i+=1

    def fitPP(self,params):
        alpha,sigmaE, sigmaM = params
        k = 3
        pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        blockNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == blockNums[0]]   
        rot = -pDat['rotation'].values[0]
        if self.flipRot:
            rot = -rot
        aims = pDat['aim'].values
                                                    
        self.aims = aims
        
        if self.hasOutcomeMeasure:
            outcomes = pDat['hit'].values
        scale = 180/np.pi                          
        
        numSims = 1000
        with mp.Pool() as pool:
            args_list = [(alpha, sigmaE, sigmaM, aims, rot) for _ in range(numSims)]
            results = pool.map(simulate_logll, args_list)
        simLogLiks = [r[0] for r in results]
        allStates = [r[1] for r in results]
        self.allStates = allStates
        numSamp = len(self.aims)
        logLikelihood = np.mean(simLogLiks)
        BIC = k * np.log(numSamp) - 2 * logLikelihood
        printing = False
        if printing:
            print(params)
                                                                                                                       
            print(logLikelihood)
        if -logLikelihood < self.negLL[self.it]:
            self.BICs[self.it] = BIC
            self.mStates[self.it] = allStates
        return -logLikelihood
    
    def rp(self,x):
        return math.radians(x)
    
    def genDat(self,params,rot,trials=np.arange(30),prepend=0):
        alpha,sigmaE,sigmaM = params
        ASM = ActionSelectionModelThreeParam(alpha,sigmaE,sigmaM)
        states = []
        for i in range(prepend):
            states.append(0)
        noise = 0
        sample = 0 + np.random.normal(0,noise)
        modelMean = 0
        for t in trials:
            states.append(sample)
            if np.abs(sample - rot) <= 3:
                outcome = 1
            else:
                outcome = 0
            ASM.updateModel(observed=sample, desiredPrev=modelMean, outcome=outcome)
            modelMean, modelVar = ASM.getModelParams()
            std = np.sqrt(modelVar)
            sample = np.random.normal(loc=modelMean, scale=std, size=1)
        return states
