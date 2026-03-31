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
    def __init__(self,stepStart,stepHeight,lr,executionVariance,ret):
        self.stepStart = stepStart
        self.stepHeight = stepHeight
        self.lr = lr
        self.ret = ret
        self.ev = executionVariance
        self.ssm_state = 0.0
       
    def modelMove(self,trialNum,pert):
        if trialNum < self.stepStart:
            return 0, self.ev
        else:
            return self.stepHeight - self.ssm_state, self.ev
       
class fitShell:
    def __init__(self,df, conVal='none',condition='none',startCap=320,fitLen=320,fitPhase='rotation',heightCap=150,rmse=False,
                 method='L-BFGS-B', flipRot=False, imp=False):
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
        self.flipRot = flipRot
        self.imp = imp
       
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
            self.pp = pp
            self.it = i
            if self.rmse:
                bounds = [(0, self.startCap), (0, self.heightCap), (0, 1), (0,1)]
                res = evolution(self.fitPP, bounds=bounds, workers=-1)
            else:
                bounds = [(0, self.startCap), (0, self.heightCap), (0, 1), (0.1, 10000), (0,1)]
                res = evolution(self.fitPP, bounds=bounds, workers=-1)
                                                            
            self.update_states(res.x)
            print(i, 'out of', len(uniqP), ' , BIC: ,', self.BIC, end='\r')
            i += 1
        print()                                               

    def fitPP(self,params):
        if self.rmse:
            stepStart,stepHeight,lr,ret = params
            executionVar = None
        else:
            stepStart,stepHeight,lr,executionVar,ret = params
        stepStart = int(np.ceil(stepStart))
        pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        blockNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == blockNums[0]]
        aims = pDat['aim'].values
        rots = pDat['rotation'].values
        if self.flipRot:
            rots = -rots
        if self.imp:
            imps = pDat['imp'].values
        else:
            imps = np.zeros_like(rots)
        stepper = Stepper(stepStart,stepHeight,lr,executionVar,ret)
        totErr = []
        sts = []
        liks = []
        for t in range(len(aims)):
            rot = rots[t] if t > 0 else 0
            aim = aims[t]
            imp = imps[t]
            mOut,modelVar = stepper.modelMove(t,rot)
            sts.append(mOut)
            if t >= stepStart:
                error = -rot 
                stepper.ssm_state = stepper.ssm_state * ret + lr * error
            if not np.isnan(aim) and not np.isnan(imp):
                aim = int(aim)
                if not self.rmse:
                    modelStd = np.sqrt(modelVar)
                    lik = norm.pdf(aim,mOut,modelStd) + 1e-12
                    liks.append(lik)
                mErr = aim - mOut
                totErr.append(mErr)
        numSamp = len(totErr)
        if numSamp == 0:
            return np.inf
        if self.rmse:
            sumSquares = np.sum(np.array(totErr)**2)
            rmse = np.sqrt(sumSquares/numSamp)
            totErr_arr = np.array(totErr)
            totErr_arr = totErr_arr[np.isfinite(totErr_arr)]
            sortedErr = np.sort(totErr_arr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            logLikelihood = np.sum(np.log(liks))
        return -logLikelihood

    def update_states(self, params):
        if self.rmse:
            stepStart,stepHeight,lr,ret = params
            executionVar = None
        else:
            stepStart,stepHeight,lr,executionVar,ret = params
        stepStart = int(np.ceil(stepStart))
        pDat = self.dat[(self.dat['participantNum'] == self.pp) & (self.dat['phase'] == self.fitPhase)]
        blockNums = pDat['blockNum'].unique()
        pDat = pDat[pDat['blockNum'] == blockNums[0]]
        aims = pDat['aim'].values
        rots = pDat['rotation'].values
        if self.flipRot:
            rots = -rots
        if self.imp:
            imps = pDat['imp'].values
        else:
            imps = np.zeros_like(rots)
        stepper = Stepper(stepStart,stepHeight,lr,executionVar,ret)
        totErr = []
        sts = []
        liks = []
        for t in range(len(aims)):
            rot = rots[t] if t > 0 else 0
            aim = aims[t]
            imp = imps[t]
            mOut,modelVar = stepper.modelMove(t,rot)
            sts.append(mOut)
            if t >= stepStart:
                error = -rot
                stepper.ssm_state = stepper.ssm_state * ret + lr * error
            if not np.isnan(aim) and not np.isnan(imp):
                aim = int(aim)
                if not self.rmse:
                    modelStd = np.sqrt(modelVar)
                    lik = norm.pdf(aim,mOut,modelStd) + 1e-12
                    liks.append(lik)
                mErr = aim - mOut
                totErr.append(mErr)
        numSamp = len(totErr)
        if numSamp == 0:
            return
        if self.rmse:
            sumSquares = np.sum(np.array(totErr)**2)
            rmse = np.sqrt(sumSquares/numSamp)
            totErr_arr = np.array(totErr)
            totErr_arr = totErr_arr[np.isfinite(totErr_arr)]
            sortedErr = np.sort(totErr_arr)
            mu, std = norm.fit(sortedErr)
            logLikelihood = np.sum(np.log(norm.pdf(sortedErr, mu, std) + 1e-12))
        else:
            logLikelihood = np.sum(np.log(liks))
        k = 4 if self.rmse else 5
        BIC = k * np.log(numSamp) - 2 * logLikelihood
        negLL = -logLikelihood
        if negLL < self.negLL[self.it]:
            self.negLL[self.it] = negLL
            self.BICs[self.it] = BIC
            self.BIC = BIC
            self.mStates[self.it] = sts
            self.allAims[self.it] = aims
            if self.rmse:
                self.xs[self.it] = stepStart,stepHeight,lr
            else:
                self.xs[self.it] = stepStart,stepHeight,lr,executionVar

    def genDat(self,params,rots,trials=np.arange(-5,35,1)):
        if len(params) == 4:
            ss,sh,lr = params
            ev = None
        else:
            ss,sh,lr,ev,ret = params
        noise = 0
        sfm = Stepper(ss,sh,lr,ev,ret)
        states = []
        for idx, t in enumerate(trials):
            if t >= 30:
                t = -100
            rot = rots[idx]
            mOut, _ = sfm.modelMove(t, rot)
            if t >= ss:
                error = -rot
                sfm.ssm_state = sfm.ssm_state * ret + lr * error
            states.append(mOut + np.random.normal(0, noise))
        return states