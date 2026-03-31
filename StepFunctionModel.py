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
    def __init__(self,stepStart,stepHeight):
        self.stepStart = stepStart
        self.stepHeight = stepHeight
        
    def modelMove(self,trialNum):
        if trialNum < self.stepStart:
            return 0
        else:
            return self.stepHeight
        
class fitShell:
    def __init__(self,df, conVal='none',condition='none',startCap=30,fitLen=320,fitPhase='rotation',heightCap=150):
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
        self.polishing = False
        
    def fitRot(self):
        pass

    def fitPP(self,params):
        pass

    def genDat(self,params,rots,trials=np.arange(-5,35,1)):
        ss,sh = params
        noise = 0
        sfm = Stepper(ss,sh)
        mOut = 0 + np.random.normal(0,noise)
        states = []
        for t in trials:
            if t >= 30:
                t = -100
            mOut = sfm.modelMove(t) + np.random.normal(0,noise)
            states.append(mOut)
        return states
