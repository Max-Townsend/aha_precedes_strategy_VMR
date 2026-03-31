import os
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import operator
import pandas as pd
from statsmodels.stats.multitest import multipletests as multest
from pingouin import pairwise_tukey
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.optimize import curve_fit
from  matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
import statsmodels.api as sm
from statsmodels.api import RLM as rlm
from scipy.stats import ttest_1samp as oneSampT
from scipy.stats import ttest_ind as indT
from scipy.stats import ttest_rel as pairedT
from pingouin import ttest
import matplotlib.colors as mcolors
import colorsys
import colorcet as cc
from matplotlib.colors import rgb2hex
               
                                                   
                                                                              
colours = ['#44AA99'] + ['#88CCEE'] + ['#FF9825'] + ['#CC6677'] + ['#AA4499']
sns.set_theme(context='notebook', style='white', palette='colorblind', font='sans-serif', font_scale=1, color_codes=True, rc=None)
           
labFontSize = 28
subTitFontSize = 50
titleFontSize = 58
panelTitSize = 36
captionFontSize = 32
legFontSize = 12
NFontSize = 36
plt.rcParams['xtick.labelsize'] = labFontSize
plt.rcParams['ytick.labelsize'] = labFontSize
plt.rcParams['axes.titlesize'] = panelTitSize
plt.rcParams['axes.labelsize'] = panelTitSize
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams["legend.markerscale"] = 1
plt.rcParams.update({
    "text.usetex": False,      
    "font.family": "sans-serif",
    "font.serif": ["Arial"],
    "font.weight": "normal"})
plt.rcParams['axes.labelweight'] = 'bold'
markerEdgeWidth = 3

def bootStrapDF(df,numPPOut,rotAsCond=True,replace=True):
    if rotAsCond:
        rots = df['blockRot'].unique() 
        rots = sorted([i for i in rots if not np.isnan(i)])
        for r in rots:
            pps = df[df['blockRot'] == r]['participantNum'].unique()
            samples = np.random.choice(pps, size=numPPOut, replace=replace)
            zdf = copy.deepcopy(df[(df['participantNum'] == samples[0]) & (df['blockRot'] == r)])
            it = 1
            for pp in samples[1:]:
                tmp = copy.deepcopy(df[(df['participantNum'] == pp) & (df['blockRot'] == r)])
                tmp['participantNum'] = tmp['participantNum']+(it*1000)
                zdf= zdf.append(tmp, ignore_index=True)
                it+=1
            if r == rots[0]:
                outDf = copy.deepcopy(zdf)
            else:
                outDf = outDf.append(zdf,ignore_index=True)
    else:
        pps = df['participantNum'].unique()
        samples = np.random.choice(pps, size=numPPOut, replace=replace)
        zdf = copy.deepcopy(df[(df['participantNum'] == samples[0])])
        it = 1
        for pp in samples[1:]:
            tmp = copy.deepcopy(df[(df['participantNum'] == pp)])
            tmp['participantNum'] = tmp['participantNum']+(it*10000)
            zdf= zdf.append(tmp, ignore_index=True)
            it+=1
        outDf = zdf
    return outDf

def addConsTrial(df):
    pps = df['participantNum'].unique()
    cts = []
    for pp in pps:
        expLen = len(df[df['participantNum'] == pp])
        ppCons = df[df['participantNum'] == pp]['consistency'].iloc[0]
        if ppCons > 1:
            cTrial = flattenJagged([np.arange(0,ppCons,1).tolist() for i in range(int(expLen/ppCons))])
        else:
            cTrial = [0]+[1]*(expLen-1)
        cts.append(cTrial)
    df['consTrial'] = flattenJagged(cts)
    
                                                         
    
def addPrevRot(df):
    pps = df['participantNum'].unique()
    postRots = []
    for pp in pps:
        bRots = df[df['participantNum'] == pp]['rotation'].tolist()
        bRots.pop()
        bRots.insert(0,np.nan)
        postRots.append(bRots)
    postRots=flattenJagged(postRots)
    df['prevRot'] = postRots

def addCycleColumn(df,cLen=8,blLen=40,ppIndicator='participantNum'):
    expLen = len(df[df[ppIndicator] == df[ppIndicator].unique()[0]]) 
    cs = flattenJagged([flattenJagged([[t] * cLen for t in np.arange(-blLen//cLen,(expLen-blLen)//cLen)]) for i in df[ppIndicator].unique()])
    df['cycle'] = cs     
    
def addBlockCycle(df,cycleLength,baselineLength,blockLength,numBlocks):
    cycles = flattenJagged([flattenJagged([[t] * cycleLength for t in np.arange(-baselineLength//cycleLength,(blockLength-baselineLength)//cycleLength)]) for i in df['participantNum'].unique()])
    tmp = [i for i in cycles]
    [tmp.append(i) for i in cycles]
    df['blockCycle'] = tmp

def addTrialNums(df):
    ts = []
    expLen = [len(df['aim'].iloc[pp]) for pp in range(len(df))]
    ts = [np.arange(expLen[pp]) for pp in range(len(df))]
    df['trialNum'] = ts  
    
def addCueMarker(df,blockSize,baselineLength,washoutLength):
    pps = df['participantNum'].unique()
    cues = []
    for pp in pps:
        trials = df[df['participantNum'] == pp]['trialNum'].tolist()[0]
        uncuedIndices = df[df['participantNum'] == pp]['uncuedIndices'].tolist()[0]
        tmp = [i for i in uncuedIndices]
        for i in uncuedIndices:
            tmp.append(i+blockSize)
        uncuedIndices = [i+baselineLength for i in tmp]
        cueMarker = np.zeros(len(trials))
        for i in range(len(cueMarker)):
            if i-1 in uncuedIndices:                        
                cueMarker[i] = 2
            elif i not in uncuedIndices and ((i >= baselineLength and i < blockSize-washoutLength) or (i >= baselineLength + blockSize and i < (2 * blockSize) - washoutLength)):
                cueMarker[i] = 1
        cues.append(cueMarker)
    df['cue'] = cues
    print(cueMarker)
        
            
    
def addTotalAngle(df):
    df['totalAngle'] = df['aim'] + df['imp']

def addBlockNum(df,secSize,expLen,newColName='blockNum',cons=False):
    if cons:
        pps = df['participantNum'].unique()
        blockList = []
        for pp in pps:
            secSize = int(df[df['participantNum'] == pp]['consistency'].iloc[0])
            expLen = len(df[df['participantNum'] == pp])
            blockList.append([[i]*secSize for i in range(int(expLen/secSize))])
        blockList = flattenJagged(flattenJagged(blockList))            
    else:
        dfLen = len(df)
        blockList = flattenJagged(flattenJagged([[[i]*secSize for i in range(int(expLen/secSize))] for pp in range(int(dfLen/expLen))]))
    df[newColName] = blockList

def addSignFlipMarker(df,secSize,expLen,newColName='flip',cons=False):
    sfmList = []
    if cons:
        pps = df['participantNum'].unique()
        for pp in pps:
            secSize = int(df[df['participantNum'] == pp]['consistency'].iloc[0])
            rot = df[df['participantNum'] == pp]['rotation'].tolist()
            expLen = len(df[df['participantNum'] == pp])
            for sec in range(int(expLen/secSize)):
                sfm = [1 if np.nanmax(rot[sec*secSize:(sec+1)*secSize]) > 0 else 0] * secSize
                sfmList.append(flattenJagged(sfm))
    else:
        dfLen = len(df)
        compList = df['rotation'].tolist()
        for sec in range(int(dfLen/secSize)):
            tmp = compList[sec*secSize:(sec+1)*secSize]
            sfm = [1 if np.nanmax(tmp) > 0 else 0] * secSize
            sfmList.append(sfm)
    df[newColName] = flattenJagged(sfmList)
    
def addBlockTrial(df,secSize,expLen,blLength,newColName='blockTrial',cons=False):
    if cons:
        pps = df['participantNum'].unique()
        blockList = []
        for pp in pps:
            secSize = int(df[df['participantNum'] == pp]['consistency'].iloc[0])
            expLen = len(df[df['participantNum'] == pp])
            blockList.append([[i for i in range(secSize)]*int(expLen/secSize)])
        blockList = flattenJagged(flattenJagged(blockList))
    else:
        rotLen = secSize - (blLength*2)                        
        dfLen = len(df)
        blockList = flattenJagged(flattenJagged([[np.arange(-blLength,rotLen+blLength) for i in range(int(expLen/secSize))] for pp in range(int(dfLen/expLen))]))
    df[newColName] = blockList

def expFlipSign(df,chgVar,coeff=-1,condVar='flip',condVal=1):
    for c in chgVar:
        if c in df.columns:
            df.loc[df[condVar] == condVal, [c]] *= coeff
            
def addBlockRot(df,secSize,expLen,newColName='flip',cons=False):
    if cons:
        pps = df['participantNum'].unique()
        brot = []
        for pp in pps:
            pbr=[]
            secSize = int(df[df['participantNum'] == pp]['consistency'].iloc[0])
            rot = df[df['participantNum'] == pp]['rotation'].tolist()
            expLen = len(df[df['participantNum'] == pp])
            for sec in range(int(expLen/secSize)):
                r = rot[sec*secSize]
                tmp = [r]*secSize
                pbr.append(tmp)
            brot.append(flattenJagged(pbr))
        sfmList=brot
    else:
        dfLen = len(df)
        compList = df['rotation'].tolist()
        sfmList = []
        for sec in range(int(dfLen/secSize)):
            tmp = compList[sec*secSize:(sec+1)*secSize]
            tmp = tmp[int(len(tmp)/2)]                              
            sfm = [tmp] * secSize
            sfmList.append(sfm)
    df[newColName] = flattenJagged(sfmList)
    
"""
def condMulSign(df,condVar,comparOp,condVal,chgVar,coeff):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq}
    df.loc[ops[comparOp](df[condVar],condVal), [chgVar]] *= coeff
    #df[chgVar]=df[chgVar].apply(lambda x: func(x,fArg1) if ops[comparOp](df[condVar],condVal) else x)
 """ 
  
def selectiveTransform(dat,comparator,condition,transform):
    """
    selectiveTransform compares the comparator to condition elementwise
    if equal, transform the element in data at the same index
    params dat & comparator: 1-d vectors of same length
    param condition: list of values to compare against
    param transform: function to apply to appropriate elements
    return: selectively transformed data
    """
    for t in range(len(dat)):
        if comparator[t] in condition:
            dat[t] = transform(dat[t])
    return dat
       
def negOne(d):
    return np.multiply(d,-1)
    
def flipSign(dat,perts):
    """
    flipSign flips the sign of dat for trials which recieved a negative perturbation
    uses selectiveTransform() internally
    params: dat & perts 1-d vector of trials for single PP
    return: dat with aim for positive perts flipped, absolute perts
    """
    toFlip = [i for i in perts if i > 0]                                   
    dat = selectiveTransform(dat,dat,toFlip,negOne)
    absPerts = np.abs(perts)
    return dat, absPerts
def dfColumnManip(df,manipFunc,colsToManip=[],pertCol='null'):
    """
Not compatible with `selectiveTransform`.
    dfColumnManip allows the use of custom matrix-based operations
    to be applied to columns of a dataframe
    param manipFunc: function to be applied to columns, e.g., flipSign
    param colsToManip: list of column headers
    return: modified df
    """
    for c in colsToManip:
        if c in df.columns:
            dat = df[c]
            if pertCol != 'null':
                perts = df[pertCol]
                nuDat,nuPerts = manipFunc(dat,perts)
                                                 
                nuDat = boundVals(nuDat,c)
                df[pertCol] = nuPerts
            else:
                nuDat = manipFunc(dat)
                nuDat = boundVals(nuDat,c)
            df[c] = nuDat
    return df

def boundVals(dList,c):
    if c == 'at' or c == 'rt':
        tmp = []
        for pp in dList:
            tmp.append([i if i > 0.0 else 0.0 for i in pp])
        dList = tmp
    elif c in ['aim','error','imp']:
        tmp = []
        for pp in dList:
            pt = []
            for i in pp:
                if i <= -180:
                    pt.append(i+360)
                elif i > 180:
                    pt.append(i-360)
                else:
                    pt.append(i)
            tmp.append(pt)
        dList = tmp
    return dList

def collapseSign(dat,perts,absPList):
    """
    dat,perts = flipSign(dat,perts):
    collapsedDat = [[]]*len(absPList)
    for i in range(len(absPList)):
    """
    return
        
    
def pertOLRemoval():
    return

def CI(data, confidence=0.95):
    a = np.array(data)
    mask = ~np.isnan(a)
    effective_n = np.sum(mask)
    if effective_n < 2:
        return np.nan, np.nan, np.nan                  
    m = np.mean(a[mask])
    se = stats.sem(a[mask])
    df = effective_n - 1
    h = se * stats.t.ppf((1 + confidence) / 2., df)
    return m, m - h, m + h


def delBads(dat,rem):
    """
    delBads deletes the entries for the indices in rem
    param dat: first dimension must be participant index
    return: data after removal
    """
    for p in list(reversed(range(len(dat)))):
        if p in rem:
            del dat[p]
    return dat
                
def negDat(dat):
    """
    negDat multiplies the data by -1. Does not require np array.
    param dat: first dimension must be participant index
    return: negative dat
    """
    pit=0
    for p in dat:
        tit=0
        for t in p:
            dat[pit][tit]=-t
            tit+=1
        pit+=1
    return dat
   

def zeroDat(dat,pList=[90,60,45,30,15]):
    """
    zeroDat subtracts perturbation from data. pList must be appropriatley ordered
    param dat: must be of shape (perturbationGroup, participant, trial)
    return: dat - perturbation
    """
    git=0
    for g in dat:
        pit=0
        for p in g:
            tit=0
            for t in p:
                dat[git][pit][tit]=t-pList[git]
                tit+=1
            pit+=1
        git+=1
    return dat

def zeroCenter(arr):
    """
    zeroCenter transforms data to be -180 to 180
    return: transformed arr
    """
    finArr = []
    for pp in arr: 
        t=[]
        for trial in pp:
            if isinstance(trial,list):
                a = trial[-1]
            else:
                a = trial
            if a > 180:
                a = a-360
            t.append(a)
        finArr.append(t)
    return finArr


def deCom(arrStr):
    """
    deCom splits on (and removes) commas
    Must conain no special characters other than comma
    return: splitted arrStr
    """
    arrStr = str(arrStr)
    return arrStr.split(',')

def arrStr2Floats(arrStr):
    """
    arrStr2Floats splits (and removes) commas from string or list of strings, then removes empty splits
    Must conain no special characters other than comma
    return: list of floats
    """
    splitted = deCom(arrStr)
    return [float(i) for i in splitted if i != '']

def str2FList(inStr):
    """
    str2FList converts a string (of numbers seperated by commas) into a list of floats
    Must conain no special characters other than comma
    param inStr: string containing list of numbers
    return: list of floats
    """
    inStr = inStr[1:len(inStr)-1]
    inStr = inStr.replace('\n','')
    inStr = inStr.replace(' ',',')
    splat = inStr.split(',')
    return [float(i) for i in splat if i != '']

def colFloater(inCol):
    """
    colFloater converts a list of strings (of lists of numbers) into a two-dimensional list of floats
    Must conain no special characters other than comma
    return: 2-d list of floats
    """
    return [str2FList(i) for i in inCol[:len(inCol)]]

def flattenJagged(l):
    """
    flattenJagged flattens a jagged list
    return: flattened list
    """
    if type(l[0]) is not list and not isinstance(l[0], np.ndarray):
        return l
    else:
        flat = []
        for sublist in l:
            for item in sublist:
                flat.append(item)
        return flat

def getValidData(pDeg,perturbation,data):
    """
    getValidData returns the elements of data for which the experienced
    perturbation == pDeg
    return:list of data for trials with perturbation == pDeg
    """
    mask = [pDeg == i for i in perturbation]
    maskD = [data[i] for i in range(len(data)) if mask[i]]
    return maskD

def angleFromLoc(targetLoc,startLoc=[0,0.3]):
    """
    angleFromLoc converts cartesion coordinates in angular location
    """
    dy = targetLoc[1] - startLoc[1]
    dx = targetLoc[0] - startLoc[0]
    radAngle = math.atan2(dy,dx)
    degAngle = math.degrees(radAngle)
    degAngle*=2
    degAngle = np.abs(degAngle)
    if degAngle > 180:
        degAngle = -(360-degAngle)
    return -degAngle

def addLists(lists,func=np.add):
    """
    recursively apply add/subtract/multiply n-length list of lists
    """
    postFunc = func(lists[0],lists[1])
    for i in range(2):
        del lists[0]
    lists.insert(0,postFunc)
    if len(lists) == 1:
        return postFunc
    else:
        return addLists(lists,func=func)

def addErrorCol(df):
    """
    addErrorCol adds an error column to an unexploded df, which is the sum of aim, rotation, and imp
    """
    toAdd = ['aim','rotation']
    if 'imp' in df.columns:
        toAdd.append('imp')
    numPP = len(df)
    errs=[]
    for pp in range(numPP):
        pDat = df[df['participantNum'] == df['participantNum'].tolist()[pp]]
        if len(pDat[toAdd[0]].tolist()[0]) != len(pDat[toAdd[1]].tolist()[0]):                                                                                
            df = df.drop(df[df.participantNum == pp].index)
        else:
            error = addLists([pDat[i].tolist()[0] for i in toAdd])
            errs.append(error.tolist())
    df['error'] = errs
    return df
def outlierPPIDs(df,medATFloor=0.1,madErrCeil=20,propNanCeil=0.2):
    """
    outlierPPIDs takes in unexploded df and calculates outlier participants
    param medATFloor: outlier if median AT below this
    param madErrCeil: outlier if mad(err) above this
    param propNanCeil: outlier if proportion nan trials above this
    return rem,ATs,Errs,nans: all indices to remove, med(AT) values, mad(error) values, count(nan) values
    """
    colNames = df.columns
    imp = True if 'imp' in colNames else False
    toAdd = ['aim','rotation']
    if imp:
        toAdd.append('imp')
    numPP = len(df)
    rem=[]
    ATs=[]
    Errs=[]
    nans=[]
    numZeroAim = 0
    numTooManyNans = 0
    numTooLowAT = 0
    numMADError = 0
    for pp in df['participantNum'].unique():
        pDat = df[df['participantNum'] == pp]
                                                               
        absAims = np.abs(pDat['aim'].tolist()[0])
        rots = pDat['rotation'].tolist()[0]
        error = addLists([absAims,rots])
        if np.nanmedian(pDat['at'].tolist()[0]) < medATFloor:
            if pp not in rem:
                rem.append(pp)
                numTooLowAT+=1
        ATs.append(np.nanmedian(pDat['at'].tolist()[0]))
        c=0
        for i in pDat['aim'].tolist()[0]:
            if np.isnan(i):
                c+=1
        if (c/len(pDat['aim'].tolist()[0])) > propNanCeil:
            if pp not in rem:
                rem.append(pp)
                numTooManyNans+=1
        nans.append(c/len(pDat['aim'].tolist()[0]))
        if stats.median_abs_deviation(error, nan_policy='omit') > madErrCeil:
            if pp not in rem:
                rem.append(pp)
                numMADError+=1
        Errs.append(stats.median_abs_deviation(error, nan_policy='omit'))
        
        aims = pDat['aim'].tolist()[0]
        filtered_aims = [aims[i] for i, r in enumerate(rots) if r != 0]
        unique_targets = list(set(pDat['targetPosition'].tolist()[0]))
        if len(unique_targets) == 1:
            aims = filtered_aims[:30]                                                   
        if np.nanmean(np.abs(aims)) <= 3:                           
            if pp not in rem:
                rem.append(pp)
                numZeroAim += 1
                                                                                                    
    print("num exluded because zero mean aim MAGNITUDE: " + str(numZeroAim))
    print("num exluded because too many nans: " + str(numTooManyNans))
    print("num exluded because too high MAD error: " + str(numMADError))
    print("AT "+str(numTooLowAT))
    return rem,ATs,Errs,nans
        
def removeIndices(df,indices,condCol='participantNum'):
    """
    removeIndices removes rows from df where condCol isin indices
    return: df with indices removed, df of only removed indices
    """
    return df[~df[condCol].isin(indices)],df[df[condCol].isin(indices)]

def nanIndices(df,metrics,iqrLimit,numParts,keepFirst):
    """
    nanIndices takes in a unexploded df for individual PP and computes trial indices which have values >3iqr away from median
    param keepFirst: number of trials to make immune from removal after perturbation change
    return: indices to be nanned, thresholds for each metric, list indicating which trial a change in perturbation occured
    """
    toNan=[]
    lowT = []
    highT = []
    mNans = []
    for col in metrics:
        tmpMNan = []
        listed = df[col].tolist()[0]
        perts = df['rotation'].tolist()[0]
        expLen = (len(listed))
        tit=0
        tmpH = []
        tmpL = []
        changeSeen = False
        chDist = 0
        pertCh = []
        for i in range(numParts):
            floor = int(expLen * (i/numParts))
            ceil = int(expLen * ((1+i)/numParts))
            sec = listed[floor:ceil]
            med = np.nanmedian(sec)
            q75, q25 = np.nanpercentile(sec, [75 ,25])
            iqr = q75-q25
            high = med + (iqrLimit*iqr)
            low = med - (iqrLimit*iqr)
            for t in sec:
                if changeSeen:
                    chDist+=1
                if chDist == keepFirst:
                    chDist = 0
                    changeSeen = False
                if perts[tit] != perts[tit-1]:
                    if not np.isnan(perts[tit]) and not np.isnan(perts[tit-1]):
                        pertCh.append(1)
                        if keepFirst > 0:
                            changeSeen = True
                    else:
                        pertCh.append(0)
                else:
                    pertCh.append(0)
                tmpH.append(high)
                tmpL.append(low)
                if not np.isnan(t):
                    if tit not in toNan:
                        if t > high or t < low:
                            if not changeSeen:
                                toNan.append(tit)
                                tmpMNan.append(tit)
                tit+=1 
        lowT.append(tmpL)
        highT.append(tmpH)
        mNans.append(tmpMNan)
    return toNan,[lowT,highT],pertCh,mNans

                                                                                                               

                                   
    

                                                                                   
                                                          
def blockDeNanRotation(df,phaseChanges,baselineVal=0,washoutVal=0):
    dfLen = len(df)
    blockSize = sum(phaseChanges)
                                                        
    for i in range(dfLen):
        if i % blockSize == 0 and i > 0:
            phaseChanges = [i + blockSize for i in phaseChanges]
        if np.isnan(df['rotation'].iloc[i]):
            print(i)
            finalTrialIndices = [i-1 for i in phaseChanges]                              
            if i in finalTrialIndices:                                                                                           
                df['rotation'].iloc[i] = df['rotation'].iloc[i-1]
            else:   
                df['rotation'].iloc[i] = df['rotation'].iloc[i+1]

    
                                                                                                  
                                                                                  
"""TODO: delete this
def insertNanRow(df,expLen,resetIndex=False):
    if resetIndex:
        df = df.reset_index(drop=True)
    for i in reversed(range(len(df)-1)):
        currTrial = df['blockTrial'].iloc[i]
        prevTrial = df['blockTrial'].iloc[i-1]
        pp = df['participantNum'].loc[i]
        #remember we are cycling through in reverse
        #else if last entry of df
        if (i == len(df)-1 and nextTrial != expLen-1):#TODO: what if multiple trials missing at end
            insertLoc = len(df)
            rot = df['rotation'].iloc[i] # at some point in the future this may need to be more sophisticated, but for any soon use-case this magic number is fine
            trialNum = df['trialNum'].iloc[i] - 1
            blockNum = df['blockTrial'].iloc[i]
            blockTrial = currTrial - 1 #get from prev trial or need to pass this in to func as argument somehow
            blockRot = df['blockRot'].iloc[i]#get from prev trial
            cycle = df['cycle'].iloc[i]#get from prev trial but will need something more sophisticated elsewhere
            phase = df['phase'].iloc[i]#get from prev trial
            experiment = df['experiment'].iloc[i]#get from prev trial
        #elif (currTrial + 1 != prevTrial and currTrial + prevTrial != 359):
            #insert row of nans here (but copy important information from
            #if
        # and simple check for final trial missing based on expLen and add final row if absent
        df.loc[insertLoc] = [np.nan,pp,rot,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,trialNum,blockNum,np.nan,blockTrial,blockRot,cycle,phase,experiment]
    df = df.sort_index().reset_index(drop=True)
    return df
"""
    
                                     
def getNanRows(df):
                                
    tmp = df[df.isna().any(axis=1)] 
    return tmp

                                     
                                                 
def nanOutlierTrials(df,listCols=[],ATFloor=0.1,iqrLimit=3,numParts=4,ignore=[],keepFirst=1):
    """
    nanOutlierTrials takes in unexploded df and computes outlying trials based on error, at, and rt
    param keepFirst: keep first trial with new perturbation, even if outside threshold
    return: df with outlying trials nanned out, list of individual trial indices to be nanned (first half ATFloor, second half iqr thresholds),
    metric names, list of individual (floor and ceiling) thresholds for each metric (shape = PP*lowHigh*metric*trial),
    a list for each participant indicating which trial a change in perturbation occured
    UPDATE TODO:also now classes as outlier if participant didn't change aim from start position
    """
                                                                                               
    [ignore.append(i) for i in ['rotation','targetPosition','aim','imp','totalAngle']]                                  
    nanIdcs=[]
    atIdcs=[]
    numPP = len(df)
    for tmp in [col for col in ['error', 'at', 'rt', 'aim', 'imp', 'totalAngle'] if col in listCols]:
                              
            df['raw'+tmp] = copy.deepcopy(df[tmp].tolist())
                                                                                         
    errs=[]
                                       
    eLen = len(df['aim'].iloc[0])
    if 'at' not in ignore:
        for pp in df['participantNum'].unique():
            at = df[df['participantNum'] == pp]['at'].tolist()[0]
            rt = df[df['participantNum'] == pp]['rt'].tolist()[0]
            rotz = df[df['participantNum'] == pp]['rotation'].tolist()[0]
            aimz = df[df['participantNum'] == pp]['aim'].tolist()[0]
            toNan = []
            atNan = []
            tit = 0
            for t in at:
                if t < ATFloor and rotz[tit] != 0:
                    if tit != 40 or eLen != 400:                                                                                                       
                        toNan.append(tit)
                        atNan.append(tit)
                tit+=1
            for col in listCols:
                if col != 'rotation' and col != 'targetPosition':
                    listed = df[df['participantNum'] == pp][col].tolist()[0]
                    for t in toNan:
                        listed[t] = np.nan
                    if col != 'error':
                        df[df['participantNum'] == pp][col] = [listed]
                    else:      
                        errs.append(listed)
            nanIdcs.append(toNan)
            atIdcs.append(atNan)
        df.drop(columns=['error'])
        df['error'] = errs
    thresholds = []
    pertChanges = []
    metIdcs = []
    for pp in df['participantNum'].unique():
                                       
        metrics = [i for i in listCols if i not in ignore]                                          
        go = True
        for m in metrics:
            if len(df[df['participantNum'] == pp][m].tolist()) == 0:
                df = df.drop(df[df['participantNum'] == pp].index)
                go = False
                break
        if go:
            toNan,threshes,pertCh,mNans = nanIndices(df[df['participantNum'] == pp],metrics,iqrLimit,numParts,keepFirst)
            thresholds.append(threshes)
            pertChanges.append(pertCh)
            for col in listCols:
                if col != 'rotation' and col != 'targetPosition':
                    listed = df[df['participantNum'] == pp][col].tolist()[0]
                    for t in toNan:
                        listed[t] = np.nan
                    df[df['participantNum'] == pp][col] = [listed]
            nanIdcs.append(toNan) 
            metIdcs.append(mNans)
                                                                                                             
    metIdcs = [[[trial for trial in metIdcs[pp][met] if trial not in atIdcs[pp]] for pp in range(len(metIdcs))] for met in range(len(metrics))]
    return df,nanIdcs,metrics,thresholds,pertChanges,atIdcs,metIdcs

def getCounts(nanIdcs,expLen):
    """
    getCounts counts number of participant whose data on that trial was outlying
    return: counts for each trial
    """
    counts = []
    for t in range(expLen):
        c = 0
        for pp in nanIdcs:
            if t in pp:
                c+=1
        counts.append(c)
    return counts
    
def plotCounts(grpCounts,expLen,nCols,nRows,titles,pertChs,numPPs,atCounts,metCounts,metrics,yCeil=0.5):
    """
    plotCounts plots the count of participants whose trial is an outlier for each trial, for each condition
    param grpCounts: counts of nanned PPs for each condition, for each trial
    param pertChs: list for each condition, for each PP, whether the trial is a change in perturbation or not
    return: multi-paned plot for all groups
    """
    alpha=0.4
    fig, axs = plt.subplots(ncols=nCols,nrows=nRows)
    fig.supxlabel('Trial',fontsize=labFontSize)
    fig.supylabel('Proportion Participants Excluded',fontsize=labFontSize)
    fig.suptitle('Proportion of outlying participants per trial',fontsize=titleFontSize)
    c,r = 0,0
    for i in range(len(grpCounts)):
        if i>0 and i % nCols == 0:
            c = 0
            r += 1
        tmpAx = axs[r,c]
        totExc = sum(grpCounts[i])
                                                                                              
        sns.lineplot(x=np.arange(expLen[i]),y=np.divide(grpCounts[i],numPPs[i]),ax=tmpAx,color='black',alpha=alpha,label=('total'),legend=False)
        sns.lineplot(x=np.arange(expLen[i]),y=np.divide(atCounts[i],numPPs[i]),ax=tmpAx,color=colours[-3],alpha=alpha,label=('noAimTrial'),legend=False)
        [sns.lineplot(x=np.arange(expLen[i]),y=np.divide(metCounts[i][j],numPPs[i]),ax=tmpAx,color=colours[j],label=metrics[j],alpha=alpha,legend=False) for j in range(len(metrics))]
        tmpAx.set_title(titles[i]+'\nN = '+str(numPPs[i])+'\nProportion trials excluded: '+str(totExc/(expLen[i]*numPPs[i]))[:5])
        tit=0
        for t in pertChs[i][0]:
            if t == 1:
                tmpAx.axvline(linewidth=2,x=tit,color='red',alpha=1,ls='--')
            tit+=1
                                
        tmpAx.axhline(linewidth=2,y=-1,color='red',alpha=1,ls='--',label='PertChange')
        if i == 0:
            tmpAx.legend()
        xTix = np.arange(expLen[i],step = 40)
        tmpAx.set_xticks(xTix) 
        tmpAx.set_xticklabels(xTix)
        tmpAx.set_ylim(0,yCeil)
        tmpAx.set_xlim(0,400)
                                    
        c+=1
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    return fig, axs

def plotIndiOutliers(ppIndices,preData,postData,nRows,nCols,thresholds,varName,pertChs,supylab,ylabs,excIdcs,floor=False,fVal=0):
    """
    plotIndiOutliers plots individual data, pre- and post- outlier nanning, along with outlier thresholds and perturbation changes,
    nCols must be same as numPPs from each grp; nRows same as num groups
    param ppIndices: which participant indices to plot for each condition
    param preData: data including all outliers (shape = condition*pp*trial)
    param postData: data excluding outliers
    param thresholds: lower and upper thresholds for excluion for each condition, for each participant, for each trial
    shape = group*PP*lowHigh*trial
    return: multi-paned plot of outlier analyses for selected individuals
    """
                                                                                                                                       
    fig, axs = plt.subplots(ncols=nCols,nrows=nRows)
    fig.supxlabel('Trial',fontsize=labFontSize)
    fig.supylabel(supylab,fontsize=labFontSize)
    fig.suptitle('Individual time series data: '+varName,fontsize=titleFontSize)
    c,r = 0,0
    xmx = 0
    for cond in range(len(ppIndices)):
        ymx = 0
        ymn = 0
        condAxes = []
        pit = 0
        c = 0
        for pp in ppIndices[cond]:
            pre = preData[cond][pp]
            ymx = np.nanmax(pre) if np.nanmax(pre) > ymx else ymx
            ymn = np.nanmin(pre) if np.nanmin(pre) < ymn else ymn
                                      
            lowT = thresholds[cond][pp][0]
            highT = thresholds[cond][pp][1]
                                      
            exc = excIdcs[cond][pp]
            post = postData[cond][pp]
            pre = [pre[i] if i in exc else np.nan for i in range(len(pre))]
            tmpAx = axs[r,c]
            condAxes.append(tmpAx)
            sns.scatterplot(x=np.arange(len(pre)),y=pre,ax=tmpAx,color='red',label='Outlier',legend=False,edgecolor="black",linewidth=1)
            sns.scatterplot(x=np.arange(len(post)),y=post,ax=tmpAx,color=colours[1],label='Included',legend=False,edgecolor="black",linewidth=1)
            if not floor:
                tmpAx.fill_between(np.arange(len(post)),lowT,-1000,color='blue',alpha=0.15,label='exclude region')
                tmpAx.fill_between(np.arange(len(post)),lowT,highT,color='orange',alpha=0.1,label='include region')
                tmpAx.fill_between(np.arange(len(post)),highT,1000,color='blue',alpha=0.15)
            else:
                tmpAx.fill_between(np.arange(len(post)),fVal,-1000,color='blue',alpha=0.15,label='exclude region')
                tmpAx.fill_between(np.arange(len(post)),1000,fVal,color='orange',alpha=0.1,label='include region')
            if pit == len(ppIndices[cond])-1:
                c = 0
                r += 1
            tit=0
            for t in pertChs[cond][0]:
                if t == 1:
                    tmpAx.axvline(linewidth=2,x=tit,color='black',alpha=0.1,ls='--')
                tit+=1
            xTix = np.arange(len(pre),step = 100)
            xmx = len(pre) if len(pre) > xmx else xmx
            tmpAx.set_xticks(xTix) 
            tmpAx.set_xticklabels(xTix)
            if pit == 0:
                tmpAx.set_ylabel(ylabs[cond])
                if r == 0:
                    tmpAx.legend()                 
            c+=1
            pit+=1
        for ax in condAxes:
            if not floor:
                ax.set_ylim((ymn-0.1)*1.1,ymx*1.1)
            else:
                ax.set_ylim(-0.1,5)
    for ax in axs.flat:
        sns.despine(bottom=not ax.is_last_row(), left=not ax.is_first_col(), ax=ax)
        if not ax.is_first_col():
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)
        if not ax.is_last_row():
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
        ax.set_xlim(0,xmx)
    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    return fig, axs

def plotGrpOutliers(dats,bounds,floorCeil,datLabs,xLabs,paneTitles,rems,Ns):
    """
    plotGroupOutliers plots the distribution of each metric in dats and shades the exclusion region
    param floorCeil: list of integers indicating whether the bound is floor (0) or ceiling (1)
    param datLabs: list of condition names
    param xLabs: list of xLabels (i.e., metric for the dependent variables)
    param paneTitles: list of pane titles (i.e., dependent variable names)
    """
    nCols = len(dats)
    pal = list(reversed(colours[:len(datLabs)]))
                                                                                  
    nRows = 1
    fig, axs = plt.subplots(ncols=nCols,nrows=nRows)
    fig.supylabel('Count',fontsize=labFontSize)
    fig.suptitle('Participant Exclusion by metric',fontsize=titleFontSize)
    for i in range(nCols):
        flatDat = flattenJagged(dats[i])
        numPPs = Ns
        rem = [len(j) for j in rems]
        remProp = [j/k for j,k in zip(rem,numPPs)]
        excBound = np.max(flatDat) if floorCeil[i] == 1 else 0
        incBound = np.max(flatDat) if floorCeil[i] == 0 else 0
        tmpAx = axs[i]
        tmpAx.set_title(paneTitles[i],fontsize=subTitFontSize)
        tmpAx.set_xlabel(xLabs[i],fontsize=labFontSize)
        sns.histplot(dats[i],ax=tmpAx,label=datLabs,legend=False,palette=pal)
        if bounds[i] < excBound and floorCeil[i] == 1:
            excRange = np.arange(bounds[i],excBound,1e-3) 
        elif bounds[i] > excBound and floorCeil[i] == 0:
            excRange = np.arange(excBound,bounds[i],1e-3) 
        else:
            excRange = np.arange(excBound,bounds[i],1e-3) 
        tmpAx.fill_between(excRange,0,60,color='blue',alpha=0.15)
        tmpAx.set_ylim(0,60)
        sns.histplot(dats[i],ax=tmpAx,label=datLabs,legend=False)
                                                                
        if i == 0:
            [datLabs.insert(0,l) for l in ['Exclude Region']]
            tmpAx.legend(datLabs)
            
        elif i == nCols -1:
            props = dict(boxstyle='round', facecolor='wheat', alpha=1)
            textstr =''.join([datLabs[j+1]+': NExc = '+str(rem[j])+'; PropExc = ' + str(remProp[j])[:5] +'\n' for j in range(len(rems))])
            tmpAx.text(0.25, 0.95, textstr, transform=tmpAx.transAxes, fontsize=14,verticalalignment='top', bbox=props)

        tmpAx.axvline(linewidth=2,x=bounds[i],color='gray',alpha=1,ls='--')
    for ax in axs.flat:
        sns.despine(bottom=not ax.is_last_row(), left=not ax.is_first_col(), ax=ax)  
        if not ax.is_first_col():
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)
        if not ax.is_last_row():
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
        ax.set_ylabel('')
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    return fig,axs
    
    
                      
                                                                                  
                                                             
def aveVar(df,tStart,tEnd,x='blockTrial',y='aim',rots=[-90,-60,-45,-30,-15],aveFunc=np.nanmean,varFunc=np.nanstd,
           confidenceIntervals=False,confidence=0.95,ppAverages=False): 
    aves, varis, cis = [],[],[]
    for rot in rots:
        if ppAverages:
            dat = df[(df[x] >= tStart) & (df[x] < tEnd) & (df['blockRot'] == rot)]
            pps = dat['participantNum'].unique()
            dat = getParticipantAverages(dat,pps,y)
        else:
            dat = df[(df[x] >= tStart) & (df[x] < tEnd) & (df['blockRot'] == rot)][y]
        aves.append(aveFunc(dat))
        varis.append(varFunc(dat))
        if confidenceIntervals:
            cis.append(CI(dat,confidence))
    if confidenceIntervals:
        return aves,varis,cis
    else:
        return aves,varis

                      
    
                                                                                    
 
def CIMN(data):
    kis=[CI(ki) for ki in data]
    mmm=[qi[0] for qi in kis]
    bnds=[qi[2] - qi[0] for qi in kis]
    print('MEANS: ', mmm, '  95ci: ', bnds) 
    for inva in [2,1,0]:
        for a in range(len(pList)//2):
            t,p = pairedT(data[inva],data[a])
            ps.append(p)
            ts.append(t)
            if inva != a and p < 0.2:
                print('rots = ' +str(rots[inva])+' and '+str(rots[a]))
                print('t = ', t)
                print('bonferroni corrected p = ', p*3)
                print('dfs = ',2, 2*len(data[0])-2)
 
                     

                          
def tTestOneSample(df,y,rots=[-90,-60,-45,-30,-15],nanPolicy='omit', alternative='two-sided',correction="auto",paired=False,
          dimensions=1,confidence=0.95,oneSampleMean=0,multipleMeans=False):
    allStats = []
    it = 0
    for rot in rots:
        dat = df[df['blockRot'] == rot]
        pps = dat['participantNum'].unique()
        datA = getParticipantAverages(dat,pps,y)
        print(np.mean(datA))
        print(np.std(datA))
        
        if multipleMeans:
            datB = oneSampleMean[it]
        else:
            datB = oneSampleMean
        res = ttest(datA,datB,paired=paired,alternative=alternative,correction=correction,confidence=confidence)
        cilab = 'CI'+ str(int(confidence*100)) + '%'
        if multipleMeans:
            res[cilab] = [[round(i - oneSampleMean[it],5) for i in res[cilab][0]]]
        else:
            res[cilab] = [[round(i - oneSampleMean,5) for i in res[cilab][0]]]
        print(np.mean(datB))
        print(np.std(datB))
        allStats.append(res)
        it+=1
    return allStats 

def tTestTwoSample(df,y,sampleLabels,rots=[-90,-60,-45,-30,-15],nanPolicy='omit', alternative='two-sided',correction="auto",paired=False,
          dimensions=1,confidence=0.95,oneSampleMean=0,multipleMeans=False):
    allStats = []
    it = 0
    for rot in rots:
        print(rot)
        dat = df[(df['blockRot'] == rot) & (df['sample'] == sampleLabels[0])]
        pps = dat['participantNum'].unique()
        datA = getParticipantAverages(dat,pps,y)
        print(np.mean(datA))
        print(np.std(datA))
        dat = df[(df['blockRot'] == rot) & (df['sample'] == sampleLabels[1])]
        pps = dat['participantNum'].unique()
        datB = getParticipantAverages(dat,pps,y)
        res = ttest(datA,datB,paired=paired,alternative=alternative,correction=correction,confidence=confidence)
        cilab = 'CI'+ str(int(confidence*100)) + '%'
        if multipleMeans:
            res[cilab] = [[round(i - oneSampleMean[it],5) for i in res[cilab][0]]]
        else:
            res[cilab] = [[round(i - oneSampleMean,5) for i in res[cilab][0]]]
        print(np.mean(datB))
        print(np.std(datB))
        allStats.append(res)
        it+=1
    return allStats 
        
    
                                                                                                                     
def tTestPhase(df,tStartA,tEndA,tStartB=0,tEndB=0,x='blockTrial',y='aim',rots=[-90,-60,-45,-30,-15],nanPolicy='omit',
               alternative='two-sided',correction="auto",paired=False,dimensions=1,confidence=0.95,oneSampleMean=0,multipleMeans=False):
    allStats = []
    it = 0
    for rot in rots:
        dat = df[(df[x] >= tStartA) & (df[x] < tEndA) & (df['blockRot'] == rot)]
        pps = dat['participantNum'].unique()
        datA = getParticipantAverages(dat,pps,y)
        if dimensions > 1:
            dat = df[(df[x] >= tStartB) & (df[x] < tEndB) & (df['blockRot'] == rot)]
            datB = getParticipantAverages(dat,pps,y)
            res = ttest(datB,datA,paired=paired,alternative=alternative,correction=correction,confidence=confidence)
        else:
            if multipleMeans:
                datB = oneSampleMean[it]
            else:
                datB = oneSampleMean
            res = ttest(datA,datB,paired=paired,alternative=alternative,correction=correction,confidence=confidence)
            cilab = 'CI'+ str(int(confidence*100)) + '%'
            if multipleMeans:
                res[cilab] = [[round(i - oneSampleMean[it],5) for i in res[cilab][0]]]
            else:
                res[cilab] = [[round(i - oneSampleMean,5) for i in res[cilab][0]]]
        allStats.append(res)
        it+=1
    return allStats

                                  
def tTestRotSize():
    return    

def tTestExperiments(df,tStartA,tEndA,expLabels=['cg','bt'],x='blockTrial',y='aim',rots=[-90,-60,-45,-30,-15],nanPolicy='omit',
               alternative='two-sided',confidence=0.95):
    allStats = []
    for rot in rots:
        datA = df[(df[x] >= tStartA) & (df[x] < tEndA) & (df['blockRot'] == rot) & (df['experiment'] == expLabels[0])]
        datB = df[(df[x] >= tStartA) & (df[x] < tEndA) & (df['blockRot'] == rot) & (df['experiment'] == expLabels[1])]
        ppsA = datA['participantNum'].unique()
        datA = getParticipantAverages(datA,ppsA,y)
        ppsB = datB['participantNum'].unique()
        datB = getParticipantAverages(datB,ppsB,y)
        res = ttest(datA,datB,paired=False,alternative=alternative,correction='auto',confidence=confidence)
        allStats.append(res)
        print(np.mean(datA))
        print(np.std(datA))
        print(np.mean(datB))
        print(np.std(datB))
    return allStats

def getParticipantAverages(df,participantList,variable):
    aveList = []
    for pp in participantList:
        aveList.append(df.loc[df['participantNum'] == pp][variable].mean())
    return aveList

def bonferroniCorrectP(pValue,numComparisons):
    return pValue*numComparisons

                                

def addPhaseIndicator(df,phaseLengths,phaseLabels=['baseline','rotation','washout']):
    participants = df['participantNum'].unique()
    numBlocks = df['blockNum'].unique()
    phaseIndicatorList = []
    for pp in participants:
        for block in numBlocks:
            for pl in range(len(phaseLengths)):
                for i in range(phaseLengths[pl]):
                    phaseIndicatorList.append(phaseLabels[pl])
    df['phase'] = phaseIndicatorList
    return df
                                                                                                       

                            
def getHeatDF(df,expLen,numBlocks,var='aim',binWidth=8,x='blockTrial',cond='blockRot',squareRootTransform=False,yLow=-180,yHigh=181):
    bins = list(reversed(np.arange(yLow,yHigh,binWidth)))                                                          
    counts = []
    uniq = sorted(df[cond].unique())
    numPPs = len(df['participantNum'].unique())
    if expLen == 60:
        df = df[(df['blockTrial'] >= -5) & (df['blockTrial'] < 35)]
    elif expLen == 10:
        df = df[(df['blockTrial'] >= 0) & (df['blockTrial'] < 10)]
    xAxis = df[df['participantNum'] ==df['participantNum'].unique()[0]][x].tolist()
    blockLen = int(expLen/numBlocks)
    xAxis = xAxis[:blockLen]
    power = 0.5 if squareRootTransform else 1
                                                                                                            
                                                                                                                           
    for t in xAxis:
        tmp = []
        for rot in uniq:
            dat = df[(df[cond] == rot) & (df[x] == t)][var]
            denom=len(dat)
            rTmp = []
            for idx in range(len(bins)-1):
                l = bins[idx+1]
                h = bins[idx]
                rTmp.append((dat[(dat >= l) & (dat < h)].count()/denom)**power)
            tmp.append(rTmp)
        counts.append(flattenJagged(tmp))
    counts = np.transpose(counts)
    nuDf = pd.DataFrame(counts,columns=xAxis)
    nuDf['binCeil'] = flattenJagged([bins[:-1]]*len(uniq))
    nuDf['binFloor'] = flattenJagged([bins[1:]]*len(uniq))
    nuDf[cond] = flattenJagged([[rot] * (len(bins)-1) for rot in uniq])
                                                  
    return nuDf

    

               
    
def heatMap(df,numBlocks,xLox,xTix,xlim=(0,35),yTix=[],minYTix=[],yLabs=[],kdeDf=None,var='aim',infoCols=['binFloor','binCeil','blockRot'],
            kde=False,kdeBW=0.05,blLen=15,panelLabs=[''],capOffset=0,caption='',rotStart=40,rotLen=320,inAx=False,axs=None,
            overlayTrialSeries=True,figSize=(20,20),fileName='tmp.svg',save=False,xTickStepSize=1,yTickStepSize=5,cbar=False,showPlot=True,
            deleteTitle=False,norm=None,ySize=360,cmap="viridis",plotTitle=None):
    binSize = df['binCeil'].iloc[0] - df['binFloor'].iloc[0]
    yLen = int(ySize/binSize)
    yAxis = df['binCeil'].iloc[:yLen].tolist()
    uniq = sorted(df['blockRot'].unique())
    nCols = int(np.ceil(len(uniq)/3))
    nRows = int(np.ceil(len(uniq)/nCols))
    if not inAx:
        fig, axs = plt.subplots(ncols=nCols,nrows=nRows,figsize=figSize,squeeze=False)
        fig.supylabel('Aim (deg)',fontsize=labFontSize)
        fig.supxlabel('Trial with respect to perturbation onset',fontsize=labFontSize)
                                                                                        
                                                                    
    df = df.loc[:, ~df.columns.duplicated()]
    datDf = df[df.columns[~df.columns.isin(infoCols)]]
    it=0
    r,c=0,0
    kdeDf = copy.deepcopy(kdeDf)
    kdeDf['aim'] = (kdeDf['aim']*-0.125) + 22.5
    for i in range(len(uniq)):
        tmpAx = axs[r,c] 
        if it < 5:
                                      
            blockDf = datDf.iloc[(i*yLen):(i+1)*yLen]  
            cols = blockDf.columns
            blockDf[cols] = blockDf[cols].replace({'0':np.nan, 0:np.nan})
                                                                                
            if c == (nCols-1):
                sns.heatmap(blockDf,ax=tmpAx,cbar=cbar,cmap=cmap,norm=norm)
            else:
                sns.heatmap(blockDf,ax=tmpAx,cbar=cbar,cmap=cmap,norm=norm)
            if cbar:
                cbar = tmpAx.collections[0].colorbar
                cbar.ax.set_yticklabels('')
            if kde:
                kdeDat = kdeDf[(kdeDf['blockRot'] == uniq[i])]
                sns.kdeplot(x=np.add(kdeDat['blockTrial'].astype(float),blLen),y=(-1*np.divide(kdeDat[var].astype(float)-180,binSize)),ax=tmpAx,bw_method=kdeBW)
                                                               
            if overlayTrialSeries:
                plotTrialSeries(kdeDf[(kdeDf['blockRot'] == uniq[i])],yZeroLn=False,x='trialNum',hue='blockRot',markSize=10,idlLines=False,ax=tmpAx,inAx=True,palette=['gray'],alpha=0.4)
                                                                                                           
                                                                              
                                      
                                     
                                                                                                                                                                                          
                                                                   
                                                                                                      
            tmpAx.tick_params(rotation=0) 
            tmpAx.set_title(panelLabs[it])
            if deleteTitle:
                tmpAx.set_title('')
            tmpAx.set_xticks(xLox,xTix)
        c+=1
        if i > 0 and c % nCols == 0:
            c = 0
            r+=1
        it+=1
                                                                  
                                                                      
    tit=0            
    for ax in axs.flat:
        if True:                   
                                                                      
            ax.set_yticks(yTix,labels=yLabs,rotation=0)
                                                    
                                                                                      
        sns.despine(ax=ax)
        if False:                        
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)
        if False:                                   
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
        ax.set_ylabel('')
        if tit == 5:
            sns.despine(ax=ax)
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
        elif tit == 2:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)
        tit+=1
        if rotLen == 30:
            ax.set_xticks(np.arange(5.5,36.5,10))
            ax.set_xticklabels(np.arange(0,31,10))
        ax.set_xlim(xlim)
        ax.xaxis.set_minor_locator(AutoMinorLocator(xTickStepSize))
        ax.yaxis.set_minor_locator(AutoMinorLocator(yTickStepSize))
    plt.tight_layout()
    if not plotTitle == None:
        plt.title(plotTitle)
                                  
    if save:
        plt.savefig(fileName)
    if not showPlot:
        plt.ioff()
        plt.close();
    plt.show()
    

def plotSwitchyPairDifference(df,x="blockTrial",yZeroLn=True, y="difference",hue='blockRot',palette=colours,markSize=100,inAx=False,
                    ax=None,ylim=0,yflr=0,yMetric='degrees',aveFunc='mean',alpha=1,printN=False,figSize=(12,10),
                    xTickStepSize=2,yTickStepSize=5,fileName='tmp.svg',save=False,
                    yLim=(-25,100),xNumP=0.2,xLim=0,fontMult=1,numBlocks=1):
    numColours = list(sorted([i for i in df[hue].unique() if not np.isnan(i)]))
    pal = list(reversed(palette[:len(numColours)]))
    if not inAx:
        fig,ax=plt.subplots(figsize=figSize)
    df = copy.deepcopy(df[df['phase'] == 'rotation'])
    uniq = list(sorted([i for i in df['rotation'].unique() if not np.isnan(i) and i != 0]))
    pMasked = df[df['participantNum'] == np.min(df['participantNum'].tolist())][x].tolist()
                                   
    newDf = df[df['cue'] == 2]
    cue0Df = df[df['cue'] == 0]
    newDf['difference'] = np.subtract(newDf['aim'].tolist(), cue0Df['aim'].tolist())
    if aveFunc == 'mean':
        t = newDf.groupby([x,hue])[y].mean().reset_index()
    elif aveFunc == 'median':
        t = newDf.groupby([x,hue])[y].median().reset_index()
    for b in range(numBlocks):
        tmp = newDf[newDf['blockNum']==b]
        sns.lineplot(data=tmp, x=x, y=y,ci=95,linestyle='',hue=hue,palette=pal,legend=False,ax=ax,alpha=1)
    sns.scatterplot(data=t, x=x, y=y,s=markSize,hue=hue,palette=pal,ax=ax,alpha=alpha,edgecolor="black",linewidth=3) 
    ax.get_legend().remove()
    if x == 'cycle':
        ax.set_xticks(np.arange(0,41,10))
        ax.set_xticklabels(np.arange(0,41,10))
    else:
        ax.set_xticks(np.arange(0,31,10))
        ax.set_xticklabels(np.arange(0,31,10))
    ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':')
    hMax = pMasked[-1] + pMasked[1]                                                                               
    if x == 'cycle':
        hMax+=1
    it=0
    for r in np.multiply(sorted(uniq),-1):                             
        ax.plot(np.arange(0-0.5,hMax+0.5,1),[r]*(hMax+1),color=pal[it],ls='--',alpha=1,linewidth=2)
        it+=1
    ax.yaxis.set_minor_locator(AutoMinorLocator(yTickStepSize))
    ax.xaxis.set_minor_locator(AutoMinorLocator(xTickStepSize))
    ax.set_ylim(yLim)
    sns.despine(ax=ax)
    plt.tight_layout()
    if save:
        plt.savefig(fileName)
    return ax
    
def plotTrialSeries(df,x="blockTrial",yZeroLn=True, y="aim",hue='blockRot',palette=colours,markSize=100,idlLines=True,inAx=False,
                    ax=None,ylim=0,yflr=0,yMetric='degrees',aveFunc='mean',alpha=1,printN=True,figSize=(12,10*0.9324),
                    xTickStepSize=2,yTickStepSize=1,fileName='tmp.svg',save=False,findStartAsymp=False,asymTest='linReg',
                    tesLen=5,yLim=(-25,100),xNumP=0.2,xLim=0,fontMult=1,colourCue=False,numBlocks=1,forcePal=False,killLabels=False,
                    yLox=[],yTix=[],customY=False,ci=95,xTix=[]):
    """
    plotTrialSeries plots the trial-series of seriesVar, with confidence intervals, broken down by condition
    """
    if colourCue:
        uniq = list(sorted([i for i in df['rotation'].unique() if not np.isnan(i)]))
    else:
        uniq = list(sorted([i for i in df['blockRot'].unique() if not np.isnan(i)]))
    numColours = list(sorted([i for i in df[hue].unique() if not np.isnan(i)]))
    pal = palette[:len(numColours)]
    if colourCue:
        pal = list(reversed(pal))
    total_N = len(df['participantNum'].unique())
    plt.rcParams['xtick.labelsize'] = labFontSize*fontMult
    plt.rcParams['ytick.labelsize'] = labFontSize*fontMult
    plt.rcParams['axes.titlesize'] = panelTitSize*fontMult
    plt.rcParams['axes.labelsize'] = panelTitSize*fontMult
    plt.rcParams["legend.markerscale"] = 1.5*fontMult

    if aveFunc == 'mean':
        t = df.groupby([x,hue,'participantNum'])[y].agg(np.nanmean).reset_index()
        t = t.groupby([x,hue])[y].agg(np.nanmean).reset_index()
        ciDF = df.groupby([x,hue,'participantNum','blockNum'])[y].agg(np.nanmean).reset_index()
    elif aveFunc == 'median':
        t = df.groupby([x,hue,'participantNum'])[y].agg(np.nanmean).reset_index()
        t = t.groupby([x,hue])[y].agg(np.nanmedian).reset_index()
        ciDF = df.groupby([x,hue,'participantNum','blockNum'])[y].agg(np.nanmedian).reset_index()
    if not inAx:
        fig,ax=plt.subplots(figsize=figSize)
    tmp = df[df['participantNum'] == np.min(df['participantNum'].tolist())][x].tolist()
    hMax = int(tmp[-1] + tmp[1])                                                                               
    if x == 'cycle':
        hMax+=1
    if yZeroLn:
        if ylim != 0:
            ax.set_ylim(yflr,ylim)
        ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':',zorder=-10)
        ax.axvline(linewidth=2,x=0-0.5,color='gray',alpha=1,zorder=-10)
        ax.axvline(linewidth=2,x=hMax-0.5,color='gray',alpha=1,zorder=-10)
        ax.set_xlabel(x+' with respect to perturbation onset',fontsize=fontMult*labFontSize)
        ax.set_ylabel(y+' ('+yMetric+')',fontsize=fontMult*labFontSize)                                                              
        ax.set_title(x+'-series plot of '+y,fontsize=fontMult*titleFontSize)
        if xLim != 0:
            ax.set_xlim(xLim)
    if not colourCue:
        sns.lineplot(data=ciDF, x=x, y=y,ci=ci,linestyle='',hue=hue,palette=pal,legend=False,ax=ax,alpha=1)
                                                              
        for i, rot in enumerate(numColours):
            t_sub = t[t[hue] == rot]
            col = pal[i]
                                       
            col_clean = col.lstrip('#')
            if len(col_clean) == 3:
                col_clean = col_clean[0]*2 + col_clean[1]*2 + col_clean[2]*2
            col_rgb = (
                int(col_clean[0:2], 16) / 255.0,
                int(col_clean[2:4], 16) / 255.0,
                int(col_clean[4:6], 16) / 255.0
            )
            col_d = tuple(c * 0.75 for c in col_rgb)
            ax.scatter(t_sub[x], t_sub[y], s=markSize, color=col, edgecolor=col_d, linewidth=3, alpha=alpha, zorder=10)
    else:
        if forcePal:
            pal = [palette[1],palette[2],palette[0]]
        for b in range(numBlocks):
            tmp = ciDF[ciDF['blockNum']==b]
            sns.lineplot(data=tmp, x=x, y=y,ci=ci,linestyle='',hue=hue,palette=pal,legend=False,ax=ax,alpha=1)
                                                              
        for i, rot in enumerate(numColours):
            t_sub = t[t[hue] == rot]
            col = pal[i]
                                       
            col_clean = col.lstrip('#')
            if len(col_clean) == 3:
                col_clean = col_clean[0]*2 + col_clean[1]*2 + col_clean[2]*2
            col_rgb = (
                int(col_clean[0:2], 16) / 255.0,
                int(col_clean[2:4], 16) / 255.0,
                int(col_clean[4:6], 16) / 255.0
            )
            col_d = tuple(c * 0.75 for c in col_rgb)
            ax.scatter(t_sub[x], t_sub[y], s=markSize, color=col, edgecolor=col_d, linewidth=3, alpha=alpha, zorder=1000)
    if len(numColours) > 1:
        if colourCue and numColours[1] == 2:
            pal = list(reversed(pal))
    if idlLines:
        it=0
        for r in np.multiply(sorted(uniq),-1):                             
            ax.plot(np.arange(0-0.5,hMax+0.5,1),[r]*(hMax+1),color=pal[it],ls='--',alpha=1,linewidth=2,zorder=-1)
            it+=1
    for rot in uniq:
        dat = df[(df[hue] == rot) & (df['rotation'] == rot)]
        if findStartAsymp:
            xAx = sorted(dat[x].unique())
            yAx = dat.groupby([x]).mean()[y].tolist()
            ps,ts=[],[]
            asymIdx = np.nan
            asymVal = np.nan
            if asymTest == 'tTest':
                for t in xAx:
                    tmp = dat[dat[x] == t].groupby([x,'participantNum']).mean()[y].tolist()
                    res = stats.ttest_1samp(tmp,np.abs(rot),nan_policy='omit')
                    ps.append(res.pvalue)
                    ts.append(res.statistic)
                for t in range(len(ps)-tesLen):
                    if sum([ps[t+q] > 0.05 for q in range(tesLen)]) == tesLen:
                        asymIdx = xAx[t+1]
                        asymVal = yAx[t+1]
                        break
            elif asymTest == 'linReg':
                for t in range(len(xAx)):
                    tx = xAx[t:]
                    ty = yAx[t:]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(tx,ty)
                    if p_value > 0.05:
                        asymIdx = xAx[t]
                        asymVal = yAx[t]
                        break
            ax.scatter(asymIdx,asymVal,color='black',s=markSize,alpha=1,edgecolor="black",linewidth=3)
    if printN:
        ax.text(0.98, 0.95, f'N={total_N}', transform=ax.transAxes, fontsize=fontMult * labFontSize, va='top', ha='left')
    if len(xTix) > 0:
        ax.set_xticks(xTix)
    else:
        if x == 'cycle':
            ax.set_xticks(np.arange(0,41,10))
            ax.set_xticklabels(np.arange(0,41,10))
        else:
            ax.set_xticks(np.arange(0,31,10))
            ax.set_xticklabels(np.arange(0,31,10))
    if customY:
        ax.set_yticks(yLox,yTix)
    ax.yaxis.set_minor_locator(AutoMinorLocator(yTickStepSize))
    ax.xaxis.set_minor_locator(AutoMinorLocator(xTickStepSize))
    ax.set_ylim(yLim)
    if killLabels:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
    sns.despine(ax=ax)
    plt.tight_layout()
    if save:
        plt.savefig(fileName)
    return ax

def multiTrialSeries(df,numRots=1,x="blockTrial", y="aim",hue='blockRot',palette=colours,markSize=10,idlLines=True,numBlocks=3):
    nCols = int(numRots/2)
    nRows = int(numRots/nCols)
    fig, axs = plt.subplots(ncols=nCols,nrows=nRows)
    uniq = sorted(df[hue].unique())
    fig.supylabel('Aiming Angle (Degrees)',fontsize=labFontSize)
    fig.supxlabel('Trial with respect to perturbation onset',fontsize=labFontSize)
    fig.suptitle('Savings',fontsize=titleFontSize)
    row,c=0,0
    it=0
    for r in uniq:
        plotTrialSeries(df[(df[hue] == r) & (df['blockNum'] < numBlocks)],hue='blockNum',markSize=100,idlLines=False,ax=axs[row,c],inAx=True)
        axs[row,c].set_ylim(0.3*r,-r*1.5)
        if it != 0:
            axs[row,c].get_legend().remove()
        numP = len(df[(df[hue] == r) & (df['blockNum'] < numBlocks)]['participantNum'].unique())
        if row == 0:
            axs[row,c].set_title('Rotation size: '+str(r)+' degrees\nN = '+str(numP))
        else:
            axs[row,c].set_title('')
        tmp = df[df['participantNum'] == np.min(df['participantNum'].tolist())][x].tolist()
        hMax = tmp[-1] + tmp[1]
        axs[row,c].plot(np.arange(-0.5,hMax+0.5,1),[-r]*(hMax+1),color='gray',ls='--',alpha=1)
        c+=1
        if it > 0 and c % nCols == 0:
            c = 0
            row+=1
        it+=1
    for ax in axs.flat:
        sns.despine(bottom=not ax.is_last_row() and not ax.is_last_col() , ax=ax)  
        if not ax.is_first_col():
                                   
                                                             
            ax.set_ylabel('')
        if not ax.is_last_row() and not ax.is_last_col():
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    
def muiltCondTrialSeries(dfs,xs,expTitles,figSize=(20,10),supTitle='',y0='none',y="aim",y2='none',
                         hue='blockRot',palette=colours,markSize=100,idlLines=True,ylim=(-180,180),
                         caption='',xLabs=[''],yLabs=[''],legTitle='',legLabs=[''],xTix=[],xLox=[],
                         capOffset=-0.35,tesLen=0,heat=False,overlayTrialSeries=False,findStartAsymp=False,
                         insetYlim=(-2,20),insetZoom=2.5,insetLeft=True,insetTrialLen=8,asymTest = 'linReg'):
    ys = [y]
    tesLen+=1                          
    if y2 != 'none':
        ys.append(y2)
    if y0 != 'none':
        ys.insert(0,y0)
    numCond = len(dfs) * len(ys)
    nRows = int(np.ceil(numCond/2))
    nCols = int(np.ceil(numCond/nRows))
    if not heat:
        fig, axs = plt.subplots(ncols=nCols,nrows=nRows,squeeze=False,figsize=figSize)
        tcaption=''
    else:
        fig, axs = plt.subplots(ncols=2,nrows=4,squeeze=False,figsize=(figSize[0],figSize[1]*2),
                                gridspec_kw={'height_ratios': [1, 0.7,0.7,0.7]})
        heatAxs = axs[1:,:]
        t = getHeatDF(dfs[1],expLen=400,numBlocks=1,var='aim',binWidth=8,x='blockTrial',cond='blockRot')                      
        txLox = [np.arange(0,400,40)[it] for it in range(len(np.arange(0,400,40))) if it%2==1]
        txTix =[np.arange(-40,360,40)[it] for it in range(len(np.arange(-40,360,40))) if it%2==1]
        yTix = [np.arange(-2.5,46,5)[it] for it in range(10) if it%2==1]
        minYTix = np.arange(0.5,45,1)
        yLabs=[np.arange(200,-181,-40)[it] for it in range(10) if it%2==1]
        texpTitles=[str(i) + 'deg Rotation Group' for i in [90,60,45,30,15]]
        tcaption = ' '
        heatMap(df=t,kdeDf=dfs[1],numBlocks=1,kde=False,kdeBW=0.04,xLox=txLox,xTix=txTix,yTix=yTix,minYTix=minYTix,yLabs=yLabs,panelLabs=texpTitles,caption=tcaption,capOffset=-0.17,inAx=True,axs=heatAxs,
                overlayTrialSeries=overlayTrialSeries)
                                          
                                                              
    fig.suptitle(supTitle,fontsize=titleFontSize)
    row,c=0,0
    supI = 0
    for y in ys:
        it=0
        dit=0
        nxs = [0.2,0.07]
        for df in dfs:
                                            
                                      
            asymIdcs = []
            tmpAx = axs[row,c]
            x = xs[dit]
            plotTrialSeries(df,x=x,y=y,hue=hue,markSize=markSize,idlLines=idlLines,ax=tmpAx,inAx=True)
                                         
            """
            if True:# row != 0 or c != 0:
                tmpAx.get_legend().remove()
            else:
                h, l = tmpAx.get_legend_handles_labels()
                tmpAx.legend(frameon=False,title='',title_fontsize=legFontSize,
                   fontsize=legFontSize,labels=legLabs,handles=h,handletextpad=-0.4)
            """
            numP = []
            rots = sorted(df[hue].unique())
            for rot in rots:
                if not np.isnan(rot):
                    dat = df[(df[hue] == rot) & (df['rotation'] == rot)]
                    numP.append(len(dat['participantNum'].unique()))
                                       
                    if findStartAsymp:
                        xAx = sorted(dat[x].unique())
                                                                                               
                        yAx = dat.groupby([x]).mean()[y].tolist()
                                                                     
                                                              
                                                 
                                                                                                               
                        ps,ts=[],[]
                        asymIdx = np.nan
                        asymVal = np.nan
                        if asymTest == 'tTest':
                            for t in xAx:
                                tmp = dat[dat[x] == t].groupby([x,'participantNum']).mean()[y].tolist()
                                                                          
                                                                                                                    
                                res = stats.ttest_1samp(tmp,np.abs(rot),nan_policy='omit')
                                ps.append(res.pvalue)
                                ts.append(res.statistic)
                            for t in range(len(ps)-tesLen):
                                if sum([ps[t+q] > 0.05 for q in range(tesLen)]) == tesLen:
                                    asymIdx = xAx[t+1]
                                    asymVal = yAx[t+1]
                                    break
                                                     
                        elif asymTest == 'linReg':
                            for t in range(len(xAx)):
                                tx = xAx[t:]
                                ty = yAx[t:]
                                slope, intercept, r_value, p_value, std_err = stats.linregress(tx,ty)
                                if p_value > 0.05:
                                    asymIdx = xAx[t]
                                    asymVal = yAx[t]
                                    break
                        tmpAx.scatter(asymIdx,asymVal,color='black',s=markSize,alpha=1,edgecolor="black",linewidth=3)
            if row == 0:
                tmpAx.set_title(expTitles[dit],fontsize=subTitFontSize)
            else:
                tmpAx.set_title('')
            nys = [0.95,0.67,0.53,0.4,0.25]
                         
                                                                                                                                                                                   
            tmpAx.set_xlabel(xLabs[c])
            tmpAx.set_ylabel(yLabs[row])
            tmpAx.set_ylim(ylim)
            tmpAx.set_xticks(xLox[c],xTix[c])
            yInd = 1.2 if row == 0 else 1.2
            xInd =  -0.2 if c == 0 else 0
                                                  
            if c == 0 and insetLeft:
                axins = zoomed_inset_axes(tmpAx, insetZoom, loc='right')
                plotTrialSeries(df,x=x,y=y,hue=hue,markSize=markSize,idlLines=idlLines,ax=axins,inAx=True)
                tStart = 30.5
                x1, x2 = tStart, tStart + insetTrialLen
                axins.set_xlim(x1, x2)
                axins.set_ylim(insetYlim)
                mark_inset(tmpAx, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                axins.set_xticklabels(axins.get_xticklabels(),fontsize=10)
                removeAxComponents(axins,xTicks=True,despine=True,title=True,xLab=True,yLab=True,legend=True)              
            if c == 1:
                markOnly = zoomed_inset_axes(tmpAx, insetZoom*8, loc='right')
                markOnly.set_aspect(0.04)
                axins = zoomed_inset_axes(tmpAx, insetZoom, loc='right')
                plotTrialSeries(df,x='blockTrial',y=y,hue=hue,markSize=markSize,idlLines=idlLines,ax=axins,inAx=True)
                tStart = 320.5
                x1, x2 = tStart, tStart + insetTrialLen
                axins.set_xlim(x1, x2)
                axins.set_ylim(insetYlim)
                markOnly.set_ylim(insetYlim)
                tStart = 39.5
                markOnly.set_xlim(tStart,int(tStart+(insetTrialLen/8)))
                mark_inset(tmpAx, markOnly, loc1=2, loc2=4, fc="none", ec="0.5")
                                          
                axins.set_xticklabels(axins.get_xticklabels(),fontsize=10)
                removeAxComponents(markOnly,yTicks=True,xTicks=True,xTickLabels=True,yTickLabels=True,despine=True,title=True,xLab=True,yLab=True)
                removeAxComponents(axins,xTicks=True,despine=True,title=True,xLab=True,yLab=True,legend=True)    
                                                                                              
            c+=1
            if it > 0 and c % nCols == 0:
                c = 0
                row+=1
            it+=1
            dit+=1
            supI +=1
    for ax in axs.flat:
        sns.despine(bottom=not ax.is_last_row(),left=not ax.is_first_col() , ax=ax) 
        if not ax.is_first_col():
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)
            ax.set_ylabel('')
        if not ax.is_last_row():
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
            ax.set_xlabel('')
    fig.tight_layout()
    plt.subplots_adjust(left=0.1)
                                                                                               
                                           

def plotRelativeBIC(xBIC,yBIC,palette=colours,figSize=(20,10),secondaryPlot=False,markerSize=100,fileName="modelComparisonBIC.svg",ylim=(-60,230)):
    relBIC = []
    for r in range(len(xBIC)):
        rotRel = []
        for p in range(len(xBIC[r])):
            rel = xBIC[r][p] - yBIC[r][p]
            rotRel.append(rel)
        relBIC.append(rotRel)
                                 
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    scats = []
    cols = []
    it = 0
    for r in list(reversed(relBIC)):
        tmp = sns.scatterplot(x=np.arange(len(r)),y=list(sorted(r)),color=palette[4-it],ax=ax,s=markerSize*2,linewidth=markerEdgeWidth,edgecolor='black')
        scats.append(tmp.collections[0])
        it+=1
    ax.axhline(y=0,color='k')
    plt.xlabel('Participant')
    plt.ylabel('Relative BIC')
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=1, alpha=0.7, zorder=0)
    sns.despine(ax=ax)
    plt.ylim(ylim)
    leg = ax.legend(handles=scats,labels=[90,60,45,30,15])
    it=0
    for i in leg.legendHandles:
        i.set_color(palette[it])
        it+=1
    plt.savefig(fileName)
    plt.show()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(figSize[0]/4,10))
    it = 0
    for r in list(reversed(relBIC)):
        _,low,high = CI(r)
        r = np.nanmean(r)
        print(it,r,low,high)
        plt.errorbar([it],r,yerr=r-low,capsize=8,elinewidth=4,capthick=4,color='black',zorder=0)
        tmp = sns.scatterplot(x=[it],y=[r],color=list(reversed(palette))[it],zorder=1,ax=ax,s=markerSize*2,linewidth=markerEdgeWidth,edgecolor='black')
        it+=1
    plt.xlim(-1,10)
    ax.axhline(y=0,color='k')
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=1, alpha=0.7, zorder=0)
                                       
    plt.xlabel('')
    plt.ylabel('')
                      
                           
    ax.set_xticks([])
    ax.set_xticklabels([])
    sns.despine(ax=ax)
    plt.ylim(ylim)
    plt.savefig('groupAverage_'+fileName)
    plt.show()
    if secondaryPlot:
        flatRel = flattenJagged(relBIC)
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
        sns.scatterplot(x=np.arange(len(flatRel)),y = list(sorted(flatRel)),color='black',ax=ax)
        ax.axhline(y=0,color='k')
        ax.xlabel('Participant')
        ax.ylabel('Relative BIC')
        sns.despine(ax=ax)
                
        plt.show()
    
def plotThreeWayRelativeBIC(xBIC, yBIC, zBIC, xdse=None,ydse=None,palette=colours, figSize=(20, 10), secondaryPlot=False, markerSize=100, fileName="modelComparisonBIC.svg",
                            ylim=(-60, 230),xlim=(-70,200),xlab='Relative BIC (Insight vs SSM)',ylab='Relative BIC (Insight vs Cashaback)',
                            averageXlim=None,targetFolder=''):
    relBIC_xy = []
    relBIC_zy = []
    
    for r in range(len(xBIC)):
        rel_xy = []
        rel_zy = []
        for p in range(len(xBIC[r])):
            rel_xy_val = xBIC[r][p] - yBIC[r][p]
            rel_zy_val = zBIC[r][p] - yBIC[r][p]
            rel_xy.append(rel_xy_val)
            rel_zy.append(rel_zy_val)
        relBIC_xy.append(rel_xy)
        relBIC_zy.append(rel_zy)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figSize)
    scats = []
    cols = []
    it = 0
    reg = np.arange(1000)
    regy = np.ones(1000)*1000
    plt.fill_between(reg, regy, where=(reg >= 0) & (regy >= 0), color='gray',alpha=0.3)
    ax.axhline(y=0, color='k',zorder=-100)
    ax.axvline(x=0, color='k',zorder=-100)
    if not xdse == None:
        for r_xy, r_zy, r_xdse, r_ydse in zip(reversed(relBIC_xy), reversed(relBIC_zy), reversed(xdse),reversed(ydse)):
                                               
            plt.errorbar(x=r_xy, y=r_zy, xerr=r_xdse, yerr=r_ydse, fmt='none', ecolor=palette[4 - it],
                         elinewidth=2, capsize=4, capthick=2,alpha=0.9,zorder=0)
            tmp = sns.scatterplot(x=r_xy, y=r_zy, color=palette[4 - it], ax=ax, s=markerSize * 2,
                                  linewidth=markerEdgeWidth, edgecolor='black',zorder=1)
            scats.append(tmp.collections[0])
        
            

            it += 1
    else:
        for r_xy, r_zy in zip(reversed(relBIC_xy), reversed(relBIC_zy)):
            tmp = sns.scatterplot(x=r_xy, y=r_zy, color=palette[4-it], ax=ax, s=markerSize*2, linewidth=markerEdgeWidth, edgecolor='black')
            scats.append(tmp.collections[0])
            it += 1
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=1, alpha=0.7, zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    sns.despine(ax=ax)
    if False:
        leg = ax.legend(handles=scats, labels=[90, 60, 45, 30, 15])
        it = 0
        
        for i in leg.legendHandles:
            i.set_color(palette[it])
            it += 1
    
    plt.savefig(targetFolder+fileName)
    plt.show()
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figSize[0]/3, figSize[1]))
    it = 0
    plt.fill_between(reg, regy, where=(reg >= 0) & (regy >= 0), color='gray',alpha=0.3)
    plt.axvline(x=0, color='k',zorder=-100)
    plt.axhline(y=0, color='k',zorder=-100)
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=1, alpha=0.7, zorder=0)
    for r_xy, r_zy in zip(reversed(relBIC_xy), reversed(relBIC_zy)):
        _, low, high = CI(r_xy)
        r_xy_mean = np.nanmean(r_xy)
        _, low_x, high_x = CI(r_xy)
        _, low_y, high_y = CI(r_zy)
        plt.errorbar([np.nanmean(r_xy)], [np.nanmean(r_zy)], xerr=[[np.nanmean(r_xy)-low_x]], yerr=[[np.nanmean(r_zy)-low_y]], capsize=7, elinewidth=3, capthick=3, color=list(reversed(palette))[it], zorder=0)

        tmp = sns.scatterplot(x=[r_xy_mean], y=[np.nanmean(r_zy)], color=list(reversed(palette))[it], zorder=1, ax=ax, s=markerSize*2, linewidth=markerEdgeWidth, edgecolor='black')
        it += 1
    xMin, xMax = plt.xlim()
    yMin, yMax = plt.ylim()
    if averageXlim == None:
        plt.xlim(xMin-3, xMax+3)
    else:
        plt.xlim(averageXlim)
    print(xlim,averageXlim)
    plt.ylim(ylim)                 
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    sns.despine()
    plt.savefig(targetFolder + 'groupAverage' + fileName)
    plt.show()
                      



def plotMultipleDatasetsRelativeBIC(datasets, palette=colours, figSize=(20, 10), secondaryPlot=False,
                                   markerSize=100,alpha=1, fileName="modelComparisonBIC.svg",
                                   ylim=None, xlim=None, xlab='Relative BIC (Insight vs SSM)',
                                   ylab='Relative BIC (Insight vs Cashaback)', averageXlim=None,
                                   targetFolder='', markerEdgeWidth=1, showLegend=True):
    """
    Plots relative BICs for multiple datasets.
   
    Parameters:
    - datasets: list of dicts, each dict contains:
        - 'xBIC': list of lists of np.arrays (rotations x [array of participants' BICs]) or list of np.arrays
        - 'yBIC': identical structure
        - 'zBIC': identical structure
        - 'rotSizes': list of int (e.g., [15, 30, 45, 60, 90] or [30])
        - 'name': str (dataset name for legend)
        - 'xdse': list of lists of np.arrays (x errors, optional; same structure as xBIC)
        - 'ydse': list of lists of np.arrays (y errors, optional; same structure)
    - Other parameters similar to original.
    """
    rotSizes = list(reversed([15, 30, 45, 60, 90]))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(datasets)]
   
    def get_participants_bic(bic_data, rotIdx):
        if len(bic_data) == 1:
            bic = bic_data[0]
        else:
            bic = bic_data[rotIdx]
        if isinstance(bic, list):
            bic = bic[0]
        return np.asarray(bic)
   
                                              
    all_rel_x = []
    all_rel_y = []
    all_mean_x = []
    for ds in datasets:
        for rotIdx, rot in enumerate(ds['rotSizes']):
            if rot not in rotSizes:
                continue
            participantsX = get_participants_bic(ds['xBIC'], rotIdx)
            participantsY = get_participants_bic(ds['yBIC'], rotIdx)
            participantsZ = get_participants_bic(ds['zBIC'], rotIdx)
            relXY = participantsX - participantsY
            relZY = participantsZ - participantsY
            all_rel_x.extend(relXY)
            all_rel_y.extend(relZY)
            rXYMean = np.nanmean(relXY)
            all_mean_x.append(rXYMean)
    
    if xlim is None and all_rel_x:
        min_x, max_x = min(all_rel_x), max(all_rel_x)
        range_x = max_x - min_x if max_x > min_x else 1
        pad_x = max(2, 0.02 * range_x)
        xlim = (min_x - pad_x, max_x + pad_x)
    else:
        xlim = (-70, 200)
    
    if ylim is None and all_rel_y:
        min_y, max_y = min(all_rel_y), max(all_rel_y)
        range_y = max_y - min_y if max_y > min_y else 1
        pad_y = max(4, 0.02 * range_y)
        ylim = (min_y - pad_y, max_y + pad_y)
    else:
        ylim = (-60, 230)
    
    if averageXlim is None and all_mean_x:
        min_mx, max_mx = min(all_mean_x), max(all_mean_x)
        range_mx = max_mx - min_mx if max_mx > min_mx else 1
        pad_mx = max(5, 0.15 * range_mx)
        averageXlim = (min_mx - pad_mx, max_mx + pad_mx)
    else:
        averageXlim = (-70, 200)
   
                                                                                    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figSize)
    reg = np.arange(10000)
    regy = np.ones(10000) * 10000
    plt.fill_between(reg, regy, where=(reg >= 0) & (regy >= 0), color='gray', alpha=0.3)
    ax.axhline(y=0, color='k', zorder=-100)
    ax.axvline(x=0, color='k', zorder=-100)
   
    for dsIdx, ds in enumerate(datasets):
        marker = markers[dsIdx]
        for rotIdx, rot in enumerate(ds['rotSizes']):
            if rot not in rotSizes:
                continue                         
            colorIdx = rotSizes.index(rot)
            color = palette[colorIdx]
           
                                                                  
            participantsX = get_participants_bic(ds['xBIC'], rotIdx)
            participantsY = get_participants_bic(ds['yBIC'], rotIdx)
            participantsZ = get_participants_bic(ds['zBIC'], rotIdx)
           
            relXY = participantsX - participantsY
            relZY = participantsZ - participantsY
           
                                                   
            hasErrors = 'xdse' in ds and ds['xdse'] is not None and 'ydse' in ds and ds['ydse'] is not None
            if hasErrors:
                rXdse = get_participants_bic(ds['xdse'], rotIdx)
                rYdse = get_participants_bic(ds['ydse'], rotIdx)
                plt.errorbar(x=relXY, y=relZY, xerr=rXdse, yerr=rYdse, fmt='none',
                             ecolor=color, elinewidth=2, capsize=4, capthick=2, alpha=0.9, zorder=0)
           
                                 
            tmp = sns.scatterplot(x=relXY, y=relZY, color=color, ax=ax, s=markerSize * 2,
                                  marker=marker, linewidth=markerEdgeWidth/2, edgecolor='black', zorder=1,alpha=alpha)
   
    plt.ylim(ylim)
    plt.xlim(xlim)
                               
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=2, alpha=0.7, zorder=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    sns.despine(ax=ax)
   
    if showLegend:
                                                       
        colorHandles = [ax.scatter([], [], c=palette[i], s=markerSize * 2,
                                     marker='o', label=f'{rot}°')
                         for i, rot in enumerate(rotSizes)
                         if any(rot in ds['rotSizes'] for ds in datasets)]
                                                   
        markerHandles = [ax.scatter([], [], c='gray', s=markerSize * 2,
                                      marker=markers[i], label=ds['name'])
                          for i, ds in enumerate(datasets)]
        ax.legend(handles=colorHandles + markerHandles, loc='upper left',
                  bbox_to_anchor=(1.05, 1))
   
    plt.savefig(targetFolder + fileName)
    plt.show()
   
                                                                          
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figSize[0] / 3, figSize[1]))
    plt.fill_between(reg, regy, where=(reg >= 0) & (regy >= 0), color='gray', alpha=0.3)
    ax.axvline(x=0, color='k', zorder=-100)
    ax.axhline(y=0, color='k', zorder=-100)
   
    for dsIdx, ds in enumerate(datasets):
        marker = markers[dsIdx]
        for rotIdx, rot in enumerate(ds['rotSizes']):
            if rot not in rotSizes:
                continue
            colorIdx = rotSizes.index(rot)
            color = palette[colorIdx]
           
                                                           
            participantsX = get_participants_bic(ds['xBIC'], rotIdx)
            participantsY = get_participants_bic(ds['yBIC'], rotIdx)
            participantsZ = get_participants_bic(ds['zBIC'], rotIdx)
            rXY = participantsX - participantsY
            rZY = participantsZ - participantsY
           
            rXYMean = np.nanmean(rXY)
            rZYMean = np.nanmean(rZY)
            _, lowX, highX = CI(rXY)
            _, lowY, highY = CI(rZY)
                                                                       
            xerr = [[rXYMean - lowX]]
            yerr = [[rZYMean - lowY]]
            plt.errorbar([rXYMean], [rZYMean], xerr=xerr, yerr=yerr, capsize=2,
                         elinewidth=1.5, capthick=1.5, color=color, zorder=0,alpha=0.7)
            tmp = sns.scatterplot(x=[rXYMean], y=[rZYMean], color=color, zorder=1,
                                  ax=ax, s=markerSize * 6, marker=marker,
                                  linewidth=markerEdgeWidth, edgecolor='black',alpha=alpha)

    
    ax.set_xlim(averageXlim)
    ax.set_ylim(ylim)
                               
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', linewidth=2, alpha=0.7, zorder=0)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    sns.despine(ax=ax)
    plt.savefig(targetFolder + 'groupAverage' + fileName)
    plt.show()




def confIntervals(data, axis=0):
    n = data.shape[axis]
    mean = np.nanmean(data, axis=axis)
    cInt = 1.96 * np.nanstd(data, axis=axis) / np.sqrt(n)                   
    return cInt

                                                            
def dataFrameCI(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)                                              
    n = len(data)
    marginOfError = stats.t.ppf(0.975, df=n - 1) * (std / np.sqrt(n))
    lowerBound = mean - marginOfError
    upperBound = mean + marginOfError
    return (mean,lowerBound,upperBound)                                                                                      

def plotClampSims(data,angles,asympFunction,numGroups,groupClamps,groupFunc,nRows=3,nCols=3,dataRaw=[],groupColours=False,s=100):
    fig, axes = plt.subplots(nRows, nCols, figsize=(50, 50))
    for row in range(nRows):
        for col in range(nCols):
            ax = axes[row,col]
            ax.plot(angles,asympFunction,color='red',alpha=0.8)
            indiData = data[(row*nCols)+col]
            errors = []
            rawErrors = []
            for g in range(numGroups):
                gDat = indiData[indiData['group']==g]
                mean = gDat.groupby('clampSize')['handAngle'].mean().reset_index()['handAngle']
                if len(dataRaw) > 0:
                    rawGDat = dataRaw[(row*nCols)+col]
                    rawGDat = rawGDat[rawGDat['group']==g]
                    rawMean = rawGDat.groupby('clampSize')['handAngle'].mean().reset_index()['handAngle']
                    [rawErrors.append(j-i) for i,j in zip(groupFunc[g],rawMean)]
                    ax.scatter(groupClamps[g],rawMean,s=s,color='green',alpha=0.1)
                ci = gDat.groupby('clampSize')['handAngle'].agg(dataFrameCI).reset_index()['handAngle']
                low = [i[1] for i in ci]
                high = [i[2] for i in ci]
                                                                                
                if groupColours:
                    ax.scatter(groupClamps[g],mean,s=s)
                else:
                    ax.scatter(groupClamps[g],mean,s=s,color='grey')
                [errors.append(j-i) for i,j in zip(groupFunc[g],mean)]
                                                              
            se = [i*i for i in errors]
            mse = np.mean(se)
            rmse = np.sqrt(mse)
            rawRmse = np.sqrt(np.mean([i*i for i in rawErrors]))
            ax.set_title('rmse of recentred data: ' + str(rmse) +'\n rmse change vs raw data \n (negative = recentred better): '+str(rmse-rawRmse))
            [ax.axvline(x=i,color='k',alpha=0.1) for i in angles]
            ax.axhline(y=0,color='k',ls='--',alpha=.3)
            ax.set_xlabel('clamp size')
            ax.set_ylabel('asymptotic learning')
            ax.set_ylim(-20,60)
    plt.suptitle('Universal clamp groups',fontsize=50)
    plt.tight_layout()
    plt.show()

def plotClampSimsDiffs(data,angles,asympFunction,numGroups,groupClamps,groupFunc,nRows=3,nCols=3,dataRaw=[]):
    fig, axes = plt.subplots(nRows, nCols, figsize=(50, 50))
    for row in range(nRows):
        for col in range(nCols):
            ax = axes[row,col]
            ax.plot(angles,asympFunction,color='red',alpha=0.8)
            indiData = data[(row*nCols)+col]
            errors = []
            rawErrors = []
            for g in range(numGroups):
                gDat = indiData[indiData['group']==g]
                mean = gDat.groupby('clampSize')['diffs'].mean().reset_index()['diffs']
                if len(dataRaw) > 0:
                    rawGDat = dataRaw[(row*nCols)+col]
                    rawGDat = rawGDat[rawGDat['group']==g]
                    rawMean = rawGDat.groupby('clampSize')['diffs'].mean().reset_index()['diffs']
                    [rawErrors.append(j-i) for i,j in zip(groupFunc[g],rawMean)]
                                                                                     
                ci = gDat.groupby('clampSize')['diffs'].agg(dataFrameCI).reset_index()['diffs']
                low = [i[1] for i in ci]
                high = [i[2] for i in ci]
                                                                                
                ax.scatter(groupClamps[g],mean,s=100,color='grey')
                [errors.append(j-i) for i,j in zip(groupFunc[g],mean)]
                                                              
            se = [i*i for i in errors]
            mse = np.nanmean(se)
            rmse = np.sqrt(mse)
            rawRmse = np.sqrt(np.nanmean([i*i for i in rawErrors]))
            print(rmse,rawRmse)
            ax.set_title('rmse of recentred data: ' + str(rmse) +'\n rmse change vs raw data \n (negative = recentred better): '+str(rmse-rawRmse))
            [ax.axvline(x=i,color='k',alpha=0.1) for i in angles]
            ax.axhline(y=0,color='k',ls='--',alpha=.3)
            ax.set_xlabel('clamp size')
            ax.set_ylabel('asymptotic learning')
            ax.set_ylim(-20,60)
    plt.suptitle('Universal clamp groups',fontsize=50)
    plt.tight_layout()
    plt.show()



def plotParams(ssm, sfm, asm, rots=[90,60,45,30,15], markerSize=8, figsize=(20, 20), sqrtVariance=True, squeezeX=False, fileName='tmp.png', allPNames=[['Learning Rate','Retention Rate','Execution noise (STD)'], ['Changepoint','Step Height','Execution noise (STD)'], ['Learning Rate','Exploratory Noise (STD)','Execution noise (STD)', 'Fourth Parameter']]):
   
    modelNames = ['SSM','Insight Model','Cashaback RL Model']
    models = ['ssm', 'sfm', 'asm']
    model_data_list = [ssm, sfm, asm]
    
                                                                              
    max_params = max(len(model_data[0][0]) for model_data in model_data_list if model_data)
    
    dfs = []
    for model_idx, model_data in enumerate(model_data_list):
        for r, rows in enumerate(model_data):
            for p, col_data in enumerate(zip(*rows)):
                                                                  
                param_name = allPNames[model_idx][p] if p < len(allPNames[model_idx]) else f'Param {p+1}'
                if sqrtVariance and 'STD' in param_name:
                    col_data = np.sqrt(col_data)
                df = pd.DataFrame({
                    'model': np.repeat(models[model_idx], len(col_data)),
                    'rot': np.repeat(rots[r], len(col_data)),
                    'param': np.repeat(p, len(col_data)),
                    'val': col_data
                })
                dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(df)
    
                                                                   
    fig, axes = plt.subplots(3, max_params, figsize=figsize)
                                                   
    if max_params == 1:
        axes = axes.reshape(3, 1)
   
    m = 0
    for model_idx, (model, model_data) in enumerate(zip(models, model_data_list)):
        num_params = len(model_data[0][0]) if model_data else 0
        for p in range(num_params):
            ax = axes[m, p]
            dat = df[(df['param'] == p) & (df['model'] == model)]
            swarmplot = sns.swarmplot(data=dat, x='rot', y='val', ax=ax, hue='rot', palette=list(reversed(colours)), s=markerSize, edgecolor='black', linewidth=1.5)
            custom_legend = swarmplot.legend_
                                              
                                   
            param_name = allPNames[m][p] if p < len(allPNames[m]) else f'Param {p+1}'
            ax.set_ylabel(param_name)
            sns.despine(ax=ax)
            if squeezeX:
                pass                           
        
                                         
        for p in range(num_params, max_params):
            axes[m, p].set_visible(False)
        
        m += 1
    
    plt.tight_layout()
    plt.savefig(fileName)

def plotMultiDatasetParams(datasets, datasetLabels, rotationMap, hasImp, allPNames, rots=[90, 60, 45, 30, 15], markerSize=6, figsize=(20, 25), sqrtVariance=True, squeezeX=False, fileName='tmp.png', debug=False, plot_type='core'):
    """
    Plots parameter fits for multiple datasets across models in a multi-panel figure.
    plot_type: 'core' for main params (up to 3 for SSM/QLEARN, 6 for HMM), 'extras' for A, B, Implicit Generalisation (params 6,7,8).
  
    Each model gets its own block of rows (at most 4 panels per row) for core; for extras, 3x3 grid (models x params).
    Datasets are distinguished by markers (style), colors by rotation size (hue).
  
    Handles flexible structures: list of lists/arrays for per-person, or list of arrays for per-param.
    Dynamically determines numP per model instance.
  
    Args:
        ...
        plot_type: 'core' or 'extras'
        debug: If True, print debug info on xs shapes and param value ranges
    """
    modelLabels = ['SSM', 'QLEARN', 'HMM']
    num_models = 3
    base_numPs = [len(names) for names in allPNames]                          
    core_max_p = [3, 4, 6]                        
    param_rows_per_model = [int(np.ceil(base_numPs[i] / 4.0)) for i in range(num_models)]
    total_rows = sum(param_rows_per_model)
                                 
    rot_palette = dict(zip(rots,(colours)))
    colors = [rot_palette.get(r, 'gray') for r in rots]
                                 
    extra_names = [r'$A_{\mathrm{imp}}$', r'$B_{\mathrm{imp}}$', r'$\sigma_{\mathrm{imp}}$']
    for m_idx in range(num_models):
        current_len = len(allPNames[m_idx])
        needed = core_max_p[m_idx] + len(extra_names) - current_len
        if needed > 0:
            allPNames[m_idx].extend(extra_names[:needed])
  
                                                              
    dfs = []
    reorder_idx = [3, 4, 0, 5, 1, 2]                                    
    for ds_idx, ds_name in enumerate(datasetLabels):
        model_lists = datasets[ds_idx]
        rotations = rotationMap[ds_idx]
        imp_flag = hasImp[ds_idx]
      
        for rot_idx, rot in enumerate(rotations):
            fit_shells = []
            for m_idx in range(num_models):
                model_fit_list = model_lists[m_idx]
                if rot_idx < len(model_fit_list):
                    fit_shells.append(model_fit_list[rot_idx])
                else:
                    fit_shells.append(None)
          
            for m_idx, fit in enumerate(fit_shells):
                if fit is None:
                    continue
                model = modelLabels[m_idx]
              
                xs = fit.xs
                if debug:
                    print(f"\n--- {model} '{ds_name}' rot {rot} ---")
                    print(f"xs type: {type(xs)}, len(xs): {len(xs) if hasattr(xs, '__len__') else 'N/A'}")
                    if len(xs) > 0:
                        first = xs[0]
                        print(f"First item type: {type(first)}, len(first): {len(first) if hasattr(first, '__len__') else 'N/A'}")
              
                                                             
                is_per_person = False
                numP_model = None
                if isinstance(xs, list) and len(xs) > 0:
                    first_item = xs[0]
                    if hasattr(first_item, '__len__'):
                        n_items = len(xs)
                        len_first = len(first_item)
                        if len_first > 0 and n_items > 0:
                            if len_first == n_items:
                                numP_model = n_items
                                if debug:
                                    print(f"Detected per-param (square): {numP_model} params")
                            elif len_first in [3, 4, 6, 7, 9]:
                                is_per_person = True
                                numP_model = len_first
                                if debug:
                                    print(f"Detected per-person: {n_items} persons, {numP_model} params each")
                                raw_person_params = xs
                            elif n_items in [3, 4, 6, 7, 9]:
                                numP_model = n_items
                                if debug:
                                    print(f"Detected per-param: {numP_model} params, {len_first} subjects each")
                                raw_person_params = None
                            else:
                                                                                                                    
                                if len_first > core_max_p[m_idx]:
                                    is_per_person = True
                                    numP_model = len_first
                                    if debug:
                                        print(f"Relaxed detected per-person with extras: {n_items} persons, {numP_model} params each")
                                    raw_person_params = xs
                                else:
                                    if debug:
                                        print(f"Warning: Unexpected - n_items={n_items}, len_first={len_first}")
                                    continue
                        else:
                            if debug:
                                print(f"Warning: Empty items")
                            continue
                    else:
                        if debug:
                            print(f"Warning: First item not len-able: {type(first_item)}")
                        continue
                elif isinstance(xs, np.ndarray) and xs.ndim == 2:
                    n_persons, n_params = xs.shape
                    if n_params in [3, 4, 6, 7, 9] or n_params > core_max_p[m_idx]:
                        is_per_person = True
                        numP_model = n_params
                        raw_person_params = [xs[i, :] for i in range(n_persons)]
                        if debug:
                            print(f"Detected 2D per-person: {n_persons} x {numP_model} (extras if >{core_max_p[m_idx]})")
                    else:
                        if debug:
                            print(f"Warning: 2D shape {xs.shape} unexpected")
                        continue
                else:
                    if debug:
                        print(f"Warning: Unsupported xs type/shape: {type(xs)}")
                    continue
              
                if debug and numP_model < core_max_p[m_idx] + 3:
                    print(f"Note: {model} has {numP_model} params (no full extras)")
              
                                                             
                if is_per_person:
                    person_params = [np.array(person) for person in raw_person_params]
                    if m_idx == 2:                                     
                        core_params = 6
                        if numP_model > core_params:
                            reordered_core = [np.take(person[:core_params], reorder_idx) for person in person_params]
                            extra_params = [person[core_params:] for person in person_params]
                            person_params = [np.concatenate([re_core, extra]) for re_core, extra in zip(reordered_core, extra_params)]
                        else:
                            person_params = [np.take(person, reorder_idx[:numP_model]) for person in person_params]
                        if debug:
                            print(f"HMM reordered core using indices: {reorder_idx[:min(6, numP_model)]}, extras: {max(0, numP_model - 6)}")
                                                                                                                   
                    if m_idx == 0 and numP_model > 3:                  
                        core_params_ssm = 3
                        core = [person[:core_params_ssm] for person in person_params]
                        extras = [person[core_params_ssm:] for person in person_params]
                        extras_len = len(extras[0]) if extras else 0
                        if extras_len >= 2:
                            reordered_extras = []
                            for extra in extras:
                                reordered_extra = extra.copy()
                                reordered_extra[0], reordered_extra[1] = extra[1], extra[0]
                                reordered_extras.append(reordered_extra)
                            person_params = [np.concatenate([c, re]) for c, re in zip(core, reordered_extras)]
                            if debug:
                                print(f"SSM reordered extras: switched indices 0 and 1 (params {core_params_ssm} and {core_params_ssm+1})")
                        else:
                            if debug:
                                print(f"SSM extras too short for reordering: length {extras_len}")
                    param_arrays = [np.array([person[j] for person in person_params]) for j in range(numP_model)]
                else:
                    raw_param_arrays = [np.array(xs[p]) for p in range(numP_model)]
                    if m_idx == 2:                                     
                        core_params = 6
                        if numP_model > core_params:
                            reordered_core = [raw_param_arrays[i] for i in reorder_idx[:core_params]]
                            extra_params = raw_param_arrays[core_params:]
                            param_arrays = reordered_core + extra_params
                        else:
                            param_arrays = [raw_param_arrays[i] for i in reorder_idx[:numP_model]]
                        if debug:
                            print(f"HMM reordered core param arrays using indices: {reorder_idx[:min(6, numP_model)]}, extras: {max(0, numP_model - 6)}")
                    else:
                        param_arrays = raw_param_arrays
                                                                                                                   
                    if m_idx == 0 and numP_model > 3:                  
                        core_params_ssm = 3
                        core = param_arrays[:core_params_ssm]
                        extras = param_arrays[core_params_ssm:]
                        extras_len = len(extras)
                        if extras_len >= 2:
                            extras[0], extras[1] = extras[1], extras[0]
                            param_arrays = core + extras
                            if debug:
                                print(f"SSM reordered extras param arrays: switched indices 0 and 1 (params {core_params_ssm} and {core_params_ssm+1})")
                        else:
                            if debug:
                                print(f"SSM extras too short for reordering: length {extras_len}")
              
                                                                
                max_p = min(numP_model, len(allPNames[m_idx]))
                for p in range(max_p):
                    col_data = param_arrays[p]
                    param_name = allPNames[m_idx][p] if p < len(allPNames[m_idx]) else f'Param {p}'
                    if sqrtVariance and 'STD' in param_name:
                        col_data = np.sqrt(col_data)
                  
                    if debug:
                        print(f"p{p} '{param_name}': min={np.min(col_data):.3f}, max={np.max(col_data):.3f}, mean={np.mean(col_data):.3f}, n={len(col_data)}")
                  
                    temp_df = pd.DataFrame({
                        'model': [model] * len(col_data),
                        'dataset': [ds_name] * len(col_data),
                        'rot': [rot] * len(col_data),
                        'param': [p] * len(col_data),
                        'val': col_data,
                        'param_name': [param_name] * len(col_data)
                    })
                    dfs.append(temp_df)
  
    if not dfs:
        print("No data to plot.")
        return
    df = pd.concat(dfs, ignore_index=True)
    if debug:
        print(f"\nDF shape: {df.shape}")
        print(df.groupby(['model', 'param_name'])['val'].agg(['min', 'max', 'mean', 'count']).round(3))
  
                                  
    if plot_type == 'extras':
                                                                             
        numP = 3               
        num_rows = 3
        num_cols = 4
        fig_width = 20
        fig_height = (25 / 4) * num_rows
        figsize = (fig_width, fig_height)                                       
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)                                  
        axes = np.atleast_2d(axes)
                            
        model_to_row = {'SSM': 0, 'QLEARN': 1, 'HMM': 2}
        extra_start_p = {modelLabels[i]: core_max_p[i] for i in range(num_models)}                                   
    else:         
                                                                                                
        df = df[df['param'] < 6]                                                                                              
        param_rows_per_model = [int(np.ceil(core_max_p[i] / 4.0)) for i in range(num_models)]
        total_rows = sum(param_rows_per_model)
        fig, axes = plt.subplots(total_rows, 4, figsize=figsize)
        axes = np.atleast_2d(axes)
        model_to_row = None                      
        extra_start_p = None
  
                        
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:len(datasetLabels)]
    marker_dict = dict(zip(datasetLabels, markers))
  
                   
    num_ds = len(datasetLabels)
    dodge_width = 0.4
    offsets = np.linspace(-dodge_width / 2, dodge_width / 2, num_ds)
    dodge_dict = dict(zip(datasetLabels, offsets))
  
               
    current_block_start = 0
    handles = []
    labels = []
    for m_idx, model in enumerate(modelLabels):
        if plot_type == 'extras':
            row_start = model_to_row[model]
            num_pr = 1
            numP = 3
            core_start = extra_start_p[model]
        else:
            numP = core_max_p[m_idx]
            num_pr = param_rows_per_model[m_idx]
            row_start = current_block_start
            core_start = 0
      
        for pr in range(num_pr):
            for pc in range(4):
                local_p = pr * 4 + pc if plot_type == 'core' else pc
                if local_p >= numP:
                    if plot_type == 'extras':
                        axes[row_start, pc].set_visible(False)
                    else:
                        axes[row_start + pr, pc].set_visible(False)
                    continue
              
                if plot_type == 'extras':
                    ax = axes[row_start, pc]
                    global_p = core_start + local_p
                else:
                    ax = axes[row_start + pr, pc]
                    global_p = local_p
              
                dat = df[(df['model'] == model) & (df['param'] == global_p)]
              
                if len(dat) == 0:
                    if debug and plot_type == 'extras':
                        print(f"No data for {model} extra param {local_p} (global {global_p}) in extras plot")
                    ax.set_visible(False)
                    continue
              
                                  
                all_y = dat['val'].values
                if len(all_y) > 0 and np.isfinite(all_y).all():
                    q_low, q_high = np.percentile(all_y, [2.5, 97.5])
                    pad = 0.1 * (q_high - q_low)
                    ax.set_ylim(q_low - pad, q_high + pad)
              
                                                                          
                plotted = False
                np.random.seed(42)
                for ds_name in datasetLabels:
                    ds_dat = dat[dat['dataset'] == ds_name]
                    if len(ds_dat) == 0:
                        continue
                  
                    offset = dodge_dict[ds_name]
                    x_pos = ds_dat['rot'].values + offset
                    jitter = np.random.uniform(-2.5, 2.5, len(x_pos))
                    x_pos += jitter
                    colors = [rot_palette.get(r, 'gray') for r in ds_dat['rot']]
                    marker = marker_dict[ds_name]
                  
                    scatter = ax.scatter(x_pos, ds_dat['val'], c=colors, marker=marker, s=markerSize,
                                         edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)
                    plotted = True
                  
                    if not any(l == ds_name for l in labels):
                        handles.append(scatter)
                        labels.append(ds_name)
              
                if not plotted:
                    ax.set_visible(False)
                    continue
              
                                  
                param_name = allPNames[m_idx][global_p] if global_p < len(allPNames[m_idx]) else f'Param {global_p}'
                ax.set_ylabel(param_name)
                ax.set_xticks(rots)
                ax.set_xticklabels([str(r) for r in rots])                           
                ax.set_xlabel('Rotation (deg)')
                sns.despine(ax=ax)
              
                                                                
                if plot_type == 'extras':
                    ax.set_title(model, pad=20, fontweight='bold', fontsize=12)
                elif pr == 0 and pc == 0:
                    ax.set_title(model, pad=20, fontweight='bold', fontsize=12)
      
        if plot_type == 'core':
            current_block_start += num_pr
  
            
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=True, fancybox=False)
  
    plt.tight_layout()
    plt.savefig(fileName, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {fileName}")

def plotSecAves(df,tStart,tEnd,x='blockRot',y='aim'):
    t = df[(df['blockTrial'] >= tStart) & (df['blockTrial'] < tEnd)]
    d = t.groupby(['participantNum',x])[y].mean().reset_index()
    sns.pointplot(data=d,x=x,y=y,join=False,color='black',dodge=True,ci=95,zorder=100)
    sns.stripplot(data=d,x=x,y=y,zorder=0)
    sns.despine()
    plt.ylabel(y+' (degrees)',fontsize=labFontSize)
    plt.xlabel('Rotation'+' (degrees)',fontsize=labFontSize)
    plt.axhline(linewidth=2,y=0,ls='--',alpha=1,color='gray')
    plt.title('Mean individual '+y+' values for final '+str(tEnd-tStart)+' trials of block',fontsize=titleFontSize)
    
def plotSecVars(df,tStart,tEnd,x='blockRot',y='aim'):
    t = df[(df['blockTrial'] >= tStart) & (df['blockTrial'] < tEnd)]
    d = t.groupby(['participantNum',x])[y].std().reset_index()
    sns.pointplot(data=d,x=x,y=y,join=False,color='black',dodge=True,ci=95,zorder=100)
    sns.stripplot(data=d,x=x,y=y,zorder=0)
    sns.despine()
    plt.ylabel(y+' (STD, degrees)',fontsize=labFontSize)
    plt.xlabel('Rotation'+' (degrees)',fontsize=labFontSize)
    plt.title('STD individual '+y+' values for final '+str(tEnd-tStart)+' trials of block',fontsize=titleFontSize)

def plotAgeAves(df,plotMean=True):
    ages = getAges(df)
    sns.swarmplot(y=ages)
    mnAge = np.nanmean(ages)
    stdAge = np.nanstd(ages)
    if plotMean:
        plt.errorbar(0.05,mnAge,stdAge,markersize=30,marker='x')
    return mnAge,stdAge,ages

def agePerformanceScatter(df,y='error',x='age',fun=np.nanmean,ylab='',ylim=None,**kwargs):
    ages = getAges(df)
    ys = getVarAsList(df,y,fun=fun,**kwargs)
    sns.scatterplot(y=ys,x=ages)
    X = sm.add_constant(ages)
                                  
    lr = sm.OLS(ys,X,missing='drop')
    res = lr.fit()
    print(res.t_test([0,1]))
    constant,coef = res.params
    low = np.nanmin(ages)
    high = np.nanmax(ages)
    X = np.arange(low,high+1,1)
    y = [(i*coef)+constant for i in X]
    plt.plot(X,y)
    plt.ylabel(ylab)
    plt.xlabel(x)
    if ylim != None:
        plt.ylim(ylim)
    sns.despine()
    plt.show()
    return

def getVarAsList(df,x,fun=np.nanmean,**kwargs):
    pps = df['participantNum'].unique()
    xs = [fun(df[df['participantNum'] == i][x].tolist(),**kwargs) for i in pps]
    return xs

def getAges(df):
    pps = df['participant'].unique()
    ages = [df[df['participant'] == i]['age'].tolist()[0] for i in pps]
    ys = []
    for x in ages:
        if type(x) is float:
            y = np.nan
        elif x.isnumeric():
            y = float(x)
        else:
            y = np.nan
        ys.append(y)
    return ys

def isInt(x):
    if isinstance(x,int):
        return True
    elif isinstance(x, float):
        return True
    return False
        

           
    
def muiltiExpTrialSeries(dfs,x,expTitles,figSize=(12,10),supTitle='',y0='none',y="aim",y2='none',
                         hue='blockRot',palette=colours,markSize=100,idlLines=True,ylim=(-180,180),
                         caption='',xLabs=[''],yLabs=[''],legTitle='',legLabs=[''],xTix=[],xLox=[],
                         capOffset=-0.35,BTIdx=0,CGIdx=1,save=False,fileName='tmp.svg',
                         xTickStepSize=2,yTickStepSize=5,despineLeft=False):

    def darken_color(color, factor=0.5):
        rgb = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l = max(0, l * factor)                                 
        return colorsys.hls_to_rgb(h, l, s)
    ys = [y]
    if y2 != 'none':
        ys.append(y2)
    if y0 != 'none':
        ys.insert(0,y0)
    if len(dfs[0][hue].unique()) == 1:
        numCond = 1
    else:
        numCond = len(dfs) * len(ys)
    nRows = int(np.ceil(numCond/2))
    nCols = int(np.ceil(numCond/nRows))
    fig, axs = plt.subplots(ncols=nCols,nrows=nRows,squeeze=False,figsize=figSize)
             
                                                            
                                          
                                                              
    fig.suptitle(supTitle,fontsize=titleFontSize)
    alpha = 1
                          
    marks = ['o','o','o']
    markLabs=['BTAim','BTImplicit','BTTotal','CGAim','CGImplicit','CGTotal']
    row,c=0,0
    supI = 0
    for i in range(2):
        dfs[i]['cgbt'] = [i]*len(dfs[i])
    tukdf = pd.concat([dfs[0], dfs[1]])
    rit = 0
    for rot in sorted(dfs[0][hue].unique()):
        it=0
        dit=0
        for y in ys:
                                            
                                      
            if y != 'aim':
                alpha=0.3
                errBar=False
            else:
                alpha=1
                errBar=True
            td = dfs[BTIdx][(dfs[BTIdx][hue] == rot) & (dfs[BTIdx]['cycle'] < 40)]
            tmpAx = axs[row,c]
            t = td.groupby([x,hue])[y].mean().reset_index()
            bt_edge = darken_color(colours[1])
            sns.scatterplot(data=t, x=x, y=y,s=markSize,color=colours[1],ax=tmpAx,marker=marks[it],alpha=alpha,label=markLabs[it],edgecolor=bt_edge,linewidth=3)
            if errBar:
                sns.lineplot(data=td, x=x, y=y,ci=95,linestyle='',color=colours[1],legend=False,ax=tmpAx,alpha=alpha)
            td = dfs[CGIdx][(dfs[CGIdx][hue] == rot) & (dfs[CGIdx]['cycle'] < 40)]
            tmpAx = axs[row,c]
            t = td.groupby([x,hue])[y].mean().reset_index()
            cg_edge = darken_color(colours[2])
            sns.scatterplot(data=t, x=x, y=y,s=markSize,color=colours[2],ax=tmpAx,marker=marks[it],alpha=alpha,label=markLabs[it+3],edgecolor=cg_edge,linewidth=3)
            if errBar:
                sns.lineplot(data=td, x=x, y=y,ci=95,linestyle='',color=colours[2],legend=False,ax=tmpAx,alpha=alpha)
            hMax = 40
            tmpAx.axvline(linewidth=2,x=0-0.5,color='gray',alpha=1)
            tmpAx.axvline(linewidth=2,x=hMax,color='gray',alpha=1)
            tmpAx.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':')
            tmpAx.plot(np.arange(-0.5,hMax+0.5,1),[-rot]*(hMax+1),color='gray',ls='--',alpha=1,linewidth=2)
            if y == 'aim':
                ps=[]
                sts=[]
                for cyc in np.arange(0,40):
                    tst = (tukdf[(tukdf['cycle'] == cyc) & (tukdf['blockRot'] == rot)].pairwise_tukey(dv='aim', between='cgbt').round(3))
                    ps.append(tst['p-tukey'][0])
                    sts.append(tst['T'][0])
                sig,p,_,_=multest(ps,method='fdr_bh')
                for point in range(len(sig)):
                    if sig[point] and point >=5 and point < 45:                                        
                        tmpAx.fill_between(x=np.arange(point-5.45,point-4.45,0.001), y1=[-17], y2=[-20],fc='black',alpha=0.8,ec='white')
                                                                                    
                                 
                                                                                       
            if True:                    
                tmpAx.get_legend().remove()
            else:
                h, l = tmpAx.get_legend_handles_labels()
                tmpAx.legend(frameon=False,title='',fontsize=16,labels=l,handles=h,handletextpad=-0.4)
          
                                                      
            tmpAx.set_title(expTitles[rit])                                                                                                  
            tmpAx.set_xlabel(xLabs[c])
            tmpAx.set_ylabel(yLabs[row])
            tmpAx.set_ylim(ylim)
            tmpAx.set_xticks(xLox[c],xTix[c])
            tmpAx.set_yticks([0,15,30,45,60,75,90],[0,15,30,45,60,75,90])
            tmpAx.set_xlim(-0.75,39.75)
            yInd = 1.2 if row == 0 else 1.2
            xInd = -0.2 if c == 0 else 0
                                                                                                               
                                                                                              
            it+=1
            dit+=1
            supI +=1
        rit += 1
                
        c+=1
        if c % nCols == 0:
            c = 0
            row +=1
    axit = 0
    for ax in axs.flat:
        col_idx = axit % nCols
        row_idx = axit // nCols
        is_first_col = col_idx == 0
        is_last_row = row_idx == nRows - 1
        is_last_col = col_idx == nCols - 1
        if numCond == 1:
            sns.despine(left=despineLeft, ax=ax)
                                                                                   
        else:
            sns.despine(bottom=(not is_last_row or (is_last_col and not save)) and axit != 3,left=not is_first_col , ax=ax)
        if despineLeft or not is_first_col:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)
            ax.set_ylabel('')
        if not is_last_row and axit !=3:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
            ax.set_xlabel('')
        if axit == 5:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
            ax.set_xlabel('')
        if save:
            ax.yaxis.set_minor_locator(AutoMinorLocator(yTickStepSize))
            ax.xaxis.set_minor_locator(AutoMinorLocator(xTickStepSize))
        ax.set_ylim(ylim)
        axit+=1
    fig.tight_layout()
    plt.subplots_adjust(left=0.1)
                 
                                                                                    
    if save:
        plt.savefig(fileName)
                                           


def componentTrialSeries(dfs, x, expTitles, figSize=(12,10), supTitle='', y0='none', y="aim", y2='none',
                         hue='blockRot', palette=colours, markSize=100, idlLines=True, ylim=(-180,180),
                         caption='', xLabs=[''], yLabs=[''], legTitle='', legLabs=[''], xTix=[], xLox=[],
                         capOffset=-0.35, BTIdx=0, CGIdx=1, save=False, fileName='tmp.svg',
                         xTickStepSize=2, yTickStepSize=5, fitModel=False, singlePane=False, rotationOnly=False):
    if rotationOnly:
        dfs = [dfs[i][dfs[i]['rotation'] != 0] for i in range(len(dfs))]
    ys = [y]
    if y2 != 'none':
        ys.append(y2)
    if y0 != 'none':
        ys.insert(0, y0)
    numCond = len(dfs) * len(ys)
    if singlePane:
        numCond = 1
    nRows = int(np.ceil(numCond / 2))
    nCols = int(np.ceil(numCond / nRows))
    fig, axs = plt.subplots(ncols=nCols, nrows=nRows, squeeze=False, figsize=figSize)
    fig.suptitle(supTitle, fontsize=titleFontSize)
    alpha = 1
    marks = ['o', 'o', 'o']
    markLabs = ['BTAim', 'BTImplicit', 'BTTotal', 'CGAim', 'CGImplicit', 'CGTotal']
    lowVals = []
    peakVals = []
    row, c = 0, 0
    supI = 0
    rit = 0
    dfs = dfs[0]
    if x == 'cycle':
        xlim = (-0.75, 39.75)
    else:
        xlim = (-0.75, 29.75)
  
                                                                                 
                                                                                                 
    numeric_cols = []
    if y0 != 'none':
        numeric_cols.append(y0)
    numeric_cols.append(y)
    if y2 != 'none':
        numeric_cols.append(y2)
    numeric_cols.append(x)
    if 'rotation' in dfs.columns:
        numeric_cols.append('rotation')
    for col in numeric_cols:
        if col in dfs.columns:
            dfs[col] = pd.to_numeric(dfs[col], errors='coerce')
  
    for rot in sorted(dfs[hue].unique()):
        it = 0
        dit = 0
        for y in ys:
            errBar = True
            td = dfs[dfs[hue] == rot].copy()                                   
                                           
            for col in numeric_cols:
                if col in td.columns:
                    td[col] = pd.to_numeric(td[col], errors='coerce')
          
            tmpAx = axs[row, c]
            t = td.groupby([x, hue, 'rotation'])[y].mean().reset_index()
            tInd = td.groupby([x, hue, 'rotation', 'participantNum'])[y].mean().reset_index()
            
                                                                     
            y_lower = y.lower()
            if 'aim' in y_lower:
                face_color = '#EE6677'
                edge_color = '#A34436'                   
            elif 'imp' in y_lower or 'implicit' in y_lower:
                face_color = '#4477AA'
                edge_color = '#1B446D'                    
            elif 'total' in y_lower:
                face_color = '#AA3377'
                edge_color = '#6D1B44'                      
            else:
                face_color = 'gray'
                edge_color = 'black'
            
            sns.scatterplot(data=t, x=x, y=y, s=markSize, color=face_color, ax=tmpAx, marker=marks[it], alpha=alpha, label=markLabs[it+3], edgecolor=edge_color, linewidth=5)
            if errBar:
                sns.lineplot(data=td, x=x, y=y, ci=95, linestyle='', color=face_color, legend=False, ax=tmpAx, alpha=alpha)
          
                                                                 
            if y == 'aim':
                td_rot = td[td['rotation'] == rot]
                tInd_rot = tInd[tInd['rotation'] == rot]
               
                tmp = t[t['rotation'] != 0]
                if len(tmp) == 0 or tmp['aim'].isna().all():
                    peak = np.nan
                    peakIdx = np.nan
                    peakTrial = np.nan
                    low = np.nan
                    aimDec = np.nan
                    peakVals = []
                    lowVals = []
                    peakTs = np.array([], dtype=float)
                    mnAfterPeak = np.nan
                    stdAfterPeak = np.nan
                else:
                    peak = tmp['aim'].max()
                    peakIdx = tmp['aim'].idxmax()
                    peakTrial = tmp[x].loc[peakIdx]
                   
                                                                                        
                    group_max_idx = tInd_rot.groupby('participantNum')['aim'].idxmax()
                    peakTs = []
                    for idx in group_max_idx.values:
                        if pd.isna(idx):
                            peakTs.append(np.nan)
                        else:
                            try:
                                peakTs.append(tInd_rot.loc[idx, x])
                            except (KeyError, IndexError):
                                peakTs.append(np.nan)
                    peakTs = np.array(peakTs)
                   
                    uniqueP = tInd_rot['participantNum'].unique()
                    actualPeaks = []
                    lowVals = []
                    for i, pp in enumerate(uniqueP):
                        highest = np.nan
                        lowest = np.nan
                       
                                                           
                        if not pd.isna(peakTs[i]):
                            sub = tInd_rot[(tInd_rot['participantNum'] == pp) & (tInd_rot[x] == peakTs[i])]
                            if len(sub) > 0:
                                aim_vals = sub['aim'].dropna()
                                if len(aim_vals) > 0:
                                    highest = aim_vals.iloc[0]
                       
                                                  
                        this_df = tInd_rot[tInd_rot['participantNum'] == pp]
                        if len(this_df) > 0:
                            last_x = this_df[x].max()
                            if not pd.isna(last_x):
                                sub_low = this_df[this_df[x] == last_x]
                                if len(sub_low) > 0:
                                    aim_vals_low = sub_low['aim'].dropna()
                                    if len(aim_vals_low) > 0:
                                        lowest = aim_vals_low.iloc[0]
                       
                        actualPeaks.append(highest)
                        lowVals.append(lowest)
                   
                    peakVals = actualPeaks
                   
                    low = tmp['aim'].iloc[-1]
                    aimDec = peak - low
                   
                    slopeDat = tmp['aim'].loc[peakIdx:].to_numpy().reshape(-1, 1)
                   
                                                                                     
                    valid = ~np.isnan(slopeDat.flatten())
                    n_valid = np.sum(valid)
                    if n_valid >= 2:                                                              
                        obj = slopeDat[valid, 0]           
                        valid_offsets = np.arange(len(slopeDat))[valid]                            
                        inp_base = (peakIdx + valid_offsets).reshape(-1, 1)
                        origX = inp_base.copy()
                        inp = sm.add_constant(inp_base)
                        huber = rlm(obj, inp)
                        results = huber.fit()
                        pred = results.predict()
                        origX = origX.flatten()
                        if x == 'cycle':
                            origX -= 5
                        else:
                            origX -= 15
                        if fitModel:
                            sns.lineplot(x=origX, y=pred.flatten(), color=colours[-1], alpha=0.9, linewidth=10)
                    else:
                                                                           
                        pass
                   
                                                                              
                    if pd.isna(peakTrial):
                        filtered_td = pd.DataFrame()        
                    else:
                        filter_col = x
                        filtered_td = td_rot[td_rot[filter_col] >= peakTrial]
                    meansAfterPeak = filtered_td.groupby('participantNum')['aim'].mean().values
                    stdAfterPeak = np.nanstd(meansAfterPeak)
                    mnAfterPeak = np.nanmean(meansAfterPeak)
          
            if x == 'cycle':
                hMax = int(xlim[1])
            else:
                hMax = int(xlim[1])
            tmpAx.axvline(linewidth=2, x=0 - 0.5, color='gray', alpha=1)
            tmpAx.axvline(linewidth=2, x=hMax - 0.5, color='gray', alpha=1)
            tmpAx.axhline(linewidth=2, y=0, color='gray', alpha=1, ls=':')
            tmpAx.plot(np.arange(-0.5, hMax + 1.5, 1), [-rot] * (hMax + 2), color='gray', ls='--', alpha=1, linewidth=2)
            if True:                      
                tmpAx.get_legend().remove()
            else:
                h, l = tmpAx.get_legend_handles_labels()
                tmpAx.legend(frameon=False, title='', fontsize=16, labels=l, handles=h, handletextpad=-0.4)
          
            tmpAx.set_title('')
            tmpAx.set_xlabel('')
            tmpAx.set_ylabel('')
            tmpAx.set_ylim(ylim)
            tmpAx.set_xticks(xLox[c], xTix[c])
            tmpAx.set_xlim(xlim)
            yInd = 1.2 if row == 0 else 1.2
            xInd = -0.2 if c == 0 else 0
            it += 1
            dit += 1
            supI += 1
        rit += 1
        c += 1
        if c % nCols == 0:
            c = 0
            row += 1
  
                                                                                                  
    axit = 0
    for i in range(nRows):
        for j in range(nCols):
            ax = axs[i, j]
            is_last_row = (i == nRows - 1)
            is_last_col = (j == nCols - 1)
            is_first_col = (j == 0)
            sns.despine(bottom=(not is_last_row or (is_last_col and not save)) and axit != 3, left=not is_first_col, ax=ax)
            if not is_first_col:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', length=0)
                ax.set_ylabel('')
            if not is_last_row and axit != 3:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
                ax.set_xlabel('')
            if axit == 5:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=0)
                ax.set_xlabel('')
            if save:
                ax.yaxis.set_minor_locator(AutoMinorLocator(yTickStepSize))
                ax.xaxis.set_minor_locator(AutoMinorLocator(xTickStepSize))
            axit += 1
  
    if len(axs.flat) > 0:
        last_ax = axs.flat[-1]                                                             
        last_ax.set_yticks([0, 15, 30, 45, 60, 75, 90], [0, 15, 30, 45, 60, 75, 90])
        last_ax.set_ylim(ylim)
        if x == 'blockTrial':
            last_ax.set_xticks(np.arange(-10, 40, 10))
            last_ax.set_xticklabels(np.arange(-10, 40, 10))
            last_ax.set_xlim(xlim)
    fig.tight_layout()
    plt.subplots_adjust(left=0.1)
    if save:
        plt.savefig(fileName)
                                                                                                             
    if 'aim' in ys and 'results' in locals():
        pars = [pred[0], results.params[1]]
    else:
        pars = [np.nan, np.nan]                                        
    return pars, peak, low, aimDec, peakTrial, mnAfterPeak, stdAfterPeak, peakVals, lowVals, peakTs                                                                                          
    
def scatterParams(groupedParams,figSize=(12,12),markSize=100,fileName='tmp.svg',rots=[90,60,45,30,15],rotLines=True,idLine=False):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    markers = ['o','s','p','P']
    git = 0
    for g in groupedParams:
        rit=0
        print(g)
        for r in g:
            if rotLines:
                ax.axhline(linewidth=2,y=rots[rit],color=colours[rit],alpha=1,ls='--')
            plt.scatter(r[1],r[0],color=colours[rit],marker=markers[git],s=markSize,edgecolors='black',linewidths=3)
            rit+=1
        git+=1
    if idLine:
        plt.plot(np.arange(90),np.arange(90),linewidth=2,ls=':',color='black')    
    ax.axhline(linewidth=2,y=0,color='gray',alpha=0.95,ls='--')
    ax.axvline(linewidth=2,x=0,color='gray',alpha=0.95,ls='--')
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    fig.tight_layout()
    sns.despine()
    plt.savefig(fileName)
    
def jointPlot(data,x,y,hue=None,xlim=(-0.02,1.02),ylim=(-0.02,1.02),fileName='tmp.png',markerSize=40,suptitle='',kind='scatter'):
    if kind != 'hist':
        sns.jointplot(data=data,x=x,y=y,hue=hue,palette=colours[-1],height=12,s=markerSize,color='black',ylim=ylim,xlim=xlim,kind=kind)
    else:
        sns.jointplot(data=data,x=x,y=y,hue=hue,palette=colours[-1],height=12,color='black',ylim=ylim,xlim=xlim,kind=kind)
    plt.suptitle(suptitle)
    plt.savefig(fileName)
    
def baseSigmoid(df,x='prevRot',y='error',hue='consistency',xLox=[],xTix=[],consTrial=1,noise=0,figSize=(20,10),xLab='',yLab='',markSize=100,fileName='tmp.svg'):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    data = df[(df['consTrial']==consTrial) & (df['noise']==0)]
    numCons = len(df['consistency'].unique())
    pal = colours[:numCons]
    mns = data.groupby([x,hue]).mean()
    for c in df[hue].unique():
        print(c)
        print(len(data[data[hue]==c]['participantNum'].unique()))
    sns.scatterplot(data=mns,x=x,y=y,hue=hue,ax=ax,palette=pal,s=markSize,alpha=0.7,edgecolor="black",linewidth=3)
    sns.lineplot(data=data, x=x, y=y,ci=95,linestyle='',hue=hue,palette=pal,legend=False,ax=ax)
    sns.despine()
    x = np.arange(np.min(xLox),np.max(xLox))
    y = np.arange(np.max(xLox),np.min(xLox),-1)
    sns.lineplot(x,y,ax=ax,color='gray')
    ax.set_xlabel(xLab)
    ax.set_ylabel(yLab)
    ax.set_xticks(xLox,xTix)
    ax.set_yticks(xLox,xTix)
    ax.legend(frameon=False)
    ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':')
    plt.tight_layout()  
    plt.savefig(fileName)
    
def baseSigSim(df,x='prevRot',y='error',y2='simImp',hue='consistency',xLox=[],xTix=[],consTrial=1,noise=0,figSize=(20,10),xLab='',yLab='',markSize=100,fileName='tmp.svg'):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    data = df[(df['consTrial']==consTrial) & (df['noise']==0)]
    numCons = len(df['consistency'].unique())
    pal = colours[:numCons]
    mns = data.groupby([x,hue]).mean()
    for c in df[hue].unique():
        print(c)
        print(len(data[data[hue]==c]['participantNum'].unique()))
    sns.scatterplot(data=mns,x=x,y=y,hue=hue,ax=ax,palette=pal,s=markSize,alpha=0.7,edgecolor="black",linewidth=3)
    sns.lineplot(data=data, x=x, y=y,ci=95,linestyle='',hue=hue,palette=pal,legend=False,ax=ax)
    sns.lineplot(data=data, x=x, y=y2,ci=None,hue=hue,palette=pal,ax=ax,linewidth=5)
    sns.despine()
    x = np.arange(np.min(xLox),np.max(xLox))
    y = np.arange(np.max(xLox),np.min(xLox),-1)
    sns.lineplot(x,y,ax=ax,color='gray')
    ax.set_xlabel(xLab)
    ax.set_ylabel(yLab)
    ax.set_xticks(xLox,xTix)
    ax.set_yticks(xLox,xTix)
    ax.legend(frameon=False)
    ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':')
    plt.ylim(-8,8)
    plt.tight_layout()  
    plt.ylabel('Implicit angle (deg) on trial 2')
    plt.xlabel('rortation (deg)')
    plt.savefig(fileName)

def comparSigmoid(df,x='prevRot',y='error',hue='noise',consistency=7,xLox=[],xTix=[],consTrial=1,noise=0,figSize=(20,10),xLab='',yLab='',markSize=100,rotLim=16,fileName='tmp.svg'):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    if hue != 'consistency':
        data = df[(df['consTrial']==consTrial) & (df['consistency']==consistency) & (np.abs(df['prevRot'])<=rotLim)]
    else:
        data = df[(df['consTrial']==consTrial) & (np.abs(df['prevRot'])<=rotLim)]
    numHue = len(df[hue].unique())
    pal = colours[:numHue]
    mns = data.groupby([x,hue]).mean()
    sns.scatterplot(data=mns,x=x,y=y,hue=hue,ax=ax,palette=pal,s=markSize,alpha=0.7,edgecolor="black",linewidth=3)
    sns.lineplot(data=data, x=x, y=y,ci=95,linestyle='',hue=hue,palette=pal,legend=False,ax=ax)
    sns.despine()
    x = np.arange(-64,64)
    y = np.arange(64,-64,-1)
    sns.lineplot(x,y,ax=ax,color='gray')
    ax.set_xlabel(xLab)
    ax.set_ylabel(yLab)
    ax.set_xticks(xLox,xTix)
    ax.set_yticks(xLox,xTix)
    ax.set_ylim(-rotLim*1.1,rotLim*1.1)
    ax.set_xlim(-rotLim*1.1,rotLim*1.1)
    ax.legend(frameon=False)
    ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':')
    plt.tight_layout()    
    plt.savefig(fileName)     
    
def barPlot(df,y='aim',rotSize=45,xlabs=[],x='blockTrial',ylim=(-10,122),xIdcs=[],markSize=10,xLab='Section',yLab='Aim (deg)',fileName='tmp.svg',figSize=(7,14)):
    xLabs = [str(np.nanmean(i)) for i in xIdcs]
    df = df[df['blockRot']==rotSize]
    zdf = df[df[x].isin(xIdcs[0])]
    zdf['xLabs'] = xLabs[0]
    if len(xIdcs) > 1:
        it=1
        for idx in xIdcs[1:]:                      
            tdf =df[df[x].isin(idx)]
            tdf['xLabs'] = xLabs[it]
            zdf=zdf.append(tdf)
            it+=1
    data = zdf.groupby(['participantNum','xLabs']).mean()[y].to_frame()
    data['participantNum'] = [i[0] for i in data.index]
    data['xLabs'] = [i[1] for i in data.index]
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    sns.barplot(data=data, x='xLabs', y=y,ci=95,palette=colours,ax=ax,errwidth=10,capsize=0.2,errcolor='darkgray')
    sns.swarmplot(data=data,x='xLabs', y=y,ax=ax,color='none',size=markSize,edgecolor='gray', linewidth=0.5)
    sns.despine()
    ax.set_xlabel(xLab,fontsize=labFontSize)
    ax.set_ylabel(yLab,fontsize=labFontSize)
    ax.set_ylim(ylim)
    ax.axhline(linewidth=2,y=-rotSize,color='black',alpha=0.8,ls='--')
                                        
    ax.legend(frameon=False)
    ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':')
    plt.tight_layout()    
    plt.savefig(fileName)
    
def allRotBarPlot(df,y='aim',x='blockTrial',hue='section',ylim=(-10,122),xIdcs=[],markSize=8,xLab='section',yLab='Aim (deg)',fileName='tmp.svg',figSize=(20,10),
                    yLox=[],yTix=[],customY=False):
                                                                                                                                       
                                     
    
                                                                                              
    def hex_to_rgb(hex_str):
        hex_str = hex_str.lstrip('#')
        lv = len(hex_str)
        if lv == 3:
            rgb = tuple(int(hex_str[i:i+1] * 2, 16) / 255.0 for i in range(0, 3))
        elif lv == 6:
            rgb = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in range(0, 6, 2))
        elif lv == 8:                           
            rgb = tuple(int(hex_str[i:i+2], 16) / 255.0 for i in range(0, 6, 2))
        elif lv == 4:                      
            rgb = tuple(int(hex_str[i:i+1] * 2, 16) / 255.0 for i in range(0, 3))
        else:
            raise ValueError(f"Invalid hex colour format: {hex_str}")
        return rgb

    xLabs = [str(np.nanmean(i)) for i in xIdcs]
    df = copy.deepcopy(df)
    df['blockRot'] = -df['blockRot']
    rots = df['blockRot'].dropna().unique()
    zdf = df[df[x].isin(xIdcs[0])]
    zdf[xLab] = xLabs[0]
    if len(xIdcs) > 1:
        it=1
        for idx in xIdcs[1:]:                      
            tdf =df[df[x].isin(idx)]
            tdf[xLab] = xLabs[it]
            zdf=zdf.append(tdf)
            it+=1
                                                                          
    zdf['participantNum'] = zdf['participantNum'].astype('category')
  
                                                                
                                                                                                                                         
                                                                                                       
                                                                                                 
    bin_size = len(xIdcs[0])
    if bin_size > 20 and 'cycle' in zdf.columns:
                                                                                          
        cycle_data = zdf.groupby(['participantNum', 'blockRot', xLab, 'cycle'])[y].agg(np.nanmean).reset_index()
        data = cycle_data.groupby(['participantNum', 'blockRot', xLab])[y].agg(np.nanmean).reset_index()
    else:
                                                        
        data = zdf.groupby(['participantNum', 'blockRot', xLab])[y].agg(np.nanmean).reset_index()
  
    data[xLab] = data[xLab]                               
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    rots_sorted = sorted(rots)
    num_rots = len(rots_sorted)
    pal = list(reversed(colours[:num_rots]))
    hues = sorted(data[xLab].unique())
    num_hues = len(hues)
    box = sns.boxplot(data=data, x='blockRot', y=y,hue=xLab,palette=None,ax=ax,notch=True,showfliers=False,linewidth=5,showcaps=False,whiskerprops={'linewidth': 8, 'color': 'k'}, legend=False)
    for i, patch in enumerate(ax.patches):
        rot_idx = i // num_hues
        colour_str = pal[rot_idx]
        patch.set_facecolor(colour_str)
        colour_rgb = hex_to_rgb(colour_str)
        darkened_rgb = tuple(c * 0.5 for c in colour_rgb)
        col_d = darkened_rgb + (1.0,)                                           
        patch.set_edgecolor(col_d)
                                              
    x_shift = 0.2
    jitter_scale = 0.02
    dodge_width = 0.2
    offsets = [(hi - (num_hues - 1) / 2.0) * dodge_width for hi in range(num_hues)]
    for ri, rot in enumerate(rots_sorted):
        colour_str = pal[ri]
        colour_rgb = hex_to_rgb(colour_str)
        darkened_rgb = tuple(c * 0.5 for c in colour_rgb)
        col_d = darkened_rgb + (1.0,)                                           
        pos = ri
        for hi, sec in enumerate(hues):
            sub_data = data[(data['blockRot'] == rot) & (data[xLab] == sec)]
            if len(sub_data) == 0:
                continue
            offset = offsets[hi]
            x_base = pos + offset + x_shift
            x_plot = x_base + np.random.normal(0, jitter_scale, len(sub_data))
            ax.scatter(x_plot, sub_data[y], color=colour_str, edgecolor=col_d, linewidth=1.5, s=np.pi * (markSize / 2)**2, alpha=1, zorder=3)
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine()
    ax.set_xlabel('rotation',fontsize=labFontSize)
    ax.set_ylabel(yLab,fontsize=labFontSize)
    ax.set_ylim(ylim)
    it=0
    for rotSize in list(reversed(rots_sorted)):
        ax.axhline(linewidth=2,y=rotSize,color=colours[it],alpha=0.8,ls='--',zorder=-10)
        it+=1
    if customY:
        ax.set_yticks(yLox,yTix)
                                        
    ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':',zorder=-10)
    plt.tight_layout()
    plt.savefig(fileName)
    
def barCompar(s0,s1,figSize=(6,3),ylim=(0,110),col2=1,filename='tmp.svg'):
                  
                 
    mns = []
    stds = []
    for s in [s0,s1]:
        m,s = getMeansAndStds(s)
        mns.append(m)
        stds.append(s)
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    ax2 = ax.twinx()
    ax.bar(np.arange(len(s1))+0.2,mns[1],color=colours[col2],edgecolor='black',width=0.5,linewidth=2)
    ax.errorbar(np.arange(len(s1))+.2,mns[1],stds[1],color='black',fmt='none',elinewidth=3)
                                 
                           
    x = flattenJagged([flattenJagged([[i] * len(s1[0][0])]) for i in np.arange(len(s1))])
    y = flattenJagged(flattenJagged([flattenJagged(i[0].tolist()) for i in s1]))
    ax2.bar(np.arange(len(s0))-0.2,mns[0],color=colours[2],edgecolor='black',width=0.5,linewidth=2)
    ax2.errorbar(np.arange(len(s0))-0.2,mns[0],stds[0],color='black',fmt='none',elinewidth=3)
    ax.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax2.set_yticks([])
                                               
    sns.despine()
    plt.savefig(filename)
   
def boxPlotDifference(s0,s1,yLab='',expLabels=['cg','bt'],markSize=6,rots=[15,30,45,60,90],figSize=(10,10),ylim=(-10,100),col2=-1,
                      yLox=[0,15,30,45,60,90],yTix=[0,15,30,45,60,90],filename='tmp.svg',squeeze=True,arrToList=False,idlLines=True):
    s0 = copy.deepcopy(s0)
    s1 = copy.deepcopy(s1)
    if squeeze:
        s0 = [r[0] for r in s0]
        s1 = [r[0] for r in s1]
    exp = np.ones_like(flattenJagged(s0)).tolist()
    tmp = np.zeros_like(flattenJagged(s1))
    [exp.append(r) for r in tmp]
    exp = flattenJagged(exp)
    rot = [[rots[r]]*len(s0[r]) for r in range(len(s0))]
    tmp = [[rots[r]]*len(s1[r]) for r in range(len(s1))]
    if arrToList:
        s0 = s0.tolist()
    [rot.append(r) for r in tmp]
    rot = flattenJagged(rot)
    [s0.append(r) for r in s1]
    s0 = flattenJagged(s0)
    pid = np.arange(len(s0))
    df = pd.DataFrame({'val':s0, 'exp':exp, 'participant':pid, 'rotation':rot})
    exp_palette = [colours[2], colours[1]]                                                  
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    box = sns.boxplot(data=df,x='rotation',y='val',hue='exp',palette=exp_palette,ax=ax,notch=True,
                      showfliers=False,bootstrap=10000,linewidth=3,showcaps=False,
                      whiskerprops={'linewidth': 6, 'color': 'k'})
    strip = sns.stripplot(data=df,x='rotation',y='val',hue='exp',palette=exp_palette,ax=ax,size=markSize,
                          edgecolor='gray',linewidth=0.75,dodge=True)
    x_shift = 0.1
   
                                                    
    for strip_artist in strip.collections:
        strip_artist.set_offsets(strip_artist.get_offsets() + [x_shift, 0])
    sns.despine()
    ax.set_ylabel(yLab)
    if idlLines:
        it=0
        for rotSize in list(reversed(sorted(rots))):
            ax.axhline(linewidth=2,y=rotSize,color=colours[it],alpha=0.8,ls='--',zorder=-10)
            it+=1
    ax.set_yticks(yLox,yTix)
                                        
                             
    ax.axhline(linewidth=2,y=0,color='gray',alpha=1,ls=':',zorder=-10)
    ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename)
    return df

    
def getMeansAndStds(v):
    mns = [np.nanmean(rot[0]) for rot in v]
    stds = [np.nanstd(rot[0]) for rot in v]
    return mns,stds      

def plotSSMState(state,aim,tot,trial,target=180,figSize=(7,7)):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    plt.plot(state,color='#E60026',linewidth=10)
    plt.axvline(aim,color='#4D4DFF',linewidth=6)
    plt.axvline(target,color='black',ls='--',linewidth=6)
    plt.axhline(-1,color='black',linewidth=6)
    plt.axvline(tot,color='#570861',linewidth=6)
    plt.ylim(-1,47)
    plt.xlim(0,360)
    plt.xticks(np.arange(0,361,90),np.arange(-180,181,90))
    plt.title(str(trial))
    sns.despine()
    plt.tight_layout()
    plt.savefig('genExampleTrial'+str(trial)+'.svg')
    
                                       
def BICxLR(ssmLR,diff,figSize=(10,7.5)):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    plt.scatter(ssmLR,diff)
    plt.axhline(y=0)
    plt.ylabel('BIC difference (higher = \nstepFucn better than SSM)')
    plt.xlabel('ssm learning rate')
    sns.despine()
    plt.tight_layout()
    plt.savefig('BICs vs ssmLR.png')

def params(data,figSize=(20,10),x='learningRate',y='retentionRate',height=10,hue=None,xlim=(0,100),ylim=(0,100),fileName = 'tmp.png',ylab = 'SSM retention parameter',xlab='SSM learning rate'):
    sns.jointplot(data=data,hue=hue,x=x,y=y,palette=list(reversed(colours)),xlim=xlim,ylim=ylim,height=height,legend=False, marginal_kws=dict(bins=10))
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    sns.despine()
    plt.tight_layout()
    plt.savefig(fileName)
    
def outputPlots(ssmStates,sfmStates,reach='',figSize=(20,10),fileName = 'tmp.svg',ylab = '',xlab=''):
    fig,axs = plt.subplots(ncols=7,nrows=7,figsize=figSize)
    rav = axs.ravel()
    for p in range(len(ssmStates)):
        ax = rav[p]
        if ssmStates != 'skip':
            ax.plot(ssmStates[p],label='ssm')
        ax.plot(sfmStates[p],label='stepFunc')
                                         
                    
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        sns.despine()
    plt.tight_layout()
    plt.savefig(fileName)

def BICComp(df,filename='BICDiff.png',figSize=(10,7.5)):
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=figSize)
    sns.swarmplot(data=df,y='BICDiff',x='rotation')
    plt.axhline(y=0)
    plt.ylabel('')
    plt.xlabel('')
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename)
    
    
"""
Force the step function to be flat --> vertical --> flat
If this is too bad then need a variable section of noise before the step up
So first do learnable params as vertical point and final height (fix first horizontal)
Fit only to rotation phase
"""

    
"""
def cycCompare(dat):
    cors=[]
    pList=[90,60,45,30,15]
    mpl.rcParams.update({'font.size': 16}) 
    mpl.rcParams['figure.dpi'] = 200
    plt.figure(figsize=(15,10))
    for git in range(5):
        sts=[]
        ps=[]
        cgmn=[]
        btmn=[]
        cgstd=[]
        btstd=[]
        if dat == 'imp': # because messed up on the import and no time to reimport data - but do fix in future
            sts=[]
            ps=[]
            cgmn=[np.nan]*5
            btmn=[np.nan]*5
            cgstd=[np.nan]*5
            btstd=[np.nan]*5
            for c in range(5,47):
                btdf = df[(df['cyc']==c) & (df['cgbt'] == 1)]
                cgdf = df[(df['cyc']==c+3) & (df['cgbt'] == 0)]
                tukdf = btdf.append(cgdf)
                tukdf = tukdf[(tukdf['rot'] == pList[git])]
                tst = (tukdf.pairwise_tukey(dv=dat, between='cgbt').round(3))
                ps.append(tst['p-tukey'][0])
                print(ps)
                sts.append(tst['T'][0])
                cgstd.append(stats.median_abs_deviation(tukdf[tukdf['cgbt'] == 0][dat],nan_policy='omit'))
                btstd.append(stats.median_abs_deviation(tukdf[tukdf['cgbt'] == 1][dat],nan_policy='omit'))
                cgmn.append(np.nanmean(tukdf[tukdf['cgbt'] == 0][dat]))
                btmn.append(np.nanmean(tukdf[tukdf['cgbt'] == 1][dat]))
        else:
            for c in range(50):
                    tdf = df[df['cyc']==c]
                    tukdf = tdf[(tdf['rot'] == pList[git])]
                    tst = (tukdf.pairwise_tukey(dv=dat, between='cgbt').round(3))
                    ps.append(tst['p-tukey'][0])
                    sts.append(tst['T'][0])
                    cgstd.append(stats.median_abs_deviation(tukdf[tukdf['cgbt'] == 0][dat],nan_policy='omit'))
                    btstd.append(stats.median_abs_deviation(tukdf[tukdf['cgbt'] == 1][dat],nan_policy='omit'))
                    cgmn.append(np.nanmean(tukdf[tukdf['cgbt'] == 0][dat]))
                    btmn.append(np.nanmean(tukdf[tukdf['cgbt'] == 1][dat]))
        sig,p,_,_=multest(ps,method='fdr_bh')
        
        #get rotation section and calculate regression coeff between BT and CG
        cgcm = cgmn[5:45]
        btcm = btmn[5:45]
        rSq = stats.spearmanr(cgcm,btcm)
        cors.append(rSq)
        
        
        sigBar=[]
        if dat == 'imp':
            sigBar=[np.nan]*5
        [sigBar.append(i) for i in sig]  
        print(p)
        print(sigBar)
        ax=plt.subplot(3,2,git+1) 
        #if pp in bads2:
        #    ax.set_facecolor('grey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        lnt=50
        if dat == 'imp':
            lnt=47
        for point in range(len(sigBar)):
            col = 0 if sigBar[point] == True else 1
            plt.scatter(x=point,y=0,marker='s',color=cols[col],s=60)
        plt.scatter(np.arange(lnt),cgmn,label='CGCycleVar',color=cols[git])
        plt.fill_between(x=np.arange(lnt),y1=np.add(cgmn,cgstd),y2=np.subtract(cgmn,cgstd),alpha=0.2,color=cols[git])
        plt.scatter(np.arange(lnt),btmn,label='BTCycleVar',color='k')
        plt.fill_between(x=np.arange(lnt),y1=np.add(btmn,btstd),y2=np.subtract(btmn,btstd),alpha=0.2,color='k')
        plt.axhline(linewidth=2,y=0,ls='--',alpha=0.3)
        plt.axhline(linewidth=2,y=pList[git],ls='--',alpha=0.3)        
        plt.axvline(linewidth=2,x=44.5,ls='--',alpha=0.3)
        plt.axvline(linewidth=2,x=4.5,ls='--',alpha=0.3)
        ax.set_xticks(np.arange(0,51,5))
        ax.set_xticklabels(np.arange(-5,50,5))  
        if dat == 'imp':
            plt.ylim((-5,30))
        else:
            plt.ylim((-15,105))
        if dat == 'at':
            plt.ylim((0,5))
        #plt.legend()
        plt.xlabel('cycle with respet to perturbation onset')
        plt.ylabel(dat)
        plt.title(str(pList[git]))
        plt.suptitle('Cycle-Series '+dat+' Comparison')#plt.legend()
    plt.tight_layout()
    plt.savefig(dat+'CGBTCycComparTukey.svg')
    return cors


cors = cycCompare('at')    
"""   

    
    
                      

def fitExponential(x,y):
    params,pcov = curve_fit(exponential,x,y)
    return params,pcov
    
def sigmoid(z):
    return 1/(1 + np.power(((1/0.99)-1),(2*z)-0.3)) 

def exponential(x,g,h,t):
    return [g+(h*(np.exp(i*-t))) for i in x]

def removeAxComponents(ax,yTicks=False,xTicks=False,xTickLabels=False,yTickLabels=False,despine=True,title=False,xLab=False,yLab=False,legend=False):
    if xTickLabels:
        ax.set_xticklabels([])
    if xTicks:
        ax.tick_params(axis='x', which='both', length=0)
    if xLab:
        ax.set_xlabel('')
    if yTickLabels:
        ax.set_yticklabels([])
    if yTicks:
        ax.tick_params(axis='y', which='both', length=0)
    if yLab:
        ax.set_ylabel('')
    if title:
        ax.set_title('')
    if despine:
        sns.despine(ax=ax)
    return
    if legend:
        ax.get_legend().remove()

def modelUpdate(aimAngle,prevState,rot,prevImp,lr=0.0385/2,retain=0.9985*0.98,sd=30):
    aimAngle = int(aimAngle)
    gaussFunc = gaussGen(sd)
    toShift =  540 - aimAngle if aimAngle >= 180 else 180 - aimAngle
                                         
    pe = prevState[aimAngle] + rot
    if not np.isnan(toShift):
        for i in range(int(toShift)):
            gaussFunc.append(gaussFunc.pop(0))
        newState = np.subtract(np.multiply(prevState,retain), np.multiply(gaussFunc,lr*pe))
    else:
        newState = prevState
    return newState,toShift
    
def gaussGen(sd):
    return [sampleGauss(i,sd) for i in np.arange(-180,180,1)]
    
def sampleGauss(x,sd):
    return np.math.exp(-(x*x)/(2*(sd*sd)))

         
    

                 
"""        


x=np.arange(0,10,0.001)
lx = [(a*(1-np.exp(-i*b)))+d for i in x]
lx = [g+(h*(np.exp(-i*t)))]
#plt.plot(x)
plt.plot(lx)

perts = [-64,-32,-16,-8,-4,-2,0,2,4,8,16,32,64]

order = np.random.choice(perts, size=len(perts), replace=False)

sevLine = np.array([[i]*7 for i in order]).flatten()

oneLine = np.array([np.random.choice(perts, size=len(perts), replace=False) for i in range(7)]).flatten()

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
plt.plot(oneLine,color=colours[-2],linewidth=3)
sns.despine()
plt.savefig('consOnePertExample.svg')


fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
plt.plot(sevLine,color=colours[-3],linewidth=3)
sns.despine()
plt.savefig('consSevenPertExample.svg')



#coloured sections

import seaborn as sns
import matplotlib.pyplot as plt
colours = ['#4D4DFF'] + ['#E60026'] + ['#570861'] + ['#09710D'] + ['#FFAD00']
allRots = [90,60,45,30,15]

#Quick plot for varying SSM LR

for eg in range(1):
    x = 0
    order = np.random.choice((allRots), size=len(allRots), replace=False).tolist()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
    for i in order:
        line = [i]*60
        print(np.argwhere(allRots==np.abs(i)))
        plt.plot(np.arange(x,x+60),line,color=colours[np.argwhere(allRots==np.abs(i))[0][0]],linewidth=8)
        x += 60
    plt.ylim(-95,95)
    sns.despine()
    plt.axhline(y=0)
    plt.savefig('VarySSMEx'+str(eg)+'.svg')
    

perts = [-45,-60,-90,90,60,45]
for eg in range(3):
    bl = [0]*15
    wash = [0]*15
    x = 0
    order = np.random.choice(perts, size=len(perts), replace=False).tolist()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
    for i in order:
        line = np.array(bl+([i]*30)+wash).flatten()
        print(np.argwhere(allRots==np.abs(i)))
        plt.plot(np.arange(x,x+60),line,color=colours[np.argwhere(allRots==np.abs(i))[0][0]],linewidth=2)
        x += 60
    plt.ylim(-95,95)
    sns.despine()
    plt.savefig('1TarPertExample'+str(eg)+'.svg')
    
perts = [-45,-30,-15,15,30,45]
for eg in range(3,6):
    bl = [0]*15
    wash = [0]*15
    x = 0
    order = np.random.choice(perts, size=len(perts), replace=False).tolist()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
    for i in order:
        line = np.array(bl+([i]*30)+wash).flatten()
        plt.plot(np.arange(x,x+60),line,color=colours[np.argwhere(allRots==np.abs(i))[0][0]],linewidth=2)
        x += 60
    sns.despine()
    plt.ylim(-95,95)
    plt.savefig('1TarPertExample'+str(eg)+'.svg')

perts = [-90,60,-45,15]
pit=0
for p in perts:
    bl = [0]*5
    pert = [p]*40
    wash=[0]*5
    line = np.array(bl+pert+wash).flatten()
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
    plt.plot(line,color=colours[np.argwhere(allRots==np.abs(p))[0][0]],linewidth=2)
    plt.ylim(-95,95)
    sns.despine()
    plt.savefig('8TarPertExample'+str(pit)+'.svg')
    pit+=1

tarPos = np.arange(0,360,45)
line = np.random.choice(tarPos, size=len(tarPos), replace=False).tolist()
fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(10,3))
plt.plot(line,color='black',linewidth=3)
sns.despine()
plt.xticks(np.arange(0,8))
plt.savefig('cycleExample.svg')
"""













































































