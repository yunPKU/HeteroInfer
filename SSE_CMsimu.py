#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    to perform the statistical inference on the true trans tree
    0828: perfomrm the MLE estimation of the k (or sigma_lambda)
    0831: added the function of hypothesis testing based on the tail branches
"""
import numpy as np
from Bio import Phylo
from io import StringIO
#import dendropy

import treeSimuFun_SSESeq0816 as tSF
import treePertbFun0831 as tPF
import treeMapFun0821 as tMF
import transEventFun0815 as tEF
import scipy as sp
import matplotlib.pyplot as plt
from heapq import nlargest,nsmallest
pathMaker = lambda x: [x];




def sseOLSimu(simu_K = 100,NUD_bd = 1.0):


    CV_Seq = np.hstack((np.array([0,0.3,0.5,0.7]),np.arange(1,5.5,0.5)))
    mu_InFC,mu_Rec  = 2.5,1;
    std_Seq = CV_Seq*mu_InFC 
    
    var_Seq         = std_Seq**2  
    SampleSize      = 100;
    popSize         = [100];
    #K = 90;
    
    
    CTTL = 0.5;
    obIILen = 1
    
    
    effNCT,ctRatio,effRND,effBnM   = (np.zeros((len(var_Seq),simu_K)) for i in range(4))
    meanBnM, meanRatio,meanRND,meanNCT\
      =  (np.zeros((len(var_Seq),len(popSize))) for i in range(4))
    stdBnM, stdRatio,stdRND,stdNCT\
      =  (np.zeros((len(var_Seq),len(popSize))) for i in range(4))
    
    
    noSrs,nceSrs,rndSrs = np.zeros((len(var_Seq),5)),np.zeros((len(var_Seq),5)),np.zeros((len(var_Seq),5))
    
    for k in range(len(popSize)):    
        for j in range(len(var_Seq)):
            var_InFC = var_Seq[j]    
            print(j)
            for i in range(simu_K):  
                recHost = 0
                while recHost<50:
                    sysSeq,sysId = [],0;
                    while sysId>=len(sysSeq):
                        a = tSF.OtbI(mu_InFC,var_InFC,mu_Rec,SampleSize)
                        sysSeq,sysId = a[4],a[5];
                    # return(hostLife,transPair,TotalHost,curt,sysSeq,sysId)
                    b = tSF.otbII_NoContrl(a[0],a[1],a[2],a[3],a[4],a[5],a[3] + obIILen,mu_InFC,var_InFC,mu_Rec)
                    # return(TotalHost,indSeq,recHost,totalHostSeries)
                    recHost = b[2];
                    
                noSrs[j,:] += (b[3] - a[2])/(b[0]-a[2]);
                # return return(TotalHost,indSeq,recHost)
                c = tSF.otbII_WithContrl_Nnd(a[0],a[1],a[2],a[3],a[4],a[5],a[3] + obIILen,b[1],CTTL,NUD_bd,'NCE')
                nceSrs[j,:] += (c[2] - a[2])/(b[0]-a[2]);
                # return return(TotalHost,ctHost,totalHostSeries,recHost) 
    
                effNCT[j,i] = (c[0] - a[2])/(b[0]-a[2]);
                ctRatio[j,i] = c[1]/c[3];
                
                # 2 calculate the performance of the RND
                c = tSF.otbII_WithContrl_Nnd(a[0],a[1],a[2],a[3],a[4],a[5],a[3] + obIILen,b[1],CTTL,ctRatio[j,i],'RND')
                rndSrs[j,:]+= (c[2] - a[2])/(b[0]-a[2]);
                effRND[j,i] = (c[0] - a[2])/(b[0]-a[2]);
                
        meanBnM[:,k], meanRatio[:,k],meanRND[:,k],meanNCT[:,k]\
                = np.nanmean(effBnM,axis = 1),np.nanmean(ctRatio,axis = 1),np.nanmean(effRND,axis = 1),\
                  np.nanmean(effNCT,axis = 1); 
                  
    nceSrs,rndSrs,noSrs = nceSrs/simu_K,rndSrs/simu_K,noSrs/simu_K
    comRe = np.hstack((np.reshape(CV_Seq,(13,1)),meanNCT,meanRND,meanRatio))
    
    return(comRe,nceSrs,rndSrs,noSrs)

def comCMPlot(CV = [3,5], simuK = 100):
    a01 = sseOLSimu(simu_K = simuK,NUD_bd = 1.0)
    a02 = sseOLSimu(simu_K = simuK,NUD_bd = 2.0)
    a03 = sseOLSimu(simu_K = simuK,NUD_bd = 3.0)
    a04 = sseOLSimu(simu_K = simuK,NUD_bd = 4.0)

    for i in range(len(CV)):
        rw = np.where(a01[0][:,0]==CV[i])
        a = np.hstack((np.reshape(np.hstack((a01[1][rw,:],a01[2][rw,:],a01[3][rw,:])),(1,15)),a01[0][rw,3]))
        b = np.hstack((np.reshape(np.hstack((a02[1][rw,:],a02[2][rw,:],a02[3][rw,:])),(1,15)),a02[0][rw,3]))
        c = np.hstack((np.reshape(np.hstack((a03[1][rw,:],a03[2][rw,:],a03[3][rw,:])),(1,15)),a03[0][rw,3]))
        d = np.hstack((np.reshape(np.hstack((a04[1][rw,:],a04[2][rw,:],a04[3][rw,:])),(1,15)),a04[0][rw,3]))
    
        a = np.vstack((a,b,c,d))
    
    
        title = 'CV='+str(CV[i])
        #
        #file = open('seq0826_series_'+title+'.txt', 'r')
        #a = np.loadtxt(file)
        #
        Num = 5
        timUni = 2.5
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.12, 0.7, 0.8],
                           xticklabels=[], ylim=(0, 1))
        
        xlb = np.linspace(0,1,num=Num)
        ax1.plot(xlb, a[0,0:Num],'C0-o',label = '$NCE \geqslant 1$ policy')
        ax1.plot(xlb, a[0,Num:2*Num],'C0--o',label = 'Random policy')
        ax1.plot(xlb, a[1,0:Num],'C1-v',label = '$NCE \geqslant 2$ policy')
        ax1.plot(xlb, a[1,Num:2*Num],'C1--v',label = 'Random policy')
        ax1.plot(xlb, a[2,0:Num],'C2-+',label = '$NCE \geqslant 3$ policy')
        ax1.plot(xlb, a[2,Num:2*Num],'C2--+',label = 'Random policy')
        ax1.plot(xlb, a[3,0:Num],'C3-*',label = '$NCE \geqslant 4$ policy')
        ax1.plot(xlb, a[3,Num:2*Num],'C3--*',label = 'Random policy')
        ax1.plot(xlb, a[3,2*Num:3*Num],'k-',label = 'No Prevention')
        
        ax1.legend(loc = 0,prop={'size':10})
        ax1.set_xlabel('time(year)')
        ax1.set_title('Relative Infection Size ('+title+')', size = 12)
        ax1.set_xticks(xlb)
        ax1.set_xticklabels(['Start of Control',0.25*timUni,0.5*timUni,0.75*timUni,1.0*timUni])
        
        ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.8],ylim=(0, 1))
        ax2.plot([0,0.01],[a[:,-1],a[:,-1]],linewidth = 5)
        ax3 = ax2.twinx()
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_ylabel('Fraction of Contact traced',size = 12)
#demo
#comCMPlot()