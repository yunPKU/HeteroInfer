#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    to perform the statistical inference on the true trans tree
    0828: perfomrm the MLE estimation of the k (or sigma_lambda)
    0831: added the function of hypothesis testing based on the tail branches
    0426: using updated correction for the estimation of gamma
"""
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from io import StringIO
#import dendropy

import treeSimuFun_SSESeq0816 as tSF
import treePertbFun0831 as tPF
import treeMapFun0821 as tMF
import transEventFun0421 as tEF
import emAlgFunFinal0421_0 as eAF
import scipy as sp

from heapq import nlargest
pathMaker = lambda x: [x];
def rmseCal(est,trueVal):
    data = np.transpose(est);
    m,n = np.shape(data);
    rmse = np.zeros(m)
    for i in range(m):
        rmse[i] = np.sqrt(np.nansum((data[i,:]- trueVal[i])**2)/n)
    return(rmse)
def Est_PartData_RS(transPairSP,hostLifeSP,censoring_flagSP,cor_ratio,spl):
    eventBranch,eventTail,Host_tailBranch,eventChild,\
                eventLevel,eventPathSet,eventDepth,eventCensor,eventAccInfo\
                = tEF.transPair2EventInfor1116(transPairSP,hostLifeSP,censoring_flagSP);
    averTail = np.mean(eventTail[eventTail>0]);
    dtime = np.percentile(eventTail[eventTail>0],cor_ratio*100)
#            dtime2 = np.percentile(hostLifeSP[:,2]-hostLifeSP[:,1],cor_ratio*100);
    K = len(np.nonzero(transPairSP[:,2]<np.max(hostLifeSP[:,2]) - dtime)[0])
#    K = 30
    transPairSP = transPairSP[0:K,:];
    hostSel = np.unique(transPairSP[:,0:2]);
    temp_id = np.isin(hostLifeSP[:,0],hostSel)
    hostLifeSP = hostLifeSP[temp_id,:];
    censoring_flagSP = censoring_flagSP[temp_id]
    censorTime = transPairSP[-1,2];
    censoring_flagSP = hostLifeSP[:,2]>=censorTime
    hostLifeSP[:,2] = np.minimum(hostLifeSP[:,2],censorTime);
    
    eventBranch,eventTail,Host_tailBranch,eventChild,\
        eventLevel,eventPathSet,eventDepth,eventCensor,eventAccInfo\
        = tEF.transPair2EventInfor1116(transPairSP,hostLifeSP,censoring_flagSP);
        
        
    # added 0401 for the estimation of gamma and use this value to the following estimation
    
    tempN = len(np.nonzero(eventCensor==0)[0]);
    totalEdge = np.sum(eventTail[eventTail>0])+np.sum(eventBranch[:,2])
    totalN    = np.sum(eventCensor>-1)
    totalRecEst2 = (tempN/spl) / (totalEdge + averTail*(totalN/(cor_ratio*spl) - totalN)); 
    muRecEst = totalRecEst2;
    
    newEvent = np.nonzero(eventLevel==1)[0]
    tailInfo = np.max(eventTail[newEvent,:],axis = 1); # tail is the independent tail infor
    tailCensor = eventCensor[newEvent,np.argmax(eventTail[newEvent,:],axis = 1)]
    pstCensor = eventCensor[newEvent,np.argmin(eventTail[newEvent,:],axis = 1)]
    paraSlc = (tEF.SlcEst(eventBranch[newEvent,2]))[0:2]; # mu,k
    
#            get the estimation from Phylo
    pstEvent = newEvent; 
    pstPath = list(map(pathMaker,list(newEvent)));
    tempassCount = len(np.nonzero(eventTail[newEvent,0]<=eventTail[newEvent,1])[0]);
#    tempassCount = 0;
    b = [paraSlc,pstEvent,eventAccInfo,pstPath,tempassCount,pstCensor]
    dep, count = np.unique(eventLevel, return_counts=True);                       
    for lv in range(2,max(dep)):
        newEvent = np.nonzero(eventLevel==lv)[0];        
        b = eAF.EMoptm1218(b[0],muRecEst,newEvent,b[2],eventBranch,\
                           eventChild,eventTail,b[1],b[3],tailInfo,\
                           b[4],b[5],eventCensor,tailCensor);

    temp_mu, temp_sig = b[0];    
#    muRecEst = b[6];
#    totalRecEst = b[7];
    return(temp_mu, temp_sig,totalRecEst2,0)

def crct_aveg(rawEst,crctRatio,Rho_SD):
    mu_est, sigma_est, rec_est, totalrec = 0,0,0,0;
    n = len(rawEst);    
    for i in range(n): # correction and then average
        temp_mu, temp_sig, temp_Rec, temp_tol = rawEst[i];
        cor_ratio = crctRatio[i]*Rho_SD;
        tempCV = temp_sig/temp_mu;
        
        tempC = temp_mu/cor_ratio + (1-np.exp(-tempCV))*(temp_Rec - temp_mu/cor_ratio)/(1+np.exp(-tempCV))
        
        mu_est      += temp_mu + (1 - cor_ratio)*tempC; # changed 0110
        sigma_est   += np.sqrt((temp_sig**2 + (1 - cor_ratio)*(temp_mu-cor_ratio*tempC)**2)/cor_ratio);
        rec_est     += temp_Rec
        totalrec    += temp_tol
    mu_est, sigma_est, rec_est, totalrec = mu_est/n, sigma_est/n, rec_est/n, totalrec/n;
    return(mu_est, sigma_est, rec_est,totalrec)

def bckEst(transPairRec,hostLifeRec,censoring_flagRec,popSize,SampleSize,spl,back_ratio):
    # this function performs backward estimation for the SID model
    tptransPairRec = transPairRec.copy();
    tphostLifeRec = hostLifeRec.copy();
    tpcensoring_flagRec = censoring_flagRec.copy();
    if spl< 1:
        tptransPairRec,tphostLifeRec,tpcensoring_flagRec = \
                        tEF.lgeSampling(transPairRec,hostLifeRec,censoring_flagRec,\
                                        popSize - SampleSize,'Random');
#    re = Est_PartData_RS(tptransPairRec,tphostLifeRec,tpcensoring_flagRec,back_ratio[0],spl);                                    
    re = [[]]*len(back_ratio);
    for i in range(len(back_ratio)):
        re[i] = Est_PartData_RS(tptransPairRec,tphostLifeRec,tpcensoring_flagRec,back_ratio[i],spl);
    return(re)

    

def pwVrySmpSize(mu_InFC = 2.5, mu_Rec  = 1,smpSeq = [20,40,60,80,100]):    
    simu_K,simu_pw          = 200,500
    CV_Seq = np.hstack((np.array([0,0.1,0.3,0.5,0.7]),np.arange(1,5.5,0.5)))
    std_Seq = CV_Seq*mu_InFC 
    var_Seq         = std_Seq**2  
    smpRatio        = np.array([0.9]);
    
    back_ratio = [0.9,0.85,0.8]
    #back_ratio = [0.99,0.95,0.85]
    ciLv = 95    
    sigmaUB = 0    
    tempSig = np.zeros(simu_pw)
        
    pwResult = np.zeros((len(smpSeq),len(var_Seq)))
    for k in range(len(smpSeq)):
        popSize         = smpSeq[k];
        print(k)
        sigmaUB = 0
        for j in range(len(var_Seq)):
            var_InFC = var_Seq[j]    
            rejCont = 0
            print(j)
            pwcont = 0;
            
            if var_InFC==0: # estimate the up-bound
                tempSig = np.zeros(simu_pw)
                for ip in range(simu_pw): 
                    hostLife,transPair,censoring_flag\
                            = tSF.OtbI(mu_InFC,var_InFC,mu_Rec,popSize)
                        # choose the first n infections to calculate the MLE based on true tree
                        # sampling the recovered
                    transPairRec,hostLifeRec,censoring_flagRec = \
                        tEF.lgeSampling(transPair,hostLife,censoring_flag,\
                                        0,'Rec');
                    tempSPratio = smpRatio[0];
                    SampleSize  = np.round(tempSPratio*popSize)
                    spl = tempSPratio;
                        
                    re = bckEst(transPairRec,hostLifeRec,censoring_flagRec,popSize,SampleSize,spl,back_ratio)
                    mu_mle,sig_mle,mu_RecEst,total_rec = crct_aveg(re,back_ratio,spl);
                    tempSig[ip] = sig_mle
                sigmaUB = np.percentile(tempSig,ciLv)
            else:
                rejCont = 0
                for i in range(simu_K):  
                # generating the tree
                    hostLife,transPair,censoring_flag\
                            = tSF.OtbI(mu_InFC,var_InFC,mu_Rec,popSize)
                    # choose the first n infections to calculate the MLE based on true tree
                    # sampling the recovered
                    transPairRec,hostLifeRec,censoring_flagRec = \
                        tEF.lgeSampling(transPair,hostLife,censoring_flag,\
                                        0,'Rec');
        
                    tempSPratio = smpRatio[0];
                    SampleSize  = np.round(tempSPratio*popSize)
                    spl = tempSPratio;
                    
                    re = bckEst(transPairRec,hostLifeRec,censoring_flagRec,popSize,SampleSize,spl,back_ratio)
                    
                    # changed 0110
                    mu_mle,sigma_mle,mu_RecEst,total_rec = crct_aveg(re,back_ratio,spl);
                    rejCont += sigma_mle > sigmaUB
                pwResult[k,j] = rejCont/simu_K
    return(pwResult)        
    
def pwVrySmplingRatio(mu_InFC = 2.5, mu_Rec  = 1,smpRatio = np.array([0.2,0.4,0.6,0.8,1])):
    simu_K,simu_pw          = 200,500    
    CV_Seq = np.hstack((np.array([0,0.1,0.3,0.5,0.7]),np.arange(1,5.5,0.5)))
    std_Seq = CV_Seq*mu_InFC     
    var_Seq         = std_Seq**2  
    
    back_ratio = [0.9,0.85,0.8]
    ciLv = 95
    sigmaUB = 0
    
    tempSig = np.zeros(simu_pw)
        
    pwResult = np.zeros((len(smpRatio),len(var_Seq)))
    for k in range(len(smpRatio)):
        popSize         = 100;
        print(k)
        sigmaUB = 0
        for j in range(len(var_Seq)):
            var_InFC = var_Seq[j]    
            rejCont = 0
            print(j)
            pwcont = 0;
            
            if var_InFC==0: # estimate the up-bound
                tempSig = np.zeros(simu_pw)
                for ip in range(simu_pw): 
                    hostLife,transPair,censoring_flag\
                            = tSF.OtbI(mu_InFC,var_InFC,mu_Rec,popSize)
                        # choose the first n infections to calculate the MLE based on true tree
                        # sampling the recovered
                    transPairRec,hostLifeRec,censoring_flagRec = \
                        tEF.lgeSampling(transPair,hostLife,censoring_flag,\
                                        0,'Rec');
                    tempSPratio = smpRatio[k];
                    SampleSize  = np.round(tempSPratio*popSize)
                    spl = tempSPratio;
                        
                    re = bckEst(transPairRec,hostLifeRec,censoring_flagRec,popSize,SampleSize,spl,back_ratio)
                    mu_mle,sig_mle,mu_RecEst,total_rec = crct_aveg(re,back_ratio,spl);
                    tempSig[ip] = sig_mle
                sigmaUB = np.percentile(tempSig,ciLv)
            else:
                rejCont = 0
                for i in range(simu_K):  
                # generating the tree
                    hostLife,transPair,censoring_flag\
                            = tSF.OtbI(mu_InFC,var_InFC,mu_Rec,popSize)
                    # choose the first n infections to calculate the MLE based on true tree
                    # sampling the recovered
                    transPairRec,hostLifeRec,censoring_flagRec = \
                        tEF.lgeSampling(transPair,hostLife,censoring_flag,\
                                        0,'Rec');
        
                    tempSPratio = smpRatio[k];
                    SampleSize  = np.round(tempSPratio*popSize)
                    spl = tempSPratio;
                    
                    re = bckEst(transPairRec,hostLifeRec,censoring_flagRec,popSize,SampleSize,spl,back_ratio)
                    
                    # changed 0110
                    mu_mle,sigma_mle,mu_RecEst,total_rec = crct_aveg(re,back_ratio,spl);
                    rejCont += sigma_mle > sigmaUB
                pwResult[k,j] = rejCont/simu_K
    return(pwResult)