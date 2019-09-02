#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    to perform the statistical inference on the true trans tree
    0828: perfomrm the MLE estimation of the k (or sigma_lambda)
    0831: added the function of hypothesis testing based on the tail branches
"""
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from io import StringIO
#import dendropy

import treeSimuFun1204 as tSF
import treePertbFun0831 as tPF
import treeMapFun0821 as tMF
import transEventFun0815 as tEF
import emAlgFunFinal0421_0 as eAF
import scipy as sp

from heapq import nlargest
pathMaker = lambda x: [x];


''' multiple simulation for a sequence of variances 
    0820 -- estimte the error of theta testimation , improve the estimation of k_root
'''


''' 
    0831 -- hypothesis testing based on the tail information
'''
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
    muRecEst = b[6];
    totalRecEst = b[7];
    return(temp_mu, temp_sig,totalRecEst2,totalRecEst)

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

def crct_avegINFP(rawEst,crctRatio,Rho_SD):
    mu_est, sigma_est, rec_est, totalrec = 0,0,0,0;
    n = len(rawEst);    
    for i in range(n): # correction and then average
        temp_mu, temp_sig, temp_Rec, temp_tol = rawEst[i];
        cor_ratio = crctRatio[i]*Rho_SD;
        tempCV = temp_sig/temp_mu;

        temp_mu = 1/temp_mu
        tempC   = temp_mu + (1-np.exp(-tempCV))*(1/temp_Rec - temp_mu)/(1+np.exp(-tempCV))
        mu_est  += 1/(temp_mu*cor_ratio**2 +  tempC*(1-cor_ratio))
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

'''
   Single Simulation 
'''
def singleSimu(mu_InFC = 2.5,mu_Rec = 1,cv_InFC = 1,smpSize = 100,smpRatio = 0.9,fntNum = 9999):
    popSize = smpSize
    var_InFC = cv_InFC*mu_InFC
    # generating the tree
    if fntNum == 9999:
        transPair,hostLife,host_infe_var,simutree,censoring_flag,BD_rate\
             = tSF.Simu_InftPop_GilleSID0515(mu_InFC,var_InFC,mu_Rec,popSize)
    else:
        transPair,hostLife,host_infe_var,simutree,censoring_flag,BD_rate\
             = tSF.Simu_InftPop_GilleSID_FntN(mu_InFC,var_InFC,mu_Rec,popSize,fntNum)

    # sampling the recovered
    transPairRec,hostLifeRec,censoring_flagRec = \
        tEF.lgeSampling(transPair,hostLife,censoring_flag,\
                        0,'Rec');
    SampleSize  = np.round(smpRatio*popSize)
    tptransPairRec,tphostLifeRec,tpcensoring_flagRec = \
                    tEF.lgeSampling(transPairRec,hostLifeRec,censoring_flagRec,\
                                    popSize - SampleSize,'Random');
    
    TreeBranch,Host_tailBranch = tSF.transPair2treebranch(transPair,hostLife)
    simutree        = tSF.treeBranch2treeView(TreeBranch,Host_tailBranch)                        
        
   
    return(tptransPairRec,tphostLifeRec,tpcensoring_flagRec,simutree)

def singleEst(tptransPairRec,tphostLifeRec,tpcensoring_flagRec,back_ratio = [0.9,0.85,0.8],smpRatio = 0.9):

    re = [[]]*len(back_ratio);
    for i in range(len(back_ratio)):
        re[i] = Est_PartData_RS(tptransPairRec,tphostLifeRec,tpcensoring_flagRec,back_ratio[i],smpRatio);
    mu_est,sigma_est,mu_RecEst,total_rec = crct_avegINFP(re,back_ratio,smpRatio)
    return(mu_est,sigma_est,mu_RecEst)
'''
    Composite Simulation
'''

def cmpEst(mu_InFC = 2.5,mu_Rec = 1,smpSize = 100,fntNum = 9999,simu_K = 100,\
           CV_Seq = np.hstack((np.array([0,0.1,0.3,0.5,0.7]),np.arange(1,5.5,0.5))),\
           smpRatio        = np.array([0.9]),back_ratio = [0.9,0.85,0.8]):    
    std_Seq = CV_Seq*mu_InFC 
    popSize = smpSize
    var_Seq         = std_Seq**2  
    #back_ratio = [0.99,0.95,0.85]
    ciLv = 95
    rlSize_mean,rlSize_Rmse,\
    ro_mean,roRmse, ro_meanCD,roRmseCD, ro_meanCU,roRmseCU,\
    mu_mean,muRmse, mu_meanCD,muRmseCD, mu_meanCU,muRmseCU,\
    sigma_mean,sigmaRmse,sigma_meanCD,sigmaRmseCD,sigma_meanCU,sigmaRmseCU, \
    cv_mean, cvRmse, cv_meanCD, cvRmseCD,cv_meanCU, cvRmseCU,totalRec_mean,totalRecRmse,\
    totalRecRmseUB,totalRecRmseLB\
     = (np.zeros((len(var_Seq),len(smpRatio))) for i in range(30))
         
    mu_ttree,sigma_ttree, cv_ttree = (np.zeros((len(var_Seq),simu_K)) for i in range(3))    
    ro_mle,ro_mleCU,ro_mleCD,rlSize_mle,\
    mu_mle,sigma_mle,mu_RecEst,mu_mleCD,sigma_mleCD,mu_RecEstCD,cv_mle,cv_mleCU,cv_mleCD,\
        mu_mleCU,sigma_mleCU,mu_RecEstCU, total_rec \
        = (np.zeros((simu_K,len(smpRatio))) for i in range(17))    
    mu_mean_tt, muRmse_tt, sigma_mean_tt,sigmaRmse_tt,cv_mean_tt,cvRmse_tt\
     = (np.zeros((len(var_Seq))) for i in range(6))    
    for j in range(len(var_Seq)):
        var_InFC = var_Seq[j]    
        print(j)
        for i in range(simu_K):  
            # generating the tree
            if fntNum == 9999:
                transPair,hostLife,host_infe_var,simutree,censoring_flag,BD_rate\
                     = tSF.Simu_InftPop_GilleSID0515(mu_InFC,var_InFC,mu_Rec,popSize)
            else:
                transPair,hostLife,host_infe_var,simutree,censoring_flag,BD_rate\
                     = tSF.Simu_InftPop_GilleSID_FntN(mu_InFC,var_InFC,mu_Rec,popSize,fntNum)
            # choose the first n infections to calculate the MLE based on true tree
            # sampling the recovered
            transPairRec,hostLifeRec,censoring_flagRec = \
                tEF.lgeSampling(transPair,hostLife,censoring_flag,\
                                0,'Rec');
                
            for k in range(len(smpRatio)):
                tempSPratio = smpRatio[k];
                SampleSize  = np.round(tempSPratio*popSize)
                spl = tempSPratio;
                
                re = bckEst(transPairRec,hostLifeRec,censoring_flagRec,popSize,SampleSize,spl,back_ratio)
                mu_mle[i,k],sigma_mle[i,k],mu_RecEst[i,k],total_rec[i,k] = crct_avegINFP(re,back_ratio,spl);
                cv_mle[i,k] = sigma_mle[i,k]/mu_mle[i,k];
                ro_mle[i,k] = mu_mle[i,k]/mu_RecEst[i,k];
    
                
        
        totalRec_mean[j,:] ,totalRecRmse[j,:] = np.nanmean(mu_RecEst,axis = 0),rmseCal(mu_RecEst,[mu_Rec]*len(smpRatio))
        totalRecRmseUB[j,:],totalRecRmseLB[j,:] = np.percentile(mu_RecEst,50+ciLv/2,axis = 0),np.percentile(mu_RecEst,50-ciLv/2,axis = 0)
        rlSize_mean[j,:],rlSize_Rmse[j,:] = np.median(rlSize_mle,axis = 0),rmseCal(rlSize_mle,[1]*len(smpRatio))
        
        gamRe = np.concatenate((np.reshape(CV_Seq,(14,1)),totalRec_mean,totalRecRmseLB,totalRecRmseLB),axis = 1);
    
        mu_mean_tt[j], muRmse_tt[j] = np.nanmean(mu_ttree[j,:]), np.sqrt(np.nansum((mu_ttree[j,:]-mu_InFC)**2)/simu_K);
        mu_mean[j,:],muRmse[j,:] = np.nanmean(mu_mle,axis = 0),rmseCal(mu_mle,[mu_InFC]*len(smpRatio));
        mu_meanCD[j,:],muRmseCD[j,:] = np.percentile(mu_mle,50-ciLv/2,axis = 0),rmseCal(mu_mleCD,[mu_InFC]*len(smpRatio));
        mu_meanCU[j,:],muRmseCU[j,:] = np.percentile(mu_mle,50+ciLv/2,axis = 0),rmseCal(mu_mleCU,[mu_InFC]*len(smpRatio));
        
        ro_mean[j,:],roRmse[j,:] = np.nanmean(ro_mle,axis = 0),rmseCal(ro_mle,[mu_InFC]*len(smpRatio));
        ro_meanCD[j,:],roRmseCD[j,:] = np.percentile(ro_mle,50-ciLv/2,axis = 0),rmseCal(ro_mleCD,[mu_InFC]*len(smpRatio));
        ro_meanCU[j,:],roRmseCU[j,:] = np.percentile(ro_mle,50+ciLv/2,axis = 0),rmseCal(ro_mleCU,[mu_InFC]*len(smpRatio));
         
        sigma_mean_tt[j],sigmaRmse_tt[j] = np.nanmean(sigma_ttree[j,:]),np.sqrt(np.nansum((sigma_ttree[j,:]- std_Seq[j])**2)/simu_K);
        sigma_mean[j,:],sigmaRmse[j,:] = np.nanmean(sigma_mle,axis = 0),rmseCal(sigma_mle,[std_Seq[j]]*len(smpRatio));     
        sigma_meanCD[j,:],sigmaRmseCD[j,:] = np.percentile(sigma_mle,50-ciLv/2,axis = 0),rmseCal(sigma_mleCD,[std_Seq[j]]*len(smpRatio));
        sigma_meanCU[j,:],sigmaRmseCU[j,:] = np.percentile(sigma_mle,50+ciLv/2,axis = 0),rmseCal(sigma_mleCU,[std_Seq[j]]*len(smpRatio));
         
        cv_mean_tt[j],cvRmse_tt[j] = np.nanmean(cv_ttree[j,:]),np.sqrt(np.nansum((cv_ttree[j,:]- CV_Seq[j])**2)/simu_K);
        cv_mean[j,:],cvRmse[j,:] = np.nanmean(cv_mle,axis = 0),rmseCal(cv_mle,[CV_Seq[j]]*len(smpRatio));     
        cv_meanCD[j,:],cvRmseCD[j,:] = np.percentile(cv_mle,50-ciLv/2,axis = 0),rmseCal(cv_mleCD,[CV_Seq[j]]*len(smpRatio));
        cv_meanCU[j,:],cvRmseCU[j,:] = np.percentile(cv_mle,50+ciLv/2,axis = 0),rmseCal(cv_mleCU,[CV_Seq[j]]*len(smpRatio));
     
    muRe = np.concatenate((np.reshape(CV_Seq,(14,1)),mu_mean,mu_meanCD,mu_meanCU),axis = 1);
    roRe = np.concatenate((np.reshape(CV_Seq,(14,1)),ro_mean,ro_meanCD,ro_meanCU),axis = 1);
    sigRe = np.concatenate((np.reshape(CV_Seq,(14,1)),sigma_mean,sigma_meanCD,sigma_meanCU),axis = 1);
    cvRe = np.concatenate((np.reshape(CV_Seq,(14,1)),cv_mean,cv_meanCD,cv_meanCU),axis = 1);        
    return(gamRe,muRe,roRe,sigRe,cvRe)


def EstRePlot(estResult,mu_InFC = 2.5,mu_Rec = 1):
    nameset = ['\gamma','\mu','R_0','\sigma','CV']
    lb_value = [mu_Rec,mu_InFC,mu_InFC/mu_Rec,estResult[0][0][0],estResult[0][0][0]]
    ub_value = [mu_Rec,mu_InFC,mu_InFC/mu_Rec,estResult[0][-1][0]*mu_InFC,estResult[0][-1][0]]
    for i in range(5):
        vabname = nameset[i]
        rawd = estResult[i]
        plt.figure()
        plt.plot(rawd[:,0], rawd[:,1],'o-',label = 'Mean of $\hat{'+vabname+'}$')
        plt.plot([rawd[0,0],rawd[-1,0]],[lb_value[i],ub_value[i]],'black',label = 'true $'+vabname+'$')
        plt.fill_between(rawd[:,0],rawd[:,2], rawd[:,3],alpha=0.5,color= 'xkcd:silver',label = '95% CI')
        plt.xlabel('$CV_{\lambda}$')
        plt.ylabel('Estimation of $'+vabname+'$')
        plt.xticks(np.arange(0,rawd[-1,0]+0.5,0.5))
        plt.legend(loc = 0,prop={'size':12})
# multiple simulaiton
re = cmpEst(mu_InFC = 1,mu_Rec = 1/2.5,simu_K = 100)
EstRePlot(re)

# single simulation
#simuG = singleSimu(mu_InFC = 1,mu_Rec = 1/2.5)
# view the phylogeny
#simutree = simuG[3]  
#Phylo.draw(Phylo.read(StringIO(simutree),"newick"))
# parameter estimation based on the single simulation
#infRe = singleEst(simuG[0],simuG[1],simuG[2])
 
