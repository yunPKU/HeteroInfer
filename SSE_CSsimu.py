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

def intvEff(InFC_State,Host_Life,ctID,udIndID,\
            mu_InFC,var_InFC,mu_Rec,transPair,BD_rate,TotalHost,curt,obIILen):
    tempInFC_State = InFC_State.copy();
    tempHost_Life = Host_Life.copy();
#            # for each ctID, control all the IDs 
    if len(ctID):
        for i_ct in range(len(ctID)):
            tempCT = ctID[i_ct]
            tempID = udIndID[tempCT]
            if tempID:
                tempInFC_State[(np.array(tempID)-1).astype(int),1] = -9999
                tempHost_Life[(np.array(tempID)-1).astype(int),2] = curt;
            
    TransPairNew,Host_LifeNew,totalHostNew,simutreeNew,censoring_flagNew = \
    tSF.Simu_InftPop_GilleSID_OtbII_ENT(mu_InFC,var_InFC,mu_Rec,\
                      np.vstack((np.array([999,1,0]),transPair)),\
                      tempHost_Life,tempInFC_State,BD_rate,TotalHost,curt,curt+obIILen)
    return(totalHostNew)


def sseCSSimu(simu_K = 100,CtRatio = 0.2):
    CV_Seq = np.hstack((np.array([0.1,0.3,0.5,0.7]),np.arange(1,5.5,0.5)))
    mu_InFC,mu_Rec  = 2.5,1;
    std_Seq = CV_Seq*mu_InFC 
    
    var_Seq         = std_Seq**2  
    SampleSize      = 100;
    popSize         = [100];
    
    NumCT = int(np.round(CtRatio*SampleSize));
    CTTL = 0.5,
    obIILen = 0.5
    effTOD,effNCT,effRND,effNUD,effUDRT   = (np.zeros((len(var_Seq),simu_K)) for i in range(5))
    meanTOD, meanNCT,meanRND,meanNUD,meanUDRT\
      =  (np.zeros((len(var_Seq),len(popSize))) for i in range(5))    
    
    for k in range(len(popSize)):
        
        for j in range(len(var_Seq)):
            var_InFC = var_Seq[j]    
            print(j)
            for i in range(simu_K):  
                # generating the tree
                NewInf_UB = 0
                while NewInf_UB<=0:            
                    transPair,hostLife,host_infe_var,simutree,censoring_flag,\
                                    Host_Life,InFC_State,BD_rate,TotalHost,curt\
                         = tSF.Simu_InftPop_GilleSID_OtbI(mu_InFC,var_InFC,mu_Rec,popSize[k])
                # 0 set the benchmark as there is no intervention
                    ctID = []
                    totalHostNew = intvEff(InFC_State,Host_Life,ctID,[],\
                            mu_InFC,var_InFC,mu_Rec,transPair,BD_rate,TotalHost,curt,obIILen)
                    NewInf_UB = totalHostNew - TotalHost;
    #            Phylo.draw(Phylo.read(StringIO(simutree),"newick"),do_show=False)     
                transPairRec,hostLifeRec,censoring_flagRec = \
                    tEF.lgeSampling(transPair,hostLife,censoring_flag,\
                                    popSize[k] - SampleSize,'Rec');
          
                numCE = tEF.brEvtCount(transPairRec,hostLifeRec,CTTL)
                numTOD = np.max(numCE[:,2])-numCE[:,2];
                numCEV = numCE[:,1];
                udIndID = tEF.udIndCountID(transPair,hostLife,censoring_flag,CTTL)
                numUD = tEF.udIndCount(transPair,hostLife,censoring_flag,CTTL)[:,1]
                numUDRT = tEF.udIndCountRate(transPair,hostLife,censoring_flag,BD_rate,CTTL)[:,2]
    #            ctInD = udIndCountID(transPair,hostLife,censoring_flag,TL)
                
                
    #           1 controlling the first n diagnosed, lease TOD
                ctID = nsmallest(NumCT, range(len(numTOD)), numTOD.take)
                totalHostNew = intvEff(InFC_State,Host_Life,ctID,udIndID,\
                        mu_InFC,var_InFC,mu_Rec,transPair,BD_rate,TotalHost,curt,obIILen)
                effTOD[j,i] = (totalHostNew - TotalHost)/NewInf_UB;
                
    ##           2 controlling the first with most numCE
                ctID = nlargest(NumCT, range(len(numCEV)), numCEV.take)
                totalHostNew = intvEff(InFC_State,Host_Life,ctID,udIndID,\
                        mu_InFC,var_InFC,mu_Rec,transPair,BD_rate,TotalHost,curt,obIILen)
                effNCT[j,i] = (totalHostNew - TotalHost)/NewInf_UB;
                
    ##           3 controlling randomly
                ctID = np.random.randint(len(numCEV),size = NumCT)
                totalHostNew = intvEff(InFC_State,Host_Life,ctID,udIndID,\
                        mu_InFC,var_InFC,mu_Rec,transPair,BD_rate,TotalHost,curt,obIILen)
                effRND[j,i] = (totalHostNew - TotalHost)/NewInf_UB;
                
    ##           4 controlling the largest UD num
                ctID = nlargest(NumCT, range(len(numUD)), numUD.take)
                totalHostNew = intvEff(InFC_State,Host_Life,ctID,udIndID,\
                        mu_InFC,var_InFC,mu_Rec,transPair,BD_rate,TotalHost,curt,obIILen)
                effNUD[j,i] = (totalHostNew - TotalHost)/NewInf_UB;
        
    ##           5 controlling the largest rate sum
                ctID = nlargest(NumCT, range(len(numUDRT)), numUDRT.take)
                totalHostNew = intvEff(InFC_State,Host_Life,ctID,udIndID,\
                        mu_InFC,var_InFC,mu_Rec,transPair,BD_rate,TotalHost,curt,obIILen)
                effUDRT[j,i] = (totalHostNew - TotalHost)/NewInf_UB;
                
        meanTOD[:,k], meanNCT[:,k],meanRND[:,k],meanNUD[:,k],meanUDRT[:,k]\
                = np.nanmean(effTOD,axis = 1),np.nanmean(effNCT,axis = 1),np.nanmean(effRND,axis = 1),\
                  np.nanmean(effNUD,axis = 1),np.nanmean(effUDRT,axis = 1);   
        rff_TOD = (1-meanTOD)/(1-meanRND);
        rff_NCT = (1-meanNCT)/(1-meanRND);
                  
    comRe = np.hstack((np.reshape(CV_Seq,(13,1)),rff_TOD,rff_NCT))
    return(comRe)
#CtRatio = np.array([0.2,0.5]); simuK = 100

def comCSPlot(CtRatio = np.array([0.2,0.5]), simuK = 100):
    a01 = sseCSSimu(simuK,CtRatio[0])
    a02 = sseCSSimu(simuK,CtRatio[1])
    
    a = np.hstack((a01,a02[:,1:]))
    #
    fig, ax1 = plt.subplots()
    ax1.plot(a[:,0], a[:,1],'*-',label = 'NCE based Prevention (p = 0.2)',color = 'xkcd:azure')
    ax1.plot(a[:,0], a[:,2],'*-',label = 'MRD based Prevention (p = 0.2)',color = 'xkcd:coral')
    ax1.plot(a[:,0], a[:,3],'v-',label = 'NCE based Prevention (p = 0.5)',color = 'xkcd:azure')
    ax1.plot(a[:,0], a[:,4],'v-',label = 'MRD based Prevention (p = 0.5)',color = 'xkcd:coral')
    ax1.set_ylim([0.5, 2])
    ax1.set_ylabel('Relative Effect',size=12)
    ax1.legend(loc = 0,prop={'size':10})
    ax1.set_xlabel('$CV_{\lambda}$')

# demo
# comCSPlot()