#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017-08-30
Modification: to add some statement to improve the data after the sample size exceeds 
            the threshold. 
   1)  when the sample size exceeds the threshold, all the active hosts continue
        infect other individuals. But the newly infected individual will not spread the disease any more
   2) make sure that there are n independent transmission sequences 

09-01: the simulation stops at the last branch event, and the life length of all the individuals are 
        coordinated to that time.
12-04: add the Simu_InftPop_SID function for SID model
"""

import numpy as np
from Bio import Phylo
from io import StringIO
import transEventFun0815 as tEF

def transPair2treebranch(TransPair,Host_Life):
    # initialization
    SampleSize = TransPair.shape[0]
    TreeBranch = np.zeros((SampleSize,3));TreeBranch[:,:2] = TransPair[:,:2]
    NodeBranchTime = np.zeros((SampleSize+1,2)); # recode the latest time of Branching event for all nodes
    NodeBranchTime[:,0] = Host_Life[:,0]; #initialized with the ID 
    NodeBranchTime[:,1] = Host_Life[:,1]; #initialized with the birth times of all nodes
    
    Host_tailBranch = np.zeros((SampleSize+1,2));
    Host_tailBranch[:,0] = Host_Life[:,0]; #initialized with the ID
    
    for i in range(SampleSize):
        donnor,recipt, temp_BTime = (TransPair[i,:]) 
        # calculate the treeBranch based on the times of this infection and lastest braching event 
        # of the donnor 

        temp_id = np.nonzero(NodeBranchTime[:,0] == np.array([donnor]))[0]
        TreeBranch[i,2] = temp_BTime - NodeBranchTime[temp_id,1]
        # update the branch time for both the donnor and the recipient 
        NodeBranchTime[temp_id,1] = temp_BTime;        
        temp_id = np.nonzero(NodeBranchTime[:,0] == np.array([recipt]))[0]
        NodeBranchTime[temp_id,1] = temp_BTime;
    
    # the tail branch is calculated based on the times of death and the last branch event of every node
    Host_tailBranch[:,1] = Host_Life[:,2] - NodeBranchTime[:,1]    
    return(TreeBranch,Host_tailBranch)

def treeBranch2treeView(TreeBranch,Host_tailBranch):
    TreeBranch[:,2] = np.round(TreeBranch[:,2],2);
    Host_tailBranch[:,1] = np.round(Host_tailBranch[:,1],2);
    NodeString = []
    LifeString = []
    for i in range(TreeBranch.shape[0]):
        temp_str = '(' + str(int(TreeBranch[i,0])) + ',' + str(int(TreeBranch[i,1])) \
                    + '):' + str(TreeBranch[i,2])
        NodeString.append(temp_str)
    for i in range(Host_tailBranch.shape[0]):
        temp_str = str(int(Host_tailBranch[i,0])) + ':' + str(Host_tailBranch[i,1])
        LifeString.append(temp_str)
    
    treeNewick = NodeString[0]
    for i in range(1,TreeBranch.shape[0]):
        id_string = ','+ str(int(TreeBranch[i,0])) + ')'    
        temp_str = treeNewick.replace(id_string,','+ NodeString[i]+ ')')
        if temp_str!=treeNewick:
            treeNewick = temp_str
        else:
            id_string = '(' + str(int(TreeBranch[i,0])) + ','
            treeNewick = treeNewick.replace(id_string,'(' + NodeString[i] + ',')
    for i in range(Host_tailBranch.shape[0]):
        id_string = ','+ str(int(Host_tailBranch[i,0])) + ')'    
        temp_str = treeNewick.replace(id_string,','+ LifeString[i] + ')')
        if temp_str!=treeNewick:
            treeNewick = temp_str
        else:
            id_string = '(' + str(int(Host_tailBranch[i,0])) + ','
            treeNewick = treeNewick.replace(id_string,'(' + LifeString[i] + ',')
    treeNewick += ';'    
    return(treeNewick)

   

  
def genrtIndSeq(mu_InFC,var_InFC,mu_Rec):
    #input the rate of infection and recovery
    #output, a serial of times, the last one is the time of dig, and the others are the times of infection
    #generate the rate of infection
    temp_rate = np.random.gamma(mu_InFC**2/var_InFC,var_InFC/mu_InFC,1) if var_InFC>0 else mu_InFC;
    #generate the times of infection
    tmpEnd = np.random.exponential(scale = 1/mu_Rec);
    tmpItv = np.random.exponential(scale = 1/temp_rate,size = np.int(np.ceil(tmpEnd*temp_rate)));
    
    tmoI   = np.cumsum(tmpItv);
    while tmoI[-1]<tmpEnd:
        tmpItv = np.append(tmpItv,np.random.exponential(scale = 1/temp_rate,size = np.int(np.ceil(tmpEnd*temp_rate))))
        tmoI   = np.cumsum(tmpItv);    
    pos = np.sum(tmoI<tmpEnd)
    tmoI[pos] = tmpEnd
    
    return(tmoI[:pos+1])
    
'''
    sysSeq = [time, type of event, owner of the event] m*3
    indSeq = [t1,t2, ..., tn]
    curt: current time 
    totalHost, number of host in sysSeq
'''    
def seqMerge(indSeq,sysSeq,curt,totalHost):

    indSeq = indSeq + curt;
    tempSeq = np.zeros((len(indSeq),3))
    tempSeq[:,0] = indSeq;
    tempSeq[:,1] = (totalHost)
    tempSeq[0:-1,2] = 1
    tempSeq[-1,2] = -1
    if len(sysSeq):
        sysSeq = np.concatenate((sysSeq,tempSeq),axis=0);
        sysSeq = sysSeq[np.argsort(sysSeq[:,0]),:]
    else:
        sysSeq = tempSeq;    
    return(sysSeq)
    

def OtbI(mu_InFC,var_InFC,mu_Rec,recSize):
    SampSize = 2*recSize
    k_InFC, theta_InFC  = (mu_InFC**2/var_InFC, var_InFC/mu_InFC) if var_InFC>0 else (0,0)
    recHost = 0
    
    while recHost < recSize :
        # ini parameters
        recHost,curt,TotalHost = 0,0,0;
        transPair       = np.zeros((SampSize,3))
        hostLife        = np.zeros((SampSize,3))
        indSeq          = [[]]*SampSize
        # infection of the first one
        indSeq[0]       = genrtIndSeq(mu_InFC,var_InFC,mu_Rec)
        TotalHost += 1;
    #    indSeq[0]       = np.array([0.1,0.3,0.5,0.7,0.9])
        sysSeq          = seqMerge(indSeq[0],[],curt,TotalHost)
        sysId,indId     = 0,TotalHost-1
        seqLen          = len(sysSeq[:,0])
        transPair[0,:]  = np.array([999,1,0])
        hostLife[0,:]   = np.array([1,0,indSeq[0][-1]]);
        
        
        while sysId < seqLen and recHost < recSize :
            if sysSeq[sysId,2]>0: # infection event
                indId += 1
                TotalHost += 1;
                curt = sysSeq[sysId,0]
                indSeq[indId]   = list(genrtIndSeq(mu_InFC,var_InFC,mu_Rec))        
                sysSeq          = seqMerge(np.array(indSeq[indId]),sysSeq,curt,TotalHost)
                
                #update variable
                seqLen += len(indSeq[indId])
                transPair[indId,:]  = np.array([sysSeq[sysId,1],TotalHost,curt])      
                hostLife[indId,:]   = np.array([indId+1,curt,curt+indSeq[indId][-1]]);
                sysId               += 1
                
                if TotalHost == len(indSeq): # the matrix is full
                    transPair= np.vstack((transPair,np.zeros((SampSize,3))));
                    hostLife = np.vstack((hostLife,np.zeros((SampSize,3)))); 
                    indSeq   += [[]]*SampSize      
                
            else: # recovery event
                curt = sysSeq[sysId,0]
                #update variable
                recHost += 1
                sysId   += 1
    hostLife,transPair = hostLife[:TotalHost,:],transPair[:TotalHost,:]
    censoring_flag = hostLife[:,2]> curt;
    return(hostLife,transPair,TotalHost,curt,sysSeq,sysId)


# given the material, for the second outbreak without prevention
def otbII_NoContrl(hostLife,transPair,TotalHost,curt,sysSeq,sysId,stopT,\
                   mu_InFC,var_InFC,mu_Rec):
    SampSize        = 10;
    transPair       = np.vstack((transPair,np.zeros((SampSize,3))));
    hostLife        = np.vstack((hostLife,np.zeros((SampSize,3)))); 
    indSeq,indId    = [[]]*SampSize, 0
    seqLen          = len(sysSeq[:,0])
    recHost         = 0
    #init for serial observation
    obSeries        = np.linspace(curt,stopT,num=5)
    totalHostSeries = np.zeros(len(obSeries));
    obPos           = 0;
    obTime          = curt;
    while curt<=stopT and sysId < seqLen:
            if sysSeq[sysId,2]>0: # infection event
                
                TotalHost += 1;
                curt = sysSeq[sysId,0]
                indSeq[indId]   = list(genrtIndSeq(mu_InFC,var_InFC,mu_Rec))        
                sysSeq          = seqMerge(np.array(indSeq[indId]),sysSeq,curt,TotalHost)
                
                #update variable
                seqLen += len(indSeq[indId])
                transPair[TotalHost-1,:]  = np.array([sysSeq[sysId,1],TotalHost,curt])      
                hostLife[TotalHost-1,:]   = np.array([TotalHost,curt,curt+indSeq[indId][-1]]);
                sysId               += 1
                indId               += 1
                
                if hostLife[-1,2] > 0: # the matrix is full
                    transPair= np.vstack((transPair,np.zeros((SampSize,3))));
                    hostLife = np.vstack((hostLife,np.zeros((SampSize,3)))); 
                    indSeq   += [[]]*SampSize      
                
            else: # recovery event
                curt = sysSeq[sysId,0]
                #update variable
                recHost += 1
                sysId   += 1
            # outcome of series
            if  curt >= obTime:
                totalHostSeries[obPos] = TotalHost # record
                # update
                obPos += 1;
                obTime = obSeries[obPos]; 
            # update the curt
            if sysId >= seqLen: break;
            curt = sysSeq[sysId,0]
    totalHostSeries[-1] = TotalHost # record the last outcome
    return(TotalHost,indSeq,recHost,totalHostSeries) # return the total infection and the sequence for the new infection

def otbII_WithContrl(hostLife,transPair,TotalHost,curt,sysSeq,sysId,stopT,indSeqBank,\
                     CTTL,ctrLb,method):
    SampSize        = 10;
    transPair       = np.vstack((transPair,np.zeros((SampSize,3))));
    hostLife        = np.vstack((hostLife,np.zeros((SampSize,3)))); 
    indId           = 0
    seqLen          = len(sysSeq[:,0])
    recHost         = 0  # number of recovery
    ctHost          = 0  # number of controlled ind
    #init for serial observation
    obSeries        = np.linspace(curt,stopT,num=5)
    totalHostSeries = np.zeros(len(obSeries));
    obPos           = 0;
    obTime          = curt;
    while curt<=stopT and sysId < seqLen:
            if sysSeq[sysId,2]>0: # infection event
                TotalHost   += 1;
                curt        = sysSeq[sysId,0]
                sysSeq      = seqMerge(np.array(indSeqBank[indId]),sysSeq,curt,TotalHost)
                
                #update variable
                seqLen += len(indSeqBank[indId])
                transPair[TotalHost-1,:]  = np.array([sysSeq[sysId,1],TotalHost,curt])      
                hostLife[TotalHost-1,:]   = np.array([TotalHost,curt,curt+indSeqBank[indId][-1]]);
                sysId               += 1
                indId               += 1    
                
                if hostLife[-1,2] > 0: # the matrix is full
                    transPair= np.vstack((transPair,np.zeros((SampSize,3))));
                    hostLife = np.vstack((hostLife,np.zeros((SampSize,3))));      
                
            else: # recovery event
                curt = sysSeq[sysId,0]
                censoring_flag = hostLife[:TotalHost,2]> curt;
                """
                1. update the phylo
                """                
                transPairRec,hostLifeRec,censoring_flagRec = \
                    tEF.lgeSampling(transPair[1:TotalHost,:],hostLife[:TotalHost,:],censoring_flag,\
                                    0,'Rec');
                if method == 'NCE':
                    """
                    2. calculate the features
                    """
                    
                    lthLfEdge,averBL,nct = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);
#                    lthLfEdge,averBL,nce = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);
                    tempID = tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1));   
                    
                    """
                    3. making decision--do intervention
                    """                
                    if nct >= ctrLb: # tempID is selected to prevention
                        ctHost += 1;
                        tempSeq = sysSeq;
                        if tempID : # tempID is not empty                        
                            for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                                tempSeq = np.delete(tempSeq, \
                                        np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                        sysSeq =   tempSeq;
                elif method == 'RND' and np.random.uniform()< ctrLb: #using the random control                                     
                    tempID = tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1));
                    ctHost += 1;
                    tempSeq = sysSeq;
                    if tempID : # tempID is not empty                        
                        for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                            tempSeq = np.delete(tempSeq, \
                                    np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                    sysSeq =   tempSeq;                   
                #update variable
                seqLen          = len(sysSeq[:,0])
                recHost += 1
                sysId   += 1
            # outcome of series
            if  curt >= obTime:
                totalHostSeries[obPos] = TotalHost # record
                # update
                obPos += 1;
                obTime = obSeries[obPos]; 
            # update the curt
            if sysId >= seqLen: break;
            curt = sysSeq[sysId,0]
    totalHostSeries[-1] = TotalHost # record the last outcome
    return(TotalHost,ctHost,totalHostSeries,recHost) # return the total infection and the sequence for the new infection
    
def otbII_WithContrl_2nd(hostLife,transPair,TotalHost,curt,sysSeq,sysId,stopT,indSeqBank,\
                     CTTL,ctrLb,method):
    SampSize        = 10;
    transPair       = np.vstack((transPair,np.zeros((SampSize,3))));
    hostLife        = np.vstack((hostLife,np.zeros((SampSize,3)))); 
    indId           = 0
    seqLen          = len(sysSeq[:,0])
    recHost         = 0  # number of recovery
    ctHost          = 0  # number of controlled ind
    #init for serial observation
    obSeries        = np.linspace(curt,stopT,num=5)
    totalHostSeries = np.zeros(len(obSeries));
    obPos           = 0;
    obTime          = curt;
    while curt<=stopT and sysId < seqLen:
            if sysSeq[sysId,2]>0: # infection event
                TotalHost   += 1;
                curt        = sysSeq[sysId,0]
                sysSeq      = seqMerge(np.array(indSeqBank[indId]),sysSeq,curt,TotalHost)
                
                #update variable
                seqLen += len(indSeqBank[indId])
                transPair[TotalHost-1,:]  = np.array([sysSeq[sysId,1],TotalHost,curt])      
                hostLife[TotalHost-1,:]   = np.array([TotalHost,curt,curt+indSeqBank[indId][-1]]);
                sysId               += 1
                indId               += 1    
                
                if hostLife[-1,2] > 0: # the matrix is full
                    transPair= np.vstack((transPair,np.zeros((SampSize,3))));
                    hostLife = np.vstack((hostLife,np.zeros((SampSize,3))));      
                
            else: # recovery event
                curt = sysSeq[sysId,0]
                censoring_flag = hostLife[:TotalHost,2]> curt;
                """
                1. update the phylo
                """                
                transPairRec,hostLifeRec,censoring_flagRec = \
                    tEF.lgeSampling(transPair[1:TotalHost,:],hostLife[:TotalHost,:],censoring_flag,\
                                    0,'Rec');
                if method == 'NCE':
                    """
                    2. calculate the features
                    """
                    
                    lthLfEdge,averBL,nct = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);
#                    lthLfEdge,averBL,nce = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);
                    tempID = tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1));
                    
                    if len(tempID) > 0:
                        sndContID = []; # the total cases to be controlled
                        for evId in tempID:
                            sndContID += tEF.udIFD_AllInd0828(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(evId-1),curt);
                        tempID += sndContID # conbine these two 
                    """
                    3. making decision--do intervention
                    """                
                    if nct >= ctrLb: # tempID is selected to prevention
                        ctHost += 1;
                        tempSeq = sysSeq;
                        if tempID : # tempID is not empty                        
                            for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                                tempSeq = np.delete(tempSeq, \
                                        np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                        sysSeq =   tempSeq;
                elif method == 'RND' and np.random.uniform()< ctrLb: #using the random control                                     
                    tempID = tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1));
                    if len(tempID) > 0:
                        sndContID = []; # the total cases to be controlled
                        for evId in tempID:
                            sndContID += tEF.udIFD_AllInd0828(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(evId-1),curt);
                        tempID += sndContID # conbine these two
                    ctHost += 1;
                    tempSeq = sysSeq;
                    if tempID : # tempID is not empty                        
                        for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                            tempSeq = np.delete(tempSeq, \
                                    np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                    sysSeq =   tempSeq;                   
                #update variable
                seqLen          = len(sysSeq[:,0])
                recHost += 1
                sysId   += 1
            # outcome of series
            if  curt >= obTime:
                totalHostSeries[obPos] = TotalHost # record
                # update
                obPos += 1;
                obTime = obSeries[obPos]; 
            # update the curt
            if sysId >= seqLen: break;
            curt = sysSeq[sysId,0]
    totalHostSeries[-1] = TotalHost # record the last outcome
    return(TotalHost,ctHost,totalHostSeries,recHost) # return the total infection and the sequence for the new infection

'''
In the 0828, we ignore the removed inds in the phylo by 
setting their diagnosis time = 9999
'''

def otbII_WithContrl0828(hostLife,transPair,TotalHost,curt,sysSeq,sysId,stopT,indSeqBank,\
                     CTTL,ctrLb,method):
    SampSize        = 10;
    transPair       = np.vstack((transPair,np.zeros((SampSize,3))));
    hostLife        = np.vstack((hostLife,np.zeros((SampSize,3)))); 
    indId           = 0
    seqLen          = len(sysSeq[:,0])
    recHost         = 0  # number of recovery
    ctHost          = 0  # number of controlled ind
    #init for serial observation
    obSeries        = np.linspace(curt,stopT,num=5)
    totalHostSeries = np.zeros(len(obSeries));
    obPos           = 0;
    obTime          = curt;
    while curt<=stopT and sysId < seqLen:
            if sysSeq[sysId,2]>0: # infection event
                TotalHost   += 1;
                curt        = sysSeq[sysId,0]
                sysSeq      = seqMerge(np.array(indSeqBank[indId]),sysSeq,curt,TotalHost)
                
                #update variable
                seqLen += len(indSeqBank[indId])
                transPair[TotalHost-1,:]  = np.array([sysSeq[sysId,1],TotalHost,curt])      
                hostLife[TotalHost-1,:]   = np.array([TotalHost,curt,curt+indSeqBank[indId][-1]]);
                sysId               += 1
                indId               += 1    
                
                if hostLife[-1,2] > 0: # the matrix is full
                    transPair= np.vstack((transPair,np.zeros((SampSize,3))));
                    hostLife = np.vstack((hostLife,np.zeros((SampSize,3))));      
                
            else: # recovery event
                curt = sysSeq[sysId,0]
                censoring_flag = hostLife[:TotalHost,2]> curt;
                """
                1. update the phylo
                """                
                transPairRec,hostLifeRec,censoring_flagRec = \
                    tEF.lgeSampling(transPair[1:TotalHost,:],hostLife[:TotalHost,:],censoring_flag,\
                                    0,'Rec');
                if method == 'NCE':
                    """
                    2. calculate the features
                    """
                    
                    lthLfEdge,averBL,nct = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);
#                    lthLfEdge,averBL,nce = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);
                    tempID = tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1));   
                    
                    """
                    3. making decision--do intervention
                    """                
                    if nct >= ctrLb: # tempID is selected to prevention
                        ctHost += 1;
                        tempSeq = sysSeq;
                        if tempID : # tempID is not empty                        
                            for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                                tempSeq = np.delete(tempSeq, \
                                        np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                        sysSeq =   tempSeq;
                        hostLife[(np.array(tempID)-1).astype(int),2] = 9999; # setting the prevented ids 0828
                elif method == 'RND' and np.random.uniform()< ctrLb: #using the random control                                     
                    tempID = tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1));
                    ctHost += 1;
                    tempSeq = sysSeq;
                    if tempID : # tempID is not empty                        
                        for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                            tempSeq = np.delete(tempSeq, \
                                    np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                    sysSeq =   tempSeq;  
                    hostLife[(np.array(tempID)-1).astype(int),2] = 9999
                #update variable
                seqLen          = len(sysSeq[:,0])
                recHost += 1
                sysId   += 1
            # outcome of series
            if  curt >= obTime:
                totalHostSeries[obPos] = TotalHost # record
                # update
                obPos += 1;
                obTime = obSeries[obPos]; 
            # update the curt
            if sysId >= seqLen: break;
            curt = sysSeq[sysId,0]
    totalHostSeries[-1] = TotalHost # record the last outcome
    return(TotalHost,ctHost,totalHostSeries,recHost) # return the total infection and the sequence for the new infection

def otbII_WithContrl_Nnd(hostLife,transPair,TotalHost,curt,sysSeq,sysId,stopT,indSeqBank,\
                     CTTL,ctrLb,method):
    SampSize        = 10;
    transPair       = np.vstack((transPair,np.zeros((SampSize,3))));
    hostLife        = np.vstack((hostLife,np.zeros((SampSize,3)))); 
    indId           = 0
    seqLen          = len(sysSeq[:,0])
    recHost         = 0  # number of recovery
    ctHost          = 0  # number of controlled ind
    #init for serial observation
    obSeries        = np.linspace(curt,stopT,num=5)
    totalHostSeries = np.zeros(len(obSeries));
    obPos           = 0;
    obTime          = curt;
    while curt<=stopT and sysId < seqLen:
            if sysSeq[sysId,2]>0: # infection event
                TotalHost   += 1;
                curt        = sysSeq[sysId,0]
                sysSeq      = seqMerge(np.array(indSeqBank[indId]),sysSeq,curt,TotalHost)
                
                #update variable
                seqLen += len(indSeqBank[indId])
                transPair[TotalHost-1,:]  = np.array([sysSeq[sysId,1],TotalHost,curt])      
                hostLife[TotalHost-1,:]   = np.array([TotalHost,curt,curt+indSeqBank[indId][-1]]);
                sysId               += 1
                indId               += 1    
                
                if hostLife[-1,2] > 0: # the matrix is full
                    transPair= np.vstack((transPair,np.zeros((SampSize,3))));
                    hostLife = np.vstack((hostLife,np.zeros((SampSize,3))));      
                
            else: # recovery event
                curt = sysSeq[sysId,0]
                censoring_flag = hostLife[:TotalHost,2]> curt;
                """
                1. update the phylo
                """                
                transPairRec,hostLifeRec,censoring_flagRec = \
                    tEF.lgeSampling(transPair[1:TotalHost,:],hostLife[:TotalHost,:],censoring_flag,\
                                    0,'Rec');
                if method == 'NCE':
                    """
                    2. calculate the features
                    """
                    
                    lthLfEdge,averBL,nct = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);
#                    lthLfEdge,averBL,nce = tEF.nodeFetres0530(transPairRec,hostLifeRec,3,CTTL,sysSeq[sysId,1]);

                    # perform the n-th tracking
                    tempID = np.array(tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1)));
                    crtContID = tempID;
                    while len(crtContID) > 0:
                        NewContID = np.array([]); # the total cases to be controlled
                        for evId in crtContID:
                            tempContID = np.array(tEF.udIFD_AllInd0828(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(evId-1),curt));
                            # 1 delete the selected ID
                            tempContID = tempContID[np.isin(tempContID,tempID,invert=True)]
                            # keep the infectious: the hostLife[:,2]> curt && < 9999
                            tempContID = tempContID[np.logical_and(hostLife[(tempContID-1).astype(int),2]>=curt,hostLife[(tempContID-1).astype(int),2]<999)] if len(tempContID)>0 else []
                            NewContID  = np.append(NewContID,tempContID)
                        tempID = np.append(tempID,NewContID) # conbine these two 
                        crtContID = NewContID;
                    """
                    3. making decision--do intervention
                    """                
                    if nct >= ctrLb: # tempID is selected to prevention
                        ctHost += 1;
                        tempSeq = sysSeq;
                        if len(tempID)>0 : # tempID is not empty                        
                            for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                                tempSeq = np.delete(tempSeq, \
                                        np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                        sysSeq =   tempSeq;
                        hostLife[(np.array(tempID)-1).astype(int),2] = 9999
                elif method == 'RND' and np.random.uniform()< ctrLb: #using the random control                                     
                    tempID = np.array(tEF.udIFD_AllInd(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(sysSeq[sysId,1]-1)));
                    crtContID = tempID;
                    while len(crtContID) > 0:
                        NewContID = np.array([]); # the total cases to be controlled
                        for evId in crtContID:
                            tempContID = np.array(tEF.udIFD_AllInd0828(transPair[1:TotalHost,:],hostLife[:TotalHost,:],CTTL,np.int(evId-1),curt));
                            # 1 delete the selected ID
                            tempContID = tempContID[np.isin(tempContID,tempID,invert=True)]
                            # keep the infectious: the hostLife[:,2]> curt && < 9999
                            tempContID = tempContID[np.logical_and(hostLife[(tempContID-1).astype(int),2]>=curt,hostLife[(tempContID-1).astype(int),2]<999)] if len(tempContID)>0 else []
                            NewContID  = np.append(NewContID,tempContID)
                        tempID = np.append(tempID,NewContID) # conbine these two 
                        crtContID = NewContID;
                        
                        
                    ctHost += 1;
                    tempSeq = sysSeq;
                    if len(tempID)>0 : # tempID is not empty                        
                        for rmvId in tempID: # select all the ind in  sysSeq, and remove them
                            tempSeq = np.delete(tempSeq, \
                                    np.where(np.logical_and(tempSeq[:,1]==rmvId, tempSeq[:,0]>curt)), 0)
                    sysSeq =   tempSeq;       
                    hostLife[(np.array(tempID)-1).astype(int),2] = 9999
                #update variable
                seqLen          = len(sysSeq[:,0])
                recHost += 1
                sysId   += 1
            # outcome of series
            if  curt >= obTime:
                totalHostSeries[obPos] = TotalHost # record
                # update
                obPos += 1;
                obTime = obSeries[obPos]; 
            # update the curt
            if sysId >= seqLen: break;
            curt = sysSeq[sysId,0]
    totalHostSeries[-1] = TotalHost # record the last outcome
    return(TotalHost,ctHost,totalHostSeries,recHost) # return the total infection and the sequence for the new infection

def Simu_InftPop_GilleSID_OtbI(mu_InFC,var_InFC,mu_Rec,recSize):
    SampSize = 2*recSize
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    host_infe_var = -999
    while host_infe_var < 0:
        BD_rate         = np.zeros((SampSize,3))
        BD_rate[:,0]    = np.arange(SampSize)
        
        BD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
        BD_rate[:,2]    = mu_Rec # the same recovery meanTime
        
        # Initialization of other parameters
        TotalHost,Active_Host = 1,1
        TransPair           = np.zeros((SampSize,3))
        Host_Life           = np.zeros((SampSize,3))
        InFC_State          = np.zeros((SampSize,2))
        
        # Initialization of the epidemic
        TransPair[TotalHost-1,:]    = np.array([999,1,0])
        Host_Life[TotalHost-1,:]    = np.array([1,0,0]);
        InFC_State[TotalHost-1,:]   = np.array([1,BD_rate[TotalHost-1,1]]);
        # InFC_State = [id, rate_i], when the host recovered, rate_i = -9999
        
        curt = 0;recHost = 0;
        while recHost < recSize:
            Active_id = np.nonzero(InFC_State[:TotalHost,1]>0)[0]
            Active_Host         = len(Active_id)
            temp_rate = np.sum(InFC_State[Active_id,1]) + Active_Host*mu_Rec;
            if temp_rate<=0: break;
            curt += np.random.exponential(scale = 1/temp_rate); 
            # choosing the host for the event
            sus_pro = (InFC_State[Active_id,1] + mu_Rec)/temp_rate;
            temp_id = Active_id[np.nonzero(np.random.multinomial(1,sus_pro))];
            # infection or recovery?
            if np.random.uniform()< mu_Rec/(InFC_State[temp_id,1] + mu_Rec): # recovery
                InFC_State[temp_id,1] = -9999;
                Host_Life[temp_id,2] = curt;
                recHost += 1;
            else: # infection
                TotalHost += 1; Active_Host += 1;
                recip_id          = TotalHost;
                donnor_id         = InFC_State[temp_id,0];
                TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,curt]);
                Host_Life[TotalHost-1,:] = np.array([recip_id,curt,0]);
                InFC_State[TotalHost-1,:]= np.array([recip_id,BD_rate[TotalHost-1,1]]);# update the recipient
                
                if TotalHost == np.shape(Host_Life)[0]: # the matrix is full
                    TransPair = np.vstack((TransPair,np.zeros((SampSize,3))));
                    Host_Life = np.vstack((Host_Life,np.zeros((SampSize,3))));
                    InFC_State = np.vstack((InFC_State,np.zeros((SampSize,2))));
                    newBD_rate = np.zeros((SampSize,3))        
                    newBD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
                    newBD_rate[:,2]    = mu_Rec # 
                    BD_rate = np.vstack((BD_rate,newBD_rate));
        # input the transPair of node 1~10 and the corresponding life time for them
        if recHost >= recSize: 
            # trimming the matrix
            Host_Life,TransPair = Host_Life[:TotalHost,:],TransPair[:TotalHost,:]
            InFC_State, BD_rate = InFC_State[:TotalHost,:], BD_rate[:TotalHost,:]
            
            
            censoring_flag = Host_Life[:,2]<= 0;
            Host_Life[censoring_flag,2] = curt;
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,\
           Host_Life,InFC_State,BD_rate,TotalHost,curt)
    """
    The function of OtbI does not output the initial event
    so when appling the OtbII, it is necessary to integrate the inital event 
    """
  
def Simu_InftPop_GilleSID_OtbII_ENT(mu_InFC,var_InFC,mu_Rec,\
                              preTransPair,preHost_Life,preInFC_State,preBD_rate,\
                              preTotalHost,precurt,\
                              endTime):
    
    #recSize -- the target sample size in the new epidemic
    #recHost -- the recovered Host in the new epidemic
    #TotalHost -- total number of Host in two epidemic
    #SampSize -- the size of expanding the recording matrix 
    #precurt -- the starting time of the epidemic 2
    #endTime -- the ending time of the epidemic 2
    
    SampSize = 10
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    host_infe_var = -999

    # Initialization of the epidemic according to the record
    TransPair = np.vstack((preTransPair,np.zeros((SampSize,3))));
    Host_Life = np.vstack((preHost_Life,np.zeros((SampSize,3))));
    InFC_State = np.vstack((preInFC_State,np.zeros((SampSize,2))));
    newBD_rate = np.zeros((SampSize,3))        
    newBD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
    newBD_rate[:,2]    = mu_Rec # 
    BD_rate = np.vstack((preBD_rate,newBD_rate));
    # Initialization of other parameters
    TotalHost = preTotalHost;
    curt = precurt;recHost = 0;
    while curt < endTime:
        Active_id = np.nonzero(InFC_State[:TotalHost,1]>0)[0]
        Active_Host         = len(Active_id)
        temp_rate = np.sum(InFC_State[Active_id,1]) + Active_Host*mu_Rec;
        if temp_rate<=0: break;
        curt += np.random.exponential(scale = 1/temp_rate); 
        # choosing the host for the event
        sus_pro = (InFC_State[Active_id,1] + mu_Rec)/temp_rate;
        temp_id = Active_id[np.nonzero(np.random.multinomial(1,sus_pro))];
        # infection or recovery?
        if np.random.uniform()< mu_Rec/(InFC_State[temp_id,1] + mu_Rec): # recovery
            InFC_State[temp_id,1] = -9999;
            Host_Life[temp_id,2] = curt;
            recHost += 1;
        else: # infection
            TotalHost += 1; Active_Host += 1;
            recip_id          = TotalHost;
            donnor_id         = InFC_State[temp_id,0];
            TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,curt]);
            Host_Life[TotalHost-1,:] = np.array([recip_id,curt,0]);
            InFC_State[TotalHost-1,:]= np.array([recip_id,BD_rate[TotalHost-1,1]]);# update the recipient
            
            if TotalHost == np.shape(Host_Life)[0]: # the matrix is full
                TransPair = np.vstack((TransPair,np.zeros((SampSize,3))));
                Host_Life = np.vstack((Host_Life,np.zeros((SampSize,3))));
                InFC_State = np.vstack((InFC_State,np.zeros((SampSize,2))));
                newBD_rate = np.zeros((SampSize,3))        
                newBD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
                newBD_rate[:,2]    = mu_Rec # 
                BD_rate = np.vstack((BD_rate,newBD_rate));
        # input the transPair of node 1~10 and the corresponding life time for them

    # trimming the matrix
    Host_Life,TransPair = Host_Life[:TotalHost,:],TransPair[:TotalHost,:]
    censoring_flag = Host_Life[:,2]<= 0;
    Host_Life[censoring_flag,2] = curt;
    TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
    simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
    
    
    return(TransPair[1:,:],Host_Life,TotalHost,simutree,censoring_flag)