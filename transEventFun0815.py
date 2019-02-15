#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 09:45:54 2017
0922 adding the censoring flag to the eventTail 
1002 new type of equation for mu and sigma
1006 add codes for the eventAccInfo 
0202 add the function for tracing
@author: zhang
"""
import numpy as np
from heapq import nlargest
import scipy as sp

def arrBeginAdd(arr,element): # added an element at the begining 
    insertele = lambda x: element + x
    c = list(map(insertele,arr));
    return(c)

'''
    calculate the undiagnosed contacts for the diagnosed INDs with a period of TL
'''

def udIndCount(transPair,hostLife,censoring_flag,TL):
    """
    Calculate the number of undiagnosed for all the recovered INDs
    """
    # TransPair,Host_Life,TL
    DgInfor = hostLife[~censoring_flag,:]
    DgID,DgRt = DgInfor[:,0],DgInfor[:,2]
    udIndNum = np.zeros((len(DgID),2))
    udIndNum[:,0] = DgID
    for i in range(len(DgID)):
        tempDg = DgID[i];
        tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt[i]-TL,transPair[:,2]<=DgRt[i]),:];
        # select the children
        tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1];
        if tempCLD.any():
            udIndNum[i,1] += np.sum(censoring_flag[(tempCLD-1).astype(int)]);        
        # select the ansestors
        tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0];
        if tempAST.any():
            udIndNum[i,1] += np.sum(censoring_flag[(tempAST-1).astype(int)]);
    return(udIndNum)

def udIndCountRate(transPair,hostLife,censoring_flag,BD_rate,TL):
    """
    Calculate the total rates of the undiagnosed for all the recovered INDs
    """
    # TransPair,Host_Life,TL
    DgInfor = hostLife[~censoring_flag,:]
    DgID,DgRt = DgInfor[:,0],DgInfor[:,2]
    udIndNum = np.zeros((len(DgID),3))
    udIndNum[:,0] = DgID
    for i in range(len(DgID)):
        tempDg = DgID[i];
        tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt[i]-TL,transPair[:,2]<=DgRt[i]),:];
        # select the children
        tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1];
        if tempCLD.any():
            udIndNum[i,1] += np.sum(censoring_flag[(tempCLD-1).astype(int)]); 
            udIndNum[i,2] += np.sum(BD_rate[(tempCLD-1).astype(int),1]*censoring_flag[(tempCLD-1).astype(int)]);
        # select the ansestors
        tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0];
        if tempAST.any():
            udIndNum[i,1] += np.sum(censoring_flag[(tempAST-1).astype(int)]);
            udIndNum[i,2] += np.sum(BD_rate[(tempAST-1).astype(int),1]*censoring_flag[(tempAST-1).astype(int)]);
    return(udIndNum)

def udIndCountID(transPair,hostLife,censoring_flag,TL):
    # TransPair,Host_Life,TL
    DgInfor = hostLife[~censoring_flag,:]
    DgID,DgRt = DgInfor[:,0],DgInfor[:,2]
    udIndID = [[]]*len(DgID)
#    udIndNum[:,0] = DgID
    for i in range(len(DgID)):
        tempDg = DgID[i];
        tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt[i]-TL,transPair[:,2]<=DgRt[i]),:];
        # select the children
        tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1];
        if tempCLD.any():
            udIndID[i] = udIndID[i]+tempCLD[0].tolist();        
        # select the ansestors
        tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0];
        if tempAST.any():
            udIndID[i] = udIndID[i]+tempAST[0].tolist();
    return(udIndID)

'''
    calculate the branch events for a node with a period of TL
'''
def brEvtCount(transPairRec,hostLifeRec,TL):
    # TransPair,Host_Life,TL
    brEvtNum = hostLifeRec;
    brEvtNum[:,1] = 0;
    
    for i in range(np.shape(brEvtNum)[0]):
        tempDg = brEvtNum[i,0];
        tempt = hostLifeRec[i,2];
        tempPair = transPairRec[np.logical_and(transPairRec[:,2]>=tempt-TL\
                        ,transPairRec[:,2]<=tempt),:];
        m = np.shape(tempPair)[0];
        for j in range(m):
            dn, rect = tempPair[m-1-j,0:2];
            if dn== tempDg:
                brEvtNum[i,1] += 1;
            elif rect == tempDg:
                brEvtNum[i,1] += 1;
                tempDg = dn;
    return(brEvtNum)   
     
def udIndCountSI(transPair,hostLife,censoring_flag,TL,ctid):
    """
    Calculate the number of undiagnosed for all the recovered INDs
    """
    # TransPair,Host_Life,TL
    DgInfor = hostLife[ctid,:]
    DgID,DgRt = DgInfor[0],DgInfor[2]
    udIndNum = 0;
    
    
    tempDg = DgID;
    tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt-TL,transPair[:,2]<=DgRt),:];
    # select the children
    tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1];
    if tempCLD.any():
        udIndNum += np.sum(censoring_flag[(tempCLD-1).astype(int)]);        
    # select the ansestors
    tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0];
    if tempAST.any():
        udIndNum += np.sum(censoring_flag[(tempAST-1).astype(int)]);
    return(udIndNum)

def udIndCountRateSI(transPair,hostLife,censoring_flag,BD_rate,TL,ctid):
    """
    Calculate the total rates of the undiagnosed for all the recovered INDs
    """
    # TransPair,Host_Life,TL
    DgInfor = hostLife[ctid,:]
    DgID,DgRt = DgInfor[0],DgInfor[2]
    udIndNum = np.zeros(2)
    
    tempDg = DgID;
    tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt-TL,transPair[:,2]<=DgRt),:];
    # select the children
    tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1];
    if tempCLD.any():
        udIndNum[0] += np.sum(censoring_flag[(tempCLD-1).astype(int)]); 
        udIndNum[1] += np.sum(BD_rate[(tempCLD-1).astype(int),1]*censoring_flag[(tempCLD-1).astype(int)]);
    # select the ansestors
    tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0];
    if tempAST.any():
        udIndNum[0] += np.sum(censoring_flag[(tempAST-1).astype(int)]);
        udIndNum[1] += np.sum(BD_rate[(tempAST-1).astype(int),1]*censoring_flag[(tempAST-1).astype(int)]);
    return(udIndNum)


def udIFD(transPair,hostLife,infe_state,TL,ctid):
    """
    Calculate the traced INDs for each recovered INDs
    """
    # TransPair,Host_Life,TL
    DgInfor = hostLife[ctid,:]
    DgID,DgRt = DgInfor[0],DgInfor[2]
    udIndID = []
#    udIndNum[:,0] = DgID
    
    tempDg = DgID;
    tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt-TL,transPair[:,2]<=DgRt),:];
    # select the children
    tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1]; # 
    if tempCLD.any():
        infCLD = tempCLD[infe_state[(tempCLD-1).astype(int),1]>0];
        if infCLD.any(): udIndID = infCLD.tolist();        
    # select the ansestors
    tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0];
    if tempAST.any():
        infAST = tempAST[infe_state[(tempAST-1).astype(int),1]>0];
        if infAST.any(): udIndID = udIndID+infAST.tolist();
    return(udIndID)


def udIFD_AllInd(transPair,hostLife,TL,ctid):
    """
    Calculate the traced INDs for each recovered INDs, whatever it is infectious or recovered
    It is a generalization of udIFD
    """
    # TransPair,Host_Life,TL
    DgInfor = hostLife[ctid,:]
    DgID,DgRt = DgInfor[0],DgInfor[2]
    udIndID = []
#    udIndNum[:,0] = DgID
    
    tempDg = DgID;
    tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt-TL,transPair[:,2]<=DgRt),:];
    # select the children
    tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1][0]; # 
    if tempCLD.any(): udIndID = tempCLD.tolist();
             
    # select the ansestors
    tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0][0];
    if tempAST.any(): udIndID = udIndID+tempAST.tolist();
    return(udIndID)
    
def udIFD_AllInd0828(transPair,hostLife,TL,ctid,curt):
    """
    Calculate the traced INDs for each recovered INDs, whatever it is infectious or recovered
    It is a generalization of udIFD
    """
    # TransPair,Host_Life,TL
    DgInfor = hostLife[ctid,:]
    DgID,DgRt = DgInfor[0],np.minimum(DgInfor[2],curt) # added 0828, for the undiagnosed cases
    udIndID = []
#    udIndNum[:,0] = DgID
    
    tempDg = DgID;
    tempPair = transPair[np.logical_and(transPair[:,2]>=DgRt-TL,transPair[:,2]<=DgRt),:];
    # select the children
    tempCLD = tempPair[np.where(tempPair[:,0]==tempDg),1][0]; # 
    if tempCLD.any(): udIndID = tempCLD.tolist();
             
    # select the ansestors
    tempAST = tempPair[np.where(tempPair[:,1]==tempDg),0][0];
    if tempAST.any(): udIndID = udIndID+tempAST.tolist();
    return(udIndID)


'''
    calculate the branch events for a node with a period of TL
    this function using the node ID
'''

def nodeFetres0220(transPairRec,hostLifeRec,kTL,TCT,nodeid):
    # TransPair,Host_Life,TL
    """
    kTL -- k step in the phylo to do average -> averBL
    TCT -- time to go back to count the nct
    nodeid -- the id of the node
    """
    temp = hostLifeRec[np.where(hostLifeRec[:,0]==nodeid)]
    lthLfEdge,averBL = 0,0
    if temp.any():
        tempDg,ret = temp[0][0],temp[0][2];    
        tempPair = transPairRec[np.logical_and(transPairRec[:,2]>=0\
                        ,transPairRec[:,2]<=ret),:];
        m = np.shape(tempPair)[0];
        kCount = 0; stepTime = np.zeros(kTL+1);
        preT = ret;
        for j in range(m):
            dn, rect, tempt = tempPair[m-1-j,0:3];
            if dn== tempDg:
                stepTime[kCount] = preT- tempt;
                kCount += 1;
                preT = tempt;
            elif rect == tempDg:
                stepTime[kCount] = preT- tempt;
                kCount += 1;
                preT = tempt;
                tempDg = dn;
            if kCount>kTL: break; 
            
        lthLfEdge = stepTime[0];
        nct    = np.sum(np.cumsum(stepTime)<=TCT)
        if kCount>1: #kCount, the number of record in stepTime, shall >1
            stepTime = stepTime[1:kCount];
            averBL = np.mean(stepTime)        
    return(lthLfEdge,averBL,nct) 


'''
0530 new calculation of NCE at 2018-05-30
'''

def nodeFetres0530(transPairRec,hostLifeRec,kTL,TCT,nodeid):
    # TransPair,Host_Life,TL
    """
    kTL -- k step in the phylo to do average -> averBL
    TCT -- time to go back to count the nct
    nodeid -- the id of the node
    """
    temp = hostLifeRec[np.where(hostLifeRec[:,0]==nodeid)]
    lthLfEdge,averBL,nce = 0,0,0
    if temp.any():
        tempDg,ret = temp[0][0],temp[0][2];    
        tempPair = transPairRec[np.logical_and(transPairRec[:,2]>=0\
                        ,transPairRec[:,2]<=ret),:];
        m = np.shape(tempPair)[0];
        kCount = 0; stepTime = np.zeros(kTL+1);
        preT = ret;
        for j in range(m):
            dn, rect, tempt = tempPair[m-1-j,0:3];
            if dn== tempDg:
                stepTime[kCount] = preT- tempt;
                kCount += 1;
                preT = tempt;
            elif rect == tempDg:
                stepTime[kCount] = preT- tempt;
                kCount += 1;
                preT = tempt;
                tempDg = dn;
            if kCount>kTL: break; 
        # calculate the NCE
        tempDg,ret = temp[0][0],temp[0][2];
        tempt = ret;
        tempPair = transPairRec[np.logical_and(transPairRec[:,2]>=tempt-TCT\
                        ,transPairRec[:,2]<=tempt),:];
        m = np.shape(tempPair)[0];
        for j in range(m):
            dn, rect = tempPair[m-1-j,0:2];
            if dn== tempDg:
                nce += 1;
            elif rect == tempDg:
                nce += 1;
                tempDg = dn;
            
        lthLfEdge = stepTime[0];
        nct    = np.sum(np.cumsum(stepTime)<=TCT)
        if kCount>1: #kCount, the number of record in stepTime, shall >1
            stepTime = stepTime[1:kCount];
            averBL = np.mean(stepTime) 
    return(lthLfEdge,averBL,nce) 

'''
    0922 add the censoring_flag to the eventTail
    eventBranch--the branch length of each event [donor,recipient, branchLength]
    eventTail -- the tail information [tail of the donor, tail of the recipient], -99 if it has a child
    Host_tailBranch
    eventChild -- the children and parent info, [total children, child event of donor, 
                        child event of recipient, parent event]
    eventLevel -- level of the event, no child (level = 1), otherwise equal 
                    to the maximum level of its child +1
    eventPathSet -- all the path from this event to the leaves, list of list
    eventDepth   -- the steps from the root
    eventCensor -- the censor flag for the donor and recipient. -99 (not end), 0 (recovery), 1(censored)
    
'''
'''
    1006 add the informaiton of eventAccInfo for the EM algorithm
'''
'''
    1116 using the donnor_id,recipt_id in the calulation of eventTail et al
'''
def transPair2EventInfor1116(TransPair,Host_Life,censoring_flag):
    # initialization
    SampleSize = TransPair.shape[0]
    eventBranch = np.zeros((SampleSize,3));eventBranch[:,:2] = TransPair[:,:2]
    NodeBranchTime = np.zeros((SampleSize+1,2)); # recode the Branching time for all nodes
    NodeBranchTime[:,0] = Host_Life[:,0]; #initialized with the ID 
    NodeBranchTime[:,1] = Host_Life[:,1]; #initialized with the birth times of all nodes
    
    Host_tailBranch = np.zeros((SampleSize+1,2));
    Host_tailBranch[:,0] = Host_Life[:,0]; #initialized with the ID
    
    eventChild = np.zeros((SampleSize,4)).astype(int); eventChild[:,1:4] = -99;
    eventLevel = np.zeros(SampleSize).astype(int)
    eventTail  = np.zeros((SampleSize,2))-99; 
    eventPathSet = [[]]*SampleSize;
    eventCensor = np.zeros((SampleSize,2)).astype(int)-99;
    eventAccInfo = np.zeros((SampleSize,2));
    
   
    for i in range(SampleSize):
        donnor,recipt, temp_BTime = (TransPair[i,:]) 
        donnor_id,recipt_id = np.where(Host_Life[:,0]==donnor)[0],np.where(Host_Life[:,0]==recipt)[0]
        temp_id = np.nonzero(NodeBranchTime[:,0] == np.array([donnor]))[0]
        eventBranch[i,2] = temp_BTime - NodeBranchTime[temp_id,1]
#        temp_length_d = [0,temp_length_d][bool(temp_length_d<998)];
        NodeBranchTime[temp_id,1] = temp_BTime;
        temp_count = 0;
        if i< SampleSize-1: # the first n-1 events
            temp_id = np.nonzero(TransPair[i+1:,0] == donnor)[0]
            if len(temp_id) :  # the donnor will give birth later
                eventChild[i,1] = np.min(temp_id) +i +1; temp_count += 1;
            else:  # the donnor will not give birth later
                eventTail[i,0] = Host_Life[donnor_id,2] - temp_BTime;   
                eventCensor[i,0] = censoring_flag[donnor_id];
            temp_id = np.nonzero(TransPair[i+1:,0] == recipt)[0]
            if len(temp_id) :  
                eventChild[i,2] = np.min(temp_id) +i +1; temp_count += 1;
            else:
                eventTail[i,1] = Host_Life[recipt_id,2] - temp_BTime;
                eventCensor[i,1] = censoring_flag[recipt_id];
            eventChild[i,0] = temp_count;
        else: # the last event has no child 
            eventTail[i,0] = Host_Life[donnor_id,2] - temp_BTime; 
            eventTail[i,1] = Host_Life[recipt_id,2] - temp_BTime;
            eventCensor[i,0] = censoring_flag[donnor_id];
            eventCensor[i,1] = censoring_flag[recipt_id];
    Host_tailBranch[:,1] = Host_Life[:,2] - NodeBranchTime[:,1];
    
    # search for the parents of the event and calculate the depth of all the events
    eventDepth = np.ones(SampleSize).astype(int);
    
    for i in range(SampleSize):
        row, col = np.where(eventChild[:,1:3] == i);
        if len(row):
            eventChild[i,3] = row;
            
        donnor,recipt, temp_BTime = (TransPair[i,:])
        if i > 0:
            temp_id = np.nonzero(TransPair[0:i,0] == donnor)[0];
            if len(temp_id) :  # the donnor gave birth before
                temp_id = np.max(temp_id);
                eventDepth[i] = eventDepth[temp_id] +1; 
            else:  # the donnor will not give birth later
                temp_id2 = np.max(np.nonzero(TransPair[0:i,1] == donnor)[0])
                eventDepth[i] = eventDepth[temp_id2] +1;
        
    
    # search for the event level. 
    # add codes for the eventAccInfo 1006
    for i in range(SampleSize):
        if not eventChild[-i-1,0]:  # no child, then this event has level 1
            eventLevel[-i-1] = 1;
#            eventAccInfo[-i-1,:] = 1, eventBranch[-i-1,2]; # no child, accInfo = branchlength
            eventAccInfo[-i-1,:] = 1, eventBranch[-i-1,2]+np.min(eventTail[-i-1,:]); # 1019shorter is better 
        elif eventChild[-i-1,0]== 1: # one child, the level is the child's level plus 1
            temp_id = np.max(eventChild[-i-1,1:3]);
            eventLevel[-i-1] = eventLevel[temp_id] + 1;
            eventAccInfo[-i-1,:] = 0, np.max(eventTail[-i-1,:]) 
#            eventAccInfo[-i-1,:] = 0, np.max(eventTail[-i-1,:]); # no child, accInfo = branchlength
        else:                       # two children, the level is the bigger one plus 1
            temp_id1, temp_id2 = eventChild[-i-1,1:3];
            eventLevel[-i-1] = np.maximum(eventLevel[temp_id1], eventLevel[temp_id2]) + 1;
    
    # search for the path set.
    for i in range(SampleSize):
        event_id = SampleSize - i -1;
        if eventLevel[-i-1] == 1:
            temp_pathset = [[event_id]];
            eventPathSet[event_id] = temp_pathset;
        elif eventChild[-i-1,0] == 1:  # only one child
            CLD1 =  np.max(eventChild[-i-1,1:3]);
            temp_pathset =  eventPathSet[CLD1];
            temp_pathset = arrBeginAdd(temp_pathset,[event_id])
            temp_pathset = temp_pathset + [[event_id]];
            eventPathSet[event_id] = temp_pathset;
        else:
            CLD1, CLD2 =  eventChild[-i-1,1:3];
            temp_pathset1 =  eventPathSet[CLD1];
            temp_pathset1 = arrBeginAdd(temp_pathset1,[event_id]);
            temp_pathset2 =  eventPathSet[CLD2];
            temp_pathset2 = arrBeginAdd(temp_pathset2,[event_id]);
            temp_pathset = temp_pathset1 + temp_pathset2;
            eventPathSet[event_id] = temp_pathset;
    
        
    
    return(eventBranch,eventTail,Host_tailBranch,eventChild,eventLevel\
           ,eventPathSet,eventDepth,eventCensor,eventAccInfo)
'''
    The function of lgeSampling performs sampling to the given set of lineage.
    The main jobs include two parts: 1) to replace M with its last child; 2) delete M  
'''
def lgeSampling(TransPair,Host_Life,censoring_flag,delSize,SPmode):
    newTransPair,newHost_Life,newCensoring_flag \
        = TransPair.copy(),Host_Life.copy(),censoring_flag.copy()
    n = np.shape(Host_Life)[0];
#    generate the unsampled ID
    if SPmode =='Random':
        unsampledID = np.random.choice(n,size = int(delSize),replace=False);
        unsampledID = Host_Life[unsampledID,0];
        m = len(unsampledID)
    elif SPmode =='Rec':
        unsampledID = np.nonzero(censoring_flag==1)[0];
        unsampledID = Host_Life[unsampledID,0];
        m = len(unsampledID)        
#    for each unsampled host, delete it and replace it with its last child
    for m_id in range(m):
#    m_id = 0
        dtHost = unsampledID[m_id];
        tempPair = np.nonzero(newTransPair[:,0]==dtHost)[0];
        if len(tempPair) >0: # the dtHost has at least one child
            lsCHD = newTransPair[np.max(tempPair),1];
            # updating the TransPair: delete and replace
            newTransPair[tempPair,0] = lsCHD;
            newTransPair[newTransPair[:,1]==dtHost,1] = lsCHD;
            newTransPair = np.delete(newTransPair, np.max(tempPair), 0)
            
            # updating the Host_Life and censoring_flag
            lsCHD_id = np.where(newHost_Life[:,0]==lsCHD)[0];
            dtHost_id = np.where(newHost_Life[:,0]==dtHost)[0];
            # the birth time of lsCHD has been changed to that of the dtHost
            if len(lsCHD_id) != len(dtHost_id):
                a = 0
            newHost_Life[lsCHD_id,1] = newHost_Life[dtHost_id,1];
            newHost_Life = np.delete(newHost_Life, dtHost_id, 0);
        else: # the dtHost has no child
            newTransPair = np.delete(newTransPair, np.where(newTransPair[:,1]==dtHost)[0], 0)
            dtHost_id = np.where(newHost_Life[:,0]==dtHost)[0];
            newHost_Life = np.delete(newHost_Life, dtHost_id, 0);
        
        newCensoring_flag = np.delete(newCensoring_flag, dtHost_id, 0);
        
    
    return(newTransPair,newHost_Life,newCensoring_flag)

'''
    select the k path with least expected_dist
'''
def pathSelect(kpath,eventBranch,eventTail,eventCensor,eventPathSet,transPair,supThes,supSprder):
    seltedPath = np.zeros((kpath,5));
    pathScore = np.zeros(kpath)-100
    n = len(eventPathSet);
    for i in range(n): # for each event
        tempPathset = eventPathSet[i];
        m = len(tempPathset);
        for j in range(m): # for each path
            tempScore = -1000;
            if len(tempPathset[j]) >supThes: # 0928 length of path larger than supThes
                baseScore = np.min(pathScore);baseid = np.argmin(pathScore)
                
                
                path = eventBranch[np.array(tempPathset[j]),2];
                tailEvent_id = (tempPathset[j])[-1];
                if sum(eventTail[tailEvent_id,:]>0) == 2: # the event has no child
                    score1 = expctDist(np.append(path,eventTail[tailEvent_id,0]),eventCensor[tailEvent_id,0]);
                    score2 = expctDist(np.append(path,eventTail[tailEvent_id,1]),eventCensor[tailEvent_id,1]);
                    tail_id = int(score2>score1);tempScore = np.maximum(score1,score2);
                elif sum(eventTail[tailEvent_id,:]>0) == 1: # the event has no child
                    tail_id = np.nonzero(eventTail[tailEvent_id,:]>0)[0];
                    tempScore = expctDist(np.append(path,eventTail[tailEvent_id,tail_id])\
                                          ,eventCensor[tailEvent_id,tail_id]);
                if tempScore>baseScore: # the present path has lower score than one in the set
                    pathScore[baseid] = tempScore;
                    seltedPath[baseid,:3] = np.array([i,j,tail_id]);
    temp_od = np.argsort(pathScore)[::-1]; # sort the score in a descent way
    seltedPath = seltedPath[temp_od,:];
    pathScore  = pathScore[temp_od];
    
    #evaluate the selected path
    for i in range(kpath):
        tempPath = (eventPathSet[seltedPath[i,0].astype(int)])[seltedPath[i,1].astype(int)]
        tempEvent = transPair[np.array(tempPath),0];
        count = np.sum(tempEvent == supSprder);
        
#        u, count = np.unique(tempEvent,return_counts = True);
        seltedPath[i,3] = np.max(count);
        seltedPath[i,4] = len(tempEvent);
    return(seltedPath,pathScore)             

'''
    the following three functions are for the MLE based on the branch length ended with a birth event,
    i.e, lambda*exp(-(lambda+gamma)*x)
'''
def SlcBeta(beta,x):
    k = beta/(beta+x);
    return(np.mean(k)*(1 - np.mean(np.log(k))) - 1)
    
def SlcEst(x): # using the iteration to search for the root of 
    a = 1e-5;
    b = 1000;
    if SlcBeta(b,x)*SlcBeta(a,x)>0:
        z = 1/x
        mu = np.median(z); sigma = np.median(abs(z - np.median(z)));
#    beta = sp.optimize.newton(SlcBeta,x0=a,args=x);
    else:
        beta = sp.optimize.bisect(SlcBeta,a,b,args=x);
        k = beta/(beta+x);
        alpha = -1/np.mean(np.log(k));
        mu = alpha/beta; sigma = np.sqrt(alpha)/beta;       
    return(mu,mu**2/sigma**2,sigma)

def AvegSlcEst(eventLevel,eventBranch,ave_order):
    dep, count = np.unique(eventLevel, return_counts=True);
    largcont = nlargest(ave_order,count);
    largDep = np.zeros(ave_order);
    mle_mu = np.zeros(ave_order); mle_sigma = np.zeros(ave_order)
    moe_mu = np.zeros(ave_order); moe_sigma = np.zeros(ave_order)
#   mle_est = np.zeros(ave_order);
    for i in range(ave_order):
        tempDep = (dep[count == largcont[i]])[0];
        temp    = (eventBranch[np.nonzero(eventLevel == tempDep),2])[0]
        z       = 1/temp
        moe_mu[i] = np.median(z)
        moe_sigma[i] = np.median(abs(z - np.median(z)))
  #      mle_est[i] = SlcEst(temp)[1];
#        moe_est[i] = np.std(1/temp,ddof = 1)
        mle_mu[i],mle_sigma[i] = SlcEst(temp);
#        if sigma_est[i] == 0 and moe_est[i]< 50:
#            sigma_est[i] = (sigma_est[i] + moe_est[i])/2
#    return(mle_mu,moe_mu,mle_sigma,moe_sigma)
    return(np.mean(mle_mu),np.median(moe_mu),np.mean(mle_sigma),np.median(moe_sigma))
        
'''
    the following functions are for the MLE based on interval distribution, i.e., 
    the EXPD(lambda + gamma) 
    
'''
def intvlMleBeta(beta,gamma,x):    
    k = beta/(beta+x);
    aHat = (np.sum(k) + gamma*beta*np.sum(np.log(k))) / np.sum(x/beta+x);
    b = np.sum(1/(aHat + gamma*beta + gamma*x)) +  np.sum(np.log(k));
    return(b)
    
def intvlMle(gamma,x): # using the iteration to search for the root of 
    a = 1e-25;
    b = 100;
    if SlcBeta(b,x)*SlcBeta(a,x)>0:
        mu = 1/np.mean(x); sigma = 0;
#    beta = sp.optimize.newton(SlcBeta,x0=a,args=x);
    else:
        beta = sp.optimize.bisect(intvlMleBeta,a,b,args=(gamma,x));
        k = beta/(beta+x);
        alpha = (np.sum(k) + gamma*beta*np.sum(np.log(k))) / np.sum(x/beta+x);
        mu = alpha/beta; sigma = np.sqrt(alpha)/beta;       
    return(mu,sigma) 

def intvlMle_MuS(para,gamma,x):
    mu,sig2 = para;
    y = np.zeros(2);
    y[0] = np.sum(np.log(mu/(mu + sig2*x))) + np.sum(sig2/(mu**2 + gamma*mu + gamma*sig2*x));
    y[1] = np.sum((mu*x - 1)/(mu + sig2*x)) + np.sum(gamma/(mu**2 + gamma*mu + gamma*sig2*x));
    return(y)

'''
    calculate the expected distance for a path
'''

def expctDist(x,censor_flag):
    N = len(x);
    n = N - 1; y = x[:-1];
    gamma = 1
    if censor_flag: # censor_flag == 1, this is a censored data 
        KL_dist = -(N-n) - np.sum(np.log(n*y/np.sum(x))) ;
    else:# censor_flag == 0, this is a recovery 
        KL_dist = N*np.log(np.sum(x)) - N*np.log(N) - np.sum(np.log(x));    
    expDist = 1/KL_dist*np.exp(-np.sum(x)*gamma);
    return(expDist);
    
'''
    select the first m (=3) suscepted event: 
        short length, high rate and belong to different host
'''
def susEventSelct(m,eventBranch,BD_meanTime):
    temp_id = np.nonzero(eventBranch[:,2]<=np.percentile(eventBranch[:,2],10))[0];
    temp_host = eventBranch[temp_id,0].astype(int);
    desc_ord = (np.argsort(BD_meanTime[temp_host-1,1]))[::-1];
    susEvent = np.zeros(m).astype(int);
    tempeventBranch = eventBranch;
    tempBDtime = BD_meanTime;
#    for i in range(m):
        #select the least branch 
def transPair2EventInfor1006(TransPair,Host_Life,censoring_flag):
    # initialization
    SampleSize = TransPair.shape[0]
    eventBranch = np.zeros((SampleSize,3));eventBranch[:,:2] = TransPair[:,:2]
    NodeBranchTime = np.zeros((SampleSize+1,2)); # recode the Branching time for all nodes
    NodeBranchTime[:,0] = Host_Life[:,0]; #initialized with the ID 
    NodeBranchTime[:,1] = Host_Life[:,1]; #initialized with the birth times of all nodes
    
    Host_tailBranch = np.zeros((SampleSize+1,2));
    Host_tailBranch[:,0] = Host_Life[:,0]; #initialized with the ID
    
    eventChild = np.zeros((SampleSize,4)).astype(int); eventChild[:,1:4] = -99;
    eventLevel = np.zeros(SampleSize).astype(int)
    eventTail  = np.zeros((SampleSize,2))-99; 
    eventPathSet = [[]]*SampleSize;
    eventCensor = np.zeros((SampleSize,2)).astype(int)-99;
    eventAccInfo = np.zeros((SampleSize,2));
    
    
    for i in range(SampleSize):
        donnor,recipt, temp_BTime = (TransPair[i,:]) 
        temp_id = np.nonzero(NodeBranchTime[:,0] == np.array([donnor]))[0]
        eventBranch[i,2] = temp_BTime - NodeBranchTime[temp_id,1]
#        temp_length_d = [0,temp_length_d][bool(temp_length_d<998)];
        NodeBranchTime[temp_id,1] = temp_BTime;
        temp_count = 0;
        if i< SampleSize-1: # the first n-1 events
            temp_id = np.nonzero(TransPair[i+1:,0] == donnor)[0]
            if len(temp_id) :  # the donnor will give birth later
                eventChild[i,1] = np.min(temp_id) +i +1; temp_count += 1;
            else:  # the donnor will not give birth later
                eventTail[i,0] = Host_Life[int(donnor)-1,2] - temp_BTime;   
                eventCensor[i,0] = censoring_flag[int(donnor)-1];
            temp_id = np.nonzero(TransPair[i+1:,0] == recipt)[0]
            if len(temp_id) :  
                eventChild[i,2] = np.min(temp_id) +i +1; temp_count += 1;
            else:
                eventTail[i,1] = Host_Life[int(recipt)-1,2] - temp_BTime;
                eventCensor[i,1] = censoring_flag[int(recipt)-1];
            eventChild[i,0] = temp_count;
        else: # the last event has no child 
            eventTail[i,0] = Host_Life[int(donnor)-1,2] - temp_BTime; 
            eventTail[i,1] = Host_Life[int(recipt)-1,2] - temp_BTime;
            eventCensor[i,0] = censoring_flag[int(donnor)-1];
            eventCensor[i,1] = censoring_flag[int(recipt)-1];
    Host_tailBranch[:,1] = Host_Life[:,2] - NodeBranchTime[:,1];
    
    # search for the parents of the event and calculate the depth of all the events
    eventDepth = np.ones(SampleSize).astype(int);
    
    for i in range(SampleSize):
        row, col = np.where(eventChild[:,1:3] == i);
        if len(row):
            eventChild[i,3] = row;
            
        donnor,recipt, temp_BTime = (TransPair[i,:])
        if i > 0:
            temp_id = np.nonzero(TransPair[0:i,0] == donnor)[0];
            if len(temp_id) :  # the donnor gave birth before
                temp_id = np.max(temp_id);
                eventDepth[i] = eventDepth[temp_id] +1; 
            else:  # the donnor will not give birth later
                temp_id2 = np.max(np.nonzero(TransPair[0:i,1] == donnor)[0])
                eventDepth[i] = eventDepth[temp_id2] +1;
        
    
    # search for the event level. 
    # add codes for the eventAccInfo 1006
    for i in range(SampleSize):
        if not eventChild[-i-1,0]:  # no child, then this event has level 1
            eventLevel[-i-1] = 1;
#            eventAccInfo[-i-1,:] = 1, eventBranch[-i-1,2]; # no child, accInfo = branchlength
            eventAccInfo[-i-1,:] = 1, eventBranch[-i-1,2]+np.min(eventTail[-i-1,:]); # 1019shorter is better 
        elif eventChild[-i-1,0]== 1: # one child, the level is the child's level plus 1
            temp_id = np.max(eventChild[-i-1,1:3]);
            eventLevel[-i-1] = eventLevel[temp_id] + 1;
            eventAccInfo[-i-1,:] = 0, np.max(eventTail[-i-1,:]) 
#            eventAccInfo[-i-1,:] = 0, np.max(eventTail[-i-1,:]); # no child, accInfo = branchlength
        else:                       # two children, the level is the bigger one plus 1
            temp_id1, temp_id2 = eventChild[-i-1,1:3];
            eventLevel[-i-1] = np.maximum(eventLevel[temp_id1], eventLevel[temp_id2]) + 1;
    
    # search for the path set.
    for i in range(SampleSize):
        event_id = SampleSize - i -1;
        if eventLevel[-i-1] == 1:
            temp_pathset = [[event_id]];
            eventPathSet[event_id] = temp_pathset;
        elif eventChild[-i-1,0] == 1:  # only one child
            CLD1 =  np.max(eventChild[-i-1,1:3]);
            temp_pathset =  eventPathSet[CLD1];
            temp_pathset = arrBeginAdd(temp_pathset,[event_id])
            temp_pathset = temp_pathset + [[event_id]];
            eventPathSet[event_id] = temp_pathset;
        else:
            CLD1, CLD2 =  eventChild[-i-1,1:3];
            temp_pathset1 =  eventPathSet[CLD1];
            temp_pathset1 = arrBeginAdd(temp_pathset1,[event_id]);
            temp_pathset2 =  eventPathSet[CLD2];
            temp_pathset2 = arrBeginAdd(temp_pathset2,[event_id]);
            temp_pathset = temp_pathset1 + temp_pathset2;
            eventPathSet[event_id] = temp_pathset;
    
        
    
    return(eventBranch,eventTail,Host_tailBranch,eventChild,eventLevel\
           ,eventPathSet,eventDepth,eventCensor,eventAccInfo)    