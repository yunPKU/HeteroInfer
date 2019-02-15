#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
 0829 : add the function of transPair2BranchLen
         used to calculate the internal and external branches 
 0831 : add the function of treePertB_NumK0831 to do sequence mutation,
         that is, only the first n 
'''


import numpy as np
import math as mt
from scipy import special as spl

'''
generation all the subset of [0,1,...,n-1] 
'''
from itertools import combinations, chain
allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))

"""
This function transforms the transPair and Host_Life information to the BL and 
    offspring information for all the nodes
input:  transPair  = [Donor, Recipient, time_of_trans]
        Host_Life  = [Host, time_of_birth, time_of_death]
Output: Node_NumOS = [Host_ID, number_of_offspring]
        Node_BL    = [Host_ID, length_of_effective_infectious]
@author: zhang
"""


def tree2NodeInfor(transPair,Host_Life):
    single_host     = Host_Life[:,0]
    n = single_host.shape[0]
    Node_BL     = np.zeros((n,2));Node_BL[:,0] = single_host
    Node_NumOS  = Node_BL.copy()
    for i in range(n):
        temp_host       = single_host[i]
        # count the number of trans events in which temp_host is donor
        temp_pair       = np.nonzero(transPair[:,0] == temp_host) 
        Node_NumOS[i,1] = temp_pair[0].size
        if Node_NumOS[i,1] >= 1: # if temp_host infected some individuals
            temp_maxBTtime  = np.max(transPair[temp_pair,2]) # the last time of infection
            Node_BL[i,1]    = temp_maxBTtime - Host_Life[i,1]# the length of infectious
    return(Node_NumOS,Node_BL)

def tree2NodeInfor0731(transPair,Host_Life):
    single_host     = Host_Life[:,0]
    n = single_host.shape[0]
    Node_BL     = np.zeros((n,2));Node_BL[:,0] = single_host
    Node_NumOS  = Node_BL.copy()
    Node_BL[:,1]= Host_Life[:,2]- Host_Life[:,1]
    for i in range(n):
        temp_host       = single_host[i]
        # count the number of trans events in which temp_host is donor
        temp_pair       = np.nonzero(transPair[:,0] == temp_host) 
        Node_NumOS[i,1] = temp_pair[0].size
    return(Node_NumOS,Node_BL)

def tree2NodeInfor1019(transPair,Host_Life):
    single_host     = Host_Life[:,0]
    n = single_host.shape[0]
    Node_BL     = np.zeros((n,2));Node_BL[:,0] = single_host
    Node_NumOS  = Node_BL.copy()
    Node_BL[:,1]= Host_Life[:,2]- Host_Life[:,1]
    for i in range(n):
        temp_host       = single_host[i]
        # count the number of trans events in which temp_host is donor
        temp_pair       = np.nonzero(transPair[:,0] == temp_host) 
        Node_NumOS[i,1] = temp_pair[0].size
        if Node_NumOS[i,1] >= 1: # if temp_host infected some individuals
            temp_maxBTtime  = np.max(transPair[temp_pair,2]) # the last time of infection
            Node_BL[i,1]    = temp_maxBTtime - Host_Life[i,1]# the length of infectious
    return(Node_NumOS,Node_BL)

'''
    This function is to calculate the length of the internal and external branches 
    added at 2017-08-29
'''

def transPair2BranchLen(TransPair,Host_Life): # add this function on 0829
    n = Host_Life.shape[0];
    IntBrach,ExtBrach,tail_type = np.zeros(n),np.zeros(n),np.zeros(n);
    for i in range(n):
        don_id = np.nonzero(TransPair[:,0] == i+1)[0];
        rec_id = np.nonzero(TransPair[:,1] == i+1)[0];
        if len(don_id)>0:
            lastbrach = np.max(TransPair[don_id,2])
        else:
            lastbrach = TransPair[rec_id,2]
        IntBrach[i] = lastbrach - Host_Life[i,1];
        ExtBrach[i] = Host_Life[i,2] - lastbrach;
    return(IntBrach,ExtBrach,)   

def transPair2treebranch(TransPair,Host_Life):
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


'''
To perform a single perturbation given a tree structure 
            and the id of the lines to do perturbation,
input:  TransPair = [Donor, Recipient, time_of_trans]
        Host_Life = [Host, time_of_birth, time_of_death]
        Rat_Pert  = the ratio of trans event that will experience pertubation, in (0,1)
Output: TransPair and Host_Life for another tree which is coincide with the input tree
        in terms of Phylogeny

'''
def treePurtB_ID(transPair,Host_Life,PB_id):
    # Sort the transPair according to the time 
#    transPair = transPair[transPair[:,2].argsort(),]
    # Sort the Host_Life according to the Host ID for the usage of searchsorted
#    Host_Life = Host_Life[Host_Life[:,0].argsort(),]
    #PB_id   = np.random.randint(m)
#    PB_id   = 0
    PB_Donr,PB_Rcpt = transPair[PB_id,0:2].astype(int)
    # replace the RV_Donr in the previous events with RV_Rcpt
    transPair[np.nonzero(transPair[:PB_id,0] == PB_Donr),0] = PB_Rcpt;
    transPair[np.nonzero(transPair[:PB_id,1] == PB_Donr),1] = PB_Rcpt;
    # exchange the trans event
    transPair[PB_id,0:2] = np.array([PB_Rcpt,PB_Donr])
    
    PB_Donr_id = np.nonzero(Host_Life[:,0] == PB_Donr)
    PB_Rcpt_id = np.nonzero(Host_Life[:,0] == PB_Rcpt)
    Host_Life[(PB_Donr_id,PB_Rcpt_id),1] = Host_Life[(PB_Rcpt_id,PB_Donr_id),1]
    return(transPair,Host_Life)

''' computing the random tree pertubation 0804
    PertNum <0, all the subset shall be included
    PertNum > 0, only PertNum subsets (randomly choosen) shall be included
'''
def treePertB_Num(transPair,hostLife,PertNum):
    SampleSize      = hostLife.shape[0]
    if PertNum <0: # all the subset shall be considered
        allPertB        = allsubsets(SampleSize - 1)
        Node_NumOS,Node_BL = tree2NodeInfor0731(transPair,hostLife)
        PertTreeInfor   = np.zeros((2,SampleSize,len(allPertB)))
        PertTreeInfor[:,:,0] = np.array([Node_NumOS[:,1],Node_BL[:,1]])
        for pt_i in range(1,len(allPertB)):
            temp_transPair,temp_hostLife = transPair.copy(),hostLife.copy()# starting from the original tree
            # making the pertubated tree
            temp_pb = np.array(allPertB[pt_i]) 
            for j in range(len(temp_pb)):
                temp_transPair,temp_hostLife= treePurtB_ID(temp_transPair,temp_hostLife,temp_pb[j])
            temp_NumOS,temp_BL              = tree2NodeInfor0731(temp_transPair,temp_hostLife)
            PertTreeInfor[:,:,pt_i]         = np.array([temp_NumOS[:,1],temp_BL[:,1]])
    else: # only a proportion of the subset shall be considered
        PertTreeInfor   = np.zeros((2,SampleSize,PertNum))
        for pt_i in range(PertNum):
            temp_transPair,temp_hostLife = transPair.copy(),hostLife.copy()# starting from the original tree
            temp_pb =  np.random.choice(SampleSize-1,size = np.random.randint(SampleSize-1),replace=False)# making the pertubated tree
            temp_pb = np.unique(temp_pb) # delete the replicatioin
            if len(temp_pb) >0: # if the temp_pb is not empty
                for j in range(len(temp_pb)):
                    temp_transPair,temp_hostLife= treePurtB_ID(temp_transPair,temp_hostLife,temp_pb[j])
            temp_NumOS,temp_BL              = tree2NodeInfor0731(temp_transPair,temp_hostLife)
            PertTreeInfor[:,:,pt_i]         = np.array([temp_NumOS[:,1],temp_BL[:,1]])
    return(PertTreeInfor)
    
def treePertB_NumK(transPair,hostLife,PertNum,PertK):
    SampleSize      = hostLife.shape[0]    
    PertTreeInfor   = np.zeros((2,SampleSize,PertK))
    for pt_i in range(PertK):
        temp_transPair,temp_hostLife = transPair.copy(),hostLife.copy()# starting from the original tree
        temp_pb =  np.random.choice(SampleSize-1,size = PertNum.astype(int),replace=False)# making the pertubated tree
        temp_pb = np.sort(temp_pb) # delete the replicatioin
        if len(temp_pb) >0: # if the temp_pb is not empty
            for j in range(len(temp_pb)):
                temp_transPair,temp_hostLife= treePurtB_ID(temp_transPair,temp_hostLife,temp_pb[j])
        temp_NumOS,temp_BL              = tree2NodeInfor0731(temp_transPair,temp_hostLife)
        PertTreeInfor[:,:,pt_i]         = np.array([temp_NumOS[:,1],temp_BL[:,1]])
    return(PertTreeInfor,temp_transPair,temp_hostLife)
    
def treePertB_NumK0831(transPair,hostLife,PertNum,PertK):
    total_size = hostLife.shape[0];
    tailSize = len(np.nonzero(hostLife[:,2]-hostLife[:,1]<0.0001)[0]);
    tail_hostlife = hostLife[-tailSize:,:];
    hostLife = hostLife[:total_size-tailSize,:];
    tail_stid =   np.min(tail_hostlife[:,0]);
    tail_transPair = transPair[transPair[:,1]>=tail_stid,:];
    transPair = transPair[transPair[:,1]<tail_stid,:];
    SampleSize      = hostLife.shape[0]    
    PertTreeInfor   = np.zeros((2,total_size,PertK))
    for pt_i in range(PertK):
        temp_transPair,temp_hostLife = transPair.copy(),hostLife.copy()# starting from the original tree
        temp_pb =  np.random.choice(SampleSize-1,size = PertNum.astype(int),replace=False)# making the pertubated tree
        temp_pb = np.sort(temp_pb) # delete the replicatioin
        if len(temp_pb) >0: # if the temp_pb is not empty
            for j in range(len(temp_pb)):
                temp_transPair,temp_hostLife= treePurtB_ID(temp_transPair,temp_hostLife,temp_pb[j])
        temp_transPair = np.vstack((temp_transPair,tail_transPair));
        temp_hostLife = np.vstack((temp_hostLife,tail_hostlife));
        temp_NumOS,temp_BL              = tree2NodeInfor0731(temp_transPair,temp_hostLife)
        PertTreeInfor[:,:,pt_i]         = np.array([temp_NumOS[:,1],temp_BL[:,1]])
    return(PertTreeInfor,temp_transPair,temp_hostLife)