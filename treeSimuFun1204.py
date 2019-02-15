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

def Simu_FTPop(mu_InFC,var_InFC,mu_Rec,Pop_N,SampSize):
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC;
    host_infe_var = -999
    while host_infe_var < 0:
        BD_meanTime         = np.zeros((SampSize,3))
        BD_meanTime[:,0]    = np.arange(SampSize)
        
        BD_meanTime[:,1]    = 1/np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
        BD_meanTime[:,2]    = 1/mu_Rec # the same recovery meanTime
        
        # Initialization of other parameters
        TotalHost,Active_Host = 1,1
        TransPair           = np.zeros((SampSize,3))
        Host_Life           = np.zeros((SampSize,3))
        InFC_State          = np.zeros((SampSize,2)); 
        Host_SusState       = np.zeros(Pop_N); # 0--sus, 1 -- Infected, -99 -- Rec
        
        # Initialization of the epidemic
        TransPair[TotalHost-1,:]    = np.array([999,1,0])
        Host_Life[TotalHost-1,:]    = np.array([1,0,
                 np.random.exponential(BD_meanTime[TotalHost-1,2])]);
        InFC_State[TotalHost-1,:]   = np.array([1,
                  np.random.exponential(BD_meanTime[TotalHost-1,1])]);
        Host_SusState[TotalHost-1]  = 1
        while TotalHost < SampSize and Active_Host>0:
            temp_Recv               = (np.nonzero((Host_Life[:TotalHost,2] \
                <InFC_State[:TotalHost,1])&(InFC_State[:TotalHost,1]>-999)))[0];
            if temp_Recv.size > 0: # Recovery
                temp_RecvId         = InFC_State[temp_Recv,0].astype(int);
                InFC_State[temp_Recv,1]  = -9999
                Active_Host         = len(np.nonzero(InFC_State[:TotalHost,1]>0)[0]) # 0830
                Host_SusState[temp_RecvId-1] = -99
            if Active_Host > 0: # Infection
                # 1)  determine the donnor
                # the following 3 lines are modified at 0619
                temp_acID = np.nonzero(InFC_State[:TotalHost,1]>0)[0]# all the active host
                InF_time  = np.amin(InFC_State[temp_acID,1]) # determine the time of Next Infection 
                temp_id   = temp_acID[np.argmin(InFC_State[temp_acID,1])]
                donnor_id = InFC_State[temp_id,0]; # determine the donnor of Next Infection 
                InFC_State[temp_id,1]    = InF_time +\
                        np.random.exponential(BD_meanTime[temp_id,1]); # update the donnor   
                
                # 2) determine the recipient ID
                recip_id = np.random.randint(Pop_N-1)+1;
                if recip_id>=donnor_id:
                    recip_id += 1
                if Host_SusState[recip_id-1] == 0: # the recipient is susceptible, it is a succesful transmission
                    TotalHost += 1; Active_Host += 1;        
                    TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,InF_time]);
                    Host_Life[TotalHost-1,:] = np.array([recip_id,InF_time,InF_time +\
                             np.random.exponential(BD_meanTime[TotalHost-1,2])]);
                    InFC_State[TotalHost-1,:]= np.array([recip_id, InF_time + \
                              np.random.exponential(BD_meanTime[TotalHost-1,1])]);# update the recipient
                    Host_SusState[recip_id-1] = 1;
        
        # input the transPair of node 1~10 and the corresponding life time for them
        Host_Life_trim = Host_Life.copy()
        Host_Life_trim[:,2] = np.fmin(Host_Life[-1,1]+0.2,Host_Life[:,2])
        if TotalHost >= SampSize:
            temp_time = np.max(TransPair[:,2]); # 0901 stops at the last branch time
            Host_Life[:,2] = np.minimum(Host_Life[:,2],temp_time);# 0901 stops at the last branch time
                        
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree = treeBranch2treeView(TreeBranch,Host_tailBranch)
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair,Host_Life,host_infe_var,simutree)



    
    
    
def Simu_InftPop(mu_InFC,var_InFC,mu_Rec,SampSize):
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    host_infe_var = -999
    while host_infe_var < 0:
        BD_meanTime         = np.zeros((SampSize,3))
        BD_meanTime[:,0]    = np.arange(SampSize)
        
        BD_meanTime[:,1]    = 1/np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
        BD_meanTime[:,2]    = 1/mu_Rec # the same recovery meanTime
        
        # Initialization of other parameters
        TotalHost,Active_Host = 1,1
        TransPair           = np.zeros((SampSize,3))
        Host_Life           = np.zeros((SampSize,3))
        InFC_State          = np.zeros((SampSize,2))
        
        # Initialization of the epidemic
        TransPair[TotalHost-1,:]    = np.array([999,1,0])
        Host_Life[TotalHost-1,:]    = np.array([1,0,
                 np.random.exponential(BD_meanTime[TotalHost-1,2])]);
        InFC_State[TotalHost-1,:]   = np.array([1,
                  np.random.exponential(BD_meanTime[TotalHost-1,1])]);
        
        
        while TotalHost < SampSize and Active_Host>0:
            temp_Recv               = (np.nonzero((Host_Life[:TotalHost,2] \
                <InFC_State[:TotalHost,1])&(InFC_State[:TotalHost,1]>0)))[0];
            if temp_Recv.size > 0: # Recovery
                temp_RecvId         = InFC_State[temp_Recv,0].astype(int);
                InFC_State[temp_Recv,1]  = -99999
                Active_Host         = len(np.nonzero(InFC_State[:TotalHost,1]>0)[0]) # 0830 # modified at 0619
            if Active_Host > 0: # Infection
                # the following 3 lines are modified at 0619
                temp_acID = np.nonzero(InFC_State[:TotalHost,1]>0)[0]
                InF_time  = np.amin(InFC_State[temp_acID,1])
                temp_id   = temp_acID[np.argmin(InFC_State[temp_acID,1])]
        
                TotalHost += 1; Active_Host += 1;
                recip_id          = TotalHost;
                donnor_id           = InFC_State[temp_id,0];
                TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,InF_time]);
                Host_Life[TotalHost-1,:] = np.array([recip_id,InF_time,InF_time +\
                         np.random.exponential(BD_meanTime[TotalHost-1,2])]);
                InFC_State[TotalHost-1,:]= np.array([recip_id, InF_time + \
                          np.random.exponential(BD_meanTime[TotalHost-1,1])]);# update the recipient
                InFC_State[temp_id,1]    = InF_time +\
                        np.random.exponential(BD_meanTime[temp_id,1]); # update the donnor   
       
        # input the transPair of node 1~10 and the corresponding life time for them
        if TotalHost >= SampSize: 
            temp_time = np.max(TransPair[:,2]); # 0901 stops at the last branch time
            censoring_flag = Host_Life[:,2]>=temp_time;
            Host_Life[:,2] = np.minimum(Host_Life[:,2],temp_time);# 0901 stops at the last branch time
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,BD_meanTime)

def Simu_InftPop_SID(mu_InFC,var_InFC,mu_Rec,recSize):
    SampSize = recSize*10
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    host_infe_var = -999
    while host_infe_var < 0:
        BD_meanTime         = np.zeros((SampSize,3))
        BD_meanTime[:,0]    = np.arange(SampSize)
        
        BD_meanTime[:,1]    = 1/np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
        BD_meanTime[:,2]    = 1/mu_Rec # the same recovery meanTime
        
        # Initialization of other parameters
        TotalHost,Active_Host,recHost = 1,1,0
        TransPair           = np.zeros((SampSize,3))
        Host_Life           = np.zeros((SampSize,3))
        InFC_State          = np.zeros((SampSize,2))
        
        # Initialization of the epidemic
        TransPair[TotalHost-1,:]    = np.array([999,1,0])
        Host_Life[TotalHost-1,:]    = np.array([1,0,
                 np.random.exponential(BD_meanTime[TotalHost-1,2])]);
        InFC_State[TotalHost-1,:]   = np.array([1,
                  np.random.exponential(BD_meanTime[TotalHost-1,1])]);
        
        
        while recHost < recSize and Active_Host>0:
            temp_Recv               = (np.nonzero((Host_Life[:TotalHost,2] \
                <InFC_State[:TotalHost,1])&(InFC_State[:TotalHost,1]>0)))[0];
            if temp_Recv.size > 0: # Recovery
                temp_RecvId         = InFC_State[temp_Recv,0].astype(int);
                InFC_State[temp_Recv,1]  = -99999
                Active_Host         = len(np.nonzero(InFC_State[:TotalHost,1]>0)[0]) # 0830 # modified at 0619
            if Active_Host > 0: # Infection
                # the following 3 lines are modified at 0619
                temp_acID = np.nonzero(InFC_State[:TotalHost,1]>0)[0]
                InF_time  = np.amin(InFC_State[temp_acID,1])
                temp_id   = temp_acID[np.argmin(InFC_State[temp_acID,1])]
        
                TotalHost += 1; Active_Host += 1;
                recip_id          = TotalHost;
                donnor_id           = InFC_State[temp_id,0];
                TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,InF_time]);
                Host_Life[TotalHost-1,:] = np.array([recip_id,InF_time,InF_time +\
                         np.random.exponential(BD_meanTime[TotalHost-1,2])]);
                InFC_State[TotalHost-1,:]= np.array([recip_id, InF_time + \
                          np.random.exponential(BD_meanTime[TotalHost-1,1])]);# update the recipient
                InFC_State[temp_id,1]    = InF_time +\
                        np.random.exponential(BD_meanTime[temp_id,1]); # update the donnor   
            recHost = TotalHost - Active_Host;
        # input the transPair of node 1~10 and the corresponding life time for them
        if recHost >= recSize: 
            temp_time = np.max(TransPair[:,2]); # 0901 stops at the last branch time
            censoring_flag = Host_Life[:,2]>=temp_time;
            Host_Life[:,2] = np.minimum(Host_Life[:,2],temp_time);# 0901 stops at the last branch time
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,BD_meanTime)

def Simu_InftPop_Const(mu_InFC,var_InFC,mu_Rec,SampSize):
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    host_infe_var = -999
    while host_infe_var < 0:
        BD_meanTime         = np.zeros((SampSize,3))
        BD_meanTime[:,0]    = np.arange(SampSize)
        BD_meanTime[:,1]    = 1/mu_InFC
        BD_meanTime[:,2]    = 1/mu_Rec
#        BD_meanTime[:,1]    = 1/np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
#        BD_meanTime[:,2]    = 1/np.random.exponential(mu_Rec) # the same recovery meanTime
        
        # Initialization of other parameters
        TotalHost,Active_Host = 1,1
        TransPair           = np.zeros((SampSize,3))
        Host_Life           = np.zeros((SampSize,3))
        InFC_State          = np.zeros((SampSize,2))
        
        # Initialization of the epidemic
        TransPair[TotalHost-1,:]    = np.array([999,1,0])
        Host_Life[TotalHost-1,:]    = np.array([1,0,
                 np.random.exponential(BD_meanTime[TotalHost-1,2])]);
        InFC_State[TotalHost-1,:]   = np.array([1,
                  np.random.exponential(BD_meanTime[TotalHost-1,1])]);
        
        
        while TotalHost < SampSize and Active_Host>0:
            temp_Recv               = (np.nonzero((Host_Life[:TotalHost,2] \
                <InFC_State[:TotalHost,1])&(InFC_State[:TotalHost,1]>0)))[0];
            if temp_Recv.size > 0: # Recovery
                temp_RecvId         = InFC_State[temp_Recv,0].astype(int);
                InFC_State[temp_Recv,1]  = -99999
                Active_Host         = len(np.nonzero(InFC_State[:TotalHost,1]>0)[0]) # 0830
            if Active_Host > 0: # Infection
                # the following 3 lines are modified at 0619
                temp_acID = np.nonzero(InFC_State[:TotalHost,1]>0)[0]
                InF_time  = np.amin(InFC_State[temp_acID,1])
                temp_id   = temp_acID[np.argmin(InFC_State[temp_acID,1])]
        
                TotalHost += 1; Active_Host += 1;
                recip_id          = TotalHost;
                donnor_id           = InFC_State[temp_id,0];
                TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,InF_time]);
                Host_Life[TotalHost-1,:] = np.array([recip_id,InF_time,InF_time +\
                         np.random.exponential(BD_meanTime[TotalHost-1,2])]);
                InFC_State[TotalHost-1,:]= np.array([recip_id, InF_time + \
                          np.random.exponential(BD_meanTime[TotalHost-1,1])]);# update the recipient
                InFC_State[temp_id,1]    = InF_time +\
                        np.random.exponential(BD_meanTime[temp_id,1]); # update the donnor   
       
        # input the transPair of node 1~10 and the corresponding life time for them
        Host_Life_trim = Host_Life.copy()
        Host_Life_trim[:,2] = np.fmin(Host_Life[-1,1]+0.2,Host_Life[:,2])
        if TotalHost >= SampSize: 
            temp_time = np.max(TransPair[:,2]); # 0901 stops at the last branch time
            Host_Life[:,2] = np.minimum(Host_Life[:,2],temp_time);# 0901 stops at the last branch time
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life_trim)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree)

'''
    Infinite Pop and complete sampling for the first n sequences
    added at 2017-08-30
'''
def Simu_InftPop_CPL(mu_InFC,var_InFC,mu_Rec,SampSize):
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    host_infe_var = -999
    while host_infe_var < 0:
        BD_meanTime         = np.zeros((SampSize,3))
        BD_meanTime[:,0]    = np.arange(SampSize)
        
        BD_meanTime[:,1]    = 1/np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
        BD_meanTime[:,2]    = 1/mu_Rec # the same recovery meanTime
        
        # Initialization of other parameters
        TotalHost,Active_Host = 1,1
        TransPair           = np.zeros((SampSize,3))
        Host_Life           = np.zeros((SampSize,3))
        InFC_State          = np.zeros((SampSize,2))
        
        # Initialization of the epidemic
        TransPair[TotalHost-1,:]    = np.array([999,1,0])
        Host_Life[TotalHost-1,:]    = np.array([1,0,
                 np.random.exponential(BD_meanTime[TotalHost-1,2])]);
        InFC_State[TotalHost-1,:]   = np.array([1,
                  np.random.exponential(BD_meanTime[TotalHost-1,1])]);
        
        
        while TotalHost < SampSize and Active_Host>0:
            temp_Recv               = (np.nonzero((Host_Life[:TotalHost,2] \
                <InFC_State[:TotalHost,1])&(InFC_State[:TotalHost,1]>0)))[0];
            if temp_Recv.size > 0: # Recovery
                temp_RecvId         = InFC_State[temp_Recv,0].astype(int);
                InFC_State[temp_Recv,1]  = -99999
                Active_Host         = len(np.nonzero(InFC_State[:TotalHost,1]>0)[0]) # 0830
#                Active_Host         -= temp_Recv.size # modified at 0619
            if Active_Host > 0: # Infection
                # the following 3 lines are modified at 0619
                temp_acID = np.nonzero(InFC_State[:TotalHost,1]>0)[0]
                InF_time  = np.amin(InFC_State[temp_acID,1])
                temp_id   = temp_acID[np.argmin(InFC_State[temp_acID,1])]
        
                TotalHost += 1; Active_Host += 1;
                recip_id          = TotalHost;
                donnor_id           = InFC_State[temp_id,0];
                TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,InF_time]);
                Host_Life[TotalHost-1,:] = np.array([recip_id,InF_time,InF_time +\
                         np.random.exponential(BD_meanTime[TotalHost-1,2])]);
                InFC_State[TotalHost-1,:]= np.array([recip_id, InF_time + \
                          np.random.exponential(BD_meanTime[TotalHost-1,1])]);# update the recipient
                InFC_State[temp_id,1]    = InF_time +\
                        np.random.exponential(BD_meanTime[temp_id,1]); # update the donnor   
        # add at 0830 to allow for the endemic outbreak
        if Active_Host>0: # there are still active hosts when samplesize has been exceeded
            # find out these active hosts
            temp_acID = np.nonzero(InFC_State[:TotalHost,1]>0)[0];
            tail_count = 0;
            tail_size = 10*np.ceil(np.sum((Host_Life[temp_acID,2]-InF_time)/BD_meanTime[temp_acID,1]));
            tail_size = tail_size.astype(int)
            tail_TransPair = np.zeros((tail_size,3))
            tail_HostLife = np.zeros((tail_size,3))
            # generate the offspring for them
            for i in range(len(temp_acID)):
                temp_id = temp_acID[i];
                donnor_id = InFC_State[temp_id,0].astype(int);
                InF_time  = InFC_State[temp_id,1];
                while InF_time<Host_Life[donnor_id-1,2] and tail_count<tail_size:
                    tail_count += 1;
                    recip_id          = SampSize + tail_count;
                    tail_TransPair[tail_count-1,:] = np.array([donnor_id,recip_id,InF_time]);                    
                    tail_HostLife[tail_count-1,:] = np.array([recip_id,InF_time,InF_time]);
                    InF_time += np.random.exponential(BD_meanTime[temp_id,1]);
                InFC_State[temp_id,1] = -99999
            Host_Life = np.vstack((Host_Life,tail_HostLife[:tail_count,:]))
            TransPair = np.vstack((TransPair,tail_TransPair[:tail_count,:]))
#             Sort the transPair according to the time 
            TransPair = TransPair[TransPair[:,2].argsort(),:]
#     Sort the Host_Life according to the Host ID for the usage of searchsorted
            Host_Life = Host_Life[Host_Life[:,0].argsort(),]
        # input the transPair of node 1~10 and the corresponding life time for them
        Host_Life_trim = Host_Life.copy()
        Host_Life_trim[:,2] = np.fmin(Host_Life[-1,1]+0.2,Host_Life[:,2])
        if TotalHost >= SampSize: 
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life_trim)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree)

def Simu_InftPop_Gille(mu_InFC,var_InFC,mu_Rec,SampSize):
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
        
        curt = 0;
        while TotalHost < SampSize:
            Active_id = np.nonzero(InFC_State[:TotalHost,1]>0)[0]
            Active_Host         = len(Active_id)
            temp_rate = np.sum(InFC_State[Active_id,1]) + Active_Host*mu_Rec;
            if temp_rate<=0: break; # if no active host,break
            curt += np.random.exponential(scale = 1/temp_rate); 
            # choosing the host for the event
            sus_pro = (InFC_State[Active_id,1] + mu_Rec)/temp_rate;
            temp_id = Active_id[np.nonzero(np.random.multinomial(1,sus_pro))];
            # infection or recovery?
            if np.random.uniform()< mu_Rec/(InFC_State[temp_id,1] + mu_Rec): # recovery
                InFC_State[temp_id,1] = -9999;
                Host_Life[temp_id,2] = curt; 
            else: # infection
                TotalHost += 1; Active_Host += 1;
                recip_id          = TotalHost;
                donnor_id         = InFC_State[temp_id,0];
                TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,curt]);
                Host_Life[TotalHost-1,:] = np.array([recip_id,curt,0]);
                InFC_State[TotalHost-1,:]= np.array([recip_id,BD_rate[TotalHost-1,1]]);# update the recipient
        # input the transPair of node 1~10 and the corresponding life time for them
        if TotalHost >= SampSize: 
            censoring_flag = Host_Life[:,2]<= 0;
            Host_Life[censoring_flag,2] = curt;
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,BD_rate)

def Simu_InftPop_Gille_Const(mu_InFC,var_InFC,mu_Rec,SampSize):
    k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    host_infe_var = -999
    while host_infe_var < 0:
        BD_rate         = np.zeros((SampSize,3))
        BD_rate[:,0]    = np.arange(SampSize)
        
        BD_rate[:,1]    = mu_InFC # random InFC meanTime
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
        
        curt = 0;
        while TotalHost < SampSize:
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
            else: # infection
                TotalHost += 1; Active_Host += 1;
                recip_id          = TotalHost;
                donnor_id         = InFC_State[temp_id,0];
                TransPair[TotalHost-1,:] = np.array([donnor_id,recip_id,curt]);
                Host_Life[TotalHost-1,:] = np.array([recip_id,curt,0]);
                InFC_State[TotalHost-1,:]= np.array([recip_id,BD_rate[TotalHost-1,1]]);# update the recipient
        # input the transPair of node 1~10 and the corresponding life time for them
        if TotalHost >= SampSize: 
            censoring_flag = (Host_Life[:,2]<= 0);
            Host_Life[censoring_flag,2] = curt;
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,BD_rate)

def Simu_InftPop_GilleSID(mu_InFC,var_InFC,mu_Rec,recSize):
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
             
            censoring_flag = Host_Life[:,2]<= 0;
            Host_Life[censoring_flag,2] = curt;
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,BD_rate)
    
def Simu_InftPop_GilleSID0515(mu_InFC,var_InFC,mu_Rec,recSize):
    SampSize = 2*recSize
    if var_InFC>0:
        k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    else:
        k_InFC, theta_InFC = 0,0
    host_infe_var = -999
    while host_infe_var < 0:
        BD_rate         = np.zeros((SampSize,3))
        BD_rate[:,0]    = np.arange(SampSize)
        if var_InFC>0:
            BD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
        else:
            BD_rate[:,1] = mu_InFC;    
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
                    if var_InFC>0:
                        newBD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
                    else:
                        newBD_rate[:,1] = mu_InFC; 
                    newBD_rate[:,2]    = mu_Rec # 
                    BD_rate = np.vstack((BD_rate,newBD_rate));
        # input the transPair of node 1~10 and the corresponding life time for them
        if recHost >= recSize: 
            # trimming the matrix
            Host_Life,TransPair = Host_Life[:TotalHost,:],TransPair[:TotalHost,:]
             
            censoring_flag = Host_Life[:,2]<= 0;
            Host_Life[censoring_flag,2] = curt;
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,BD_rate)


def Simu_InftPop_GilleSID_FntN(mu_InFC,var_InFC,mu_Rec,recSize,fntN):
    SampSize = 2*recSize
    if var_InFC>0:
        k_InFC, theta_InFC  = mu_InFC**2/var_InFC, var_InFC/mu_InFC
    else:
        k_InFC, theta_InFC = 0,0
    host_infe_var = -999
    while host_infe_var < 0:
        BD_rate         = np.zeros((SampSize,3))
        BD_rate[:,0]    = np.arange(SampSize)
        if var_InFC>0:
            BD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
        else:
            BD_rate[:,1] = mu_InFC;    
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
                # make a selection first
                if fntN>0: 
                    tempid = np.random.randint(fntN);
                else:
                    tempid = 999999    
                if tempid>TotalHost:  # if the selected is a sus individual                   
                    # if select a susceptible ind
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
                        if var_InFC>0:
                            newBD_rate[:,1]    = np.random.gamma(k_InFC, theta_InFC,SampSize) # random InFC meanTime
                        else:
                            newBD_rate[:,1] = mu_InFC; 
                        newBD_rate[:,2]    = mu_Rec # 
                        BD_rate = np.vstack((BD_rate,newBD_rate));
        # input the transPair of node 1~10 and the corresponding life time for them
        if recHost >= recSize: 
            # trimming the matrix
            Host_Life,TransPair = Host_Life[:TotalHost,:],TransPair[:TotalHost,:]
             
            censoring_flag = Host_Life[:,2]<= 0;
            Host_Life[censoring_flag,2] = curt;
            
            TreeBranch,Host_tailBranch = transPair2treebranch(TransPair[1:,:],Host_Life)
            simutree        = treeBranch2treeView(TreeBranch,Host_tailBranch)            
            single_host,host_infe_count    = np.unique(TransPair[:,0],return_counts=True)
            host_infe_var   = np.var(host_infe_count)
    return(TransPair[1:,:],Host_Life,host_infe_var,simutree,censoring_flag,BD_rate)