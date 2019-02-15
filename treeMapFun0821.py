#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as mt
from scipy import special as spl
import treePertbFun0831 as tPF
import scipy as sp
"""
Created on Wed Jul 26 15:50:52 2017
Evalueation of Gradient
input: Node_NumOS, Node_BL from the output of function tree2NodeInfor,
        without the Host id information (modified at 2017-07-28)
       the k_InFC and theta_InFC as the present value of unkown parameters
Output: DL_DK and DL_Dtheta, the gradient of the likelihood function L at the 
        present values
@author: zhang
"""
def GradtEval(Node_NumOS,Node_BL,k_InFC,theta_InFC):
    n   = Node_BL.shape[0]
    Com_Fun = np.ones(n)
    KC_Fun  = np.zeros(n)
    TC_Fun  = np.zeros(n)
    for node_i in range(n):
        ni,BLi  = Node_NumOS[node_i],Node_BL[node_i] # modified at 0728
#        if ni > 0: # 0821
        Com_Fun[node_i] = np.prod(np.arange(k_InFC,k_InFC + ni)*theta_InFC/(1 + theta_InFC*BLi))\
                            /((1 + theta_InFC*BLi)**k_InFC)
#            Com_Fun[node_i] = np.prod(np.arange(k_InFC,k_InFC + ni)*theta_InFC/(1 + theta_InFC*BLi))\
#                                /((1 + theta_InFC*BLi)**k_InFC)
        KC_Fun[node_i]  = spl.polygamma(0,ni + k_InFC) - spl.polygamma(0,k_InFC) \
                            -mt.log(1 + theta_InFC*BLi)
        TC_Fun[node_i]  = (ni - k_InFC*theta_InFC*BLi)/(theta_InFC+theta_InFC*theta_InFC*BLi)            
    Dens        = np.prod(Com_Fun)
    DL_DK       = np.sum(KC_Fun)
    DL_Dtheta   = np.sum(TC_Fun)
    return(DL_DK, DL_Dtheta,Dens)

'''
    using the log-likelihood (LogDens) instead of the likelihood (Dens)
'''
def GradtEval0831(Node_NumOS,Node_BL,k_InFC,theta_InFC):
    n   = Node_BL.shape[0]
    Com_Fun = np.ones(n)
    KC_Fun  = np.zeros(n)
    TC_Fun  = np.zeros(n)
    LogLK   = np.zeros(n)
    for node_i in range(n):
        ni,BLi  = Node_NumOS[node_i],Node_BL[node_i] # modified at 0728
        if BLi > 0: # 08311
#            Com_Fun[node_i] = np.prod(np.arange(k_InFC,k_InFC + ni)*theta_InFC/(1 + theta_InFC*BLi))\
#                                /((1 + theta_InFC*BLi)**k_InFC)
    #            Com_Fun[node_i] = np.prod(np.arange(k_InFC,k_InFC + ni)*theta_InFC/(1 + theta_InFC*BLi))\
    #                                /((1 + theta_InFC*BLi)**k_InFC)
            KC_Fun[node_i]  = spl.polygamma(0,ni + k_InFC) - spl.polygamma(0,k_InFC) \
                                -mt.log(1 + theta_InFC*BLi)
            TC_Fun[node_i]  = (ni - k_InFC*theta_InFC*BLi)/(theta_InFC+theta_InFC*theta_InFC*BLi)    
            LogLK[node_i]   = np.sum(np.log(np.arange(k_InFC,k_InFC + ni))) + ni*np.log(theta_InFC) \
                                - (ni+k_InFC)*np.log(1+theta_InFC*BLi)
    Dens        = np.prod(Com_Fun)
    DL_DK       = np.sum(KC_Fun)
    DL_Dtheta   = np.sum(TC_Fun)
    LogDens     = np.sum(LogLK)
    return(DL_DK, DL_Dtheta,LogDens)
'''This function calculate the KL distance of data to the exponential distribution
    Node_BL: the observaed data, to be test as the iid from Exp distribution
    m: the parameter need in KL distance calculation
'''
def TOE_KL(Node_BL,m):
#    Node_BL,m = np.random.uniform(size = 3),1;
    n = len(Node_BL);
    rawdata = np.sort(Node_BL);
    tail = np.ones(m)*rawdata[0];
    head = np.ones(m)*rawdata[-1];
    rawdata = np.append(tail,np.append(rawdata,head))
    Hraw = np.zeros(n)
    for i in range(n):
        Hraw[i] = rawdata[i+2*m] - rawdata[i];
    Hmn = np.sum(np.log(Hraw*n/(2*m)))/n
    KL = np.exp(Hmn - np.log(np.mean(Node_BL)) - 1);
    return(KL)

def GradEval_MuK(k_InFC,Node_NumOS,Node_BL,mu_InFC):
    n           = Node_BL.shape[0]
    theta_InFC  = mu_InFC/k_InFC
    LiK_Fun     = np.ones(n)
    KC_Fun      = np.zeros(n)
    for node_i in range(n):
        ni,BLi  = Node_NumOS[node_i],Node_BL[node_i] # modified at 0728
#        if ni > 0: # 0821
        KC_Fun[node_i]  = spl.polygamma(0,ni + k_InFC) - spl.polygamma(0,k_InFC)\
           + mt.log(k_InFC/(k_InFC + mu_InFC*BLi)) + (mu_InFC*BLi - ni)/(k_InFC + mu_InFC*BLi)
    Dl_DK       = np.sum(KC_Fun)
    return(Dl_DK)

# 0731 added some lines to increase the stablization of the code when k is huge and small
def GradEval_MuK0731(k_InFC,Node_NumOS,Node_BL,mu_InFC):
    n           = Node_BL.shape[0]
    theta_InFC  = mu_InFC/k_InFC
    KC_Fun      = np.zeros(n)
    for node_i in range(n):
        ni,BLi  = Node_NumOS[node_i],Node_BL[node_i] # modified at 0728
#        if ni > 0:
        KC_Fun[node_i]  = np.sum(1/np.arange(k_InFC,k_InFC + ni)) + mt.log(1 - mu_InFC*BLi/(k_InFC + mu_InFC*BLi)) + (mu_InFC*BLi - ni)/(k_InFC + mu_InFC*BLi)
    Dl_DK       = np.sum(KC_Fun)
    return(Dl_DK)

def fprime_MuK(k_InFC,Node_NumOS,Node_BL,mu_InFC):
    n           = Node_BL.shape[0]
    theta_InFC  = mu_InFC/k_InFC
    KC_Fun      = np.zeros(n)
    for node_i in range(n):
        ni,BLi  = Node_NumOS[node_i],Node_BL[node_i] # modified at 0728
#        if ni > 0:
        KC_Fun[node_i]  = mu_InFC*BLi/(k_InFC*(k_InFC + mu_InFC*BLi)) - np.sum(1/np.arange(k_InFC,k_InFC + ni)**2) + (ni - mu_InFC*BLi)/((k_InFC + mu_InFC*BLi)**2)
    Dl2DK2      = np.sum(KC_Fun)
    return(Dl2DK2)




''' computing the noisy gradient 0804
    PertNum <0, all the subset shall be included
    PertNum > 0, only PertNum subsets shall be included
'''

def noisyGradient(k_InFC,PertNum,transPair,hostLife,mu_InFC): 
    # input: k_InFC, the prime para of interest; PertNum, times of pertubatioin when evaluation
    #        transPair,hostLife,mu_InFC: other parameters for the computing of gradient
    # output: the value of gradient
    PertTreeInfor   = treePertb(PertNum,transPair,hostLife)
    re = 0
    n = PertTreeInfor.shape[2]
    for pt_i in range(n):        
        re += GradEval_MuK0731(k_InFC,PertTreeInfor[0,:,pt_i],PertTreeInfor[1,:,pt_i],mu_InFC)
    return(re)
'''
    to compute the density (or likelihood) of all trees
    input: k_InFC, mu_InFc and 
    PertTreeInfor-- 3 dimension matrix, the first 2 dimension contains the tree information
                    (the first row is Num, the second row is BL)
    output: treeDens -- the density for all trees, and treeRelDens-- the relative density
'''
def AllTreeDens(k_InFC,mu_InFC,PertTreeInfor):
    theta_InFC  = mu_InFC/k_InFC;
    treeNum = PertTreeInfor.shape[2];
    treeDens= np.zeros(treeNum);
    treeRelDens = np.zeros(treeNum);
    treeKL = np.zeros(treeNum);
    for t_id in range(treeNum):
        # using the log-likelihood instead of likelihood 0831
        treeDens[t_id]  = GradtEval0831(PertTreeInfor[0,:,t_id],PertTreeInfor[1,:,t_id],k_InFC,theta_InFC)[2];
#        treeDens[t_id]    = np.std(PertTreeInfor[1,:,t_id]);
#        treeKL[t_id]    = TOE_KL(PertTreeInfor[1,:,t_id],3);
        treeKL[t_id]    = 1
    treeRelDens = treeDens/np.sum(treeDens)
    treeRelKL   = treeKL/np.sum(treeKL)
    return(treeDens,treeKL, treeRelDens,treeRelKL)
'''
    This function return the measurement of the tree averaging 
    over all the same number of pertubation
'''
def PartSimu(Pert_ID,pSimu_k,PertNum,transPair,hostLife,k_InFC,mu_InFC):
    PertTreeInfor = tPF.treePertB_NumK(transPair,hostLife,PertNum[Pert_ID],pSimu_k)[0];
    treeDens,treeKL, treeRelDens,treeRelKL = AllTreeDens(k_InFC,mu_InFC,PertTreeInfor);
    return(np.sum(treeDens,axis = 0),np.sum(treeKL,axis = 0),\
           np.sum(treeRelDens,axis = 0),np.sum(treeRelKL,axis = 0),)
        
'''
    This function return the MLE of the k para
'''
def MLE_K(k_lowb,k_upb,treeInfor,mu_InFC):
    a = k_lowb;
    bmax = k_upb;
    Node_NumOS, Node_BL = treeInfor[0:2,:];
    b = a + 1
    fa = GradEval_MuK0731(a,Node_NumOS,Node_BL,mu_InFC)
    fb = GradEval_MuK0731(b,Node_NumOS,Node_BL,mu_InFC)
    while (fa*fb > 0) & (b< bmax):
        a,fa = b,fb # 0820
        b += 10
        fb = GradEval_MuK0731(b,Node_NumOS,Node_BL,mu_InFC)
    if b<= bmax or a == 0.001:   
        k_root = sp.optimize.bisect(GradEval_MuK0731,a,b,args=(Node_NumOS,Node_BL,mu_InFC))
    else: 
        k_root = bmax    
    Maxi_LKHo = GradtEval0831(Node_NumOS,Node_BL,k_root,mu_InFC/k_root)[2];
    return(k_root,Maxi_LKHo) 

'''
    Simplifying the root search
    adding the return of maximized likelihood
'''

def MLE_K0828(k_lowb,k_upb,treeInfor,mu_InFC):
    a = k_lowb;
    bmax = k_upb;
    Node_NumOS, Node_BL = treeInfor[0:2,:];
    fa = GradEval_MuK0731(a,Node_NumOS,Node_BL,mu_InFC)
    fb = GradEval_MuK0731(bmax,Node_NumOS,Node_BL,mu_InFC)
    if fa*fb < 0:
        k_root = sp.optimize.bisect(GradEval_MuK0731,a,bmax,args=(Node_NumOS,Node_BL,mu_InFC))
    else:
        k_root = bmax  
    Maxi_LKHo = GradtEval(Node_NumOS,Node_BL,k_root,mu_InFC/k_root)[2];
    return(k_root,Maxi_LKHo) 
          
          