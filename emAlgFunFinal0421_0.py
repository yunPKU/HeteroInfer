#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:31:50 2017
1007 maximization the likelihood function, not search for the root of the gradient
@author: zhang
"""
'''
    newEvent: the list of event that to be calculated,
    eventAccInfo = [num of birth event, length of path], the accumulated infor for all nodes
            for those events with only one child, the default is the tail information;
    paraInfo = [para_mu,para_k, mu_Rec], the first two are for the prior distribution, 
            the third for the recovery rate.
    output:
        DlDa,DlDb, and prbZmis. All are of length nZmis 
        representing the gradient of likelihood over alpha and beta 
        and the probability of the missing Zmis Prob{Zmiss | X, parainfo}
        
'''
'''
1202 -- add the estimation of muRec in the function of EMoptm
'''
'''
0401 -- add the muRec to the MAP and Marginal Probability
'''
import numpy as np
import scipy as sp
from scipy.stats import gamma, expon
'''
    This two function for the first level of optimazation: with data lv = 1
'''



'''
    This two functions are for the calculation of probability of Zmis and determining the value
    of Zmis (return of prbZmis)
'''
def prbFun(paraInfo,mu_Rec,li,lmbd,ni,bli):
    para_mu, para_sigma = paraInfo;
    if para_sigma>0:
        para_k = para_mu**2/para_sigma**2;
    else:
        para_k = 1e5;
    
    total_rate = mu_Rec + lmbd
    prb_max = expon.pdf(li, scale = 1/total_rate)*gamma.pdf(lmbd, para_k,scale = para_mu/para_k);
    prb_avg = np.sum(np.log(np.arange(para_k,para_k + ni))) + ni*np.log(para_mu) + para_k*np.log(para_k)\
            - (ni + para_k)*np.log(para_k+para_mu*bli) - mu_Rec*bli;
    prb_L2  = abs(np.log(li*lmbd));
    return(prb_max,np.exp(prb_avg),prb_L2)
#
def prbFun1016(paraInfo,mu_Rec,li,lmbd,ni,bli):
    para_mu, para_sigma = paraInfo;
    if para_sigma>0:
        para_k = para_mu**2/para_sigma**2;
        rate_est= ni/bli;  rate1 = 1/li;
        prb_map = rate_est**ni*np.exp(-rate_est*bli)*gamma.pdf(rate_est, para_k,scale = para_mu/para_k)*np.exp(- mu_Rec*bli);
        prb_avg = np.sum(np.log(np.arange(para_k,para_k + ni))) + ni*np.log(para_mu) + para_k*np.log(para_k)\
                - (ni + para_k)*np.log(para_k+para_mu*bli) - mu_Rec*bli;
        prb_avg = np.exp(prb_avg)
        prb_exp  = rate1**(ni-1)*np.exp(-bli*rate1);
    else:
        prb_avg = para_mu**ni*np.exp(-(mu_Rec+para_mu)*bli);
        prb_map = prb_avg
        
    return(prb_map,prb_avg)
#def prbFun1016(paraInfo,mu_Rec,li,lmbd,ni,bli):
#    para_mu, para_sigma = paraInfo;
#    if para_sigma>0:
#        para_k = para_mu**2/para_sigma**2;
#    else:
#        para_k = 1e5;
##    para_mu, para_k = paraInfo;
#    rate_est= ni/bli;  rate1 = 1/li;
#    prb_map = rate_est**ni*np.exp(-rate_est*bli)*gamma.pdf(rate_est, para_k,scale = para_mu/para_k)*np.exp(- mu_Rec*bli);
#    prb_avg = np.sum(np.log(np.arange(para_k,para_k + ni))) + ni*np.log(para_mu) + para_k*np.log(para_k)\
#            - (ni + para_k)*np.log(para_k+para_mu*bli) - mu_Rec*bli;
#    prb_exp  = rate1**(ni-1)*np.exp(-bli*rate1);
#    return(prb_map,np.exp(prb_avg))

def prbZmis(paraInfo,mu_Rec,newEvent,eventAccInfo,eventBranch,eventChild,eventTail):
    n = len(newEvent);  kb = 2;#nZmis = 2**n;
    # compute the gradient for each Zmis
    prb_evt = np.zeros((n,2*kb));#prbZmis = np.zeros(nZmis);
    sec = np.ones(n);
    for j in range(n): # for each node
        tempEvt = newEvent[j];
        selCHD1,selCHD2 = eventChild[tempEvt,1:3];
        # adding info to the first child
        if selCHD1<0: 
            selCHD1 = tempEvt; lmbd =  1#1/eventAccInfo[selCHD1,1];# the selCHD1 is a tail, replace it with the event infor          
            n1,bl1 = 1+eventAccInfo[selCHD1,0],eventBranch[tempEvt,2]+eventTail[tempEvt,0];
        else:
            lmbd = eventAccInfo[selCHD1,0]/eventAccInfo[selCHD1,1];
            n1,bl1 = 1+eventAccInfo[selCHD1,0],eventBranch[tempEvt,2]+eventAccInfo[selCHD1,1];
        li     = eventBranch[tempEvt,2]
        
            
        prb_evt[j,0:kb]    = np.array(prbFun1016(paraInfo,mu_Rec,li,lmbd,n1,bl1));
        
        if selCHD2<0: 
            selCHD2 = tempEvt; lmbd =  1#lmbd =  1/eventAccInfo[selCHD2,1];# the selCHD1 is a tail, replace it with the event infor          
            n2,bl2 = 1+eventAccInfo[selCHD2,0],eventBranch[tempEvt,2]+eventTail[tempEvt,1];
        else:
            lmbd =  eventAccInfo[selCHD2,0]/eventAccInfo[selCHD2,1];
            n2,bl2 = 1+eventAccInfo[selCHD2,0],eventBranch[tempEvt,2]+eventAccInfo[selCHD2,1];
        
        prb_evt[j,kb:2*kb] = np.array(prbFun1016(paraInfo,mu_Rec,li,lmbd,n2,bl2));
        temp_cst = prb_evt[j,0:kb] + prb_evt[j,kb:2*kb];
        if np.any(temp_cst==0):
            prb_evt[j,0:kb] = 0.5   
            prb_evt[j,kb:2*kb] = 0.5     
        else:
            prb_evt[j,0:kb] = prb_evt[j,0:kb]/temp_cst;   
            prb_evt[j,kb:2*kb] = prb_evt[j,kb:2*kb]/temp_cst;            
        
        temp_cst = np.argmax(prb_evt[j,:]);
        if np.max(prb_evt[j,:])>0.9: # follow the dominant one
            sec[j] = temp_cst//3
        else:
            sec[j] = 1 - int(prb_evt[j,1]>0.5);
    return(sec)

'''
    the following functions are for the optimization of score function 
    based on the initial value and the given Zmis
    ObjFun,GrdFun and HesFun to do the calculation for each ni, bli (each host) 
    SInfoFun: Based on the sltZmis to combine the information from newEvent to old information
            eventAccInfo
    SinObj,SinJac and SinHes are used for the optimization
    EMoptm: the optimization loop based on SinObj,SinJac and SinHes
'''
'''
    This function merges the newEvent information to the present Event and path, generating
    prsEvt,prsAccInfo,prsPath
'''    
def mergeInfo(newEvent,eventAccInfo,eventBranch,eventChild,sltZmis,pstEvent,pstPath):
    # output: prsEvt,prsAccInfo,prsPath
    n = len(newEvent);
    prsEvt = pstEvent.copy();prsAccInfo = eventAccInfo.copy(); prsPath = pstPath;
    for j in range(n): # for each new event
        tempEvt = newEvent[j];
        selCHD  = eventChild[tempEvt,sltZmis[j]+1];
        elsCHD = eventChild[tempEvt,2 - sltZmis[j]];        
        if selCHD <0: # tempEvt goes to extinct at this time
            selCHD  = tempEvt;  # the selCHD1 is a tail, replace it with the event infor 
            prsAccInfo[tempEvt,:] = 1+eventAccInfo[selCHD,0],\
                eventBranch[tempEvt,2]+eventAccInfo[selCHD,1];
            prsEvt = np.append(prsEvt,tempEvt);
            prsPath += [[tempEvt]];

        elif elsCHD<0: # tempEvt connects with its child
            prsAccInfo[tempEvt,:] = 1+eventAccInfo[selCHD,0],\
                eventBranch[tempEvt,2]+eventAccInfo[selCHD,1]; 
            prsAccInfo[selCHD,:]  = eventAccInfo[tempEvt,:];
            prsEvt = np.append(prsEvt,tempEvt);
            prsPath = pathExtd(prsPath,selCHD,[tempEvt]);
            prsPath += [[-99,tempEvt]];
        else:
            prsAccInfo[tempEvt,:] = 1+eventAccInfo[selCHD,0],\
                eventBranch[tempEvt,2]+eventAccInfo[selCHD,1]; # added the branch of tempEvt to its child
            np.place(prsEvt, prsEvt==selCHD, tempEvt); # replace the child id in prsEvt with its parent
            prsPath = pathExtd(prsPath,selCHD,[tempEvt]);
    return(prsEvt,prsAccInfo,prsPath)

def mergeInfo1218(newEvent,eventAccInfo,eventBranch,eventChild,sltZmis,pstEvent,pstPath,pstCensor,eventCensor):
    # output: prsEvt,prsAccInfo,prsPath
    # need to be
    n = len(newEvent);
    prsEvt = pstEvent.copy();prsAccInfo = eventAccInfo.copy(); prsPath = pstPath;
    prsCensor = pstCensor.copy();
    for j in range(n): # for each new event
        tempEvt = newEvent[j];
        selCHD  = eventChild[tempEvt,sltZmis[j]+1];
        elsCHD = eventChild[tempEvt,2 - sltZmis[j]];        
        if selCHD <0: # tempEvt goes to extinct at this time
            selCHD  = tempEvt;  # the selCHD1 is a tail, replace it with the event infor 
            prsAccInfo[tempEvt,:] = 1+eventAccInfo[selCHD,0],\
                eventBranch[tempEvt,2]+eventAccInfo[selCHD,1];
            prsEvt = np.append(prsEvt,tempEvt);
            # choose the larger element in eventCensor[tempEvt,:]
            prsCensor = np.append(prsCensor,np.max(eventCensor[tempEvt,:]));             
            prsPath += [[tempEvt]];

        elif elsCHD<0: # tempEvt connects with its child
            prsAccInfo[tempEvt,:] = 1+eventAccInfo[selCHD,0],\
                eventBranch[tempEvt,2]+eventAccInfo[selCHD,1]; 
            prsAccInfo[selCHD,:]  = eventAccInfo[tempEvt,:];
            prsEvt = np.append(prsEvt,tempEvt);
            # choose the larger element in eventCensor[tempEvt,:]
            prsCensor = np.append(prsCensor,np.max(eventCensor[tempEvt,:]));
            prsPath = pathExtd(prsPath,selCHD,[tempEvt]);            
            prsPath += [[-99,tempEvt]];
        else:
            prsAccInfo[tempEvt,:] = 1+eventAccInfo[selCHD,0],\
                eventBranch[tempEvt,2]+eventAccInfo[selCHD,1]; # added the branch of tempEvt to its child
            np.place(prsEvt, prsEvt==selCHD, tempEvt); # replace the child id in prsEvt with its parent
            prsPath = pathExtd(prsPath,selCHD,[tempEvt]);
    return(prsEvt,prsAccInfo,prsPath,prsCensor)

def SinGrd_theta(theta,ni,bli):
    n = len(ni);
    k_hat = np.sum(ni/(1+theta*bli))/np.sum(theta*bli/(1+theta*bli));
    re = 0;
    for i in range(n):
        re += np.sum(1/np.arange(k_hat,k_hat+ni[i])) - np.log(1+theta*bli[i])
#    for i in range(len(tailInfo)):
#        re += - np.log(1+theta*tailInfo[i])
    return(re)

def SinGrd_Est(pstEvent,eventAccInfo,tailInfo):
    ni,bli = eventAccInfo[pstEvent,0],eventAccInfo[pstEvent,1];
    ni = np.append(ni,np.zeros_like(tailInfo));
    bli = np.append(bli,tailInfo);
    a,b = 1e-5, 1e5;
    fa,fb = SinGrd_theta(a,ni,bli),SinGrd_theta(b,ni,bli);
    if fa*fb>0: 
        mu = np.sum(ni)/np.sum(bli); sigma = 0;
    else:
        theta = sp.optimize.bisect(SinGrd_theta,a,b,args=(ni,bli));
        k_hat = np.sum(ni/(1+theta*bli))/np.sum(theta*bli/(1+theta*bli));        
        mu = k_hat*theta; sigma = np.sqrt(k_hat)*theta;   
    return(mu,sigma)  


def EMoptm1218(paraInfo,mu_Rec,newEvent,eventAccInfo,\
           eventBranch,eventChild,eventTail,pstEvent,pstPath,tailInfo,assAcc,pstCensor,eventCensor,tailCensor):
    # get the prbZmis for each missing value.
    epsilon = 1e-5;     
    Zmis = (prbZmis(paraInfo,mu_Rec,newEvent,eventAccInfo\
                    ,eventBranch,eventChild,eventTail)).astype(int);
    prsEvt,prsAccInfo,prsPath, prsCensor = mergeInfo1218(newEvent,eventAccInfo,\
                                          eventBranch,eventChild,Zmis,pstEvent,pstPath,pstCensor,eventCensor);
    EMroot = np.array(SinGrd_Est(prsEvt,prsAccInfo,tailInfo));
    ite_count = 0;newparaInfo = 0;
    newassAcc = assAcc
    while np.sqrt(np.sum((EMroot-newparaInfo)**2)) > epsilon and ite_count<100:
        newparaInfo = np.array(EMroot);
        Zmis = prbZmis(newparaInfo,mu_Rec,newEvent,eventAccInfo\
                       ,eventBranch,eventChild,eventTail).astype(int);
        prsEvt,prsAccInfo,prsPath, prsCensor = mergeInfo1218(newEvent,eventAccInfo,\
                                          eventBranch,eventChild,Zmis,pstEvent,pstPath,pstCensor,eventCensor);
        EMroot = np.array(SinGrd_Est(prsEvt,prsAccInfo,tailInfo))
        ite_count += 1;
        newassAcc = assAcc + len(np.nonzero(Zmis==0)[0])
    
    # estimation of muRec
    
#    temp_infor = prsAccInfo[prsEvt,1];
#    temp_infor = temp_infor[(1- prsCensor).astype(bool)];
#    temp_infor = np.append(temp_infor,tailInfo[(1-tailCensor).astype(bool)])
#    muRecEst = 1/np.mean(temp_infor);
    
    # added 0306 for the estimation of total Rec
#    temp_infor = prsAccInfo[prsEvt,1];
#    temp_infor = np.append(temp_infor,tailInfo[(1-tailCensor).astype(bool)])
#    totalRec = (np.sum(1- prsCensor)+np.sum(1-tailCensor))/np.sum(temp_infor);
        
    return(EMroot,prsEvt,prsAccInfo,prsPath,newassAcc,prsCensor,0,0)
    


    


def EMoptmTtree(paraInfo,mu_Rec,Node_NumOS,Node_BL):
    # get the prbZmis for each missing value.
    epsilon = 1e-5; n = len(Node_NumOS[:,0]); 
    bnds = ((1e-3, 100), (1e-3, 100));
    prsEvt = np.arange(0,n)
    prsAccInfo = np.transpose(np.vstack((Node_NumOS[:,1],Node_BL[:,1])))
    EMroot = SinGrd_Est(prsEvt,prsAccInfo,np.array([0]));
    return(EMroot)
    
def pathExtd(path,selCHD,tempEvt): # added an element at the begining 
    insertele = lambda x: tempEvt + x if x[0]==selCHD else x;
    c = list(map(insertele,path));
    return(c)