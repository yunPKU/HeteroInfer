 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 08:17:03 2018

@author: zhang
"""
import sys
import numpy as np
import re as re
import treeSimuFun1204 as tSF
import treePertbFun0831 as tPF
import treeMapFun0821 as tMF
import transEventFun0815 as tEF
import emAlgFunFinal0421_0 as eAF
import scipy as sp
import matplotlib.pyplot as plt
from heapq import nlargest
pathMaker = lambda x: [x];

def sackCal1013(tranPairEst,normal='Null'):
    nNode = len(tranPairEst[:,0])+1
    skVal = np.zeros(nNode);
    for i in range(nNode):
    #i = 2
        tempNd = i+1;
        bID = np.where(tranPairEst[:,0]==tempNd)[0];
        if len(bID)>0: # tempNd is a donner in some events
            tempCnt = len(bID);
            bID = np.min(bID);            
        else:
            bID = np.where(tranPairEst[:,1]==tempNd)[0][0]+1; 
            tempCnt = 0;
        for j in range(bID):
            dn,rec = tranPairEst[bID-1-j,:2];
            if dn == tempNd:
                tempCnt += 1;
            elif rec == tempNd:
                tempCnt += 1;
                tempNd = dn;
        skVal[i] = tempCnt;  
    raSk = np.sum(skVal);
    if normal == 'PDA':
        return(raSk/nNode**1.5)
    elif normal == 'Yule':
        yc = 2*nNode*np.sum(1/np.arange(2,nNode+1))
        return((raSk-yc)/yc)
    else:
        return(raSk)

def treeEst(a):
    
    # find out all the position of '(' and ')'
    lftPos = np.array([i for i, ltr in enumerate(a) if ltr == "("])
    rgtPos = np.array([i for i, ltr in enumerate(a) if ltr == ")"])
    cmaPos = np.array([i for i, ltr in enumerate(a) if ltr == ","]+[len(a)])
    nMax = len(lftPos)
    tranPairEst = np.zeros((nMax,3))
    tailInfor = np.zeros(nMax+1)
    recCount = 0;
    while recCount<nMax:
        lft_id = 0;
        temp_rgt = rgtPos[lft_id];
        temp_lft = np.max(lftPos[lftPos<temp_rgt])
        if len(rgtPos)>1: 
            temp_end = np.minimum(np.min(cmaPos[cmaPos>temp_rgt]),rgtPos[1]);
        else:
            temp_end = len(a);
        new = a[temp_lft:temp_end]
        newnum = re.findall(r"[-+]?\d*\.\d+|\d+", new)
        tempSeq = (np.array(newnum)).astype(np.float)
        tranPairEst[nMax-1-recCount,:] = np.array([tempSeq[0],tempSeq[2],tempSeq[4]])
        if tempSeq[1]!=9999: tailInfor[tempSeq[0].astype(int)-1] = tempSeq[1];
        if tempSeq[3]!=9999: tailInfor[tempSeq[2].astype(int)-1] = tempSeq[3];
        b = str(tempSeq[0])+':9999'
        c = a.replace(new,b);
        recCount += 1;
        a = c;
        lftPos = np.array([i for i, ltr in enumerate(a) if ltr == "("])
        rgtPos = np.array([i for i, ltr in enumerate(a) if ltr == ")"])
        cmaPos = np.array([i for i, ltr in enumerate(a) if ltr == ","]+[len(a)])
    
    # reconstruct the transPair and hostLife information
    for i in range(1,nMax): # nMax is the number of transPair
        dn,rec = tranPairEst[i,0:2];
        dnID = np.where(tranPairEst[:i,0]==dn)[0];
        if len(dnID)>0: 
            dnID = np.max(dnID)
        else:
            dnID = np.where(tranPairEst[:i,1]==dn)[0];
            dnID = np.max(dnID)
        tranPairEst[i,2] += tranPairEst[dnID,2]
    hostLife = np.zeros((nMax+1,3)); # the number of nodes shall be nMax+1
    hostLife[:,0] = np.arange(1,nMax+2);
    for i in range(nMax+1):
        dnID = np.where(tranPairEst[:,1]==i+1)[0];
        if len(dnID)>0:
#            dnID = np.min(dnID); 
            hostLife[i,1] = tranPairEst[dnID,2];
            hostLife[i,2] = tranPairEst[dnID,2];
            
        dnID = np.where(tranPairEst[:,0]==i+1)[0];
        if len(dnID)>0:
            dnID = np.max(dnID);
            hostLife[i,2] = tranPairEst[dnID,2];
    
    hostLife[:,2] = hostLife[:,2] + tailInfor;
#    file.close()
    return(tranPairEst,hostLife)
def Est_PartData_RS(transPairSP,hostLifeSP,censoring_flagSP,cor_ratio,spl):
    eventBranch,eventTail,Host_tailBranch,eventChild,\
                eventLevel,eventPathSet,eventDepth,eventCensor,eventAccInfo\
                = tEF.transPair2EventInfor1116(transPairSP,hostLifeSP,censoring_flagSP);
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
    averTail = np.mean(eventTail[eventTail>0]);
    tempN = len(np.nonzero(eventCensor==0)[0]);
    totalEdge = np.sum(eventTail[eventTail>0])+np.sum(eventBranch[:,2])
    totalRecEst2 = tempN / (totalEdge + averTail*(tempN/(cor_ratio*spl) - tempN)); 
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
    resu = [[]]*len(back_ratio);
    for i in range(len(back_ratio)):
        resu[i] = Est_PartData_RS(tptransPairRec,tphostLifeRec,tpcensoring_flagRec,back_ratio[i],spl);
    return(resu)

'''
    Analysis of single tree
'''
def anaRealData(filename,back_ratio = [0.9,0.85,0.8]):
    spl = 0.9; nfile = 10001;resu = [[]]*len(back_ratio);
    # read the file
    filenm = filename;
    tempstr = open(filenm,'r').read()     
    transPairRec,hostLifeRec = treeEst(tempstr + ':0.2')
    transPairRec = transPairRec[np.argsort(transPairRec[:,2]),:]
    censoring_flagRec = np.zeros(len(hostLifeRec[:,0]), dtype=bool)
    #     analysis    
    for i in range(len(back_ratio)):
        resu[i] = Est_PartData_RS(transPairRec,hostLifeRec,censoring_flagRec,back_ratio[i],spl);
    mu_mle,sigma_mle,mu_RecEst,b = crct_aveg(resu,back_ratio,spl);
    saIndexPDA = sackCal1013(transPairRec,'PDA')
    return(mu_mle,sigma_mle,mu_RecEst,saIndexPDA)

'''
    Analysis of the three data sets
'''

def anaRealData_IDUAE(path,back_ratio = [0.9,0.85,0.8]):
    spl = 0.9; nfile = 10001;resu = [[]]*len(back_ratio);
    mu_mle,sigma_mle,mu_RecEst,cv,Ro,saIndexPDA,saIndexYL = (np.zeros(nfile) for i in range(7))
    for j in range(nfile):   
        if not (np.array([269,319,355,409,435,497,498,596,931,1204,1846,1868,1934,\
                          2108,2114,2428,2501,2553,2580,2728,2747,2791,3036,3231,\
                          3249,3341,3399,3512,3521,3607,3958,4455,4675,4813,4955,\
                          4992,5101,5328,5363,5424,5472,5809,6028,6284,6602,6984,\
                          7369,7663,7851,8211,8261,8297,8301,9021,9489,9543,9697,\
                          9753,9828,9947])==j).any(): 
            filenm = path + '/iduAtrees/iduA'\
                +str(j+1)+'.txt'
            tempstr = open(filenm,'r').read()     
            transPairRec,hostLifeRec = treeEst(tempstr + ':0.2')
            transPairRec = transPairRec[np.argsort(transPairRec[:,2]),:]
            censoring_flagRec = np.zeros(len(hostLifeRec[:,0]), dtype=bool)
        #     analysis    
            for i in range(len(back_ratio)):
                resu[i] = Est_PartData_RS(transPairRec,hostLifeRec,censoring_flagRec,back_ratio[i],spl);
            mu_mle[j],sigma_mle[j],mu_RecEst[j],b = crct_aveg(resu,back_ratio,spl);
            if mu_RecEst[j]>0:
                cv[j] = sigma_mle[j]/mu_mle[j]
                Ro[j] = mu_mle[j]/mu_RecEst[j]
                # sackin Index
                saIndexPDA[j] = sackCal1013(transPairRec,'PDA')    
    totalRe = np.transpose(np.vstack((mu_RecEst,Ro,cv,saIndexPDA)))
    totalRe = totalRe[mu_RecEst>0,:]
    return(totalRe)
    
def anaRealData_IDUB(path,back_ratio = [0.9,0.85,0.8]):
    back_ratio = [0.9,0.85,0.8];
    spl = 0.9; nfile = 10001;resu = [[]]*len(back_ratio);
    mu_mle,sigma_mle,mu_RecEst,cv,Ro,saIndexPDA,saIndexYL = (np.zeros(nfile) for i in range(7))
    for j in range(nfile):   
        if not (np.array([990,4876,5022])==j).any(): 
            filenm = path+'/iduBtrees/iduB'\
                +str(j+1)+'.txt'
            tempstr = open(filenm,'r').read()     
            transPairRec,hostLifeRec = treeEst(tempstr + ':0.2')
            transPairRec = transPairRec[np.argsort(transPairRec[:,2]),:]
            censoring_flagRec = np.zeros(len(hostLifeRec[:,0]), dtype=bool)
        #     analysis    
            for i in range(len(back_ratio)):
                resu[i] = Est_PartData_RS(transPairRec,hostLifeRec,censoring_flagRec,back_ratio[i],spl);
            mu_mle[j],sigma_mle[j],mu_RecEst[j],b = crct_aveg(resu,back_ratio,spl);
            if mu_RecEst[j]>0:
                cv[j] = sigma_mle[j]/mu_mle[j]
                Ro[j] = mu_mle[j]/mu_RecEst[j]
                # sackin Index
                saIndexPDA[j] = sackCal1013(transPairRec,'PDA')
    totalRe = np.transpose(np.vstack((mu_RecEst,Ro,cv,saIndexPDA)))
    totalRe = totalRe[mu_RecEst>0.05,:]
    return(totalRe)
    
def anaRealData_MSMB(path,back_ratio = [0.9,0.85,0.8]):
    back_ratio = [0.9,0.85,0.8];
    spl = 0.9; nfile = 10001;resu = [[]]*len(back_ratio);
    mu_mle,sigma_mle,mu_RecEst,cv,Ro,saIndexPDA,saIndexYL = (np.zeros(nfile) for i in range(7))
    for j in range(nfile):   
        if not (np.array([846,949,1869,3421,4253,6404,6692,7449,7568,7789,8086,\
                          8432,9912,10000])==j).any(): 
            filenm = path+'/msmC1trees/msmC'\
                +str(j+1)+'.txt'
            tempstr = open(filenm,'r').read()     
            transPairRec,hostLifeRec = treeEst(tempstr + ':0.2')
            transPairRec = transPairRec[np.argsort(transPairRec[:,2]),:]
            censoring_flagRec = np.zeros(len(hostLifeRec[:,0]), dtype=bool)
        #     analysis    
            for i in range(len(back_ratio)):
                resu[i] = Est_PartData_RS(transPairRec,hostLifeRec,censoring_flagRec,back_ratio[i],spl);
            mu_mle[j],sigma_mle[j],mu_RecEst[j],b = crct_aveg(resu,back_ratio,spl);
            if mu_RecEst[j]>0:
                cv[j] = sigma_mle[j]/mu_mle[j]
                Ro[j] = mu_mle[j]/mu_RecEst[j]
                # sackin Index
                saIndexPDA[j] = sackCal1013(transPairRec,'PDA')
    
    totalRe = np.transpose(np.vstack((mu_RecEst,Ro,cv,saIndexPDA)))
    totalRe = totalRe[mu_RecEst>0,:]
    return(totalRe)

def rePlot(idua,idub,msmB):
    msmc1 = msmB
    
    fig, axes = plt.subplots(2, 2, sharex='col')
    
    margin = [[0,1],[1,6],[0,1.5],[0.5,2.5]]
    
    para = ['Recovery rate ($\gamma$)', 'Basic Rerpoduction Number ($R_0$)','Coefficeint of Variation (CV)','Sackin Index']
    
    for i in range(4):
        i1= int(i>1)
        i2 = i - i1*2
        g=sns.violinplot(data = [idua[:,i],idub[:,i],msmc1[:,i]],ax = axes[i1][i2],\
                         palette = ['red','blue','gray',],saturation = 1,bw= 0.5)
        g.set(ylim=margin[i])
        g.set_xticklabels(['A','B','C'])
        g.set(title = para[i])

#path = 'C:/Users/blues/Documents/pic'
#idua  = anaRealData_IDUAE(path);
#idub  = anaRealData_IDUB(path);
#msmB  = anaRealData_MSMB();
## plot the results 
#rePlot(idua,idub,msmB)
filename = 'C:/Users/blues/Documents/pic/iduAtrees/iduA1.txt'
re = anaRealData(filename)