#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:48:51 2020

@author: macbjmu
"""

import simuEst as SE


""" 
   Single Simulation and analysis
""" 
simuG = SE.singleSimu(mu_InFC = 1,mu_Rec = 1/2.5);
inferRe 	= SE.singleEst(simuG, back_ratio = [0.9,0.85,0.8], smpRatio = 0.9)


""" multiple simulaiton and analysis
"""
res = SE.cmpEst(mu_InFC = 1,mu_Rec = 1/2.5,simu_K = 2)
HI.EstRePlot(re)

""" Analyze tree files 
"""

res = SE.anaTree(filename = filenm)
