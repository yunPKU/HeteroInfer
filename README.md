# HeteroInfer
# Introduction
This is the homepage of HeteroInfer, a set of python files for the inference of transmission heterogeneity as well as other epidemiological parameters (such as the basic reproduction number and the recovery rate) from a virus phylogeny. 

The input is a dated phylogeny (or genealogy), where leaves correspond to pathogens isolated from the infected and sampled hosts. The main output is the estimated amount of heterogeneity in transmissibility, i.e., the coefficient of variation of the transmissibility rate. Also there are some codes for the simualtion study of phylogeny-guided prevention.

# Usage
1. Import the main function
```
import HeteroInfer as HI
```
2. Simulate and analyze a single tree
```
simuG = HI.singleSimu(mu_InFC, mu_Rec, cv_InFC, smpSize, smpRatio, fntNum); # simulating a tree under the given setting 
inferRe 	= HI.singleEst(simuG, back_ratio = [0.9,0.85,0.8], smpRatio = 0.9) # Analyzing the tree under the given setting 
```
The meaning and default values of these parameters are as follows:

parameter	|meaning	|Default value

3. Analyze a tree file
```
res = HI.anaTree(filename = filenm)
```
4. Composite simulation
```
re = HI.cmpEst(mu_InFC = 1,mu_Rec = 1/2.5,simu_K = 2)
HI.EstRePlot(re)
```
