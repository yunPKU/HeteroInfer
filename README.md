# HeteroInfer
# Introduction
This is the homepage of HeteroInfer, a set of python files for the inference of transmission heterogeneity as well as other epidemiological parameters (such as the basic reproduction number and the recovery rate) from a virus phylogeny. 

The input is a dated phylogeny (or genealogy), where leaves correspond to pathogens isolated from the infected and sampled hosts. The main output is the estimated amount of heterogeneity in transmissibility, i.e., the coefficient of variation of the transmissibility rate. Also there are some codes for the simualtion study of phylogeny-guided prevention.

# Usage
1. Import the main function and the related package 
```
import simuEst as SE

```
2. Simulate and analyze a single tree
```
simuG   = SE.singleSimu(mu_InFC, mu_Rec, cv_InFC, smpSize, smpRatio, fntNum); # simulating a tree under the given setting 
inferRe = SE.singleEst(simuG, back_ratio = [0.9,0.85,0.8], smpRatio = 0.9) # Analyzing the tree under the given setting 
```
3. Analyze a tree file
```
res = SE.anaTree(filename)
```
4. Composite simulation to show the performance of the proposed method
```
rec = SE.cmpEst(mu_InFC, mu_Rec, smpSize, fntNum, simu_K)
HI.EstRePlot(rec)
```
The meaning and default values of these used parameters are as follows:

parameter	| symbol  |  meaning	| Default value
--------- |---------|  -------  | -------
mu_InFC	  | μ       | the average transmission rate |	1
mu_Rec	  | γ       | the rate of being diagnosis (or removal)        | 1/2.5
cv_InFC	  | CV      | the coefficient of variation of the transmission rate | 1
smpSize	  | n       | the number of being diagnosed at the stop of the simulation |100
smpRatio	| ρ       | the sequencing ratio	| 0.9
fntNum	  | N       | the size of susceptible individuals	| math.inf -- standing for infinite population size
back_ratio| p       | the percentile for the lengths of the external branches |  [0.9,0.85,0.8]
simu_K    |         | the number of simulations | 100
filename  |         | the txt file that contains the tree in the "Newick" format
