#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:46:15 2020

@author: do19150
"""

import numpy as np
from scipy.stats import multivariate_normal
from DoneLC import Done_LC

#set no. of lightcurves to simulate
N = int(input("Number of Light Curves: "))

#determine how many QPEs to include in lightcurve. The 1+ eliminates curves without QPEs
QPE_No = 1 + np.random.poisson(size=N)

#set timeframe for observation
T = int(input("Time period (10,000): ") or 10000)
dT = int(input("Binning time (1): ") or 1)
steps = T//dT

#set time gap between QPEs
t_rec = T//QPE_No

#set time of first QPE peak for each curve
t_initial = np.random.uniform(0, t_rec, size=N)

#set width & height of QPEs for each curve
width = np.random.uniform(high=0.25, size=N) * t_rec
#modify heights to reflect a wider time frame and a lower gaussian.
height = (1+np.random.poisson(lam=5, size=N))*(T/(10*QPE_No))

results = np.zeros((int(2*N), steps))
for i in range(N):
    print(i)
    t_peaks = np.arange(t_initial[i], stop=T, step=t_rec[i]) #set times of peaks for curve
        
    time, x_t, fft_f, pow_spec = Done_LC(T,dT) #create base curve
    
    #create peak signature for first peak
    peaks = 1 + height[i]*multivariate_normal.pdf(time, t_peaks[0], width[i]**2)
    
    if QPE_No[i] != 1:
        for j in np.arange(start=1, stop=QPE_No[i]):
            peaks += height[i]*multivariate_normal.pdf(time, t_peaks[j],width[i]**2)
    #add in peak signature    
    x_t *= peaks
    
    results[2*i] = time
    results[2*i+1] = x_t

    
    
filename = "/home/do19150/Git_repos/ML-LCVariability/LightCurves/QPEN" + str(N) + "T" + str(T) + "dT" + str(dT) + "LC.csv"
np.savetxt(filename,results, delimiter=',')
