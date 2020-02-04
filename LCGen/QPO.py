#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:46:15 2020

@author: do19150
"""

import numpy as np
from DoneLC import Done_LC

#set no. of lightcurves to simulate
N = int(input("Number of Light Curves: "))

#determine how many half-waves to include in lightcurve. The 1+ eliminates curves without QPOs
QPO_No = 1 + np.random.poisson(size=N)

#set timeframe for observation
T = int(input("Time period (10,000): ") or 10000)
dT = int(input("Binning time (1): ") or 1)
steps = T//dT

#set time period for QPOs
t_period = 2*(T//QPO_No)

#set phase delays for curves
t_initial = np.random.uniform(0,2*np.pi, size=N)

results = np.zeros((int(2*N), steps))
for i in range(N):
    print(i)
    time, x_t, fft_f, pow_spec = Done_LC(T,dT) #create base curve
    
    #determine amplitude of QPO as random fraction of the max amplitude of 'dirty' wave
    amp = max(x_t)*np.abs(np.random.randn())
    
    #create QPO waveform. Amplitude is a random fraction of the amplitude for the dirty wave.
    QPO = amp*(1+np.cos(2*np.pi*((time/t_period[i])+t_initial[i])))

    #add QPO to dirty wave
    x_t += QPO
    
    results[2*i] = time
    results[2*i+1] = x_t
    
filename = "/home/do19150/Git_repos/ML-LCVariability/LightCurves/QPON" + str(N) + "T" + str(T) + "dT" + str(dT) + "LC.csv"
np.savetxt(filename,results, delimiter=',')
