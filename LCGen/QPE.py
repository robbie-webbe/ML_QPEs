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

#determine how many QPEs to include in lightcurve
QPE_No = np.random.poisson(size=N)

#set timeframe for observation
T = int(input("Time period: "))
dT = int(input("Binning time: "))
steps = T//dT

#set time gap between QPEs
t_rec = T//QPE_No


#set width of QPEs
width = np.random.uniform(low=0.0, high=0.5)

#set height of QPEs
height = np.random.

#set time of first QPE peak

results = np.zeros((int(2*N), steps))
for i in range(N):
    time, x_t, fft_f, pow_spec = Done_LC(T,dT)
    results[2*i] = time
    results[2*i+1] = x_t
filename = "/home/do19150/Git_repos/ML-LCVariability/LightCurves/NoVarN" + str(N) + "T" + str(T) + "dT" + str(dT) + "LC.csv"
np.savetxt(filename,results, delimiter=',')