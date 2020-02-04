#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:15:45 2020

@author: do19150
"""

import numpy as np

from DoneLC import Done_LC


N = int(input("Number of Light Curves: "))
T = int(input("Time period (10,000): ") or 10000)
dT = int(input("Binning time (1): ") or 1)
steps = T//dT
results = np.zeros((int(2*N), steps))

for i in range(N):
    time, x_t, fft_f, pow_spec = Done_LC(T,dT)
    results[2*i] = time
    results[2*i+1] = x_t
    print(i)
    
filename = "/home/do19150/Git_repos/ML-LCVariability/LightCurves/NoVarN" + str(N) + "T" + str(T) + "dT" + str(dT) + "LC.csv"
np.savetxt(filename,results, delimiter=',')