#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:20:28 2022

@author: do19150
"""

import sys
sys.path.append('/Users/do19150/Gits/Analysis_Funcs/LC_Sim')
sys.path.append('/home/do19150/Gits/Analysis_Funcs/LC_Sim')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pdb

from scipy.stats import exponnorm
from DoneLC import Done_LC
from CurveFit import NGaussFixT

#set the total number of LCs to create
N_lcs = int(input('How many lightcurves are to be simulated: '))

#import population data needed for creation of lightcurves
#Base lightcurves on XMM-Newton curves (as used in QPE observations and cands)
#time binning of 10s and total length of 100ks.

Period = float(input('Enter length of lightcurves: '))
tbin = float(input('Enter length of lightcurve time bins: '))
t = np.arange(0,Period,tbin)

#pdb.set_trace()
#create empty array for generated lightcurves
wo_qpe_arr = np.empty((int((N_lcs/2)+1),len(t)))
qpe_arr = np.empty((int((N_lcs/2)+1),len(t)))

#set time as the first row in each output array
wo_qpe_arr[0,:] = t
qpe_arr[0,:] = t

#Create a random set of values for phi and beta to use in generating the
#simulated lightcurves. Phi will be drawn from a uniform distribution,
#while beta will be drawn from a gaussian around the average value for AGN
#power law slopes.

slope_mean = float(input('Mean value for power law slopes to be drawn: '))
slope_std = float(input('Standard deviation for power law slopes to be drawn: '))

#generate random power law slopes for the simulated lcs, ensuring all are positive
slopes = np.abs(np.random.normal(slope_mean,slope_std,N_lcs))

#generate simulated lightcurves for the lcs without qpes
for i in range(int(N_lcs/2)):
    lc = Done_LC(Period,tbin,slopes[i],phi_type='u')
    plt.plot(t,lc[1])
    plt.show()
    wo_qpe_arr[i+1,:] = lc[1]
    
#save the lcs for the sample without qpes
np.savetxt('LCGen/no_qpe_sample.csv',wo_qpe_arr,delimiter=',')

#generate simulated lightcurves for the lcs with qpes
for i in range(int(N_lcs/2)):
    lc = Done_LC(Period,tbin,slopes[i],phi_type='u')
    plt.plot(t,lc[1])
    plt.show()
    qpe_arr[i+1,:] = lc[1]

#import the eruption characteristics file
eruption_chars = pd.read_csv('Obs/eruption_profiles.csv',index_col=0)
duty_cycles = [,0.41,0.19]

#for qpe sample create distributions for amplitude, duration, and duty cycle
#all variables are to be fit to exponentially modified gaussian distributions
#in order to provide the tail including eRO-QPE1
amp_dist = exponnorm.fit(eruption_chars['Amplitude'].values)
amplitudes = np.abs(exponnorm.rvs(amp_dist[0],loc=amp_dist[1],scale=amp_dist[2],size=int(N_lcs/2)))
dur_dist = exponnorm.fit(eruption_chars['Duration'].values)
durations = np.abs(exponnorm.rvs(amp_dist[0],loc=amp_dist[1],scale=amp_dist[2],size=int(N_lcs/2)))
dc_dist = exponnorm.fit(eruption_chars['Amplitude'].values)
duty_cycles = np.abs(exponnorm.rvs(amp_dist[0],loc=amp_dist[1],scale=amp_dist[2],size=int(N_lcs/2)))

#determine the recurrence times 
trec = durations/duty_cycles

for i in range(int(N_lcs/2)):
    #for each curve determine the peak locations
    rec = trec[i]
    #set the first peak anywhere within a range of size trec centred at 0
    first_peak = np.random.uniform(low=-rec/2,high=rec/2)
    #determine the no of peaks such that either the start or end could be affected
    #by a peak starting or ending
    no_peaks = int((Period // rec) + 1)
    
    #for each qpe curve create the underlying gaussian eruptions
    peak_profile = NGaussFixT(t, no = no_peaks, amplitude = amplitudes[i], width = durations[i],
                             position = first_peak, recurrence = trec[i])

#convolve the eruptions with the power law lcs


#save the lcs for the sample with qpes
np.savetxt('LCGen/qpe_sample.csv',qpe_arr,delimiter=',')


