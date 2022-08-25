#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:20:28 2022

@author: do19150
"""

import sys
sys.path.append('/Users/do19150/Gits/Analysis_Funcs/LC_Sim')
sys.path.append('/home/do19150/Gits/Analysis_Funcs/LC_Sim')
from tqdm import tqdm

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import pdb

from scipy.stats import exponnorm
from DoneLC import Done_LC

def NGaussFixT(domain,no=2,amplitude=1,width=1,position=0,recurrence=1):
    
    '''
    Function to create a repeating Gaussian curve, with the peak of
    the first Gaussian at a specific point, and recurring with a
    certain time.
    
    Parameters:
    no - number of gaussians in the specified curve
    constant - base level for the curve(s) above zero
    amplitude - height of the gaussian curve above the base constant level
    width - FWHM of the gaussian curve
    position - location of the first peak of the curve
    recurrence - distance between peaks of Gaussians
    '''
    
    no = int(no)
    
    #create a peak waveform for the correct number of gaussians
    peak_times = np.arange(position,position+no*recurrence,recurrence)
    w = ((0.5*width)**2)/(np.log(2))
    combined_gaussians = np.zeros(np.shape(domain))
    for i in range(no):
        combined_gaussians += np.abs(amplitude)*np.exp(-((domain-peak_times[i])**2)/w)
    
    #add the constant to shift in range
    rng = combined_gaussians
    
    return rng



#set the total number of LCs to create
N_lcs = int(input('How many lightcurves are to be simulated: '))

#import population data needed for creation of lightcurves
#Base lightcurves on XMM-Newton curves (as used in QPE observations and cands)
#time binning of 10s and total length of 100ks.

Period = float(input('Enter length of lightcurves: '))
tbin = float(input('Enter length of lightcurve time bins: '))
t = np.arange(0,Period,tbin)

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
print('Without QPEs')
for i in tqdm(range(int(N_lcs/2))):
    lc = Done_LC(Period,tbin,slopes[i],phi_type='u')
    wo_qpe_arr[i+1,:] = lc[1]
    
#save the lcs for the sample without qpes
np.savetxt('LCGen/no_qpe_sample.csv',wo_qpe_arr,delimiter=',')

#generate simulated lightcurves for the lcs with qpes
print('Pre QPEs')
for i in tqdm(range(int(N_lcs/2))):
    lc = Done_LC(Period,tbin,slopes[i],phi_type='u')
    qpe_arr[i+1,:] = lc[1]

#import the eruption characteristics file
eruption_chars = pd.read_csv('Obs/eruption_profiles.csv',index_col=0)
dc_vals = pd.read_csv('Obs/eruption_dcs.csv',index_col=0)

amp_vals = list(eruption_chars['Amplitude'].values)
dur_vals = list(eruption_chars['Duration'].values)

#for qpe sample create distributions for amplitude, duration, and duty cycle
#all variables are to be fit to exponentially modified gaussian distributions
#in order to provide the tail. Remove eRO-QPE1 for the fitting.
amp_vals = amp_vals[0:18] + amp_vals[19:]
dur_vals = dur_vals[0:18] + dur_vals[19:]
amp_dist = exponnorm.fit(amp_vals)
amplitudes = np.abs(exponnorm.rvs(amp_dist[0],loc=amp_dist[1],scale=amp_dist[2],size=int(N_lcs/2)))
dur_dist = exponnorm.fit(dur_vals)
durations = np.abs(exponnorm.rvs(dur_dist[0],loc=dur_dist[1],scale=dur_dist[2],size=int(N_lcs/2)))
dc_dist = exponnorm.fit(dc_vals['DC'].values,floc=0)
duty_cycles = exponnorm.rvs(dc_dist[0],loc=dc_dist[1],scale=dc_dist[2],size=int(N_lcs/2))

#determine the recurrence times 
trec = durations/duty_cycles

print("Adding in QPE features.")
for i in tqdm(range(int(N_lcs/2))):
    #for each curve determine the peak locations
    rec = trec[i]
    #set the first peak anywhere within a range of size trec centred at 0
    first_peak = np.random.uniform(low=-rec/2,high=rec/2)
    #determine the no of peaks such that either the start or end could be affected
    #by a peak starting or ending
    if first_peak < 0:
        no_peaks = int((Period // rec) + 2)
    else:
        no_peaks = int((Period // rec) + 1)
    
    #for each qpe curve create the underlying gaussian eruptions
    peak_profile = 1 + NGaussFixT(qpe_arr[0], no = no_peaks, amplitude = amplitudes[i], 
                             width = durations[i], position = first_peak, recurrence = trec[i])
    
    print(amplitudes[i],durations[i],first_peak,trec[i])

#    fix,ax1 = plt.subplots()
#    ax2 = ax1.twinx()
#    ax1.plot(qpe_arr[0],qpe_arr[i+1],color='b')
#    ax2.plot(qpe_arr[0],peak_profile,color='r')
#    plt.show()

    #convolve the eruptions with the power law lcs
    qpe_arr[i+1] *= peak_profile
    
#    plt.plot(qpe_arr[0],qpe_arr[i+1],color='k')
#    plt.show()

#save the lcs for the sample with qpes
np.savetxt('LCGen/qpe_sample.csv',qpe_arr,delimiter=',')


