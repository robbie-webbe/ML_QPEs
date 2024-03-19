#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:39:18 2024

@author: do19150
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from stingray import Lightcurve


for file in sorted(os.listdir('real_obs/')):
    if file.startswith('P'):
        hdul = fits.open('real_obs/'+file)
        time = hdul[1].data.field('TIME')
        rate = hdul[1].data.field('RATE')
        error = hdul[1].data.field('ERROR')
        gtis = []
        for i in hdul[2].data:
            gtis.append([i[0],i[1]])
        dt = hdul[1].data.field('TIMEDEL')[0]
        lc = Lightcurve(time[np.isfinite(rate)],dt*rate[np.isfinite(rate)],err=error[np.isfinite(rate)]*dt,gti=gtis)
        lc.apply_gtis()
        lc250 = lc.rebin(250)
        lc250 = lc250.shift(-lc250.time[0])
        pos_idx = np.where(lc250.counts >= 0.0)
        fig, axs = plt.subplots(figsize=(12,8))
        axs.errorbar(lc250.time[pos_idx]/1000, lc250.countrate[pos_idx],yerr=lc250.countrate_err[pos_idx],xerr=0.125,color='k',ls='')
        axs.set(xlabel='Time (ks)',ylabel='X-ray Brightness')
        fig.savefig('zooniverse_lc_sample/'+file[:-3]+'png')
        plt.show()
        
        

non_qpes = np.loadtxt('/Users/do19150/Gits/ML_QPE/LCGen/Diff_dt/no_qpe_sample_dt250.csv',delimiter=',')
#fake_qpe = np.loadtxt('/Users/do19150/Gits/ML_QPE/LCGen/Diff_dt/qpe_sample_dt250.csv',delimiter=',')

#determine lightcurve lengths
fake_lc_lengths = 4*10**(np.random.normal(1.5,size=10))
fake_lc_lengths[np.where(fake_lc_lengths >= 400)] = 400
#sort out the minimum length, with 50ks for QPE obs, and 1ks for others
fake_lc_lengths[np.where(fake_lc_lengths <= 4)] = 4
#fake_lc_lengths[np.where(fake_lc_lengths <= 200)] = 200

#pick out the fake lightcurves to create plots
fake_lcs = sorted(np.random.choice(np.arange(50000),size=10,replace=False))

for i in range(10):
    #pick out the length of the new, faked lightcurve
    length = int(fake_lc_lengths[i])
    #pick out the corresponding faked full lightcurve
    lc = non_qpes[fake_lcs[i]+1][:length]
    
    #rescale the lightcurve to a reasonable average count rate, between 0.004 and 20 /s
    new_avg = 10**(np.random.uniform(np.log10(1),np.log10(5000)))
    lc *= (new_avg/np.average(lc))
    
    #now discretise the number of counts in each bin
    lc = lc // 1
    
    #create a fake 'old sampling' rate for the lightcurves as though they were real
    old_sampling = np.random.uniform(250/5,250/0.8)    

    #create fake scaling factors for the y error bars
    old_err_factors = np.sqrt(old_sampling)
    
    #create a time array
    time = np.arange(length)
    #determine the 'gappy' length of the lightcurve, and the indices of points to be kept
    #for the QPE sample the minimmum sampling is 0.8
    noisy_lc_length = np.random.randint(int(length*0.6),length)
    #noisy_lc_length = np.random.randint(int(length*0.8),length)
    noisylc_idx = np.random.choice(np.arange(length),size=noisy_lc_length,replace=False)
    noisylc_idx = sorted(noisylc_idx)
    #plot the 'gappy' lightcurve
    fig, axs = plt.subplots(figsize=(12,8))
    axs.errorbar(0.25*time[noisylc_idx],lc[noisylc_idx]/250,yerr=np.sqrt(lc[noisylc_idx])/(250*old_err_factors),xerr=0.125,color='k',ls='')
    axs.set(xlabel='Time (ks)',ylabel='X-ray Brightness')
    fig.savefig('zooniverse_lc_sample/fake_noqpe'+str(i)+'.png')
    #fig.savefig('zooniverse_lc_sample/fake_withqpe'+str(i)+'.png')
    plt.show()
    
    
    