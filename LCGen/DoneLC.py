#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:19:16 2019

@author: do19150
"""

import numpy as np



def Done_LC(T,delT,beta=1.0,phi_type='g'):
    
    '''
    Function designed to generate a power spectrum as per the algorithm included in 
    Appendix B of the Done et al 1992 paper. The arguments of the function are: Total
    time for the observation to be simulated; Width of time bins in the light curve 
    to be simulated; Index of the power law  rule to be applied to the frequencies
    used in the algorithm; Type of phase shift to be applied to each cosine wave.
    
    Phase shifting: There are three types of phase shift currently that can be
    incorporated in the model: Gaussian, Uniform or no shifting.
    '''

    #Initialise frequencies to be used in generation
    fmin = 1./T
    fmax = 2/delT
    nf = (fmax)/(fmin)
    
    freq = np.linspace(fmin,fmax,nf)
    
    #Initialise time and light curve outputs
    time = np.arange(0,T,delT)
    x_t = np.zeros((np.shape(time)))
    
    for i in range(len(freq)):
        
        #Scaling coefficient as per Appendix B
        coeff = ((np.sin(np.pi*freq[i]*delT))/(np.pi*freq[i]*delT))
        
        #Power law component 
        pow_f = (freq[i]**(-0.5*beta))
        
        #Select phase shift for that frequency        
        if phi_type == 'None':
            phase_shift = 0.0
        elif phi_type == 'u':
            phase_shift = np.pi*np.random.random()
        else:
            phase_shift = np.pi*np.random.randn()
            
        cos_curve = np.cos((2*np.pi*freq[i]*time)-phase_shift)
        #create curve for specific frequency
        x_f = coeff*pow_f*cos_curve
        #sum over all frequencies
        x_t += x_f
        
    x_min = min(x_t)
    x_t -= x_min
    
    #create power for fourier frequencies and the frequency labels themselves
    pow_spec = np.abs(np.fft.rfft(x_t))
    fft_f = (np.fft.fftfreq(x_t.size,d=delT))[0:len(pow_spec)-1]

    return (time, x_t, fft_f, pow_spec)
