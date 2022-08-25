#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:45:53 2022

@author: do19150
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, iqr
from stingray import Lightcurve
from stingray.crosscorrelation import AutoCorrelation


def lcfeat(lc,qpe=0):
    '''
    Function to extract a number of variability measures from either a real or 
    simulated lightcurve for use with ML classification methods. Features are:
        - Standard Deviation/Mean
        - Proportion of points at least 1, 2, 3, 4, 5 and 6 sigma from the mean
        - kurtosis
        - skew
        - reverse autocorrelation
        - Autocorrelation max after first 0
        - Interquartile Range / Standard Deviation
        - CSSD (Consecutive Same Sign Deviations from the mean)
        - Von Neumann ratio

    Parameters
    ----------
    lc : Lightcurve to be analysed for features, as [time,count(rate)]
    qpe : Do the lightcurves contain QPEs? 1 = Yes. 0 = No.
    

    Returns
    -------
    features - Array of features extracted from lightcurve.

    '''
    
    #Initialise an empty output array 
    features = np.zeros(15)
    
    plt.plot(lc[0],lc[1])
    plt.show()
    
    #calculate the mean and standard deviation of the curve
    mean = np.average(lc[1])
    std = np.std(lc[1])
    
    #output the first feature as std/mean
    features[0] = std/mean
    
    #determine the proportions of points more than x stds from the mean
    features[1] = len(np.where((lc[1] >= mean+std)|(lc[1] <= mean-std))[0])/len(lc[1])
    features[2] = len(np.where((lc[1] >= mean+2*std)|(lc[1] <= mean-2*std))[0])/len(lc[1])
    features[3] = len(np.where((lc[1] >= mean+3*std)|(lc[1] <= mean-3*std))[0])/len(lc[1])
    features[4] = len(np.where((lc[1] >= mean+4*std)|(lc[1] <= mean-4*std))[0])/len(lc[1])
    features[5] = len(np.where((lc[1] >= mean+5*std)|(lc[1] <= mean-5*std))[0])/len(lc[1])
    features[6] = len(np.where((lc[1] >= mean+6*std)|(lc[1] <= mean-6*std))[0])/len(lc[1])
    
    #determine the skew and kurtosis of the datasets and pass to the output
    features[7] = kurtosis(lc[1])
    features[8] = skew(lc[1])
    
    #reverse the lightcurve
    lc_reverse = np.flip(lc[1])
    features[9] = np.sum(((lc_reverse-mean)/std)*((lc[1]-mean)/std))
    print(features)
    
    #create the AutoCorrelation for the lightcurve
    acf = AutoCorrelation(Lightcurve(time=lc[0],counts=lc[1]),mode='full')
    #split the autocorrelation for positive lags
    acf_pos = acf.corr[np.where(acf.time_lags >= 0)[0]]
    
    #determine the zero value for correlation and the first time when the curve goes -ve
    acf0 = acf_pos[0]
    print(acf0)
    first_zero = np.where(acf_pos < 0)[0][0]
    print(first_zero)
    
    #pick the maximum value after the first zero and out put to array
    print(acf_pos)
    print(acf_pos[first_zero:])
    acf_2nd_max = max(acf_pos[first_zero:])
    features[10] = acf_2nd_max / acf0
    
    #calculate the IQR / STD
    features[11] = iqr(lc[1])/std
    
    #calculate the deviations of the lightcurve from its mean
    deviations = lc[1] - mean
    CSSD_count = 0
    
    #for each set of three points check if all three are above or below the mean
    for k in range(len(lc[1])-2):
        if (deviations[k] > 0) and (deviations[k+1] > 0) and (deviations[k+2] > 0):
            CSSD_count += 1
        elif (deviations[k] < 0) and (deviations[k+1] < 0) and (deviations[k+2] < 0):
            CSSD_count += 1
    
    #normalise the CSSD count by N-2
    features[12] = CSSD_count / (len(lc[1])-2)
    
    #calculate the numerator for the Von Neumann ratio
    VN_num = 0
    for j in range(len(lc[1])-1):
        VN_num += (lc[1][j+1] - lc[1][j])**2
    features[13] = (VN_num/(len(lc[1])-1))/(std**2)
    
    if qpe == 1:
        features[14] = 1
    elif qpe == 0:
        features[14] = 0
    else:
        return print('Must select 0 - No QPEs or 1 - QPEs present.')
    
    return features