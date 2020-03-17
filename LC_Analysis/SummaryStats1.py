#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:31:43 2020

@author: do19150
"""

import numpy as np

def sumstat(lightcurve,label=0,obs_id='Sim'):
    '''
    Function designed to calculate the following summary statistics about
    a lightcurve for implementation with TensorFlow.
    0. Label - Does the curve contain a QPE or not? (To be defined at implementation)
    1. Proportion of points in curve which are above or below the mean.
    2. Standard deviaiton of the curve as a proportion of the mean value.
    3. Maximum deviation in the curve as a proportion of the standard deviation.
    4. Proportion of points in the curve who are at least 1 SD from the mean.
    5. Proportion of points in the curve who are at least 2 SD from the mean.
    6. Proportion of points in the curve who are at least 3 SD from the mean.
    7. Proportion of points in the curve who are at least 4 SD from the mean.
    
    The function takes a lightcurve of any length to return these statistics
    as a 1 x 8 array.
    '''
    
    mean = np.mean(lightcurve)
    stdev = np.std(lightcurve)
    maxdev = max(np.abs(lightcurve - mean))
    count_over = np.shape(np.where(lightcurve >= mean))[1]
    N = len(lightcurve)
    
    s1 = count_over/(N - count_over)
    s2 = stdev/mean
    s3 = maxdev/stdev
    s4 = (np.shape(np.where(lightcurve >= mean+stdev))[1] + np.shape(np.where(lightcurve <= mean-stdev))[1])/N
    s5 = (np.shape(np.where(lightcurve >= mean+2*stdev))[1] + np.shape(np.where(lightcurve <= mean-2*stdev))[1])/N
    s6 = (np.shape(np.where(lightcurve >= mean+3*stdev))[1] + np.shape(np.where(lightcurve <= mean-3*stdev))[1])/N
    s7 = (np.shape(np.where(lightcurve >= mean+4*stdev))[1] + np.shape(np.where(lightcurve <= mean-4*stdev))[1])/N
    
    return [label, s1, s2, s3, s4, s5, s6, s7, obs_id]

    
def sumstatobs(obsid,lightcurve,errors,label=0):
    '''
    Function designed to calculate the following summary statistics about
    a lightcurve for implementation with TensorFlow.
    0. Label - Does the curve contain a QPE or not? (To be defined at implementation)
    1. Proportion of points in curve which are above or below the mean.
    2. Standard deviaiton of the curve as a proportion of the mean value.
    3. Maximum deviation in the curve as a proportion of the standard deviation.
    4. Proportion of points in the curve who are at least 1 SD from the mean.
    5. Proportion of points in the curve who are at least 2 SD from the mean.
    6. Proportion of points in the curve who are at least 3 SD from the mean.
    7. Proportion of points in the curve who are at least 4 SD from the mean.
    
    The last column will give the name and observation ID of each lightcurve.
    The function takes a lightcurve of any length to return these statistics
    as a 1 x 8 array.
    '''
    
    mean = np.mean(lightcurve)
    stdev = np.sqrt(np.sum(errors**2))
    maxdev = max(np.abs(lightcurve - mean))
    count_over = np.shape(np.where(lightcurve >= mean))[1]
    N = len(lightcurve)
    
    s1 = count_over/(N - count_over)
    s2 = stdev/mean
    s3 = maxdev/stdev
    s4 = (np.shape(np.where(lightcurve >= mean+stdev))[1] + np.shape(np.where(lightcurve <= mean-stdev))[1])/N
    s5 = (np.shape(np.where(lightcurve >= mean+2*stdev))[1] + np.shape(np.where(lightcurve <= mean-2*stdev))[1])/N
    s6 = (np.shape(np.where(lightcurve >= mean+3*stdev))[1] + np.shape(np.where(lightcurve <= mean-3*stdev))[1])/N
    s7 = (np.shape(np.where(lightcurve >= mean+4*stdev))[1] + np.shape(np.where(lightcurve <= mean-4*stdev))[1])/N

    return [label, s1, s2, s3, s4, s5, s6, s7, obsid]
  
