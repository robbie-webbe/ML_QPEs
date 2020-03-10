#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:48:18 2020

@author: do19150
"""

import numpy as np
import pandas as pd
import os
from astropy.io import fits
from SummaryStats1 import sumstatobs, sumstat

#determine the object whose lcs are to be loaded
AGN_name = input('Enter the name of the object whose lightcurves are to be loaded: ')

#point to that object in scratch5
directory = '/data/scratch5/Robbie/' + AGN_name + '/'

#capture the observations of that object on scratch5
obsids = os.listdir(directory)

no_obs = len(obsids)

#create a list with the locations of all processed lightcurves
files = []
for i in range(no_obs):
    files.append(directory + obsids[i] + '/PROC/PN_corr.fits')

features = np.zeros((len(files),9),dtype=object)

#for observation: open fits file, send to pandas DF, process data using sumstat.
for i in range(no_obs):
    hdul = fits.open(files[i])
    lc = (pd.DataFrame(hdul[1].data))['RATE']
    err = (pd.DataFrame(hdul[1].data))['ERROR']
    obs_name = AGN_name + '_' + obsids[i]
    features[i] = sumstatobs(obs_name,lc,err)

filename=AGN_name+'_Obs_Features_werr.csv'
np.savetxt(filename,features ,fmt=['%.2e','%.15e','%.15e','%.15e','%.15e','%.15e','%.15e','%.15e','%s'] ,delimiter=',')


#create a list with the locations of all processed lightcurves
files = []
for i in range(no_obs):
    files.append(directory + obsids[i] + '/PROC/PN_corr.fits')

features = np.zeros((len(files),9),dtype=object)

#for observation: open fits file, send to pandas DF, process data using sumstat.
for i in range(no_obs):
    hdul = fits.open(files[i])
    lc = (pd.DataFrame(hdul[1].data))['RATE']
    err = (pd.DataFrame(hdul[1].data))['ERROR']
    obs_name = AGN_name + '_' + obsids[i]
    features[i] = sumstat(lc,obs_id=obs_name)

filename=AGN_name+'_Obs_Features_woerr.csv'
np.savetxt(filename,features ,fmt=['%.2e','%.15e','%.15e','%.15e','%.15e','%.15e','%.15e','%.15e','%s'] ,delimiter=',')
