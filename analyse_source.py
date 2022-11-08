#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:00:42 2022

@author: do19150
"""

import os
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import time

from stingray import Lightcurve
from lcfeaturegen import lcfeat

#pick the obs and src id from input
obsid = sys.argv[1]
srcno = sys.argv[2]

#download the observation
url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+str(obsid)+"&sourceno="+str(srcno)+"&level=PPS&instname=PN&extension=FTZ&name=SRCTSR"
os.system('wget -O source_lc.tar "'+url+'"')
os.system('tar -xvf source_lc.tar')
os.system('rm source_lc.tar')

while not os.path.exists(os.getcwd()+'/'+str(obsid)+'/pps/'):
    time.sleep(1)
    print('Download pending.')
    
print('Download complete')

files = os.listdir(os.getcwd()+'/'+str(obsid)+'/pps/')

#for each time series which has been downloaded
for file in files:
    print(file)
    hdul = fits.open(os.getcwd()+'/'+str(obsid)+'/pps/'+file)
        
    #pick out start and end times and check if the exposure is long enough
    tstart = hdul[1].data.field('TIME')[0]
    tstop = hdul[1].data.field('TIME')[-1]
    
    if tstop-tstart < 10000:
        print('The exposure of this file is not long enough for a good quality analysis.',file)
        
    else:
        #if long enough then pick out the times and rates
        times = hdul[1].data.field('TIME')
        rate = hdul[1].data.field('RATE')
        #correct any negative rate values
        negative_rate = np.where(rate < 0)
        rate[negative_rate] = 0.0
        #remove any points which fall outside the GTIs
        good_indices = np.where(np.isnan(rate) == False)[0]
        #create a well behaved lightcurve
        lc = Lightcurve(times[good_indices],rate[good_indices],gti=hdul[2].data,input_counts=False,skip_checks=True)
        #rebin the lightcurve to get it at a minimum of 50s
        lc = lc.rebin(50)
        
        plt.plot(lc.time,lc.countrate)
        plt.show()
        
        #determine the features of the source in question
        srcfeats = lcfeat([lc.time,lc.countrate],qpe='?')
        print(obsid,srcno,srcfeats)
        

        


