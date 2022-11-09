#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:08:07 2022

@author: do19150
"""

import os
from astropy.io import fits

cat = fits.open(os.getcwd()+'/4XMMSSC/4XMM_DR12cat_v1.0.fits')

detids = cat[1].data.field('DETID')

for i in range(100):
    
    obsid = str(cat[1].data.field('OBS_ID')[i])
    srcnum = hex(cat[1].data.field('SRC_NUM')[i]).upper().replace('X','')
    if len(srcnum) == 2:
        srcnum = '0'+srcnum
    elif len(srcnum) == 4:
        srcnum = srcnum[1:]
        
    
    print(obsid,srcnum)
    
    if cat[1].data.field('CCDPN')[i] != -32768 :
        if cat[1].data.field('TSERIES')[i]:
            exposure_length = (cat[1].data.field('MJD_STOP')[i] - cat[1].data.field('MJD_START')[i])*24*3600
            print(exposure_length)
            if exposure_length < 10000:
                print('This source was not detected for long enough.',exposure_length)
            else:
                run_string = 'python analyse_source.py '+obsid+' '+srcnum + ' PN'
                os.system(run_string)
                os.system('rm -r '+obsid+'/')
            
        else:
            print('There is no PN time series for this detection.')
        
    else:
        print('There is no EPIC pn data for this detection.')
        
    if cat[1].data.field('CCDPN')[i] != -32768 :
        if cat[1].data.field('TSERIES')[i]:
            exposure_length = (cat[1].data.field('MJD_STOP')[i] - cat[1].data.field('MJD_START')[i])*24*3600
            print(exposure_length)
            if exposure_length < 10000:
                print('This source was not detected for long enough.',exposure_length)
            else:
                run_string = 'python analyse_source.py '+obsid+' '+srcnum + ' M1'
                os.system(run_string)
                os.system('rm -r '+obsid+'/')
            
        else:
            print('There is no MOS1 time series for this detection.')
        
    else:
        print('There is no EPIC MOS1 data for this detection.')

    