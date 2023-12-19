#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:03:15 2023

@author: do19150
"""

import numpy as np
from stingray import Lightcurve
import os
from astropy.io import fits
from lcfeaturegen import lcfeat
#import pdb

def cleanup(obsid):
    '''
    Function to clean up after downloaded files.
    '''
    hdul = 0
    os.system('rm -r _dl_temp_/'+obsid+'/')
    os.system('rm _dl_temp_/source_lc.tar')
    
    

def det_qpe_feats(obsid,src_num,inst='PN',full_band=False):
    '''
    Function which downloads lightcurves for a given source number within an observation and
    creates features which can then be used for prediction as to whether a lightcurve contains
    quasi-periodic eruptions.

    Parameters
    ----------
    obsid : The obsid of the detection being considered.
    src_num : The source number within the given observation.
    inst: Instrument for which a lightcurve is needed. Options are PN, M1 and M2.
    full_band: Should the features be used for the full XMM 0.2-12.0keV band of 0.2-2.0keV soft band.

    Returns
    -------
    lc50 : Stingray Lightcurve object binned at 50s
    lc250 : Stingray Lightcurve object binned at 250s
    lc1000 : Stingray Lightcurve object binned at 1000s
    features: The values of the 14 features for the lightcurves as a list of arrays [feats50, feats250, feats1000]

    '''

    
    #create the url for the download
    url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+obsid+"&sourceno="+src_num+"&level=PPS&instname="+inst+"&extension=FTZ&name=SRCTSR"
    
    #try and download data
    os.system('wget -O _dl_temp_/source_lc.tar "'+url+'"')
    
    #if the file is non-zero in size
    if os.path.getsize(os.getcwd()+'/_dl_temp_/source_lc.tar') != 0:
        #extract the files from the zipped archive
        os.chdir('_dl_temp_/')
        os.system('tar -xf source_lc.tar')
        os.chdir(os.getcwd()[:-10])
        
        #load the 'X' file which will contain the count rates split by energy binning
        dlfile_list = os.listdir('_dl_temp_/'+obsid+'/pps/')
        filename = []
        for file in dlfile_list:
            if 'PNX' in file:
                filename.append(file)
        hdul = fits.open('_dl_temp_/'+obsid+'/pps/'+filename[0])
        
        #extract the timestamps for the lightcurve
        time = hdul['RATE'].data.field('TIME')
        
        #extract the GTI information
        gtis = []
        for i in hdul['SRC_GTIS'].data:
            gtis.append([i[0],i[1]])
        
        #extract the appropriate rate information
        if full_band:
            rate = hdul['RATE'].data.field('RATE')
        else:
            rate1 = hdul['RATE'].data.field('RATE1')
            rate2 = hdul['RATE'].data.field('RATE2')
            rate3 = hdul['RATE'].data.field('RATE3')
            rate = rate1 + rate2 + rate3
            
        #mask any invalid count rate values
        valid_idxs = np.isfinite(rate)
            
        #create the lightcurve with GTIs, and apply the GTIs
        base_lc = Lightcurve(time[valid_idxs],rate[valid_idxs],input_counts=False,gti=gtis)
        base_lc.apply_gtis()
        #shift the lightcurve to zero time at the start
        base_lc = base_lc.shift(-base_lc.time[0])
        
        #check if the original binning is greater than 50s, if so move on to next source
        if base_lc.dt > 50.0:
            cleanup(obsid)
            return print("Lightcurve time binning is too long to be rebinned.")
        #check the lightcurve is at least 10ks in length before trunctaing
        if base_lc.tseg < 10000:
            cleanup(obsid)
            return print("Lightcurve time binning is too short to be analysed")
        
        
        #check if truncating for the first and last 5ks will remove any gtis
        if base_lc.gti[-1][1] < 5000:
            cleanup(obsid)
            return print("Lightcurve trunctation would remove any GTIs")
        if base_lc.gti[0][0] > (base_lc.time[-1] - 5000):
            cleanup(obsid)
            return print("Lightcurve trunctation would remove any GTIs")
        
        #pdb.set_trace()
        #try and truncate the lightcurve
        trunc_start = 5000
        trunc_stop = base_lc.time[-1] - 5000
        try:
            base_lc = base_lc.truncate(start=trunc_start,stop=trunc_stop,method='time')
        except:
            cleanup(obsid)
            return print("Lightcurve trunctation cannot be performed.")
        else:
            base_lc = base_lc.truncate(start=trunc_start,stop=trunc_stop,method='time')
        
        #try and rebin the lightcurves to the three rates needed
        try:
            lc50 = base_lc.rebin(50)
        except:
            cleanup(obsid)
            return print("Lightcurve cannot be appropriately rebinned at 50s. Moving on to next source.")
        else:
            lc50 = base_lc.rebin(50)
            try:
                lc250 = base_lc.rebin(250)
            except:
                cleanup(obsid)
                return print("Lightcurve cannot be appropriately rebinned at 250s. Moving on to next source.")
            else:
                lc250 = base_lc.rebin(250)
                try:
                    lc1000 = base_lc.rebin(1000)
                except:
                    cleanup(obsid)
                    return print("Lightcurve cannot be appropriately rebinned at 1ks. Moving on to next source.")
                else:
                    lc1000 = base_lc.rebin(1000)
            
        #mask any negative count rates in any of the three lightcurves
        mask50_idxs = np.where(lc50.countrate >= 0)
        mask250_idxs = np.where(lc250.countrate >= 0)
        mask1000_idxs = np.where(lc1000.countrate >= 0)
        
        t50 = lc50.time[mask50_idxs]
        r50 = lc50.countrate[mask50_idxs]
        t250 = lc250.time[mask250_idxs]
        r250 = lc250.countrate[mask250_idxs]
        t1000 = lc1000.time[mask1000_idxs]
        r1000 = lc1000.countrate[mask1000_idxs]
        
        #do a final check for very low countrates
        if np.mean(r50) < 0.005 or np.mean(r250) < 0.005 or np.mean(r1000) < 0.005:
            cleanup(obsid)
            return print("Count rate is too low to not be affected by Poissonian noise. Moving on to next source.")
        
        lc50 = Lightcurve(t50,r50,input_counts=False,gti = lc50.gti)
        lc250 = Lightcurve(t250,r250,input_counts=False,gti = lc250.gti)
        lc1000 = Lightcurve(t1000,r1000,input_counts=False,gti = lc1000.gti)
        
        #determine the features from the lightcurves
        feats_50 = np.asarray([list(lcfeat([lc50.time,lc50.countrate],qpe='?'))])
        feats_250 = np.asarray([list(lcfeat([lc250.time,lc250.countrate],qpe='?'))])
        feats_1000 = np.asarray([list(lcfeat([lc1000.time,lc1000.countrate],qpe='?'))])
    
    #if the file does not exist, exit and move on
    else:
        return print("The lightcurve data could not be downloaded.")
    
    #output the features to the appropriate output list of arrays
    features = [feats_50,feats_250,feats_1000]
                    
    cleanup(obsid)
    
    return lc50, lc250, lc1000, features





def src_qpe_feats(srcid,inst='PN',full_band=False):
    '''
    Function which downloads all lightcurves for a given source ID, creates a long-term lightcurve
    and profiles that for the existence of QPEs

    Parameters
    ----------
    srcid: Unique source ID as per the 4XMM catalogue.
    inst: Instrument for which a lightcurve is needed. Options are PN, M1 and M2.
    full_band: Should the features be used for the full XMM 0.2-12.0keV band of 0.2-2.0keV soft band.

    Returns
    -------
    features: The values of the 14 features for the lightcurves as a list of arrays [feats50, feats250, feats1000]
    predictions: The QPE probabilities for the lightcurves at the three time binnings.

    '''
    
    features = 0
    
    return features
    
    