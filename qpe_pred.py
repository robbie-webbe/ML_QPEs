#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:03:15 2023

@author: do19150
"""

import numpy as np
from stingray import Lightcurve
import os
import requests
import tarfile
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

    
def cleanup_LT():
    hdul = 0
    os.system('rm /Users/do19150/Gits/ML_QPE/_dl_temp_/LT_temp/source_lc.tar')
    obs_list = os.listdir('/Users/do19150/Gits/ML_QPE/_dl_temp_/LT_temp/')
    for directory in obs_list:
        os.system('rm -r /Users/do19150/Gits/ML_QPE/_dl_temp_/LT_temp/'+directory)
    
    

def det_qpe_feats(obsid,src_num,inst='PN',full_band=False,po_noiselim=True):
    '''
    Function which downloads lightcurves for a given source number within an observation and
    creates features which can then be used for prediction as to whether a lightcurve contains
    quasi-periodic eruptions.

    Parameters
    ----------
    obsid : The obsid of the detection being considered.
    src_num : The source number within the given observation.
    inst : Instrument for which a lightcurve is needed. Options are PN, M1 and M2.
    full_band : Should the features be used for the full XMM 0.2-12.0keV band of 0.2-2.0keV soft band.
    poisson_noise_limit : Include a cut to remove lightcurves with very low count rates?

    Returns
    -------
    lc50 : Stingray Lightcurve object binned at 50s
    lc250 : Stingray Lightcurve object binned at 250s
    lc1000 : Stingray Lightcurve object binned at 1000s
    features : The values of the 14 features for the lightcurves as a list of arrays [feats50, feats250, feats1000]

    '''

    
    #create the url for the download
    url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+obsid+"&sourceno="+src_num+"&level=PPS&instname="+inst+"&extension=FTZ&name=SRCTSR"
    
    #try and download data
    try:
        dl_dets = requests.get(url, timeout=(10,600))
    except requests.exceptions.ReadTimeout:
        try:
            #if it fails give it a second go
            dl_dets = requests.get(url, timeout=(10,6000))
        except requests.exceptions.ReadTimeout:
            print("There was an error while downloading the data.")
            return 0,0,0,0
        else:
            dl_dets = requests.get(url, timeout=(10,6000))
    else:
        dl_dets = requests.get(url, timeout=(10,600))
        
    #if the download is good then save to file
    if dl_dets.status_code == 200:
        with open('_dl_temp_/source_lc.tar','wb') as saved_file:
            saved_file.write(dl_dets.content)
        
    else:
        print("The lightcurve for this observation cannot be downloaded.")
        return 0,0,0,0
        
    
    #if the file is non-zero in size
    if os.path.getsize(os.getcwd()+'/_dl_temp_/source_lc.tar') != 0:
        
        #extract the needed 'PNX' file from the zipped archive
        tar = tarfile.open(os.getcwd()+'/_dl_temp_/source_lc.tar')
        lc_file = 0
        for file in tar.getnames():
            if 'PNX' in file:
                lc_file = file
                
        tar.extract(lc_file,path='_dl_temp_/')
        tar.close()
        
        #open the fits file within a context
        with fits.open('_dl_temp_/'+lc_file) as hdul:
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
        base_lc = Lightcurve(time[valid_idxs],rate[valid_idxs],input_counts=False,gti=gtis,skip_checks=True)
        base_lc.apply_gtis()
        #shift the lightcurve to zero time at the start
        base_lc = base_lc.shift(-base_lc.time[0])
        
        #check if the original binning is greater than 50s, if so move on to next source
        if base_lc.dt > 50.0:
            cleanup(obsid)
            print("Lightcurve time binning is too long to be rebinned.")
            return 0,0,0,0
        #check the lightcurve is at least 10ks in length before trunctaing
        if base_lc.tseg < 10000:
            cleanup(obsid)
            print("Lightcurve is too short to be analysed")
            return 0,0,0,0
        
        
        #check if truncating for the first and last 5ks will remove any gtis
        if base_lc.gti[-1][1] < 5000 or base_lc.gti[0][0] > (base_lc.time[-1] - 5000):
            cleanup(obsid)
            print("Lightcurve trunctation would remove any GTIs")
            return 0,0,0,0
        
        #pdb.set_trace()
        #try and truncate the lightcurve
        trunc_start = 5000
        trunc_stop = base_lc.time[-1] - 5000
        try:
            base_lc = base_lc.truncate(start=trunc_start,stop=trunc_stop,method='time')
        except:
            cleanup(obsid)
            print("Lightcurve trunctation cannot be performed.")
            return 0,0,0,0
        else:
            base_lc = base_lc.truncate(start=trunc_start,stop=trunc_stop,method='time')
        
        #try and rebin the lightcurves to the three rates needed
        try:
            lc50 = base_lc.rebin(50)
        except:
            cleanup(obsid)
            print("Lightcurve cannot be appropriately rebinned at 50s. Moving on to next source.")
            return 0,0,0,0
        else:
            lc50 = base_lc.rebin(50)
            try:
                lc250 = base_lc.rebin(250)
            except:
                cleanup(obsid)
                print("Lightcurve cannot be appropriately rebinned at 250s. Moving on to next source.")
                return 0,0,0,0
            else:
                lc250 = base_lc.rebin(250)
                try:
                    lc1000 = base_lc.rebin(1000)
                except:
                    cleanup(obsid)
                    print("Lightcurve cannot be appropriately rebinned at 1ks. Moving on to next source.")
                    return 0,0,0,0
                else:
                    lc1000 = base_lc.rebin(1000)
            
        #mask any negative count rates in any of the three lightcurves
        lc50.counts[lc50.counts<0] = 0
        lc250.counts[lc250.counts<0] = 0
        lc1000.counts[lc1000.counts<0] = 0
        
        #do a final check for very low countrates
        if po_noiselim == True:
            if np.mean(lc50.counts) < 0.005 or np.mean(lc250.counts) < 0.005 or np.mean(lc1000.counts) < 0.005:
                cleanup(obsid)
                print("Count rate is too low to not be affected by Poissonian noise. Moving on to next source.")
                return 0,0,0,0
        
        #determine the features from the lightcurves
        feats_50 = np.asarray([list(lcfeat([lc50.time,lc50.countrate],qpe='?'))])
        feats_250 = np.asarray([list(lcfeat([lc250.time,lc250.countrate],qpe='?'))])
        feats_1000 = np.asarray([list(lcfeat([lc1000.time,lc1000.countrate],qpe='?'))])
    
    #if the file does not exist, exit and move on
    else:
        print("The downloaded lightcurve data is 0KB in size. Moving on.")
        return 0,0,0,0
    
    #output the features to the appropriate output list of arrays
    features = [feats_50,feats_250,feats_1000]
                    
    cleanup(obsid)
    
    return lc50, lc250, lc1000, features





def src_qpe_feats(obsid_list,srcnum_list,inst='PN',full_band=False):
    '''
    Function which downloads all lightcurves for a given source, identified as a list of
    source numbers for a list of observation IDs, creates a long-term lightcurve and 
    creates features which can then be used for prediction as to whether the long-term 
    lightcurve contains quasi-periodic eruptions.

    Parameters
    ----------
    obsids : List of OBSIDs for the source in question
    srcnum_list : List of source numbers within those observations. Must be same size as obsids list.
    inst : Instrument for which a lightcurve is needed. Options are PN, M1 and M2.
    full_band : Should the features be used for the full XMM 0.2-12.0keV band of 0.2-2.0keV soft band.

    Returns
    -------
    lc50 : Stingray Lightcurve object binned at 50s
    lc250 : Stingray Lightcurve object binned at 250s
    lc1000 : Stingray Lightcurve object binned at 1000s
    features : The values of the 14 features for the lightcurves as a list of arrays [feats50, feats250, feats1000]

    '''
    
    lc_list = []
    
    #for each observation and src num combination
    for i in range(len(obsid_list)):
        obsid = obsid_list[i]
        src_num = srcnum_list[i]
    
        #download the data
        url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+obsid+"&sourceno="+src_num+"&level=PPS&instname="+inst+"&extension=FTZ&name=SRCTSR"
    
        #unpack and untar the observation details
        os.system('wget -O _dl_temp_/LT_temp/source_lc.tar "'+url+'" --timeout=300')
        if os.path.getsize(os.getcwd()+'/_dl_temp_/LT_temp/source_lc.tar') == 0:
            return print("The lightcurve data could not be downloaded.")
            continue
        
        os.chdir('_dl_temp_/LT_temp/')
        os.system('tar -xf source_lc.tar')
        os.chdir(os.getcwd()[:-17])
        
        #load the 'X' file which will contain the count rates split by energy binning
        dlfile_list = os.listdir('_dl_temp_/LT_temp/'+obsid+'/pps/')
        filename = []
        for file in dlfile_list:
            if 'PNX' in file:
                filename.append(file)
        hdul = fits.open('_dl_temp_/LT_temp/'+obsid+'/pps/'+filename[0])
        
        #load each lightcurve to a Lightcurve
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
            
        #create the lightcurve with GTIs
        base_lc = Lightcurve(time[valid_idxs],rate[valid_idxs],input_counts=False,gti=gtis)
        base_lc.apply_gtis()
        
        #check if the original binning is greater than 50s, if so move on to next source
        if base_lc.dt > 50.0:
            cleanup_LT()
            return print("Lightcurve time binning is too long to be rebinned for at least one observation.")
        #check the lightcurve is at least 10ks in length before trunctaing
        if base_lc.tseg < 10000:
            cleanup_LT()
            return print("Lightcurve length is too short to be analysed for at least one observation")
        
        
        #check if truncating for the first and last 5ks will remove any gtis
        if base_lc.gti[-1][1] < base_lc.time[0] + 5000:
            cleanup_LT()
            return print("Lightcurve trunctation would remove any GTIs for at least one observation.")
        if base_lc.gti[0][0] > (base_lc.time[-1] - 5000):
            cleanup_LT()
            return print("Lightcurve trunctation would remove any GTIs for at least one observation.")
        
        #pdb.set_trace()
        #try and truncate the lightcurve
        trunc_start = base_lc.time[0] + 5000
        trunc_stop = base_lc.time[-1] - 5000
        try:
            base_lc = base_lc.truncate(start=trunc_start,stop=trunc_stop,method='time')
        except:
            cleanup_LT()
            return print("Lightcurve trunctation cannot be performed for at least one observation.")
        else:
            base_lc = base_lc.truncate(start=trunc_start,stop=trunc_stop,method='time')
            
        lc_list.append(base_lc)
    
    #rebin all to common time binning, of 50s, 250s, 1000s
    lc_list_50s = []
    lc_list_250s = []
    lc_list_1000s = []
    for lc in lc_list:
        try:
            lc50 = lc.rebin(50)
            lc250 = lc.rebin(250)
            lc1000 = lc.rebin(1000)
        except:
            print("One or more lightcurves cannot be rebinned at 50s, 250s or 1ks. Moving on to next observation...")
            continue
        else:
            lc50 = lc.rebin(50)
            lc250 = lc.rebin(250)
            lc1000 = lc.rebin(1000)
            lc_list_50s.append(lc50)
            lc_list_250s.append(lc250)
            lc_list_1000s.append(lc1000)
    
            
    #check how many lightcurves could be rebinned to 50s
    if len(lc_list_50s) < 2:
        cleanup_LT()
        return print("There are not enough lightcurves binned at 50s to continue.")
    
    #join all lightcurves together
    full_lc50 = lc_list_50s[0]
    full_lc250 = lc_list_250s[0]
    full_lc1000 = lc_list_1000s[0]
    for k in range(len(lc_list_50s)-1):
        full_lc50 = full_lc50.join(lc_list_50s[k+1])
        full_lc250 = full_lc250.join(lc_list_250s[k+1])
        full_lc1000 = full_lc1000.join(lc_list_1000s[k+1])
        
    #apply gti filtering to the full lightcurve and shift to zero timing
    full_lc50 = full_lc50.shift(-full_lc50.time[0])
    full_lc250 = full_lc250.shift(-full_lc250.time[0])
    full_lc1000 = full_lc1000.shift(-full_lc1000.time[0])
    
    #sort the lightcurves incase the segments were downloaded out of order
    full_lc50.sort(inplace=True)
    full_lc250.sort(inplace=True)
    full_lc1000.sort(inplace=True)

    #mask any negative count rates in any of the three lightcurves
    mask50_idxs = np.where(full_lc50.counts >= 0)
    mask250_idxs = np.where(full_lc250.counts >= 0)
    mask1000_idxs = np.where(full_lc1000.counts >= 0)
    
    t50 = full_lc50.time[mask50_idxs]
    r50 = full_lc50.counts[mask50_idxs]
    t250 = full_lc250.time[mask250_idxs]
    r250 = full_lc250.counts[mask250_idxs]
    t1000 = full_lc1000.time[mask1000_idxs]
    r1000 = full_lc1000.countrate[mask1000_idxs]
    
    lc50 = Lightcurve(t50,r50,gti = full_lc50.gti)
    lc250 = Lightcurve(t250,r250,gti = full_lc250.gti)
    lc1000 = Lightcurve(t1000,r1000,gti = full_lc1000.gti)
    
    #determine the features from the lightcurves
    feats_50 = np.asarray([list(lcfeat([lc50.time,lc50.counts],qpe='?'))])
    feats_250 = np.asarray([list(lcfeat([lc250.time,lc250.counts],qpe='?'))])
    feats_1000 = np.asarray([list(lcfeat([lc1000.time,lc1000.counts],qpe='?'))])
    
    features = [feats_50,feats_250,feats_1000]
    cleanup_LT()
    
    return lc50, lc250, lc1000, features
    
    