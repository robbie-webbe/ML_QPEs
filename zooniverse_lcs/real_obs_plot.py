#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:15:39 2024

@author: rwebbe
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stingray as st
from astropy.io import fits
from astropy.table import Table

cat = Table.read(os.getcwd()[:-14] + '/4XMMSSC/4XMM_DR13cat_v1.0.fits')
cat_df = cat.to_pandas()
classes = fits.open(os.getcwd()[:-14] + '/classification_DR12_with_input.fits')

class_src_ids = classes[1].data.field('SRCID')[np.where((classes[1].data.field('prediction') == 0)|(classes[1].data.field('prediction') == 4))[0]]
#find only the detections in DR13 which match the old classifications from DR12
class_idxs = cat_df[cat_df['SRCID'].isin(class_src_ids)].index

#find only the observations where the source is variable and there is at least 50ks of PN exposure time and no flag
var_50ks_exp_idxs = np.where((cat['VAR_FLAG'][class_idxs] == True)&(cat['PN_ONTIME'][class_idxs]<=50000)&(cat['PN_ONTIME'][class_idxs]>=20000)&(cat['TSERIES'][class_idxs] == True)&
                             (cat['SUM_FLAG'][class_idxs]<=3)&(cat['CCDPN'][class_idxs] != -32768))[0]

# # #find the count rates for these detections
# rates = cat['PN_8_RATE'][var_50ks_exp_idxs]
#
# #sort the rates and pick the no_lcs th from the top value. Subtract a tiny amount to ensure it is included
# rate_cut = sorted(rates)[-no_lcs] - 0.00001
#
# #pick the top no_lcs highest rate detections
# det_idxs = var_50ks_exp_idxs[np.where(rates >= rate_cut)[0]]

# create a set of indices for the objects which meet all criteria
det_idxs = class_idxs[var_50ks_exp_idxs]

print(len(det_idxs))

filenames = []
avg_rates = []
exposures = []
frac_vars = []
chi2probs = []

#for each detection
for i in det_idxs:

    print(cat['SRCID'][i])
    #pick out the obsid
    obsid = str(cat['OBS_ID'][i])
    
    #pick out and convert the src_num to an appropriate hexadecimal for location in archives
    srcnum = hex(cat['SRC_NUM'][i]).upper().replace("X", "")
    if len(srcnum) == 2:
        srcnum = '0'+srcnum
    elif len(srcnum) == 4:
        srcnum = srcnum[1:]
        
    print(obsid, srcnum)
    
    outfile_name = obsid+'_'+srcnum+'.png'
        
    #if working locally then find the appropriate file on the server
    
    
    #if working remotely then identify the file in the archive and download for processing
    pn_url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+obsid+"&sourceno="+srcnum+"&level=PPS&instname=PN&extension=FTZ&name=SRCTSR"
    os.system('wget -O _dl_temp_/source_pnlc.tar "'+pn_url+'"')
    
    #extract the downloaded file
    if os.path.getsize('_dl_temp_/source_pnlc.tar') == 0:
        os.system('rm _dl_temp_/source_pnlc.tar')
        continue
    else:
        os.system('tar -xvf _dl_temp_/source_pnlc.tar')
    
    #open the time series data
    lc_hdu = fits.open(obsid+'/pps/P'+obsid+'PNX000SRCTSR8'+srcnum+'.FTZ',memmap=False)
    
    time = lc_hdu[1].data.field('TIME')
    rate = lc_hdu[1].data.field('RATE')
    error = lc_hdu[1].data.field('ERROR')
    gtis = []
    max_gti_length = 0
    dt = lc_hdu[1].data.field('TIMEDEL')[0]
    
    for j in lc_hdu[2].data:
        gtis.append([j[0],j[1]])
        if j[1] - j[0] > max_gti_length:
            max_gti_length = j[1] - j[0]

    if max_gti_length <= 250:
        lc_hdu.close()
        os.system('rm _dl_temp_/source_pnlc.tar')
        os.system('rm -r ' + obsid + '/')
        continue
    
    #pick out all good time points on their rate and error values
    good_rates = np.where((np.isfinite(rate))&(np.isfinite(error)))[0]
    
    #create the lightcurve
    lc = st.Lightcurve(time[good_rates],dt*rate[good_rates],err=dt*error[good_rates],gti=gtis)
    
    #rebin to 250s with GTIs imposed, and shift to zero time
    lc.apply_gtis()
    lc = lc.shift(-lc.time[0])
    lc = lc.rebin(250)


    #plot the time series data
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(13.89, 8.33))
    plt.errorbar(lc.time/1000, lc.countrate - np.average(lc.countrate), xerr=0.125, yerr=lc.countrate_err, color='black',ls='none')
    plt.xlabel('Time (ks)',fontsize=24)
    plt.ylabel('X-ray Brightness',fontsize=24)
    plt.savefig('real_obs_plots/'+outfile_name)
    plt.close('all')
    
    filenames.append(outfile_name)
    avg_rates.append(lc.meanrate)
    exposures.append(lc.tseg)
    frac_vars.append(cat['PN_FVAR'][i])
    chi2probs.append(cat['PN_CHI2PROB'][i])
    
    #remove any remainder files
    lc_hdu.close()
    os.system('rm _dl_temp_/source_pnlc.tar')
    os.system('rm -r '+obsid+'/')
    
#write file information to output
out_df = pd.DataFrame(columns = ['Filename','Avg. Rate','PN Exp.','Variance','Chi2 Prob.'],dtype=object)
out_df['Filename'] = filenames
out_df['Avg. Rate'] = avg_rates
out_df['PN Exp.'] = exposures
out_df['Variance'] = frac_vars
out_df['Chi2 Prob.'] = chi2probs

out_df.to_csv('real_obs_file_info_20ks.csv',index=False)
