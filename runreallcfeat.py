#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:58:09 2022

@author: do19150
"""

from lcfeaturegen import lcfeat
import pandas as pd
import os
import sys
sys.path.append('/Users/do19150/Gits/Analysis_Funcs/General/')
sys.path.append('/home/do19150/Gits/Analysis_Funcs/General/')

from fitsloader import XMMtolc

#create a list of all obs to be profiled
dirlist = sorted(os.listdir('Obs/'))
obslist = []
for file in dirlist:
    if file.endswith('.lc'):
        obslist.append(file)
        
#initialise an output dataframe with columns for name, the 14 features and for QPE?
out_df = pd.DataFrame(columns=['ObsID','STD/Mean','Prop > 1STD','Prop > 2STD','Prop > 3STD',
                               'Prop > 4STD','Prop > 5STD','Prop > 6STD','IQR/STD','Skew',
                               'Kurtosis','Rev CCF','2nd ACF','CSSD','Von Neumann Ratio',
                               'QPE?'])

obsids = []
rb = float(input('Enter new time binning for lightcurves for features (Default 50s): '))
#for each observation inj the list
for i in range(len(obslist)):
    #extract the obsid
    obsid = obslist[i][0:10]
    #check if the lightcurve can be rebinned to the right length
    if rb:
        lc = XMMtolc('Obs/'+obslist[i],t_bin=10)
        try:
            lc.rebin(rb)
            obsids.append(obsid)
        except:
            print(obsid+' cannot be rebinned to that length.')
    
out_df['ObsID'] = obsids

print(obsids)

#for each observation in the list
for i in range(len(obsids)):
    print(obsids[i])
    
    #create a lightcurve for the observation, rebin it to 50s and zero-time
    try:
        lc = XMMtolc('Obs/'+obsids[i]+'_.2-2.0_t10_pn.lc',t_bin=10)
        lc = lc.shift(-lc.time[0]).rebin(rb)
        lc.plot()
    except:
        lc = XMMtolc('Obs/'+obsids[i]+'_.2-2.0_t10_pn_filt.lc',t_bin=10)
        lc = lc.shift(-lc.time[0]).rebin(rb)
        lc.plot()
    
    #set the first column to be the name of the obsid
    contains_qpe = int(input('Does this lightcurve contain a QPE? (1 - QPE. 0 - No QPE): '))
    out_df.iloc[i,1:] = lcfeat([lc.time,lc.counts],
                               qpe=contains_qpe)

if rb:
    out_df.to_csv('Features/realobs_test_data_dt'+str(rb)+'.csv',index=False)
else:
    out_df.to_csv('Features/realobs_test_data.csv',index=False)
