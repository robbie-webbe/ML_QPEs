#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:11:51 2023

@author: do19150
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from astropy.io import fits
sys.path.append(os.getcwd()[:-6]+'Analysis_Funcs/General/')
from qpe_pred import det_qpe_feats, cleanup

#load the saved 14 feature dt50, dt250 and dt1000 models
dt50_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt50')
dt250_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt250')
dt1000_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt1000')
dt50_prob_model = tf.keras.Sequential([dt50_model, tf.keras.layers.Softmax()])
dt250_prob_model = tf.keras.Sequential([dt250_model, tf.keras.layers.Softmax()])
dt1000_prob_model = tf.keras.Sequential([dt1000_model, tf.keras.layers.Softmax()])

#create empty lists for output
agn_num = []
srcids = []
obsids = []
src_nums = []
IAU_names = []
pn_exps = []
avg_rates = []
preds_dt50 = []
preds_dt250 = []
preds_dt1000 = []
features_dt50 = []
features_dt250 = []
features_dt1000 = []

#import the type classification catalogue
type_cat = fits.open('classification_DR12_with_input.fits')

#import the full XMMSSC catalogue
SSC_cat = fits.open('4XMMSSC/4XMM_DR13cat_v1.0.fits')


#iterate AGN by AGN
#find any instances of the AGN from the Tranin classification in the XMMSSC
agn_idxs = np.where(type_cat[1].data.field('prediction') == 4)[0]
k = 1
for i in agn_idxs:
    print(k,'/',len(agn_idxs))
    k += 1
    #pick out the source ID to crossmatch with the XMMSSC catalogue
    agn_srcid = type_cat[1].data[i][0]
    
    #find any entries in the XMMSSC
    SSC_idxs = np.where(SSC_cat[1].data.field('SRCID') == agn_srcid)[0]
    m = 1

    #for each of these lightcurves create predictions for QPEs
    for j in SSC_idxs:
        print(m,'/',len(SSC_idxs))
        m += 1
        #pickout the obs id, and the source number in that obs
        obsid = str(SSC_cat[1].data.field('OBS_ID')[j])
        #convert the source number to hexadecimal for the download script
        srcnum = hex(SSC_cat[1].data.field('SRC_NUM')[j]).upper().replace('X','')
        if len(srcnum) == 2:
            srcnum = '0'+srcnum
        elif len(srcnum) == 4:
            srcnum = srcnum[1:]
            
        agn_num.append(k)
        srcids.append(agn_srcid)
        obsids.append(obsid)
        src_nums.append(srcnum)
        IAU_names.append(SSC_cat[1].data.field('IAUNAME')[j])
        pn_exps.append(int(SSC_cat[1].data.field('PN_ONTIME')[j]))
            
        #create lightcurves and features for that detection
        #if there is a PN detection, process the lightcurves
        if SSC_cat[1].data.field('CCDPN')[j] != -32768 and SSC_cat[1].data.field('PN_ONTIME')[j] >= 10000:
            #determine the features from the lightcurves
            
            lc_50, lc_250, lc_1000, feature_list = det_qpe_feats(obsid,srcnum)
            
            if lc_50 == lc_250 == lc_1000 == feature_list == 0:
                avg_rates.append(-0)
                features_dt50.append(np.zeros(14,dtype=int))
                features_dt250.append(np.zeros(14,dtype=int))
                features_dt1000.append(np.zeros(14,dtype=int))
                preds_dt50.append(-0)
                preds_dt250.append(-0)
                preds_dt1000.append(-0)
                continue
            else:
                print("Features created successfully. Lightcurve will be classified.")
                
            feats_50 = feature_list[0]
            feats_250 = feature_list[1]
            feats_1000 = feature_list[2]
            
            #find the prediction for the two feature sets and NNs
            pred_50 = dt50_prob_model.predict(feats_50)[0]
            pred_250 = dt250_prob_model.predict(feats_250)[0]
            pred_1000 = dt1000_prob_model.predict(feats_1000)[0]
            
            #add the srcid, obsid and srcnum details to outputs
            avg_rates.append(np.mean(lc_1000.countrate))
            features_dt50.append(feats_50)
            features_dt250.append(feats_250)
            features_dt1000.append(feats_1000)
            preds_dt50.append(pred_50[1])
            preds_dt250.append(pred_250[1])
            preds_dt1000.append(pred_1000[1])
            
            #Then create an output plot for the lightcurve
            outfile_name = str(agn_srcid)+'_'+obsid+'_'+srcnum+'_pnX.pdf'
            fig, axs = plt.subplots(3,1,sharex=True)
            axs[0].plot(lc_50.time,lc_50.countrate,color='b')
            axs[1].plot(lc_250.time,lc_250.countrate,color='b')
            axs[2].plot(lc_1000.time,lc_1000.countrate,color='b')
            axs[1].set(ylabel='Count rate')
            axs[2].set(xlabel='Time (s)')
            fig.suptitle('SRCID '+str(agn_srcid)+' Observation '+obsid+' Source '+str(SSC_cat[1].data.field('SRC_NUM')[j])+' PN')
            fig.savefig('classified_QPE_preds/AGN_plots/'+outfile_name)
            plt.close()
        
        #if no PN lightcurve information
        else:
            print("No lightcurve available for this detection. CCD PN: "+str(SSC_cat[1].data.field('CCDPN')[j])+". PN ONTIME: "+str(SSC_cat[1].data.field('PN_ONTIME')[j])+".")
            avg_rates.append(-0)
            features_dt50.append(np.zeros(14,dtype=int))
            features_dt250.append(np.zeros(14,dtype=int))
            features_dt1000.append(np.zeros(14,dtype=int))
            preds_dt50.append(-0)
            preds_dt250.append(-0)
            preds_dt1000.append(-0)


#output the final information to a file
df = pd.DataFrame(columns=['SRCID','OBSID','SRCNUM','IAUNAME','PN_EXP','AVG_RATE','FEATS50','FEATS250','FEATS1000','PRED50','PRED250','PRED1000'])
df['SRCID'] = srcids
df['OBSID'] = obsids
df['SRCNUM'] = src_nums
df['IAUNAME'] = IAU_names
df['PN_EXP'] = pn_exps
df['AVG_RATE'] = avg_rates
df['FEATS50'] = features_dt50
df['FEATS250'] = features_dt250
df['FEATS1000'] = features_dt1000
df['PRED50'] = preds_dt50
df['PRED250'] = preds_dt250
df['PRED1000'] = preds_dt1000
df.to_csv('classified_QPE_preds/AGN_preds.csv',index=False)

