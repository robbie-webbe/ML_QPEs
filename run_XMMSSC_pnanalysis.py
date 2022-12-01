#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:08:07 2022

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
from fitsloader import XMMtolc
from lcfeaturegen import lcfeat

#load the XMMSSC catalogue
cat = fits.open(os.getcwd()+'/4XMMSSC/4XMM_DR12cat_v1.0.fits')

#load the saved 14 feature dt250 and dt1000 models
dt250_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt250')
dt1000_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt1000')
dt250_prob_model = tf.keras.Sequential([dt250_model, tf.keras.layers.Softmax()])
dt1000_prob_model = tf.keras.Sequential([dt1000_model, tf.keras.layers.Softmax()])

#create empty lists for wip
preds_dt250 = []
preds_dt1000 = []
top_cands = []

#determine which detections have time series that are long enough
indices = list(np.where((cat[1].data.field('TSERIES') == True)&(cat[1].data.field('EP_ONTIME')>=50000))[0])
no_objs = len(indices)

for i in range(no_objs):
    print(i,'/',no_objs)
    index = indices[i]
    
    #pick out the source ID for collecting info on individual objects
    srcid = str(cat[1].data.field('SRCID')[index])
    #pickout the obs id, and the source number in that obs
    obsid = str(cat[1].data.field('OBS_ID')[index])
    #convert the source number to hexadecimal for the download script
    srcnum = hex(cat[1].data.field('SRC_NUM')[index]).upper().replace('X','')
    if len(srcnum) == 2:
        srcnum = '0'+srcnum
    elif len(srcnum) == 4:
        srcnum = srcnum[1:]
        
    #if there is a PN detection, try and download the data
    if cat[1].data.field('CCDPN')[i] != -32768 :
        #create a url for the pn data if it does exist
        pn_url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+obsid+"&sourceno="+srcnum+"&level=PPS&instname=PN&extension=FTZ&name=SRCTSR"
        #try and download the data
        os.system('wget -O _dl_temp_/source_pnlc.tar "'+pn_url+'"')
        
        #if the file is non-zero in size
        if os.path.getsize(os.getcwd()+'/_dl_temp_/source_pnlc.tar') != 0:
            #extract the files from the zipped archive
            os.chdir('_dl_temp_/')
            os.system('tar -xf source_pnlc.tar')
            os.chdir(os.getcwd()[:-10])
            
            #for each downloaded lc, load the lightcurve to a stingray lc object
            for file in os.listdir('_dl_temp_/'+obsid+'/pps/'):
                lc = XMMtolc('_dl_temp_/'+obsid+'/pps/'+file)
                
                try:
                    lc == last_lc
                except:
                    pass
                else:
                    if lc == last_lc:
                        continue
                    
                last_lc = lc
                
                #if the time binning for the pn lightcurve is greater than 250s then move on to the next file
                if lc.dt > 250:
                    continue
                #if the length of the pn curve is less than 50ks then move to the next file
                if lc.tseg < 50000:
                    continue
                
                #zero time the lightcurve and its gtis
                lc = lc.shift(-lc.time[0])
                
                #check there are still valid gtis
                try:
                    (lc.gti == []).all()
                except:
                    pass
                else:
                    if (lc.gti == []).all():
                        continue
                
                #remove the first and last 15ks
                try:
                    lc = lc.truncate(start=15000,stop=(lc.time[-1]-15000),method='time')
                except:
                    continue
                else:
                    lc = lc.truncate(start=15000,stop=(lc.time[-1]-15000),method='time')
                
                #rebin the lightcurves to make them the right length
                try:
                    lc_250 = lc.rebin(250)
                except:
                    continue
                else:
                    lc_250 = lc.rebin(250)
                try:
                    lc_1000 = lc.rebin(1000)
                except:
                    continue
                else:
                    lc_1000 = lc.rebin(1000)
                                
                #determine the features from the lightcurves
                feats_250 = np.asarray([list(lcfeat([lc_250.time,lc_250.countrate],qpe='?'))])
                feats_1000 = np.asarray([list(lcfeat([lc_1000.time,lc_1000.countrate],qpe='?'))])
                
                #find the prediction for the two feature sets and NNs
                pred_250 = dt250_prob_model.predict(feats_250)[0]
                pred_1000 = dt1000_prob_model.predict(feats_1000)[0] 
                
                preds_dt250.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
                                    'PN'+file[13],pred_250[1]])
                preds_dt1000.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
                                    'PN'+file[13],pred_1000[1]])
                
                #if both predictions are greater than 90% QPE then the details to
                #the strong candidate df output array
                if pred_250[1] > 0.8 and pred_1000[1] > 0.8:
                    top_cands.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
                                        'PN'+file[13],pred_250[1],pred_1000[1]])
                    #and save the plots to a folder
                    outfile_name = srcid+'_'+obsid+'_'+srcnum+'_pn'+file[13]+'.pdf'
                    fig, axs = plt.subplots(2,1,sharex=True)
                    axs[0].plot(lc_250.time,lc_250.countrate,color='b')
                    axs[1].plot(lc_1000.time,lc_1000.countrate,color='b')
                    axs[0].set(ylabel='Count rate')
                    axs[1].set(xlabel='Time (s)',ylabel='Count rate')
                    fig.suptitle('SRCID '+srcid+' Observation '+obsid+' Source '+str(cat[1].data.field('SRC_NUM')[index])+' PN')
                    if pred_250[1] >= 0.99 and pred_1000[1] >= 0.99:
                        fig.savefig('4XMMSSC/top_cand_plots/conf_99/'+outfile_name)
                    elif pred_250[1] >= 0.95 and pred_1000[1] >= 0.95:
                        fig.savefig('4XMMSSC/top_cand_plots/conf_95/'+outfile_name)
                    elif pred_250[1] >= 0.90 and pred_1000[1] >= 0.90:
                        fig.savefig('4XMMSSC/top_cand_plots/conf_90/'+outfile_name)
                    else:
                        fig.savefig('4XMMSSC/top_cand_plots/conf_80/'+outfile_name)
                    plt.close()
                        
        #remove any temporarily downloaded files                                
            os.system('rm -r _dl_temp_/'+obsid+'/')
        os.system('rm _dl_temp_/source_pnlc.tar')
        
        
    # #if there is a M1 detection, try and download the data
    # if cat[1].data.field('CCDM1')[i] != -32768 :
    #     pn_url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+obsid+"&sourceno="+srcnum+"&level=PPS&instname=M1&extension=FTZ&name=SRCTSR"
    #     os.system('wget -O _dl_temp_/source_m1lc.tar "'+pn_url+'"')
    #     if os.path.getsize(os.getcwd()+'/_dl_temp_/source_m1lc.tar') != 0:
    #         os.chdir('_dl_temp_/')
    #         os.system('tar -xf source_m1lc.tar')
    #         os.chdir(os.getcwd()[:-10])
            
    #         for file in os.listdir('_dl_temp_/'+obsid+'/pps/'):
    #             lc = XMMtolc('_dl_temp_/'+obsid+'/pps/'+file)
                
    #             try:
    #                 lc == last_lc
    #             except:
    #                 pass
    #             else:
    #                 if lc == last_lc:
    #                     continue
                    
    #             last_lc = lc    
    #             if lc.dt > 250:
    #                 continue
    #             if lc.tseg < 50000:
    #                 continue
                
    #             lc = lc.shift(-lc.time[0])
                
    #             try:
    #                 lc = lc.truncate(start=15000,stop=(lc.time[-1]-15000),method='time')
    #             except:
    #                 continue
    #             else:
    #                 lc = lc.truncate(start=15000,stop=(lc.time[-1]-15000),method='time')
                
    #             try:
    #                 lc_250 = lc.rebin(250)
    #             except:
    #                 continue
    #             else:
    #                 lc_250 = lc.rebin(250)
    #             try:
    #                 lc_1000 = lc.rebin(1000)
    #             except:
    #                 continue
    #             else:
    #                 lc_1000 = lc.rebin(1000)
                                
    #             feats_250 = np.asarray([list(lcfeat([lc_250.time,lc_250.countrate],qpe='?'))])
    #             feats_1000 = np.asarray([list(lcfeat([lc_1000.time,lc_1000.countrate],qpe='?'))])
                
    #             pred_250 = dt250_prob_model.predict(feats_250)[0]
    #             pred_1000 = dt1000_prob_model.predict(feats_1000)[0] 
                
    #             preds_dt250.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
    #                                 'M1'+file[13],pred_250[1]])
    #             preds_dt1000.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
    #                                 'M1'+file[13],pred_1000[1]])
                
    #             if pred_250[1] > 0.8 and pred_1000[1] > 0.8:
    #                 top_cands.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
    #                                     'M1'+file[13],pred_250[1],pred_1000[1]])
    #                 outfile_name = srcid+'_'+obsid+'_'+srcnum+'_m1'+file[13]+'.pdf'
    #                 fig, axs = plt.subplots(2,1,sharex=True)
    #                 axs[0].plot(lc_250.time,lc_250.countrate,color='b')
    #                 axs[1].plot(lc_1000.time,lc_1000.countrate,color='b')
    #                 axs[0].set(ylabel='Count rate')
    #                 axs[1].set(xlabel='Time (s)',ylabel='Count rate')
    #                 fig.suptitle('SRCID '+srcid+' Observation '+obsid+' Source '+str(cat[1].data.field('SRC_NUM')[index])+' M1')
    #                 if pred_250[1] >= 0.99 and pred_1000[1] >= 0.99:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_99/'+outfile_name)
    #                 elif pred_250[1] >= 0.95 and pred_1000[1] >= 0.95:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_95/'+outfile_name)
    #                 elif pred_250[1] >= 0.90 and pred_1000[1] >= 0.90:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_90/'+outfile_name)
    #                 else:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_80/'+outfile_name)
    #                 plt.close()
                                                    
    #         os.system('rm -r _dl_temp_/'+obsid+'/')
    #     os.system('rm _dl_temp_/source_m1lc.tar')
        
    # #if there is a M1 detection, try and download the data
    # if cat[1].data.field('CCDM2')[i] != -32768 :
    #     pn_url = "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="+obsid+"&sourceno="+srcnum+"&level=PPS&instname=M2&extension=FTZ&name=SRCTSR"
    #     os.system('wget -O _dl_temp_/source_m2lc.tar "'+pn_url+'"')
    #     if os.path.getsize(os.getcwd()+'/_dl_temp_/source_m2lc.tar') != 0:
    #         os.chdir('_dl_temp_/')
    #         os.system('tar -xf source_m2lc.tar')
    #         os.chdir(os.getcwd()[:-10])
            
    #         for file in os.listdir('_dl_temp_/'+obsid+'/pps/'):
    #             lc = XMMtolc('_dl_temp_/'+obsid+'/pps/'+file)
                
    #             try:
    #                 lc == last_lc
    #             except:
    #                 pass
    #             else:
    #                 if lc == last_lc:
    #                     continue
                    
    #             last_lc = lc    
    #             if lc.dt > 250:
    #                 continue
    #             if lc.tseg < 50000:
    #                 continue
                
    #             lc = lc.shift(-lc.time[0])
                
    #             try:
    #                 lc = lc.truncate(start=15000,stop=(lc.time[-1]-15000),method='time')
    #             except:
    #                 continue
    #             else:
    #                 lc = lc.truncate(start=15000,stop=(lc.time[-1]-15000),method='time')
                
    #             try:
    #                 lc_250 = lc.rebin(250)
    #             except:
    #                 continue
    #             else:
    #                 lc_250 = lc.rebin(250)
    #             try:
    #                 lc_1000 = lc.rebin(1000)
    #             except:
    #                 continue
    #             else:
    #                 lc_1000 = lc.rebin(1000)
                                
    #             feats_250 = np.asarray([list(lcfeat([lc_250.time,lc_250.countrate],qpe='?'))])
    #             feats_1000 = np.asarray([list(lcfeat([lc_1000.time,lc_1000.countrate],qpe='?'))])
                
    #             pred_250 = dt250_prob_model.predict(feats_250)[0]
    #             pred_1000 = dt1000_prob_model.predict(feats_1000)[0] 
                
    #             preds_dt250.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
    #                                 'M2'+file[13],pred_250[1]])
    #             preds_dt1000.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
    #                                 'M2'+file[13],pred_1000[1]])
                
    #             if pred_250[1] > 0.8 and pred_1000[1] > 0.8:
    #                 top_cands.append([srcid,obsid,cat[1].data.field('SRC_NUM')[index],cat[1].data.field('EP_ONTIME')[index],
    #                                     'M2'+file[13],pred_250[1],pred_1000[1]])
    #                 outfile_name = srcid+'_'+obsid+'_'+srcnum+'_m2'+file[13]+'.pdf'
    #                 fig, axs = plt.subplots(2,1,sharex=True)
    #                 axs[0].plot(lc_250.time,lc_250.countrate,color='b')
    #                 axs[1].plot(lc_1000.time,lc_1000.countrate,color='b')
    #                 axs[0].set(ylabel='Count rate')
    #                 axs[1].set(xlabel='Time (s)',ylabel='Count rate')
    #                 fig.suptitle('SRCID '+srcid+' Observation '+obsid+' Source '+str(cat[1].data.field('SRC_NUM')[index])+' M2')
    #                 if pred_250[1] >= 0.99 and pred_1000[1] >= 0.99:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_99/'+outfile_name)
    #                 elif pred_250[1] >= 0.95 and pred_1000[1] >= 0.95:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_95/'+outfile_name)
    #                 elif pred_250[1] >= 0.90 and pred_1000[1] >= 0.90:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_90/'+outfile_name)
    #                 else:
    #                     fig.savefig('4XMMSSC/top_cand_plots/conf_80/'+outfile_name)
    #                 plt.close()
                                                    
    #         os.system('rm -r _dl_temp_/'+obsid+'/')
    #     os.system('rm _dl_temp_/source_m2lc.tar')
        
#send the data to an array
preds_dt250 = np.asarray(preds_dt250)
preds_dt1000 = np.asarray(preds_dt1000)
top_cands = np.asarray(top_cands)

#create the output dataframes
predictions_dt250 = pd.DataFrame(data=preds_dt250, columns=['SRCID','OBSID','SRC_NUM','EP_ONTIME','INST','QPE_CONF'],dtype=object)
predictions_dt1000 = pd.DataFrame(data=preds_dt1000, columns=['SRCID','OBSID','SRC_NUM','EP_ONTIME','INST','QPE_CONF'],dtype=object)
top_candidates = pd.DataFrame(data=top_cands, columns=['SRCID','OBSID','SRC_NUM','EP_ONTIME','INST','CONF_DT250','CONF_DT1000'])

predictions_dt250.to_csv('4XMMSSC/predictions_pndt250.csv')
predictions_dt1000.to_csv('4XMMSSC/predictions_pndt1000.csv')
top_candidates.to_csv('4XMMSSC/top_QPE_cands_pn.csv')



