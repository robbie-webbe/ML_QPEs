#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:51:05 2023

@author: do19150
"""

from scipy.stats import kstest
import numpy as np
import pandas as pd

def simfeat_ks(feature_no,dt=250):
    '''
    Function to determine the ks test probability that the simulated feature distribution
    for a given feature and time binning are the same.
    '''
    
    nq_df = pd.read_csv('Features/no_qpe_feats_dt'+str(dt)+'.csv',header=None)    
    q_df = pd.read_csv('Features/qpe_feats_dt'+str(dt)+'.csv',header=None)
    
    nqf = nq_df[int(feature_no-1)].values
    qf = q_df[int(feature_no-1)].values
    
    results = kstest(nqf,qf)
    
    return results[1]


def realfeat_ks(feature_no,dt=250):
    '''
    Function to determine the ks test probability that the real observation feature distribution
    for a given feature and time binning are the same.
    '''
    
    rq_df = pd.read_csv('Features/reallc_qpe_dt'+str(dt)+'.csv',header=None)
    rn_df = pd.read_csv('Features/reallc_noqpe_dt'+str(dt)+'.csv',header=None)
    
    qf = rq_df[int(feature_no)].values
    nf = rn_df[int(feature_no)].values
    
    results = kstest(qf,nf)
    
    return results[1]



def class_deviation(obsid,dt):
    '''
    Function to determine the deviation of a particular observation's features from the
    simulated averages.
    '''
    
    sim_qpe_feats = pd.read_csv('Features/qpe_feats_dt'+str(dt)+'.csv',header=None)
    sim_nqpe_feats = pd.read_csv('Features/no_qpe_feats_dt'+str(dt)+'.csv',header=None)
    real_feats = pd.read_csv('Features/realobs_test_data_dt'+str(dt)+'.csv',dtype='object')
    
    obs_feats = real_feats[real_feats['ObsID'] == obsid].values[0][1:-1].astype('float')
    print(obs_feats)
    
    qpe_feat_dev = []
    nqpe_feat_dev = []
    
    for i in range(14):
        
        qpe_feat_dev.append(abs((obs_feats[i] - np.average(sim_qpe_feats.iloc[:,i]))/np.std(sim_qpe_feats.iloc[:,i])))
        nqpe_feat_dev.append(abs((obs_feats[i] - np.average(sim_nqpe_feats.iloc[:,i]))/np.std(sim_nqpe_feats.iloc[:,i])))

    qpe_mean_dev = np.average(qpe_feat_dev)
    qpe_spread_dev = np.std(qpe_feat_dev)
    nqpe_mean_dev = np.average(np.ma.masked_invalid(nqpe_feat_dev))
    nqpe_spread_dev = np.std(np.ma.masked_invalid(nqpe_feat_dev))
    
    return qpe_mean_dev, qpe_spread_dev, nqpe_mean_dev, nqpe_spread_dev


def feat_quality(feature,dt=250):
    '''
    Function to determine the average accuracy of classifiers using that feature for various 
    combinations of features.
    '''
    
    f1_df = pd.read_csv('NN_results/1feats_dt'+str(dt)+'_overall_accuracy.csv')
    f1opt_df = pd.read_csv('CO_results/conf_opt_1feats_dt'+str(dt)+'.csv')
    f2_df = pd.read_csv('NN_results/2feats_dt'+str(dt)+'_overall_accuracy.csv')
    f2opt_df = pd.read_csv('CO_results/conf_opt_2feats_dt'+str(dt)+'.csv')
    f12_df = pd.read_csv('NN_results/12feats_dt'+str(dt)+'_overall_accuracy.csv')
    f12opt_df = pd.read_csv('CO_results/conf_opt_12feats_dt'+str(dt)+'.csv')
    f13_df = pd.read_csv('NN_results/13feats_dt'+str(dt)+'_overall_accuracy.csv')
    f13opt_df = pd.read_csv('CO_results/conf_opt_13feats_dt'+str(dt)+'.csv')
    
    f1_accs = []
    f1_f1s = []
    f1opt_accs = []
    
    f2_accs = []
    f2_f1s = []
    f2opt_accs = []
    
    f12_accs = []
    f12_f1s = []
    f12opt_accs = []
    
    f13_accs = []
    f13_f1s = []
    f13opt_accs = []
    
    for i in range(len(f1_df)):
        combo_list = f1_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f1_accs.append(f1_df['Real Test Accuracy'][i])
            f1_f1s.append(f1_df['Metric Value'][i])
    for i in range(len(f2_df)):
        combo_list = f2_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f2_accs.append(f2_df['Real Test Accuracy'][i])
            f2_f1s.append(f2_df['Metric Value'][i])        
    for i in range(len(f12_df)):
        combo_list = f12_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f12_accs.append(f12_df['Real Test Accuracy'][i])
            f12_f1s.append(f12_df['Metric Value'][i])
    for i in range(len(f13_df)):
        combo_list = f13_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f13_accs.append(f13_df['Real Test Accuracy'][i])
            f13_f1s.append(f13_df['Metric Value'][i])
            
    for i in range(len(f1opt_df)):
        combo_list = f1opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f1opt_accs.append(f1opt_df['Opt Acc.'][i])
    for i in range(len(f2_df)):
        combo_list = f2opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f2opt_accs.append(f2opt_df['Opt Acc.'][i])
    for i in range(len(f12_df)):
        combo_list = f12opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f12opt_accs.append(f12opt_df['Opt Acc.'][i])
    for i in range(len(f13_df)):
        combo_list = f13opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f13opt_accs.append(f13opt_df['Opt Acc.'][i])
    
    out_data = [[1,np.average(f1_accs),np.average(f1_f1s),np.average(f1opt_accs)],
                [2,np.average(f2_accs),np.average(f2_f1s),np.average(f2opt_accs)],
                [12,np.average(f12_accs),np.average(f12_f1s),np.average(f12opt_accs)],
                [13,np.average(f13_accs),np.average(f13_f1s),np.average(f13opt_accs)]]
    quality_df = pd.DataFrame(data = np.asarray(out_data),columns=['No. Features','Avg Conf 0.5 Acc','Avg Conf 0.5 F1','Avg Opt Acc'])
    
    return quality_df
    


# def real_simfeat_ks(feature_no,dt=250,qpe=True):
    # '''
    # Function to determine whether the real and simulated feature distributions for a given
    # feature and time binning are the same via a ks test.
    # '''
#     results[1]