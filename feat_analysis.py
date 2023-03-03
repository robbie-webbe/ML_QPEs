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
    f3_df = pd.read_csv('NN_results/3feats_dt'+str(dt)+'_overall_accuracy.csv')
    f3opt_df = pd.read_csv('CO_results/conf_opt_3feats_dt'+str(dt)+'.csv')
    f4_df = pd.read_csv('NN_results/4feats_dt'+str(dt)+'_overall_accuracy.csv')
    f4opt_df = pd.read_csv('CO_results/conf_opt_4feats_dt'+str(dt)+'.csv')
    f5_df = pd.read_csv('NN_results/5feats_dt'+str(dt)+'_overall_accuracy.csv')
    f5opt_df = pd.read_csv('CO_results/conf_opt_5feats_dt'+str(dt)+'.csv')
    f6_df = pd.read_csv('NN_results/6feats_dt'+str(dt)+'_overall_accuracy.csv')
    f6opt_df = pd.read_csv('CO_results/conf_opt_6feats_dt'+str(dt)+'.csv')
    f7_df = pd.read_csv('NN_results/7feats_dt'+str(dt)+'_overall_accuracy.csv')
    f7opt_df = pd.read_csv('CO_results/conf_opt_7feats_dt'+str(dt)+'.csv')
    f8_df = pd.read_csv('NN_results/8feats_dt'+str(dt)+'_overall_accuracy.csv')
    f8opt_df = pd.read_csv('CO_results/conf_opt_8feats_dt'+str(dt)+'.csv')
    f9_df = pd.read_csv('NN_results/9feats_dt'+str(dt)+'_overall_accuracy.csv')
    f9opt_df = pd.read_csv('CO_results/conf_opt_9feats_dt'+str(dt)+'.csv')
    f10_df = pd.read_csv('NN_results/10feats_dt'+str(dt)+'_overall_accuracy.csv')
    f10opt_df = pd.read_csv('CO_results/conf_opt_10feats_dt'+str(dt)+'.csv')
    f11_df = pd.read_csv('NN_results/11feats_dt'+str(dt)+'_overall_accuracy.csv')
    f11opt_df = pd.read_csv('CO_results/conf_opt_11feats_dt'+str(dt)+'.csv')
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
    
    f3_accs = []
    f3_f1s = []
    f3opt_accs = []
    
    f4_accs = []
    f4_f1s = []
    f4opt_accs = []
    
    f5_accs = []
    f5_f1s = []
    f5opt_accs = []
    
    f6_accs = []
    f6_f1s = []
    f6opt_accs = []
    
    f7_accs = []
    f7_f1s = []
    f7opt_accs = []
    
    f8_accs = []
    f8_f1s = []
    f8opt_accs = []
    
    f9_accs = []
    f9_f1s = []
    f9opt_accs = []
    
    f10_accs = []
    f10_f1s = []
    f10opt_accs = []
    
    f11_accs = []
    f11_f1s = []
    f11opt_accs = []
    
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
    for i in range(len(f3_df)):
        combo_list = f3_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f3_accs.append(f3_df['Real Test Accuracy'][i])
            f3_f1s.append(f3_df['Metric Value'][i])  
    for i in range(len(f4_df)):
        combo_list = f4_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f4_accs.append(f4_df['Real Test Accuracy'][i])
            f4_f1s.append(f4_df['Metric Value'][i])  
    for i in range(len(f5_df)):
        combo_list = f5_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f5_accs.append(f5_df['Real Test Accuracy'][i])
            f5_f1s.append(f5_df['Metric Value'][i])
    for i in range(len(f6_df)):
        combo_list = f6_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f6_accs.append(f6_df['Real Test Accuracy'][i])
            f6_f1s.append(f6_df['Metric Value'][i])      
    for i in range(len(f7_df)):
        combo_list = f7_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f7_accs.append(f7_df['Real Test Accuracy'][i])
            f7_f1s.append(f7_df['Metric Value'][i])  
    for i in range(len(f8_df)):
        combo_list = f8_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f8_accs.append(f8_df['Real Test Accuracy'][i])
            f8_f1s.append(f8_df['Metric Value'][i]) 
    for i in range(len(f9_df)):
        combo_list = f9_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f9_accs.append(f9_df['Real Test Accuracy'][i])
            f9_f1s.append(f9_df['Metric Value'][i]) 
    for i in range(len(f10_df)):
        combo_list = f10_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f10_accs.append(f10_df['Real Test Accuracy'][i])
            f10_f1s.append(f10_df['Metric Value'][i])  
    for i in range(len(f11_df)):
        combo_list = f11_df['Features Used'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f11_accs.append(f11_df['Real Test Accuracy'][i])
            f11_f1s.append(f11_df['Metric Value'][i])  
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
    for i in range(len(f3_df)):
        combo_list = f3opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f3opt_accs.append(f3opt_df['Opt Acc.'][i])
    for i in range(len(f4_df)):
        combo_list = f4opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f4opt_accs.append(f4opt_df['Opt Acc.'][i])
    for i in range(len(f5opt_df)):
        combo_list = f5opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f5opt_accs.append(f5opt_df['Opt Acc.'][i])
    for i in range(len(f6_df)):
        combo_list = f6opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f6opt_accs.append(f6opt_df['Opt Acc.'][i])
    for i in range(len(f7_df)):
        combo_list = f7opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f7opt_accs.append(f7opt_df['Opt Acc.'][i])
    for i in range(len(f8_df)):
        combo_list = f8opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f8opt_accs.append(f8opt_df['Opt Acc.'][i])
    for i in range(len(f9_df)):
        combo_list = f9opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f9opt_accs.append(f9opt_df['Opt Acc.'][i])
    for i in range(len(f10_df)):
        combo_list = f10opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f10opt_accs.append(f10opt_df['Opt Acc.'][i])
    for i in range(len(f11_df)):
        combo_list = f11opt_df['Combo'][i].replace('(','').replace(')','').split(',')
        if '' in combo_list:
            combo_list.remove('')
        for j in range(len(combo_list)):
            combo_list[j] = int(combo_list[j])
        if feature in combo_list:
            f11opt_accs.append(f11opt_df['Opt Acc.'][i])
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
                [3,np.average(f3_accs),np.average(f3_f1s),np.average(f3opt_accs)],
                [4,np.average(f4_accs),np.average(f4_f1s),np.average(f4opt_accs)],
                [5,np.average(f5_accs),np.average(f5_f1s),np.average(f5opt_accs)],
                [6,np.average(f6_accs),np.average(f6_f1s),np.average(f6opt_accs)],
                [7,np.average(f7_accs),np.average(f7_f1s),np.average(f7opt_accs)],
                [8,np.average(f8_accs),np.average(f8_f1s),np.average(f8opt_accs)],
                [9,np.average(f9_accs),np.average(f9_f1s),np.average(f9opt_accs)],
                [10,np.average(f10_accs),np.average(f10_f1s),np.average(f10opt_accs)],
                [11,np.average(f11_accs),np.average(f11_f1s),np.average(f11opt_accs)],
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