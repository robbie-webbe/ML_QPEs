#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:54:44 2022

@author: do19150
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv('NN_results/1feats_dt50_overall_accuracy.csv')
df2 = pd.read_csv('NN_results/2feats_dt50_overall_accuracy.csv')
df3 = pd.read_csv('NN_results/3feats_dt50_overall_accuracy.csv')
df11 = pd.read_csv('NN_results/11feats_dt50_overall_accuracy.csv')
df12 = pd.read_csv('NN_results/12feats_dt50_overall_accuracy.csv')
df13 = pd.read_csv('NN_results/13feats_dt50_overall_accuracy.csv')

def feat_q(feat):
    print('Feature '+str(feat))
    df1_idx_not = []
    df2_idx_in = []
    df2_idx_not = []
    df3_idx_in = []
    
    df11_idx_not = []
    df12_idx_in = []
    df12_idx_not = []
    df13_idx_in = []
    
    for i in range(len(df1)):
        if feat != int(df1['Features Used'][i][1:-2]):
            df1_idx_not.append(i)
            
    for i in range(len(df2)):
        feature_list = df2['Features Used'][i][1:-1].split(',')
        for j in range(len(feature_list)):
            feature_list[j] = int(feature_list[j])
        if feat in feature_list:
            df2_idx_in.append(i)
        else:
            df2_idx_not.append(i)
    
    for i in range(len(df3)):
        feature_list = df3['Features Used'][i][1:-1].split(',')
        for j in range(len(feature_list)):
            feature_list[j] = int(feature_list[j])
        if feat in feature_list:
            df3_idx_in.append(i)
            
    for i in range(len(df11)):
        feature_list = df11['Features Used'][i][1:-1].split(',')
        for j in range(len(feature_list)):
            feature_list[j] = int(feature_list[j])
        if feat not in feature_list:
            df11_idx_not.append(i)
            
    for i in range(len(df12)):
        feature_list = df12['Features Used'][i][1:-1].split(',')
        for j in range(len(feature_list)):
            feature_list[j] = int(feature_list[j])
        if feat in feature_list:
            df12_idx_in.append(i)
        else:
            df12_idx_not.append(i)
            
    for i in range(len(df13)):
        feature_list = df13['Features Used'][i][1:-1].split(',')
        for j in range(len(feature_list)):
            feature_list[j] = int(feature_list[j])
        if feat in feature_list:
            df13_idx_in.append(i)
        
    f1_change1 = []            
    for i in range(len(df1_idx_not)):
        pre_f1 = df1['F1 Score'][df1_idx_not[i]]
        post_f1 = df2['F1 Score'][df2_idx_in[i]]
        delta_f1 = post_f1 - pre_f1
        f1_change1.append(delta_f1)
        
    #print('Average f1 change on adding this feature to one other: \n',np.average(f1_change1))
    
    f1_change2 = []            
    for i in range(len(df2_idx_not)):
        pre_f1 = df2['F1 Score'][df2_idx_not[i]]
        post_f1 = df3['F1 Score'][df3_idx_in[i]]
        delta_f1 = post_f1 - pre_f1
        f1_change2.append(delta_f1)
        
    #print('Average f1 change on adding this feature to combinations of two others: \n',np.average(f1_change2))
    
    f1_change11 = []            
    for i in range(len(df11_idx_not)):
        pre_f1 = df11['F1 Score'][df11_idx_not[i]]
        post_f1 = df12['F1 Score'][df12_idx_in[i]]
        delta_f1 = post_f1 - pre_f1
        f1_change11.append(delta_f1)
        
    #print('Average f1 change on adding this feature to combinations of eleven others: \n',np.average(f1_change2))
    
    f1_change12 = []            
    for i in range(len(df12_idx_not)):
        pre_f1 = df12['F1 Score'][df12_idx_not[i]]
        post_f1 = df13['F1 Score'][df13_idx_in[i]]
        delta_f1 = post_f1 - pre_f1
        f1_change12.append(delta_f1)
        
    #print('Average f1 change on adding this feature to combinations of twelve others: \n',np.average(f1_change2))
    
    
    weighted_change = (sum(f1_change1) + sum(f1_change2) + sum(f1_change11) + sum(f1_change12))/(len(f1_change1) + len(f1_change2) + len(f1_change11) + len(f1_change12))
    
    print('Average change on adding this feature: ',weighted_change)
        
    return f1_change1, f1_change2, f1_change11, f1_change12
        
    
            
    
    