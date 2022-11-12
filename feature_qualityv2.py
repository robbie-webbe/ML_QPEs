#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:02:43 2022

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

def tuplestr_to_list(tuplestring):
    tuplestring = tuplestring.replace("(", "")
    tuplestring = tuplestring.replace(")", "")
    tuplestring = tuplestring.replace(",", " ")
    tuplelist = tuplestring.split()
    tuplelist = list(map(int, tuplelist))
    return tuplelist

feats1_combos = []
feats2_combos = []
feats3_combos = []
feats11_combos = []
feats12_combos = []
feats13_combos = []

for i in range(len(df1)):
    feats1_combos.append(tuplestr_to_list(df1['Features Used'].values[i]))
for i in range(len(df2)):
    feats2_combos.append(tuplestr_to_list(df2['Features Used'].values[i]))
for i in range(len(df3)):
    feats3_combos.append(tuplestr_to_list(df3['Features Used'].values[i]))
for i in range(len(df11)):
    feats11_combos.append(tuplestr_to_list(df11['Features Used'].values[i]))
for i in range(len(df12)):
    feats12_combos.append(tuplestr_to_list(df12['Features Used'].values[i]))
for i in range(len(df13)):
    feats13_combos.append(tuplestr_to_list(df13['Features Used'].values[i]))


def feat_qual2(feat):
    
    #create an empty list for the differences between f1 scores
    f1_diff = []
    
    #for all of the combinations for 2 features identify those which include
    #the feature needed
    for i in range(len(feats2_combos)):
        combo = feats2_combos[i]
        if feat in combo:
            #then create the list of features without the one being analysed
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            
            #if that combination is in the 1 feature df
            if reduced_combo in feats1_combos:
                #find the locattion in the list of 1 combos and calculate the mv difference.
                idx = feats1_combos.index(reduced_combo)
                f1_diff.append(df2['Metric Value'].values[i] - df1['Metric Value'].values[idx])
                
    for i in range(len(feats3_combos)):
        combo = feats3_combos[i]
        if feat in combo:
            #then create the list of features without the one being analysed
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            
            #if that combination is in the 1 feature df
            if reduced_combo in feats2_combos:
                #find the locattion in the list of 1 combos and calculate the mv difference.
                idx = feats2_combos.index(reduced_combo)
                f1_diff.append(df3['Metric Value'].values[i] - df2['Metric Value'].values[idx])                             
                
    for i in range(len(feats12_combos)):
        combo = feats12_combos[i]
        if feat in combo:
            #then create the list of features without the one being analysed
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            
            #if that combination is in the 1 feature df
            if reduced_combo in feats11_combos:
                #find the locattion in the list of 1 combos and calculate the mv difference.
                idx = feats11_combos.index(reduced_combo)
                f1_diff.append(df12['Metric Value'].values[i] - df11['Metric Value'].values[idx])
                
    for i in range(len(feats13_combos)):
        combo = feats13_combos[i]
        if feat in combo:
            #then create the list of features without the one being analysed
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            
            #if that combination is in the 1 feature df
            if reduced_combo in feats12_combos:
                #find the locattion in the list of 1 combos and calculate the mv difference.
                idx = feats12_combos.index(reduced_combo)
                f1_diff.append(df13['Metric Value'].values[i] - df12['Metric Value'].values[idx])
                
    return np.average(f1_diff)

for i in range(14):
    print(str(i),feat_qual2(i))
    