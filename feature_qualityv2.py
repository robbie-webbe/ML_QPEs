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
df4 = pd.read_csv('NN_results/4feats_dt50_overall_accuracy.csv')
df10 = pd.read_csv('NN_results/10feats_dt50_overall_accuracy.csv')
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
feats4_combos = []
feats10_combos = []
feats11_combos = []
feats12_combos = []
feats13_combos = []

for i in range(len(df1)):
    feats1_combos.append(tuplestr_to_list(df1['Features Used'].values[i]))
for i in range(len(df2)):
    feats2_combos.append(tuplestr_to_list(df2['Features Used'].values[i]))
for i in range(len(df3)):
    feats3_combos.append(tuplestr_to_list(df3['Features Used'].values[i]))
for i in range(len(df4)):
    feats4_combos.append(tuplestr_to_list(df4['Features Used'].values[i]))
for i in range(len(df10)):
    feats10_combos.append(tuplestr_to_list(df10['Features Used'].values[i]))
for i in range(len(df11)):
    feats11_combos.append(tuplestr_to_list(df11['Features Used'].values[i]))
for i in range(len(df12)):
    feats12_combos.append(tuplestr_to_list(df12['Features Used'].values[i]))
for i in range(len(df13)):
    feats13_combos.append(tuplestr_to_list(df13['Features Used'].values[i]))

out_df = pd.DataFrame(columns=['Feature','Overall Change','No. 1->2','Avg 1->2','No. 2->3','Avg 2->3',
                               'No. 3->4','Avg 3->4','No. 10->11','Avg 10->11','No. 11->12','Avg 11->12',
                               'No. 12->13','Avg 12->13'])
out_df['Feature'] = np.arange(14)

def feat_qual2(feat):
    
    #create an empty list for the differences between f1 scores
    f1_diff = []
    
    #for all of the combinations for 2 features identify those which include
    #the feature needed
    #create a counter for number of 1->2 combinations considered
    no_1_2 = 0
    #create an empty list for the differences from 1->2 feats
    diff_1_2 = []
    for i in range(len(feats2_combos)):
        combo = feats2_combos[i]
        if feat in combo:
            #then create the list of features without the one being analysed
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            
            #if that combination is in the 1 feature df
            if reduced_combo in feats1_combos:
                #find the location in the list of 1 combos and calculate the mv difference.
                idx = feats1_combos.index(reduced_combo)
                no_1_2 += 1
                f1_diff.append(df2['Metric Value'].values[i] - df1['Metric Value'].values[idx])
                diff_1_2.append(df2['Metric Value'].values[i] - df1['Metric Value'].values[idx])
    
    #for all of the combinations for 3 features identify those which include
    #the feature needed
    no_2_3 = 0
    diff_2_3 = []
    for i in range(len(feats3_combos)):
        combo = feats3_combos[i]
        if feat in combo:
            #then create the list of features without the one being analysed
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            
            #if that combination is in the 2 feature df
            if reduced_combo in feats2_combos:
                #find the location in the list of 2 combos and calculate the mv difference.
                idx = feats2_combos.index(reduced_combo)
                no_2_3 += 1
                f1_diff.append(df3['Metric Value'].values[i] - df2['Metric Value'].values[idx])                             
                diff_2_3.append(df3['Metric Value'].values[i] - df2['Metric Value'].values[idx])
                
    #for all of the combinations for 3 features identify those which include
    #the feature needed
    no_3_4 = 0
    diff_3_4 = []
    for i in range(len(feats4_combos)):
        combo = feats4_combos[i]
        if feat in combo:
            #then create the list of features without the one being analysed
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            
            #if that combination is in the 2 feature df
            if reduced_combo in feats3_combos:
                #find the location in the list of 2 combos and calculate the mv difference.
                idx = feats3_combos.index(reduced_combo)
                no_3_4 += 1
                f1_diff.append(df4['Metric Value'].values[i] - df3['Metric Value'].values[idx])                             
                diff_3_4.append(df4['Metric Value'].values[i] - df3['Metric Value'].values[idx])
    
    no_10_11 = 0
    diff_10_11 = []            
    for i in range(len(feats11_combos)):
        combo = feats11_combos[i]
        if feat in combo:
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            if reduced_combo in feats10_combos:
                idx = feats10_combos.index(reduced_combo)
                no_10_11 += 1
                f1_diff.append(df11['Metric Value'].values[i] - df10['Metric Value'].values[idx])
                diff_10_11.append(df11['Metric Value'].values[i] - df10['Metric Value'].values[idx])
                
    no_11_12 = 0
    diff_11_12 = []            
    for i in range(len(feats12_combos)):
        combo = feats12_combos[i]
        if feat in combo:
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            if reduced_combo in feats11_combos:
                idx = feats11_combos.index(reduced_combo)
                no_11_12 += 1
                f1_diff.append(df12['Metric Value'].values[i] - df11['Metric Value'].values[idx])
                diff_11_12.append(df12['Metric Value'].values[i] - df11['Metric Value'].values[idx])
    
    no_12_13 = 0
    diff_12_13 = []
    for i in range(len(feats13_combos)):
        combo = feats13_combos[i]
        if feat in combo:
            reduced_combo = combo.copy()
            reduced_combo.remove(feat)
            if reduced_combo in feats12_combos:
                idx = feats12_combos.index(reduced_combo)
                no_12_13 += 1
                f1_diff.append(df13['Metric Value'].values[i] - df12['Metric Value'].values[idx])
                diff_12_13.append(df13['Metric Value'].values[i] - df12['Metric Value'].values[idx])
                
    return [np.average(f1_diff), no_1_2, np.average(diff_1_2), no_2_3, np.average(diff_2_3),
            no_3_4, np.average(diff_3_4), no_10_11, np.average(diff_10_11), no_11_12, 
            np.average(diff_11_12), no_12_13, np.average(diff_12_13)]

for i in range(14):
    quality = feat_qual2(i)
    out_df.iloc[i,1] = quality[0]
    out_df.iloc[i,2] = quality[1]
    out_df.iloc[i,3] = quality[2]
    out_df.iloc[i,4] = quality[3]
    out_df.iloc[i,5] = quality[4]
    out_df.iloc[i,6] = quality[5]
    out_df.iloc[i,7] = quality[6]
    out_df.iloc[i,8] = quality[7]
    out_df.iloc[i,9] = quality[8]
    out_df.iloc[i,10] = quality[9]
    out_df.iloc[i,11] = quality[10]
    out_df.iloc[i,12] = quality[10]
    out_df.iloc[i,13] = quality[11]
    
out_df.to_csv('feature_qual.csv')
print(out_df)
    