#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:54:44 2022

@author: do19150
"""

import pandas as pd
import numpy as np
from operator import itemgetter


df1 = pd.read_csv('NN_results/1feats_dt50_overall_accuracy.csv')
df2 = pd.read_csv('NN_results/2feats_dt50_overall_accuracy.csv')
df3 = pd.read_csv('NN_results/3feats_dt50_overall_accuracy.csv')
df11 = pd.read_csv('NN_results/11feats_dt50_overall_accuracy.csv')
df12 = pd.read_csv('NN_results/12feats_dt50_overall_accuracy.csv')
df13 = pd.read_csv('NN_results/13feats_dt50_overall_accuracy.csv')

def tuplestr_to_tuple(tuplestring):
    tuplestring = tuplestring.replace("(", "")
    tuplestring = tuplestring.replace(")", "")
    tuplestring = tuplestring.replace(",", " ")
    tuplelist = tuplestring.split()
    tuplelist = list(map(int, tuplelist))
    return tuple(tuplelist)

def feat_quality(feat):
    
    feat1without = []
    feat2with = []
    feat2without = []
    feat3with = []

    feat11without = []
    feat12with = []
    feat12without = []
    feat13with = []
    
    for i in range(len(fu1)):
        if feat not in fu1[i]:
            feat1without.append(i)
    for i in range(len(fu2)):
        if feat not in fu2[i]:
            feat2without.append(i)
        else:
            feat2with.append(i)
    for i in range(len(fu3)):
        if feat in fu3[i]:
            feat3with.append(i)
            
    
    for i in range(len(fu11)):
        if feat not in fu11[i]:
            feat11without.append(i)
    for i in range(len(fu12)):
        if feat not in fu12[i]:
            feat12without.append(i)
        else:
            feat12with.append(i)     
    for i in range(len(fu13)):
        if feat in fu13[i]:
            feat13with.append(i)        
    
    
    feat1wo_f1 = np.array(itemgetter(*feat1without)(df1['Metric Value']))
    feat2w_f1 = np.array(itemgetter(*feat2with)(df2['Metric Value']))
    feat2wo_f1 = np.array(itemgetter(*feat2without)(df2['Metric Value']))
    feat3w_f1 = np.array(itemgetter(*feat3with)(df3['Metric Value']))
    
    feat11wo_f1 = np.array(itemgetter(*feat11without)(df11['Metric Value']))
    feat12w_f1 = np.array(itemgetter(*feat12with)(df12['Metric Value']))
    feat12wo_f1 = np.array(itemgetter(*feat12without)(df12['Metric Value']))
    feat13w_f1 = np.array(itemgetter(*feat13with)(df13['Metric Value']))     
    
    avg_f1wout = np.average(np.concatenate((feat1wo_f1,feat2wo_f1,feat11wo_f1,feat12wo_f1),axis=None))
    avg_f1w = np.average(np.concatenate((feat2w_f1,feat3w_f1,feat12w_f1,feat13w_f1),axis=None))
    
    f1_diff = avg_f1w - avg_f1wout
    
    return f1_diff

fu1 = df1['Features Used'].values
fu2 = df2['Features Used'].values
fu3 = df3['Features Used'].values
fu11 = df11['Features Used'].values
fu12 = df12['Features Used'].values
fu13 = df13['Features Used'].values


for i in range(len(fu1)):
    fu1[i] = tuplestr_to_tuple(fu1[i])        
for i in range(len(fu2)):
    fu2[i] = tuplestr_to_tuple(fu2[i])
for i in range(len(fu3)):
    fu3[i] = tuplestr_to_tuple(fu3[i])

for i in range(len(fu11)):
    fu11[i] = tuplestr_to_tuple(fu11[i])
for i in range(len(fu12)):
    fu12[i] = tuplestr_to_tuple(fu12[i])
for i in range(len(fu13)):
    fu13[i] = tuplestr_to_tuple(fu13[i])
    
for i in range(14):
    print(str(i),feat_quality(i))
    
