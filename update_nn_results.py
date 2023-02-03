#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:55:31 2023

@author: do19150
"""
import pandas as pd
import numpy as np
import os
import sys

feat_no = str(int(sys.argv[1]))
tbin = str(int(sys.argv[2]))

full_results = pd.read_csv('NN_results/'+feat_no+'feats_dt'+tbin+'_overall_accuracy.csv')

file_list = sorted(os.listdir('NN_results'))

for file in file_list:
    if feat_no in file:
        if tbin in file:
            if 'combos' in file:
                combo_df = pd.read_csv('NN_results/'+file,na_filter=False)
                combo_line = np.where(combo_df['Metric Value'] != '')[0]
                full_results.iloc[combo_line,:] = combo_df.iloc[combo_line,:]
                os.system('rm NN_results/'+file)
                
full_results.to_csv('NN_results/'+feat_no+'feats_dt'+tbin+'_overall_accuracy.csv',index=False)
                