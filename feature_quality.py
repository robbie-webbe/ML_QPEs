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

def feat_quality(feat):
    
    
    