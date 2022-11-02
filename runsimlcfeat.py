#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:41:15 2022

@author: do19150
"""

import numpy as np
from lcfeaturegen import lcfeat
from tqdm import tqdm

qpe_data = np.loadtxt('LCGen/Diff_dt/qpe_sample.csv',delimiter=',')
qpe_features = np.zeros((len(qpe_data)-1,15))

for i in tqdm(range(50000)): 
    lc = [qpe_data[0],qpe_data[i+1]]
    qpe_features[i] = lcfeat(lc,qpe=1)

np.savetxt('Features/qpe_feats_dt50.csv',qpe_features,delimiter=',')



#
no_qpe_data = np.loadtxt('LCGen/Diff_dt/no_qpe_sample.csv',delimiter=',')
no_qpe_features = np.zeros((len(no_qpe_data)-1,15))
#    
for i in tqdm(range(50000)): 
    lc = [no_qpe_data[0],no_qpe_data[i+1]]
    no_qpe_features[i] = lcfeat(lc,qpe=0)
#
np.savetxt('Features/no_qpe_feats_dt50.csv',no_qpe_features,delimiter=',')
