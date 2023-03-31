#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:36:24 2023

@author: do19150
"""

lcs = np.loadtxt('LCGen/Diff_dt/no_qpe_sample_dt50.csv',delimiter=',')
po_lcs = np.zeros(lcs.shape)
po_lcs[0] = lcs[0]
lcs_avg = np.average(lcs[1:])/1000
conversion = 1/lcs_avg
for i in tqdm(range(len(lcs)-1)):
    raw_lc = lcs[i+1]
    raw_lc *= conversion
    if i in range(20):
        print(np.average(raw_lc))
    for j in range(len(raw_lc)):
        po_lcs[i+1][j] = poisson.rvs(raw_lc[j])
qpe_features = np.zeros((len(po_lcs)-1,15))
for i in tqdm(range(50000)): 
    lc = [po_lcs[0],po_lcs[i+1]]
    qpe_features[i] = lcfeat(lc,qpe=0)
np.savetxt('Features/no_qpe_feats_dt50_po1.csv',qpe_features,delimiter=',')