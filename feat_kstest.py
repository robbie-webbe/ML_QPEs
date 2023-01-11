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
    nq_df = pd.read_csv('Features/no_qpe_feats_dt'+str(dt)+'.csv',header=None)    
    q_df = pd.read_csv('Features/qpe_feats_dt'+str(dt)+'.csv',header=None)
    
    nqf = nq_df[int(feature_no-1)].values
    qf = q_df[int(feature_no-1)].values
    
    results = kstest(nqf,qf)
    
    return results[1]


def realfeat_ks(feature_no,dt=250):
    rq_df = pd.read_csv('Features/reallc_qpe_dt'+str(dt)+'.csv',header=None)
    rn_df = pd.read_csv('Features/reallc_noqpe_dt'+str(dt)+'.csv',header=None)
    
    qf = rq_df[int(feature_no)].values
    nf = rn_df[int(feature_no)].values
    
    results = kstest(qf,nf)
    
    return results[1]


def real_simfeat_ks(feature_no,dt=250,qpe=True):
    
    results[1]