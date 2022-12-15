#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:58:13 2022

@author: do19150
"""

import numpy as np

dt=1000

simq = np.loadtxt('Features/qpe_feats_dt'+str(int(dt))+'.csv',delimiter=',')
simnq = np.loadtxt('Features/no_qpe_feats_dt'+str(int(dt))+'.csv',delimiter=',')

#split the simulated data into 90/10 training and validation/testing
train_val_simq = simq[0:45000]
test_simq = simq[45000:]

train_val_simnq = simnq[0:45000]
test_simnq = simnq[45000:]

train_val_data = np.concatenate((train_val_simq,train_val_simnq))
test_data = np.concatenate((test_simq,test_simnq))

np.savetxt('Features/train_val_data_dt'+str(int(dt))+'.csv',train_val_data,delimiter=',')
np.savetxt('Features/simtest_data_dt'+str(int(dt))+'.csv',test_data,delimiter=',')