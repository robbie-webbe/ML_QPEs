#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:32:35 2022

@author: do19150
"""

import numpy as np
import pandas as pd
import sys
import os

from tqdm import tqdm

def f1_opt(input_file, print_out = False):
    '''
    Function to take an input file of predictions from a neural network
    and iterate over possible confidence values in order to find the 
    best cut at which to determine QPE or not QPE.
    '''
    
    df = pd.read_csv(input_file)
    real_labels = df['Real Label'].values
    probabilities = df['Probabilities'].values
    for i in range(len(probabilities)):
        prob_string = probabilities[i]
        prob_string = prob_string.replace('[','')
        prob_string = prob_string.replace(']','')
        prob_string = prob_string.replace(',',' ')
        prob_string = prob_string.split()
        prob_string = list(map(float, prob_string))
        probabilities[i] = prob_string
        
    
    confidence_vals = np.arange(0,1.00001,0.00001)
    accuracies = np.zeros(confidence_vals.shape)
    purities = np.zeros(len(confidence_vals))
    completenesses = np.zeros(len(confidence_vals))
    f1scores = np.zeros(len(confidence_vals))
    
    for i in range(len(confidence_vals)):
        pred_labels = []
        for j in range(len(df)):
            if probabilities[j][1] >= confidence_vals[i]:
                pred_labels.append(1)
            else:
                pred_labels.append(0)
                
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        
        for j in range(len(real_labels)):
            if real_labels[j] == 1:
                if pred_labels[j] == 1:
                    true_pos += 1
                else:
                    false_neg += 1
            else:
                if pred_labels[j] == 0:
                    true_neg += 1
                else:
                    false_pos += 1
                    
        accuracy = (true_pos + true_neg)/len(real_labels)
        if true_pos + false_pos != 0:
            purity = true_pos / (true_pos + false_pos)
        else:
            purity = 0
        completeness = true_pos / (true_pos + false_neg)
        if (purity + completeness) != 0:
            f1score = 2*(purity*completeness)/(purity + completeness)
        else:
            f1score = 0
            
        accuracies[i] = accuracy
        purities[i] = purity
        completenesses[i] = completeness
        f1scores[i] = f1score
        
    opt_acc = np.where(accuracies == max(accuracies))[0]
    #opt_pur = np.where(purities == max(purities))[0]
    #opt_com = np.where(completenesses == max(completenesses))[0]
    opt_f1s = np.where(f1scores == max(f1scores))[0]
    
    if print_out:
        print('The maximum accuracy was: ',accuracies[opt_acc[0]])
        if len(opt_acc) > 1:
            print('This was achieved at a confidence values between: ',confidence_vals[opt_acc[0]],' and ',confidence_vals[opt_acc[-1]],'\n')
        else:
            print('This was achieved at a confidence value of: ',confidence_vals[opt_acc],'\n')
        
        
        print('The maximum F1 score was: ',f1scores[opt_f1s[0]])
        if len(opt_f1s) > 1:
            print('This was achieved at a confidence values between: ',confidence_vals[opt_f1s[0]],' and ',confidence_vals[opt_f1s[-1]])
        else:
            print('This was achieved at a confidence value of: ',confidence_vals[opt_f1s])
        print('This was achieved with a purity of ',purities[opt_f1s[0]],' and a completeness of ',completenesses[opt_f1s[0]],'\n','\n','\n')
        
    return np.asarray((confidence_vals,accuracies,purities,completenesses,f1scores))
    
                

#determine number of features and time binning being optimised
feat_nos = int(sys.argv[1])
tbin = int(sys.argv[2])

#load overall features information
old_df = pd.read_csv('NN_results/'+str(feat_nos)+'feats_dt'+str(tbin)+'_overall_accuracy.csv')

#pick all files with that many features
all_files = sorted(os.listdir('NN_results/'+str(feat_nos)+'feats/'))
files = []
for file in all_files:
    if ('_dt'+str(tbin)+'_') in file:
        files.append(file)
        
out_arr = np.empty((len(files),18),dtype=object)

for i in tqdm(range(len(files))):
    combo_number = int(files[i].split('_')[0][12:])
    opt_results = f1_opt('NN_results/'+str(feat_nos)+'feats/'+files[i])
    opt_acc_idxs = np.where(opt_results[1] == max(opt_results[1]))[0]
    out_arr[i,0] = old_df['Features Used'][int(combo_number)]
    out_arr[i,1] = opt_results[1][opt_acc_idxs[0]]
    out_arr[i,2] = [opt_results[0][opt_acc_idxs[0]],opt_results[0][opt_acc_idxs[-1]]]
    out_arr[i,3] = opt_results[1,80000]
    out_arr[i,4] = opt_results[2,80000]
    out_arr[i,5] = opt_results[3,80000]
    out_arr[i,6] = opt_results[1,90000]
    out_arr[i,7] = opt_results[2,90000]
    out_arr[i,8] = opt_results[3,90000]
    out_arr[i,9] = opt_results[1,95000]
    out_arr[i,10] = opt_results[2,95000]
    out_arr[i,11] = opt_results[3,95000]
    out_arr[i,12] = opt_results[1,99000]
    out_arr[i,13] = opt_results[2,99000]
    out_arr[i,14] = opt_results[3,99000]
    out_arr[i,15] = opt_results[1,99900]
    out_arr[i,16] = opt_results[2,99900]
    out_arr[i,17] = opt_results[3,99900]
    

out_df = pd.DataFrame(data=out_arr,columns=['Combo','Opt Acc.','Conf Range','80% Acc','80% Pur','80% Comp',
                               '90% Acc','90% Pur','90% Comp','95% Acc','95% Pur','95% Comp',
                               '99% Acc','99% Pur','99% Comp','99.9% Acc','99.9% Pur','99.9% Comp'],dtype=object)

out_df.to_csv('CO_results/conf_opt_'+str(feat_nos)+'feats_dt'+str(tbin)+'.csv',index=False)
