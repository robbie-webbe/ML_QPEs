#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:32:35 2022

@author: do19150
"""

import numpy as np
import pandas as pd

def f1_opt(input_file):
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
        
    
    confidence_vals = np.arange(0,1.001,0.001)
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
    
    print('The maximum accuracy was: ',accuracies[opt_acc[0]])
    if len(opt_acc) > 1:
        print('This was achieved at a confidence values between: ',confidence_vals[opt_acc[0]],confidence_vals[opt_acc[-1]],'\n')
    else:
        print('This was achieved at a confidence value of: ',confidence_vals[opt_acc],'\n')
    
    
    print('The maximum F1 score was: ',f1scores[opt_f1s[0]])
    if len(opt_f1s) > 1:
        print('This was achieved at a confidence values between: ',confidence_vals[opt_f1s[0]],confidence_vals[opt_f1s[-1]])
    else:
        print('This was achieved at a confidence value of: ',confidence_vals[opt_f1s])
    print('This was achieved with a purity of ',purities[opt_f1s[0]],' and a completeness of ',completenesses[opt_f1s[0]],'\n','\n','\n')
    
                
    
    
    
