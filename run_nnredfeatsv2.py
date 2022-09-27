#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:25:14 2022

@author: do19150
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from random import sample, shuffle
from itertools import combinations


dt = 50

#import the training/validation data and the real & simulated testing data
if dt == 50:
    train_val_data = np.loadtxt(os.getcwd()+'/Features/train_val_data.csv',delimiter=',')
elif dt == 250:
    train_val_data = np.loadtxt(os.getcwd()+'/Features/train_val_data_dt250.csv',delimiter=',')
else:
    train_val_data = np.loadtxt(os.getcwd()+'/Features/train_val_data_dt1000.csv',delimiter=',')
    
if dt == 50:
    simlc_test_data = np.loadtxt(os.getcwd()+'/Features/simtest_data.csv',delimiter=',')
elif dt == 250:
    simlc_test_data = np.loadtxt(os.getcwd()+'/Features/simtest_data_dt250.csv',delimiter=',')
else:
    simlc_test_data = np.loadtxt(os.getcwd()+'/Features/simtest_data_dt1000.csv',delimiter=',')
    
if dt == 50:
    reallc_test_data = pd.read_csv(os.getcwd()+'/Features/realobs_test_data.csv',dtype='object')
elif dt == 250:
    reallc_test_data = pd.read_csv(os.getcwd()+'/Features/realobs_test_data_dt250.0.csv',dtype='object')
else:
    reallc_test_data = pd.read_csv(os.getcwd()+'/Features/realobs_test_data_dt1000.0.csv',dtype='object')
    
reallc_test_data = reallc_test_data.astype({'ObsID':'str','STD/Mean':'float32','Prop > 1STD':'float32','Prop > 2STD':'float32',
                                            'Prop > 3STD':'float32','Prop > 4STD':'float32','Prop > 5STD':'float32','Prop > 6STD':'float32',
                                            'IQR/STD':'float32','Skew':'float32','Kurtosis':'float32','Rev CCF':'float32','2nd ACF':'float32',
                                            'CSSD':'float32','Von Neumann Ratio':'float32','QPE?':'float32'})

col_names = ['STD/Mean','Prop > 1STD','Prop > 2STD','Prop > 3STD','Prop > 4STD','Prop > 5STD','Prop > 6STD','IQR/STD',
            'Skew','Kurtosis','Rev CCF','2nd ACF','CSSD','Von Neumann Ratio','QPE?']


#determine the best architecture for such an NN with up to 2 hidden layers.
def model_builder(hp):
    model = keras.Sequential()

    # Tune the number of dense layers between 1 and 3
    for i in range(hp.Int('num_layers', 1, 2)):
        model.add(
            layers.Dense(
                # Tune number of units separately, between 2 and 196
                units=hp.Int(f'units_{i}', min_value=3, max_value=196, step=1),
                activation='relu'))
    
    #add an output layer
    model.add(layers.Dense(2,activation='relu'))

    #select the best learning rate
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def build_test_NN(no_feats,combo_min=0,combo_max=10000):
    
    combos = list(combinations(np.arange(14),no_feats))

    #set up an output df which will contain: features used; sim test accuracy; real test accuracy.
    output_df = pd.DataFrame(columns=['Features Used','Validation Accuracy','Sim Test Accuracy','Real Test Accuracy','Real Test Completeness','Real Test Purity'])
    output_df['Features Used'] = list(combinations(np.arange(14),no_feats))
    
    combos_to_use = np.arange(combo_min,combo_max)
    
    for i in combos_to_use:
        
        #create sub-data sets for this combination of columns
        feature_combination = combos[i]
        
        combo_columns = []
        for j in feature_combination:
            combo_columns.append(col_names[j])
        print('Combination '+str(i),'\n')
        print(combo_columns)
            
        #create empty containers for features and labels
        input_data = []
        check_data = []
        simtest_data = []
        realtest_data = []
        input_labels = []
        check_labels = []
        simtest_labels = []
        realtest_labels = []
        realtest_obsids = []
        
        #split the training data to train and validation sets
        input_indices = sample(list(np.arange(len(train_val_data))),int(0.8*len(train_val_data)))
        check_indices = []
        for j in np.arange(len(train_val_data)):
            if j not in input_indices:
                check_indices.append(j)
        
        #shuffle the order of the feature sets
        shuffle(input_indices)
        shuffle(check_indices)
        simtest_indices = np.arange(len(simlc_test_data))
        realtest_indices = np.arange(len(reallc_test_data))
        shuffle(simtest_indices)
        shuffle(realtest_indices)
        
        #for each object in the whole training data, add the required features to a set for use with the NN
        for j in input_indices:
            #create empty list for features
            obj_feats = []
            #for each feature selected in this combination
            for k in feature_combination:
                obj_feats.append(train_val_data[j][k])
            input_data.append(obj_feats)
            input_labels.append(int(train_val_data[j][-1]))
            
        #do the same for the validation data
        for j in check_indices:
            obj_feats = []
            for k in feature_combination:
                obj_feats.append(train_val_data[j][k])
            check_data.append(obj_feats)
            check_labels.append(int(train_val_data[j][-1]))
                
        #and for the simulated testing data
        for j in simtest_indices:
            obj_feats = []
            for k in feature_combination:
                obj_feats.append(simlc_test_data[j][k])
            simtest_data.append(obj_feats)
            simtest_labels.append(int(simlc_test_data[j][-1]))
            
        #and the same for the real testing data
        for j in realtest_indices:
            realtest_obsids.append(reallc_test_data['ObsID'][j])
            obj_feats = []
            for k in feature_combination:
                obj_feats.append(reallc_test_data.iloc[j,(k+1)])
            realtest_data.append(obj_feats)
            realtest_labels.append(int(reallc_test_data.iloc[j,-1]))
            
        input_data = np.asarray(input_data)
        input_labels = np.asarray(input_labels)
        check_data = np.asarray(check_data)
        check_labels = np.asarray(check_labels)
        simtest_data = np.asarray(simtest_data)
        simtest_labels = np.asarray(simtest_labels)
        realtest_data = np.asarray(realtest_data)
        realtest_labels = np.asarray(realtest_labels)
        
        #create a model for the subset
        tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3, 
                             directory=str(no_feats)+'feats_wip', project_name='combo'+str(i), overwrite=True)
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        #find the best confiuratioon for this set
        tuner.search(input_data, input_labels, validation_data=(check_data,check_labels), epochs=50, callbacks=[stop_early],verbose=0)
        
        #get the best hyperparameters for the model
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        
        #determine the best number of epochs for training
        history = model.fit(input_data, input_labels, epochs=50, validation_data=(check_data,check_labels),verbose=0)
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        
        #perform the final model creation, training and validation
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(input_data, input_labels, epochs=best_epoch, validation_data=(check_data,check_labels),verbose=0)
        
        #save the model to the relevant directory
        best_model.save('saved_models/'+str(no_feats)+'_feats/feature_set'+str(i)+'_dt'+str(int(dt)))
        
        valid_loss, valid_acc = best_model.evaluate(check_data,check_labels,verbose=0)
        print('\nValidation accuracy:', valid_acc)
        output_df.iloc[i,1] = valid_acc
        
        simtest_loss, simtest_acc = best_model.evaluate(simtest_data, simtest_labels, verbose=0)
        print('\nSimulated test accuracy:', simtest_acc)
        output_df.iloc[i,2] = simtest_acc
        
        realtest_loss, realtest_acc = best_model.evaluate(realtest_data, realtest_labels, verbose=0)
        print('\nReal test accuracy:', realtest_acc)
        output_df.iloc[i,3] = realtest_acc
        
        probability_model = tf.keras.Sequential([best_model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(realtest_data)
        
        realtest_preds = []
        for j in range(len(predictions)):
            realtest_preds.append(np.argmax(predictions[j]))
            
        real_preds_out = pd.DataFrame(columns=['OBSID','Real Label','Pred Label','Probabilities'],dtype='object')
        real_preds_out['OBSID'] = realtest_obsids
        real_preds_out['Pred Label'] = realtest_preds
        real_preds_out['Probabilities'] = predictions.tolist()
        for j in range(len(real_preds_out)):
            real_preds_out.iloc[j,1] = int(realtest_labels[j])
            
        label_pairs = []
        for j in range(len(real_preds_out)):
            label_pairs.append([real_preds_out['Real Label'].values[j],real_preds_out['Pred Label'].values[j]])
            
        no_true_pos = label_pairs.count([1,1])
        no_false_neg = label_pairs.count([1,0])
        no_false_pos = label_pairs.count([0,1])
            
        output_df.iloc[i,4] = no_true_pos / (no_true_pos + no_false_neg)
        
        if (no_false_pos + no_true_pos) == 0:
            output_df.iloc[i,5] = 0
        else:
            output_df.iloc[i,5] = no_true_pos / (no_false_pos + no_true_pos)
            
        real_preds_out.to_csv('NN_results/'+str(no_feats)+'feats/featurecombo'+str(i)+'_dt'+str(int(dt))+'_realtest.csv',index=False)
        best_model.build(input_shape=(None,no_feats))
        best_model.summary()
        
        print('\nReal test completeness:', output_df.iloc[i,4])
        print('\nReal test purity:', output_df.iloc[i,5])
        
    output_df.to_csv('NN_results/'+str(no_feats)+'feats_dt'+str(int(dt))+'_overall_accuracy.csv',index=False)
    
    return output_df
    
build_test_NN(int(sys.argv[1]),combo_min=int(sys.argv[2]),combo_max=int(sys.argv[3]))


