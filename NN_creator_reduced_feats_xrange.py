#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:13:09 2022

@author: do19150
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from random import sample, shuffle
from itertools import combinations


#y = np.arange(3,12,1)
y = [1,13,2,12,3,11,4,10,5,9,6,8,7]
dt = 50

for x in y:
    combs_sum = 0
    for i in np.arange(0,x):
        combs_sum += len(list(combinations(np.arange(x),i)))
    
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
    
    col_names = ['STD/Mean','Prop > 1STD','Prop > 2STD','Prop > 3STD','Prop > 4STD','Prop > 5STD','Prop > 6STD','IQR/STD',
                'Skew','Kurtosis','Rev CCF','2nd ACF','CSSD','Von Neumann Ratio','QPE?']
    
    #split the train/validation data 80%/20% into training and validation
    index_range = list(np.arange(len(train_val_data)))
    train_indices = sorted(sample(index_range,int(0.8*len(index_range))))
    valid_indices = index_range
    for i in train_indices:
        valid_indices.remove(i)
        
    #create the feature sets
    training_data = train_val_data[train_indices]
    valid_data = train_val_data[valid_indices]
    
    #separate the training and validation features and labels and shuffle their order
    all_input_data = []
    input_labels = []
    input_indices = np.arange(len(training_data))
    shuffle(input_indices)
    for i in input_indices:
        all_input_data.append(list(training_data[i][0:14]))
        input_labels.append([training_data[i][14]])
        
    all_check_data = []
    check_labels = []
    check_indices = np.arange(len(valid_data))
    shuffle(check_indices)
    for i in check_indices:
        all_check_data.append(list(valid_data[i][0:14]))
        check_labels.append([valid_data[i][14]])
        
    #create the simulated testing feature and label sets
    all_simtest_data = []
    simtest_labels = []
    simtest_indices = np.arange(len(simlc_test_data))
    shuffle(simtest_indices)
    for i in simtest_indices:
        all_simtest_data.append(list(simlc_test_data[i][0:14]))
        simtest_labels.append([simlc_test_data[i][14]])
    
    #extract the real testing feature and label sets
    realtest_obsids = []
    all_realtest_data = []
    realtest_labels = []
    realtest_indices = np.arange(len(reallc_test_data))
    shuffle(realtest_indices)
    for i in realtest_indices:
        realtest_obsids.append(reallc_test_data.iloc[i,0])
        all_realtest_data.append(list(reallc_test_data.iloc[i,1:15].astype('float32')))
        realtest_labels.append([float(reallc_test_data.iloc[i,15])])
        
    #determine the best architecture for such an NN with up to 3 hidden layers.
    def model_builder(hp):
        model = keras.Sequential()
    
        model.add(layers.Dense(
                # Tune number of units separately, between 2 and 196
                units=hp.Int(f'units_0', min_value=1, max_value=min(combs_sum,196), step=1),
                activation='relu'))
    
        # Tune the number of dense layers between 1 and 2
        for i in range(hp.Int('num_layers', 0, 1)):
            model.add(
                layers.Dense(
                    # Tune number of units separately, between 2 and 196
                    units=hp.Int(f'units_{i}', min_value=2, max_value=196, step=1),
                    activation='relu'))
            
        model.add(layers.Dense(2,activation='relu'))
    
        #select the best learning rate
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    
        return model
    
    combos = list(combinations(np.arange(14),x))
    
    #set up an output df which will contain: features used; sim test accuracy; real test accuracy.
    output_df = pd.DataFrame(columns=['Features Used','Validation Accuracy','Sim Test Accuracy','Real Test Accuracy'])
    output_df['Features Used'] = list(combinations(np.arange(14),x))
        
    for i in range(len(combos)):
        
        #create sub-data sets for this combination of columns
        feature_combination = combos[i]
        print(feature_combination)
        
        combo_columns = []
        for j in feature_combination:
            combo_columns.append(col_names[j])
        print(combo_columns)
            
        input_data = []
        check_data = []
        simtest_data = []
        realtest_data = []
        
        for j in range(len(all_input_data)):
            feats = []
            for k in feature_combination:
                feats.append(all_input_data[j][k])
            input_data.append(feats)
            
        for j in range(len(all_check_data)):
            feats = []
            for k in feature_combination:
                feats.append(all_check_data[j][k])
            check_data.append(feats)
            
        for j in range(len(all_simtest_data)):
            feats = []
            for k in feature_combination:
                feats.append(all_simtest_data[j][k])
            simtest_data.append(feats)
            
        for j in range(len(all_realtest_data)):
            feats = []
            for k in feature_combination:
                feats.append(all_realtest_data[j][k])
            realtest_data.append(feats)    
        
        #create a model for the subset
        tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3, 
                             directory='my_dir', project_name='working'+str(x)+'_'+str(i), overwrite=True)
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        #find the best confiuratioon for this set
        tuner.search(input_data, input_labels, validation_data=(check_data,check_labels), epochs=50, callbacks=[stop_early],verbose=0)
        
        #get the best hyperparameters for the model
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        
        #determine the best number of epochs for training
        history = model.fit(input_data, input_labels, epochs=50, validation_data=(check_data,check_labels), verbose=0)
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        
        #perform the final model creation, training and validation
        best_model = tuner.hypermodel.build(best_hps)
        best_model.fit(input_data, input_labels, epochs=best_epoch, validation_data=(check_data,check_labels), verbose=0)
        
        #save the model to the relevant directory
        best_model.save('saved_models/'+str(x)+'_feats/feature_set'+str(i)+'_dt'+str(int(dt)))
        
        valid_loss, valid_acc = best_model.evaluate(check_data,check_labels,verbose=2)
        print('\nValidation accuracy:', valid_acc)
        output_df.iloc[i,1] = valid_acc
        
        simtest_loss, simtest_acc = best_model.evaluate(simtest_data, simtest_labels, verbose=2)
        print('\nTest accuracy:', simtest_acc)
        output_df.iloc[i,2] = simtest_acc
        
        realtest_loss, realtest_acc = best_model.evaluate(realtest_data, realtest_labels, verbose=2)
        print('\nTest accuracy:', realtest_acc)
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
            real_preds_out.iloc[j,1] = int(realtest_labels[j][0])
            
        real_preds_out.to_csv('NN_results/'+str(x)+'feats/featurecombo'+str(i)+'_dt'+str(int(dt))+'_realtest.csv',index=False)
        best_model.build(input_shape=(None,x))
        best_model.summary()
        
    output_df.to_csv('NN_results/'+str(x)+'feats_dt'+str(int(dt))+'_overall_accuracy.csv',index=False)
    



