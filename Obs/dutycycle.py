#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:16:03 2022

@author: do19150
"""

import os
import sys
import fnmatch
sys.path.append('/Users/do19150/Gits/Analysis_Funcs/General/')
sys.path.append('/home/do19150/Gits/Analysis_Funcs/General/')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fitsloader import XMMtolc
from CurveFit import GaussCurve, NGaussFixT
from astropy.modeling.models import Const1D
#from stingray import Lightcurve
from stingray.modeling import GaussianLogLikelihood, ParameterEstimation
#import pdb



#initialise frame with eRO QPE duty cycles
df = pd.DataFrame(columns=['Obs','DC'])

#create list of obsids with multiple eruptions
obslist = ['0823680101','0831790701','0851180401','0851180501','0864330101','0864560101','0872390101','0893810501']
df['Obs'] = obslist
start_pars = list([[20,110,2000,12000,32000],
                  [10,100,2000,2000,32000],
                  [10,80,2000,2000,30000],
                  [5,60,2000,6000,50,2000,25000,85,2000,39000],
                  [25,40,2000,30000,30,2000,58000,50,2000,88000,40,2000,122000],
                  [5,60,2000,22000,60,2000,45000,60,2000,55000,65,2000,75000,55,2000,90000,80,2000,110000,80,2000,120000],
                  [5,25,2000,2000,10000],
                  [5,80,2000,0,10000]]
                  )
#pdb.set_trace()

for i in range(len(obslist)):
    obs = obslist[i]
    print(obs)
    fname = list(filter(lambda x: obs in x, os.listdir('Obs/')))
    lc = XMMtolc('Obs/'+fname[0])
    lc = lc.shift(-lc.time[0])
    #lc.plot()
    
    model = Const1D()
    
    if obs == '0823680101':
        model += NGaussFixT(no=2)
        model.no_1.fixed=True
    elif obs == '0831790701':
        model += NGaussFixT(no=5)
        model.no_1.fixed=True
    elif obs == '0851180401':
        model += NGaussFixT(no=5)
        model.no_1.fixed=True
    elif obs == '0851180501':
        model += GaussCurve() + GaussCurve() + GaussCurve()
    elif obs == '0864330101':
        model += GaussCurve() + GaussCurve() + GaussCurve() + GaussCurve() 
    elif obs == '0864560101':
        model += GaussCurve() + GaussCurve() + GaussCurve() + GaussCurve() + GaussCurve() + GaussCurve() + GaussCurve()    
    elif obs == '0872390101':
        model += NGaussFixT(no=9)
        model.no_1.fixed=True
    elif obs == '0893810501':
        model += NGaussFixT(no=3)
        model.no_1.fixed=True
        
        
        
    loglike = GaussianLogLikelihood(lc.time,lc.counts,lc.counts_err,model)
    parest = ParameterEstimation(fitmethod='BFGS', max_post=False)
    res = parest.fit(loglike,start_pars[i],scipy_optimize_options={'options':{'maxiter':100000,'gtol':1e-10}})
    res.print_summary(loglike)
    
    plt.scatter(lc.time,lc.counts,color='b',s=0.5)
    plt.plot(lc.time,res.mfit,color='r')
    plt.show()
    
    if len(fnmatch.filter(model.param_names, 'recurrence_?')) != 0:
        t_rec = model.param_sets[model.param_names.index('recurrence_1')]
        t_dur = model.param_sets[model.param_names.index('width_1')]
    else:
        pos_names = fnmatch.filter(model.param_names, 'position_?')
        t_tot = model.param_sets[model.param_names.index(pos_names[-1])] - model.param_sets[model.param_names.index(pos_names[0])]
        t_rec = t_tot / len(pos_names)
        dur_names = fnmatch.filter(model.param_names, 'width_?')
        dur_indxs = []
        for j in dur_names:
            dur_indxs.append(model.param_names.index(j))
        durs = model.param_sets[dur_indxs]
        t_dur = np.average(durs)

        
    duty = t_dur / t_rec
    df.loc[i] = [obs,duty[0]]
    
df.to_csv('eruption_dcs.csv')
    
    
    
    
