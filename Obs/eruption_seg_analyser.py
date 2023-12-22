#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:24:18 2022

@author: do19150
"""

import os
#import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/do19150/Gits/Analysis_Funcs/LC_Sim/')
sys.path.append('/home/do19150/Git_repos/Analysis_Funcs/LC_Sim/')

from fitsloader import XMMtolc
from CurveFit import GaussCurve
from astropy.modeling.models import Const1D
#from stingray import Lightcurve
from stingray.modeling import GaussianLogLikelihood, ParameterEstimation
import matplotlib.pyplot as plt
#import pdb

segs_dir = os.getcwd()+'/Eruption_lcsegs/'
seg_list = sorted(os.listdir(segs_dir))

df = pd.DataFrame(columns=['Seg','Amplitude','Duration'])
df['Seg'] = seg_list

for i in range(len(seg_list)):
    file = seg_list[i]
    lc = XMMtolc(segs_dir + file,t_bin=10)
    lc = lc.rebin(200)
    lc.plot()
    
    model = Const1D() + GaussCurve()
    start_pars = [10,75,2000,(lc.time[-1]-lc.time[0])/2]
    loglike = GaussianLogLikelihood(lc.time,lc.counts,lc.counts_err,model)
    parest = ParameterEstimation(fitmethod='BFGS', max_post=False)
    res = parest.fit(loglike,start_pars,scipy_optimize_options={'options':{'maxiter':100000,'gtol':1e-10,'disp':True}})
    #res.print_summary(loglike)
    res.mfit /= lc.dt
    #pdb.set_trace()
    fig, ax = plt.subplots()
    plt.scatter(lc.time,lc.countrate)
    plt.plot(lc.time,res.mfit)
    plt.show()
    df.iloc[i,1] = model.amplitude_1.value/model.amplitude_0.value
    df.iloc[i,2] = model.width_1.value
    
df.to_csv(os.getcwd()+'/eruption_profiles_auto.csv')
display(df)
