#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:00:42 2022

@author: do19150
"""

import os
import sys
from astropy.io import fits

from lcfeaturegen import lcfeat

obsid = sys.argv[1]
srcno = sys.argv[2]

run_statement = 'bash dl_srclc.sh '+str(obsid) + ' ' + str(srcno)
os.system(run_statement)


