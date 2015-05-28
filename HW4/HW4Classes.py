# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:25:30 2015

@author: amesser
"""

import sklearn
import os
import itertools
import numpy as np

def DataImport(activities=range(1,20),subjects=range(1,9),segments=range(1,13)):
    data=[]
    labels=[]
    for activity,activitycount in zip(activities,range(len(activities))):
        
        for subject in subjects:
            for segment in segments:
                data.append(np.ravel(np.loadtxt("data/a%02d/p%d/s%02d.txt" %(activity,subject,segment),delimiter=",")))
                labels.append(np.array([activity,activitycount,subject,segment]))
    return data,np.array(labels)