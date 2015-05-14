# -*- coding: utf-8 -*-
"""
Created on Sat May  9 08:19:43 2015

@author: cham
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
import os
import HW3Subs
datasetnames=["ARCENE","DEXTER","DOROTHEA","GISETTE","MADELON"]
results=[]
grid_search_results=[]
for i in range(1,6):
    results.append(np.load("results%d.npy" %i))
    grid_search_results.append(np.load("grid_search_results%d.npy" %i))
    
    
featurenames=['feature_selection__percentile']
classvariable=['class__C']
for datasetnum in range(5):
    for model in range(5):
        print (datasetnames[datasetnum],model)
        print "-1 & %d & %d \\\\" %(results[model][datasetnum][0,0][0].tolist(),results[model][datasetnum][0,0][1].tolist())
        print "1 & %d & %d \\\\" %(results[model][datasetnum][0,1][0].tolist(),results[model][datasetnum][0,1][1].tolist())
        print "-1 & %d & %d \\\\" %(results[model][datasetnum][1,0][0].tolist(),results[model][datasetnum][1,0][1].tolist())
        print "1 & %d & %d \\\\" %(results[model][datasetnum][1,1][0].tolist(),results[model][datasetnum][1,1][1].tolist())
        print HW3Subs.Youdens_matfunc(results[model][datasetnum][1].astype(float))
        n_comps,scores=[],[]
        for datapoint in grid_search_results[model][datasetnum]:
            n_comps.append([datapoint[0]['feature_selection__percentile'],datapoint[1]])
        values = set(map(lambda x:x[0],n_comps))
        plotcomps=list(values)
        plotcomps.sort()
        n_comps=np.array(n_comps)
        for comp in plotcomps:
            scores.append(max(n_comps[[i[0] for i in zip(range(n_comps.shape[0]),n_comps[:,0].tolist()) if i[1]==comp],1]))
        print max(scores)
        plt.figure()
        plt.plot(plotcomps,scores)
        plt.savefig(os.path.join(os.getcwd(),'Comp_vs_scores'+datasetnames[datasetnum]+'%d.pdf' %model))
        plt.close()
            
