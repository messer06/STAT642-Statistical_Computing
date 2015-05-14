# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:26:52 2015

@author: amesser
"""

import HW4Classes
import sklearn.cluster
import sklearn.metrics
import scipy
import munkres
import numpy as np

    
try:
    data
except:
    activities=[1,2,5,15,16]
    data,labels=HW4Classes.DataImport(activities=activities,segments=range(1,6))
    data=np.array(data)
    labels=np.array(labels)

testcluster=sklearn.cluster.KMeans(n_clusters=5,verbose=10)
Clusters=testcluster.fit_predict(data,labels[:,1])
confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
testmunk=munkres.Munkres()
indices = np.array(testmunk.compute((-confusion).transpose()))
testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
confustion2=sklearn.metrics.confusion_matrix(labels[:,1],testcorrected)
score=sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted")

scoreSpec=[]
for gamma in np.logspace(-8,0,10,base=np.e):
    testcluster=sklearn.cluster.SpectralClustering(n_clusters=5,gamma=gamma)
    Clusters=testcluster.fit_predict(data)
    confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
    testmunk=munkres.Munkres()
    indices = np.array(testmunk.compute((-confusion).transpose()))
    testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
    scoreSpec.append(sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted"))
    
testcluster=sklearn.cluster.AgglomerativeClustering(n_clusters=5)
Clusters=testcluster.fit_predict(data,labels[:,1])
confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
testmunk=munkres.Munkres()
indices = np.array(testmunk.compute((-confusion).transpose()))
testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
confustion2=sklearn.metrics.confusion_matrix(labels[:,1],testcorrected)
scoreAgg=sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted")
