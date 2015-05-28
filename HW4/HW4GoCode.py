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
    n_clusters=len(activities)
confusion2=[]
testcluster=sklearn.cluster.KMeans(n_clusters=n_clusters,n_init=10,verbose=10)
Clusters=testcluster.fit_predict(data)
confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
testmunk=munkres.Munkres()
indices = np.array(testmunk.compute((-confusion).transpose()))
testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
confusion2.append(sklearn.metrics.confusion_matrix(labels[:,1],testcorrected))
score=sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted")

scoreSpec=[]
maxScore=0
for gamma in np.logspace(-15,3,5,base=np.e):
    testcluster=sklearn.cluster.SpectralClustering(n_clusters=n_clusters,affinity="cosine",gamma=gamma,eigen_solver='arpack')
    Clusters=testcluster.fit_predict(data)
    confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
    testmunk=munkres.Munkres()
    indices = np.array(testmunk.compute((-confusion).transpose()))
    testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
    if maxScore<sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted"):
        confusionSpec=sklearn.metrics.confusion_matrix(labels[:,1],testcorrected)
        maxScore=sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted")
    scoreSpec.append(sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted"))
        
confusion2.append(confusionSpec)
testcluster=sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
Clusters=testcluster.fit_predict(data)
confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
testmunk=munkres.Munkres()
indices = np.array(testmunk.compute((-confusion).transpose()))
testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
confusion2.append(sklearn.metrics.confusion_matrix(labels[:,1],testcorrected))
scoreAgg=sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted")

testcluster=sklearn.cluster.Birch(n_clusters=n_clusters)
Clusters=testcluster.fit_predict(data)
confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
testmunk=munkres.Munkres()
indices = np.array(testmunk.compute((-confusion).transpose()))
testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
confusion2.append(sklearn.metrics.confusion_matrix(labels[:,1],testcorrected))
scoreBirch=sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted")

testcluster=sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters,linkage="complete",affinity="cosine")
Clusters=testcluster.fit_predict(data)
confusion=sklearn.metrics.confusion_matrix(labels[:,1],Clusters)
testmunk=munkres.Munkres()
indices = np.array(testmunk.compute((-confusion).transpose()))
testcorrected=np.interp(Clusters,indices[:,0],indices[:,1]).astype(int)
confusion2.append(sklearn.metrics.confusion_matrix(labels[:,1],testcorrected))
scoreAgg2=sklearn.metrics.f1_score(labels[:,1],testcorrected,average="weighted")
