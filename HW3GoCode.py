# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:14:20 2015

@author: amesser
"""

import HW3Classes
import HW3Subs
import numpy as np
from sklearn import decomposition as decomp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import feature_selection as featsel

datasetnames=["ARCENE","DEXTER","DOROTHEA","GISETTE","MADELON"]
numdims=[300,20000,100000,5000,500]
BinaryData=[False,False,True,False,True]
subsetter=[(0),1,2,3,4]
maxdims=[300,5000,5000,5000,500]

results=[]
for datasetname,numdims,binvars,subset in zip(datasetnames,numdims,BinaryData,subsetter):
    if datasetname in ["ARCENE","GISETTE","MADELON"]:
        datasets=[np.loadtxt(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_train.data"),
              np.loadtxt(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_valid.data"),
                np.loadtxt(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_test.data")]
        labels=[np.loadtxt(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_train.labels"),
                np.loadtxt(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_valid.labels")]
        data1= HW3Classes.DataSet('Dataset1',data=datasets,labels=labels,numdims=numdims)
        sparseversion=False
    elif datasetname in ["DEXTER","DOROTHEA"]:
        labels=[np.loadtxt(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_train.labels"),
                np.loadtxt(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_valid.labels")]
        data1= HW3Classes.DataSet('Dataset1',labels=labels,numdims=numdims)        
        data1.add_data("datatrain",HW3Subs.ImportSparse(datasetname,"train",binvars=binvars,numcols=numdims)) 
        data1.add_data("dataval",HW3Subs.ImportSparse(datasetname,"valid",binvars=binvars,numcols=numdims)) 
        data1.add_data("datatest",HW3Subs.ImportSparse(datasetname,"test",binvars=binvars,numcols=numdims))           
        sparseversion=False
    print datasetname
    
#    HW3Subs.PltCorrMatrix(data1.corr,datasetname)
    data1.PCA()
    results.append(HW3Subs.RunClassifiers(data1.princompscores[0],data1.princompscores[1],data1.princompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
    pcaresults=[]    
    for i in range(1,maxdims[subset]):
        pcaresults.append(HW3Subs.RunClassifiers(data1.princompscores[0][:,:i],data1.princompscores[1][:,:i],data1.princompscores[2][:,:i],data1.labelstrain,data1.labelsval,["Linear SVM"]))
        print pcaresults[i-1][1]
#    data1.ICA()
#    results.append(HW3Subs.RunClassifiers(data1.indcompscores[0],data1.indcompscores[1],data1.indcompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
#    data1.KernelPCA("RBF",sparseversion=sparseversion)
#    results.append(HW3Subs.RunClassifiers(data1.kernelprincompscores[0],data1.kernelprincompscores[1],data1.kernelprincompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
#    data1.KernelPCA("sigmoid",sparseversion=sparseversion)
#    results.append(HW3Subs.RunClassifiers(data1.kernelprincompscores[0],data1.kernelprincompscores[1],data1.kernelprincompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))

    
#    
    data1.FA()
    results.append(HW3Subs.RunClassifiers(data1.FAcompscores[0],data1.FAcompscores[1],data1.FAcompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
    FAresults=[]    
    for i in range(1,maxdims[subset]):
        FAresults.append(HW3Subs.RunClassifiers(data1.FAcompscores[0][:,:i],data1.FAcompscores[1][:,:i],data1.FAcompscores[2][:,:i],data1.labelstrain,data1.labelsval,["Linear SVM"]))
        print FAresults[i-1][1]
#    plt.plot(range(len(data1.princomp.explained_variance_ratio_)),data1.princomp.explained_variance_ratio_)
#    plt.plot([0,len(data1.princomp.explained_variance_ratio_)-1],data1.princomp.explained_variance_ratio_[[0,-1]])
#    data1.princomp.fit(data1.data)
#    data1.princomp.explained_variance_ratio_
#    
#    results=HW3Subs.RunClassifiers(data1.datatrain,data1.dataval,data1.datatest,data1.labelstrain,data1.labelsval,["Linear SVM"])


