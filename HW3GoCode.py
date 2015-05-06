# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:14:20 2015

@author: amesser
"""

import HW3Classes
import HW3Subs
import numpy as np
import sklearn
from sklearn import decomposition as decomp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import feature_selection as featsel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA

datasetnames=["ARCENE","DEXTER","DOROTHEA","GISETTE","MADELON"]
numdims=[300,20000,100000,5000,500]
BinaryData=[False,False,True,False,True]
subsetter=[(0),1,2,3,4]
maxdims=[100,5000,5000,5000,500]

results1,results2,results3,results4=[],[],[],[]
model1,model2=[],[]
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
#    data1.PCA()
#    results.append(HW3Subs.RunClassifiers(data1.princompscores[0],data1.princompscores[1],data1.princompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
#    pcaresults=[]    
#    for i in range(1,maxdims[subset]):
#        pcaresults.append(HW3Subs.RunClassifiers(data1.princompscores[0][:,:i],data1.princompscores[1][:,:i],data1.princompscores[2][:,:i],data1.labelstrain,data1.labelsval,["AdaBoost"]))
#        print pcaresults[i-1][1]
#    data1.ICA()
#    results.append(HW3Subs.RunClassifiers(data1.indcompscores[0],data1.indcompscores[1],data1.indcompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
#    data1.KernelPCA("RBF",sparseversion=sparseversion)
#    results.append(HW3Subs.RunClassifiers(data1.kernelprincompscores[0],data1.kernelprincompscores[1],data1.kernelprincompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
#    data1.KernelPCA("sigmoid",sparseversion=sparseversion)
#    results.append(HW3Subs.RunClassifiers(data1.kernelprincompscores[0],data1.kernelprincompscores[1],data1.kernelprincompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))


    ####PCA AND LOGISTIC####
    FS = sklearn.decomposition.PCA()
    clf = sklearn.linear_model.LogisticRegression()
    filterclf = Pipeline([('feature_selection',FS),('class',clf)])
    param_grid=dict(feature_selection__n_components=range(10,101,10),
                    class__C=np.logspace(-4,4,3))
    grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=2,cv=2)
    grid_search.fit(np.vstack((data1.datatrain,data1.dataval)),np.hstack((data1.labelstrain,data1.labelsval)))
    results1.append([sklearn.metrics.confusion_matrix(data1.labelstrain,grid_search.best_estimator_.predict(data1.datatrain)),
                     sklearn.metrics.confusion_matrix(data1.labelsval,grid_search.best_estimator_.predict(data1.dataval))])
        
    ####Feature Selection, PCA, AND LDA####
    FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
    dec = sklearn.decomposition.PCA()
    clf = sklearn.svm.LinearSVC()
    filterclf = Pipeline([('feature_selection',FS),('PCA',dec),('class',clf)])
    param_grid=dict(feature_selection__percentile=range(2,50,6),
                    class__C=np.logspace(-4,4,3))
    grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=2,cv=2)
    grid_search.fit(np.vstack((data1.datatrain,data1.dataval)),np.hstack((data1.labelstrain,data1.labelsval)))
    results2.append([sklearn.metrics.confusion_matrix(data1.labelstrain,grid_search.best_estimator_.predict(data1.datatrain)),
                     sklearn.metrics.confusion_matrix(data1.labelsval,grid_search.best_estimator_.predict(data1.dataval))])
#    
#kbest=featsel.SelectKBest(featsel.f_regression,k=20)
#FS = LinearSVC(penalty='l1',dual=False)
#clf = LDA()
#filterclf = Pipeline([('PCA',sklearn.decomposition.PCA()),('feature_selection',FS),('class',clf)])
#param_grid=dict(PCA__n_components=range(1,60),
#                feature_selection__C=np.exp(np.linspace(-5,5,10)))
#grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10)
#grid_search.fit(data1.datatrain,data1.labelstrain)
#filterclf.fit(data1.datatrain,data1.labelstrain)
#filterclf.score(data1.dataval,data1.labelsval)
#
#kbest=featsel.SelectKBest(featsel.f_regression,k=20)
#clf = LinearSVC()
#filterclf = Pipeline([('FA',sklearn.decomposition.FactorAnalysis()),('feature_selection',kbest),('classification',clf)])
#filterclf.fit(data1.datatrain,data1.labelstrain)
#filterclf.score(data1.dataval,data1.labelsval)

#    data1.FA()
#    results.append(HW3Subs.RunClassifiers(data1.FAcompscores[0],data1.FAcompscores[1],data1.FAcompscores[2],data1.labelstrain,data1.labelsval,["Linear SVM"]))
#    FAresults=[]    
#    for i in range(1,maxdims[subset]):
#        FAresults.append(HW3Subs.RunClassifiers(data1.FAcompscores[0][:,:i],data1.FAcompscores[1][:,:i],data1.FAcompscores[2][:,:i],data1.labelstrain,data1.labelsval,["Linear SVM"]))
#        print FAresults[i-1][1]
#    plt.plot(range(len(data1.princomp.explained_variance_ratio_)),data1.princomp.explained_variance_ratio_)
#    plt.plot([0,len(data1.princomp.explained_variance_ratio_)-1],data1.princomp.explained_variance_ratio_[[0,-1]])
#    data1.princomp.fit(data1.data)
#    data1.princomp.explained_variance_ratio_
#    
#    results=HW3Subs.RunClassifiers(data1.datatrain,data1.dataval,data1.datatest,data1.labelstrain,data1.labelsval,["Linear SVM"])

