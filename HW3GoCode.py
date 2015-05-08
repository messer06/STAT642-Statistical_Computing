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
skipper=[False,False,True,True,False]


results1,results2,results3,results4,results5=[],[],[],[],[]
grid_search_results1,grid_search_results2,grid_search_results3,grid_search_results4,grid_search_results5=[],[],[],[],[]
model1,model2=[],[]
for datasetname,numdims,binvars,subset,skip in zip(datasetnames,numdims,BinaryData,subsetter,skipper):
    if not skip:
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
        HW3Subs.PltCorrMatrix(data1.corr,datasetname)

        ####PCA THEN Percentile Best AND LOGISTIC####
        data1.PCA()
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.linear_model.LogisticRegression()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(2,50,6),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=2)
        grid_search.fit(np.vstack((data1.princompscores[0],data1.princompscores[1])),np.hstack((data1.labelstrain,data1.labelsval)))
        results1.append([sklearn.metrics.confusion_matrix(data1.labelstrain,grid_search.best_estimator_.predict(data1.princompscores[0])),
                         sklearn.metrics.confusion_matrix(data1.labelsval,grid_search.best_estimator_.predict(data1.princompscores[1]))])
        grid_search_results1.append(grid_search.grid_scores_)
        
        ####Feature Selection, PCA, AND LinearSVM####
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        dec = sklearn.decomposition.ProbabilisticPCA()
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('PPCA',dec),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(10,30,6),
                        class__C=np.logspace(-4,4,3))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=2)
        grid_search.fit(np.vstack((data1.datatrain,data1.dataval)),np.hstack((data1.labelstrain,data1.labelsval)))
        results2.append([sklearn.metrics.confusion_matrix(data1.labelstrain,grid_search.best_estimator_.predict(data1.datatrain)),
                         sklearn.metrics.confusion_matrix(data1.labelsval,grid_search.best_estimator_.predict(data1.dataval))])
        grid_search_results2.append(grid_search.grid_scores_)
        
        ####PPCA, AND linearSVM####
        data1.KernelPCA('RBF')
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(2,50,6),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=2)
        grid_search.fit(np.vstack((data1.kernelprincompscores[0],data1.kernelprincompscores[1])),np.hstack((data1.labelstrain,data1.labelsval)))
        results3.append([sklearn.metrics.confusion_matrix(data1.labelstrain,grid_search.best_estimator_.predict(data1.kernelprincompscores[0])),
                         sklearn.metrics.confusion_matrix(data1.labelsval,grid_search.best_estimator_.predict(data1.kernelprincompscores[1]))])
        grid_search_results3.append(grid_search.grid_scores_)    
        
        ####FA, AND LDA####
        data1.KernelPCA('Linear')
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(2,20,6),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=2)
        grid_search.fit(np.vstack((data1.kernelprincompscores[0],data1.kernelprincompscores[1])),np.hstack((data1.labelstrain,data1.labelsval)))
        results4.append([sklearn.metrics.confusion_matrix(data1.labelstrain,grid_search.best_estimator_.predict(data1.kernelprincompscores[0])),
                         sklearn.metrics.confusion_matrix(data1.labelsval,grid_search.best_estimator_.predict(data1.kernelprincompscores[1]))])
        grid_search_results4.append(grid_search.grid_scores_)    
        
        ####LDA ON ORIGINAL######
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(2,20,6),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=2)
        grid_search.fit(np.vstack((data1.datatrain,data1.dataval)),np.hstack((data1.labelstrain,data1.labelsval)))
        results5.append([sklearn.metrics.confusion_matrix(data1.labelstrain,grid_search.best_estimator_.predict(data1.datatrain)),
                         sklearn.metrics.confusion_matrix(data1.labelsval,grid_search.best_estimator_.predict(data1.dataval))])
        grid_search_results5.append(grid_search.grid_scores_)        

