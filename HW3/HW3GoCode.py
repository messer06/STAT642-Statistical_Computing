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
from time import time


datasetnames=["ARCENE","DEXTER","DOROTHEA","GISETTE","MADELON"]
numdims=[300,20000,100000,5000,500]
BinaryData=[False,False,True,False,True]
subsetter=[(0),1,2,3,4]
maxdims=[100,5000,10000,5000,500]
skipper=[True,True,True,True,False]
youndenJ=sklearn.metrics.make_scorer(HW3Subs.Youdens_func)

results1,results2,results3,results4,results5=[],[],[],[],[]
grid_search_results1,grid_search_results2,grid_search_results3,grid_search_results4,grid_search_results5=[],[],[],[],[]
#model1,model2=[],[]
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
#        HW3Subs.PltCorrMatrix(data1.corr,datasetname)
        crossvalsplit=sklearn.cross_validation.PredefinedSplit(test_fold=np.hstack((-np.ones((1,data1.datatrain.shape[0])),np.zeros((1,data1.dataval.shape[0])))).transpose())
        ####PCA THEN Percentile Best AND LOGISTIC####
        data1.PCA()
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.linear_model.LogisticRegression()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(2,60,4),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=-1,cv=crossvalsplit,refit=False,scoring=youndenJ)
        grid_search.fit(np.vstack((data1.princompscores[0],data1.princompscores[1])),np.hstack((data1.labelstrain,data1.labelsval)))
        kwargs=grid_search.best_params_
        filterclf.set_params(**kwargs)
        filterclf.fit(data1.princompscores[0],data1.labelstrain)
        results1.append([sklearn.metrics.confusion_matrix(data1.labelstrain,filterclf.predict(data1.princompscores[0])),
                         sklearn.metrics.confusion_matrix(data1.labelsval,filterclf.predict(data1.princompscores[1]))])
        grid_search_results1.append(grid_search.grid_scores_)
        
        ####Feature Selection, PCA, AND LinearSVM####
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        dec = sklearn.decomposition.PCA()
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('PPCA',dec),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(5,40,2),
                        class__C=np.logspace(-4,4,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=-1,cv=crossvalsplit,scoring=youndenJ)
        grid_search.fit(np.vstack((data1.datatrain,data1.dataval)),np.hstack((data1.labelstrain,data1.labelsval)))
        
        kwargs=grid_search.best_params_
        filterclf.set_params(**kwargs)
        filterclf.fit(data1.datatrain,data1.labelstrain)
        results2.append([sklearn.metrics.confusion_matrix(data1.labelstrain,filterclf.predict(data1.datatrain)),
                         sklearn.metrics.confusion_matrix(data1.labelsval,filterclf.predict(data1.dataval))])
        grid_search_results2.append(grid_search.grid_scores_)
        
        ####PPCA, AND linearSVM####
        data1.KernelPCA('RBF')
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(5,50,2),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=crossvalsplit,scoring=youndenJ)
        grid_search.fit(np.vstack((data1.kernelprincompscores[0],data1.kernelprincompscores[1])),np.hstack((data1.labelstrain,data1.labelsval)))
        kwargs=grid_search.best_params_
        filterclf.set_params(**kwargs)
        filterclf.fit(data1.kernelprincompscores[0],data1.labelstrain)
        results3.append([sklearn.metrics.confusion_matrix(data1.labelstrain,filterclf.predict(data1.kernelprincompscores[0])),
                         sklearn.metrics.confusion_matrix(data1.labelsval,filterclf.predict(data1.kernelprincompscores[1]))])
        grid_search_results3.append(grid_search.grid_scores_)    
        
        ####FA, AND LDA####
        data1.KernelPCA('Linear')
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(2,40,2),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=crossvalsplit,scoring=youndenJ)
        grid_search.fit(np.vstack((data1.kernelprincompscores[0],data1.kernelprincompscores[1])),np.hstack((data1.labelstrain,data1.labelsval)))
        kwargs=grid_search.best_params_
        filterclf.set_params(**kwargs)
        filterclf.fit(data1.kernelprincompscores[0],data1.labelstrain)
        results4.append([sklearn.metrics.confusion_matrix(data1.labelstrain,filterclf.predict(data1.kernelprincompscores[0])),
                         sklearn.metrics.confusion_matrix(data1.labelsval,filterclf.predict(data1.kernelprincompscores[1]))])
        grid_search_results4.append(grid_search.grid_scores_)    
        
        ####LDA ON ORIGINAL######
        FS = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
        clf = sklearn.svm.LinearSVC()
        filterclf = Pipeline([('feature_selection',FS),('class',clf)])
        param_grid=dict(feature_selection__percentile=range(2,40,2),
                        class__C=np.logspace(-4,2,6))
        grid_search = sklearn.grid_search.GridSearchCV(filterclf,param_grid=param_grid,verbose=10,n_jobs=1,cv=crossvalsplit,scoring=youndenJ)
        grid_search.fit(np.vstack((data1.datatrain,data1.dataval)),np.hstack((data1.labelstrain,data1.labelsval)))
        kwargs=grid_search.best_params_
        filterclf.set_params(**kwargs)
        filterclf.fit(data1.datatrain,data1.labelstrain)
        results5.append([sklearn.metrics.confusion_matrix(data1.labelstrain,filterclf.predict(data1.datatrain)),
                         sklearn.metrics.confusion_matrix(data1.labelsval,filterclf.predict(data1.dataval))])
        grid_search_results5.append(grid_search.grid_scores_)        


np.save("results1",results1)
np.save("results2",results2)
np.save("results3",results3)
np.save("results4",results4)
np.save("results5",results5)

np.save("grid_search_results1",grid_search_results1)
np.save("grid_search_results2",grid_search_results2)
np.save("grid_search_results3",grid_search_results3)
np.save("grid_search_results4",grid_search_results4)
np.save("grid_search_results5",grid_search_results5)