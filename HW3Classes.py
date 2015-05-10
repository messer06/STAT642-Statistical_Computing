# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:00:22 2015

@author: amesser
"""
import sklearn
from sklearn import decomposition as decomp
import numpy as np
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn import feature_selection


class DataSet:
    def __init__(self,name,data=[[],[],[]],labels=[],numdims=[]):
        self.name = name
        self.datatrain,self.dataval,self.datatest = data
        self.labelstrain,self.labelsval = labels
        self.test=[]
        self.numdims=numdims
        if data != [[],[],[]]:
            self.corr = np.corrcoef(np.vstack((self.labelstrain.transpose(),self.datatrain.transpose())))

    def add_data(self,setname=[],data=[]):
        if setname=="datatrain":
            self.datatrain=data
            self.labelstrain.transpose()
#            self.corr = np.corrcoef(np.vstack((self.labelstrain.transpose(),self.datatrain.transpose())))
        elif setname=="dataval": self.dataval=data
        elif setname=="datatest": self.datatest=data
        
    def PCA(self,n_comps=[]):
        if n_comps==[]: n_comps=self.datatrain.shape[1]
        self.princomp=decomp.PCA()
        self.princomp.fit(self.datatrain)
        if n_comps==self.numdims:
            scree = np.vstack((np.array(range(len(self.princomp.explained_variance_))),self.princomp.explained_variance_))
            X2=scree[:,0]
            X1=scree[:,-1]
            distance = np.dot((X2*np.ones([len(self.princomp.explained_variance_),2]))-scree.transpose(),(X1*np.ones([len(self.princomp.explained_variance_),2])-scree.transpose()).transpose())
            distance=distance/np.dot(X2-X1,(X2-X1).transpose())
            distance=np.diag(distance)
            self.princomp.n_components=min(enumerate(distance),key=itemgetter(1))[0]+1
        self.princompscores=[self.princomp.transform(self.datatrain),self.princomp.transform(self.dataval),self.princomp.transform(self.datatest)]
        self.princompscores[0]=np.nan_to_num(self.princompscores[0])
        self.princompscores[1]=np.nan_to_num(self.princompscores[1])
        self.princompscores[2]=np.nan_to_num(self.princompscores[2])
        
    def KernelPCA(self,kernel,n_comps=[],sparseversion=False):
        if sparseversion:
            if n_comps==[]: n_comps=self.numdims
            self.kernelprincomp=decomp.KernelPCA(kernel,remove_zero_eig=False)
            self.kernelprincomp.fit(self.datatrain.toarray())
            self.kernelprincompscores=[self.kernelprincomp.transform(self.datatrain.toarray()),self.kernelprincomp.transform(self.dataval.toarray()),self.kernelprincomp.transform(self.datatest.toarray())]
            
        else:
            if n_comps==[]: n_comps=self.numdims
            self.kernelprincomp=decomp.KernelPCA(kernel,remove_zero_eig=False)
            self.kernelprincomp.fit(self.datatrain)
            self.kernelprincompscores=[self.kernelprincomp.transform(self.datatrain),self.kernelprincomp.transform(self.dataval),self.kernelprincomp.transform(self.datatest)]
        self.kernelprincompscores[0]=np.nan_to_num(self.kernelprincompscores[0])
        self.kernelprincompscores[1]=np.nan_to_num(self.kernelprincompscores[1])
        self.kernelprincompscores[2]=np.nan_to_num(self.kernelprincompscores[2])
    
    def ICA(self,n_comps=[]):
        if n_comps==[]: n_comps=self.datatrain.shape[1]
        self.indcomp=decomp.FastICA()
        self.indcomp.fit(self.datatrain)
        self.indcompscores=[self.indcomp.transform(self.datatrain),self.indcomp.transform(self.dataval),self.indcomp.transform(self.datatest)]
    
    def FA(self,n_comps=[]):
        if n_comps==[]: n_comps=self.datatrain.shape[1]
        self.FAcomp=decomp.FactorAnalysis()
        self.FAcomp.fit(self.datatrain)
        self.FAcompscores=[self.FAcomp.transform(self.datatrain),self.FAcomp.transform(self.dataval),self.FAcomp.transform(self.datatest)]
        
    def SparsePCA(self,n_comps=[]):
        if n_comps==[]: n_comps=self.datatrain.shape[1]
        self.sparsecomp=decomp.SparsePCA()
        self.sparsecomp.fit(self.datatrain)
#        if n_comps==self.numdims:
#            scree = np.vstack((np.array(range(len(self.princomp.explained_variance_))),self.princomp.explained_variance_))
#            X2=scree[:,0]
#            X1=scree[:,-1]
#            distance = np.dot((X2*np.ones([len(self.princomp.explained_variance_),2]))-scree.transpose(),(X1*np.ones([len(self.princomp.explained_variance_),2])-scree.transpose()).transpose())
#            distance=distance/np.dot(X2-X1,(X2-X1).transpose())
#            distance=np.diag(distance)
#            self.princomp.n_components=min(enumerate(distance),key=itemgetter(1))[0]+1
        self.sparsecompscores=[self.sparsecomp.transform(self.datatrain),self.sparsecomp.transform(self.dataval),self.sparsecomp.transform(self.datatest)]
        
