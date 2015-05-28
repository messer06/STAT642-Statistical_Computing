import sklearn
from sklearn import decomposition as decomp
import numpy as np
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from time import time
from sklearn import feature_selection
from sklearn.linear_model import SGDClassifier
from scipy import sparse

def Youdens_func(y,y_pred):
    confmat=sklearn.metrics.confusion_matrix(y,y_pred).astype(float)
    return (confmat[0,0]/(confmat[0,0]+confmat[1,0])+confmat[1,1]/(confmat[1,1]+confmat[0,1])-1)

def Youdens_matfunc(confmat):
    return (confmat[0,0]/(confmat[0,0]+confmat[1,0])+confmat[1,1]/(confmat[1,1]+confmat[0,1])-1)

    
def ImportSparse(datasetname,settype="train",binvars=False,numcols=[]):
    ds=[]
    with open(datasetname+"/"+datasetname+"/"+datasetname.lower()+"_"+settype+".data",'r') as file:
        for line in file:
            ds.append(line.split(" ")[:-2]) 
    i=0
    rows,cols,vals=[],[],[]
    for d in ds:
        for loc in d:
            rows.append(int(i))
            if binvars:
                cols.append(int(loc))
                vals.append(int(1))
            else:
                entry = loc.split(":")
                cols.append(int(entry[0]))
                vals.append(int(entry[1]))
        i+=1
    if numcols==[]:numcols=max(cols)+1
    return sparse.csc_matrix((vals,(rows,cols)),shape=(i,numcols)).toarray()
        
def PltCorrMatrix(corr,datasetname=[]):
    corrfig=plt.figure()
    plt.imshow(corr, cmap='hot', interpolation='none',vmin=0,vmax=1)
    plt.colorbar()
    plt.savefig(os.path.join(os.getcwd(),'Corr'+datasetname+'.pdf'))
    plt.close()
    
def PltHistograms(datamatrix,dataset=-1,title="Histogram"):
    numrows = int(math.ceil(scipy.sqrt(len(datamatrix[1,:]))))
    
    mng=plt.get_current_fig_manager()
    mng.full_screen_toggle()
    histfig=plt.figure()
    for row in range(numrows):
        for col in range(numrows):
            subplotnum=row*(numrows)+col
            if subplotnum <= len(datamatrix[1,:]):
                plt.subplot(numrows,numrows,subplotnum)           
                plt.hist(datamatrix[:,subplotnum],10)
                plt.xticks(fontsize='4')
                plt.yticks(fontsize='4')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(os.getcwd(),'Histogram%d.pdf' %dataset))
    plt.close()
    
def PltXYScatters(xmatrix,yvector,dataset=-1):
    numrows = int(math.ceil(scipy.sqrt(len(xmatrix[1,:]))))
#    mng=plt.get_current_fig_manager()
#    mng.full_screen_toggle()
    xyscatterfig=plt.figure()
    
    for row in range(numrows):
        for col in range(numrows):
            subplotnum=row*(numrows)+col
            if subplotnum <= len(xmatrix[1,:]):
                plt.subplot(numrows,numrows,subplotnum)           
                plt.plot(xmatrix[:,subplotnum],yvector)
                plt.xticks(fontsize='4')
                plt.yticks(fontsize='4')
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(os.getcwd(),'XYScatter%d.pdf' %dataset))
    plt.close()
 
def PltResidHist(residuals,title,dataset,datasetsize,regtype):
    histfig=plt.figure()        
    plt.hist(residuals,10)
    plt.title(title + 'Residuals',fontsize=20)
    plt.savefig(os.path.join(os.getcwd(),regtype+ title + 'ResidHist'+datasetsize+'%d.pdf' %dataset))
    plt.close()
    
def RunClassifiers(X_train,X_val,X_test,y_train,y_val,names):
    masternames = ["Linear SVM","RBF SVM","Decision Tree", "QDA","AdaBoost","Stochastic Descent"] #, , , , "Random Forest",  "Naive Bayes", "LDA"] #"AdaBoost",
    masterclassifiers = [SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1),DecisionTreeClassifier(max_depth=5),QDA(),AdaBoostClassifier(),SGDClassifier(loss="hinge",penalty="l2")]#, , , , RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  GaussianNB(), LDA(), QDA()] #AdaBoostClassifier(),    
    classifiers = []
    for name in names:
        classifiers.append(masterclassifiers[masternames.index(name)])
    LinSVMTestLevels=np.linspace(0.25,10,30)
    DecisionTreeTestLevels=range(1,8)
    QDATestLevels=np.linspace(-5,5,100)
    AdaBoostTestLevels=range(1,50,4)
    RBFTestLevelsC=np.linspace(0.25,2,10)
    RBFTestLevelsgamma=np.linspace(0,.2,10)
    StochDescTestLevels=[]    
    a,b=np.meshgrid(RBFTestLevelsC,RBFTestLevelsgamma)
    RBFTestLevelsgamma=np.linspace(0,3,.2)
    RBFTestLevels=np.vstack([a.ravel(),b.ravel()])
    confusion_matrices_train,confusion_matrices_test,confusion_matrices_val=[],[],[]
    predicttrain,predicttest,predictval=[],[],[]
    times=[]
    for name, clf in zip(names, classifiers):
        timestart=time()
        if name not in ["Decision Tree","QDA","AdaBoost","Linear SVM","RBF SVM"]:
            clf.fit(X_train, y_train) 

        elif name =="Linear SVM":
            score_train=[]
            score_val=[]        
            for test_param in LinSVMTestLevels:
    #            print test_param
                clf=SVC(kernel="linear", C=test_param)
                clf.fit(X_train, y_train) 
                score_train.append(clf.score(X_train, y_train))
                score_val.append(clf.score(X_val, y_val))
            score_maxidx,score_max = max(enumerate(score_val),key=itemgetter(1))
            print LinSVMTestLevels[score_maxidx]
            clf=SVC(kernel="linear", C=LinSVMTestLevels[score_maxidx])
        elif name =="Decision Tree":
            score_train=[]
            score_val=[]        
            for num_levels in DecisionTreeTestLevels:
                clf=DecisionTreeClassifier(max_depth=num_levels)
                clf.fit(X_train, y_train) 
                score_train.append(clf.score(X_train, y_train))
                score_val.append(clf.score(X_val, y_val))
            score_minidx,score_min = min(enumerate(score_val),key=itemgetter(1))
            clf=DecisionTreeClassifier(max_depth=DecisionTreeTestLevels[score_minidx])
            print DecisionTreeTestLevels[score_minidx]
        elif name =="QDA":
            score_train=[]
            score_val=[]        
            for test_param in QDATestLevels: 
                np.exp(test_param)
                clf=QDA(reg_param=np.exp(test_param))
                clf.fit(X_train, y_train) 
                score_train.append(clf.score(X_train, y_train))
                score_val.append(clf.score(X_val, y_val))
            score_maxidx,score_max = max(enumerate(score_val),key=itemgetter(1))
            clf=QDA(reg_param=np.exp(QDATestLevels[score_maxidx]))
            print np.exp(QDATestLevels[score_maxidx])
        elif name =="RBF SVM":
            score_train=[]
            score_val=[]        
            for test_param in range(RBFTestLevels.shape[1]):
                clf=SVC(gamma=RBFTestLevels[1,test_param], C=RBFTestLevels[0,test_param])
                clf.fit(X_train, y_train) 
                score_train.append(clf.score(X_train, y_train))
                score_val.append(clf.score(X_val, y_val))
            score_maxidx,score_max = max(enumerate(score_val),key=itemgetter(1))
            clf=SVC(gamma=RBFTestLevels[1,score_maxidx], C=RBFTestLevels[0,score_maxidx])
            print RBFTestLevels[:,score_maxidx]
        elif name =="AdaBoost":
            score_train=[]
            score_val=[]        
            for test_param in AdaBoostTestLevels:
                clf=AdaBoostClassifier(n_estimators=test_param)
                clf.fit(X_train, y_train) 
                score_train.append(clf.score(X_train, y_train))
                score_val.append(clf.score(X_val, y_val))
            score_maxidx,score_max = max(enumerate(score_val),key=itemgetter(1))
            clf=AdaBoostClassifier(n_estimators=AdaBoostTestLevels[score_maxidx])
            print AdaBoostTestLevels[score_maxidx]
        clf.fit(X_train, y_train) 
        score_train=clf.score(X_train, y_train)
        print score_train
        score_val=clf.score(X_val,y_val)
        try:
            predicttrain=vstack((predicttrain,np.transpose(np.array(clf.predict(X_train)))))
            predicttest=vstack((predicttest,np.transpose(np.array(clf.predict(X_test)))))
            predictval=vstack((predictval,np.transpose(np.array(clf.predict(X_val)))))
        except:
            predicttrain=np.transpose(np.array(clf.predict(X_train)))
            predicttest=np.transpose(np.array(clf.predict(X_test)))
            predictval=np.transpose(np.array(clf.predict(X_val)))
        try:
            scores=vstack([scores,[score_train,score_val]])
        except:
            scores=np.array([score_train,score_val])
        confusion_matrices_train.append(confusion_matrix(y_train,clf.predict(X_train)))
        confusion_matrices_val.append(confusion_matrix(y_val,clf.predict(X_val)))
        times.append(time()-timestart)
    return confusion_matrices_train,confusion_matrices_val,predicttrain,predictval,predicttest
    
