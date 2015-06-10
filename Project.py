"""
Beating the Benchmark
West Nile Virus Prediction @ Kaggle
__author__ : Abhihsek
"""

#%%
import pandas as pd
import numpy as np
import sklearn.cluster
import sklearn 
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.qda
import pylab as pl
from itertools import compress
def Youdens_func(y,y_pred,weight=0.3):
    confmat=sklearn.metrics.confusion_matrix(y,y_pred).astype(float)
    print confmat
    result= 2*(weight*confmat[0,0]/(confmat[0,0]+confmat[0,1])+(1-weight)*confmat[1,1]/(confmat[1,1]+confmat[1,0]))
    return np.nan_to_num(result)-1

def Youdens_matfunc(confmat):
    return (confmat[0,0]/(confmat[0,0]+confmat[1,0])+confmat[1,1]/(confmat[1,1]+confmat[0,1])-1)

class ItemSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
        
youdenJ=sklearn.metrics.make_scorer(Youdens_func)    
#%%
  
##################### Load dataset ##############################
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
spray = pd.read_csv('input/spray.csv')
sample = pd.read_csv('input/sampleSubmission.csv')
weather = pd.read_csv('input/weather.csv')

print "Data in"
#train=train.sort('WnvPresent')
# Get labels
labels = train.WnvPresent.values
sample2= train.WnvPresent

# Split station 1 and 2 and join horizontally
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')


# replace some missing values and T with -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', 0.001)
weather = weather.replace(' T', 0.001)
weather = weather.replace('  T', 0.001)
weather = weather.ix[:,(weather != -1).any(axis=0)]
weather = weather.drop(['CodeSum_x','CodeSum_y'],axis=1)
weather['T_20day']=pd.ewma(weather.Tmin_x,20)
weather['Dewpoint_20day']=pd.ewma(weather.DewPoint_x,20)
weather['Precip_20day']=pd.ewma(weather.PrecipTotal_x,20)
weather['T_40day']=pd.ewma(weather.Tmin_x,40)
weather['Dewpoint_40day']=pd.ewma(weather.DewPoint_x,40)
weather['Precip_40day']=pd.ewma(weather.PrecipTotal_x,40)
weather['T_60day']=pd.ewma(weather.Tmin_x,60)
weather['Dewpoint_60day']=pd.ewma(weather.DewPoint_x,60)
weather['Precip_60day']=pd.ewma(weather.PrecipTotal_x,60)

labels2=labels
#### Spray data
spray['Date2']=pd.to_datetime(spray['Date'])
train['Date2']=pd.to_datetime(train['Date'])
test['Date2']=pd.to_datetime(test['Date'])
train['Spray']=labels*0
test['Spray']=sample['WnvPresent']*0

#%%
######################## SPRAY DATA#############################
numadded=0
for row,obs in train.iterrows():
    validspray=spray[((spray['Date2'] >= obs.Date2-pd.tseries.offsets.DateOffset(days=14))
    * (spray['Date2']  <= obs.Date2-pd.tseries.offsets.DateOffset(days=1)))]
    train.Spray=0
    if validspray.shape[0]>0:
        distancetovalidspray=sklearn.metrics.pairwise.pairwise_distances(np.array([validspray[:].Latitude.values,validspray[:].Longitude.values]).transpose(),np.array([obs.Latitude,obs.Longitude]).transpose())
        if min(distancetovalidspray) <= .25:
            train.Spray=1  
            labels2[row]=min([1,labels[row]])
            numadded+=1
print "Number added: %d" %numadded
  

#%%

# Functions to extract month and day from dataset
# You can also use parse_dates of Pandas.
def create_month(x):
    return int(x.split('-')[1])

def create_day(x):
    return int(x.split('-')[2])
       

def create_year(x):
    return int(x.split('-')[0])
    
def create_intensity(x):
    year = int(x.split('-')[0])
    intensity=np.interp(year,[2006,2007,2008,2009,2010,2011,2012,2013,2014],[25,139,50,19,50,57,200,136,200])
    return intensity
    
def create_test_folds(x):
    test_fold=np.interp(x,[2007,2009,2011,2013],[0,1,2,3])
    return test_fold


train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
train['intensity'] = train.Date.apply(create_intensity)
train['year'] = train.Date.apply(create_year)
train['week'] = train.Date2.dt.weekofyear
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)
test['intensity'] = test.Date.apply(create_intensity)
test['year'] = test.Date.apply(create_year)
test['week'] = test.Date2.dt.weekofyear
# Add integer latitude/longitude columns
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)

# drop address columns
train = train.drop(['Address', 'AddressNumberAndStreet','WnvPresent'], axis = 1)
test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis = 1)

weather_n_clusters=range(2,16,2)
loc_n_clusters=range(2,40,4)


weathercols=['Tmax_x', 'Tmin_x', 'Tavg_x', 'Depart_x', 'DewPoint_x',
       'WetBulb_x', 'Heat_x', 'Cool_x', u'Sunrise_x', 'Sunset_x',
       'Depth_x', 'SnowFall_x', 'PrecipTotal_x', 'StnPressure_x',
       'SeaLevel_x', 'ResultSpeed_x', 'ResultDir_x', 'AvgSpeed_x',
       'Tmax_y', 'Tmin_y', u'Tavg_y', 'DewPoint_y', 'WetBulb_y', u'Heat_y',
       'Cool_y', 'PrecipTotal_y','StnPressure_y', 'SeaLevel_y',
       'ResultSpeed_y', 'ResultDir_y', 'AvgSpeed_y', 'T_20day',
       'Dewpoint_20day', 'Precip_20day', 'T_40day',
       'Dewpoint_40day', 'Precip_40day']


loccols=['Latitude','Longitude']
passcols=['Tmax_x', 'Tmin_x', 'Tavg_x', 'Depart_x', 'DewPoint_x',
       'WetBulb_x', 'Heat_x', 'Cool_x', u'Sunrise_x', 'Sunset_x',
       'Depth_x', 'SnowFall_x', 'PrecipTotal_x', 'StnPressure_x',
       'SeaLevel_x', 'ResultSpeed_x', 'ResultDir_x', 'AvgSpeed_x',
        'T_20day',  'Dewpoint_20day', 'Precip_20day', 'T_40day',
       'Dewpoint_40day', 'Precip_40day','T_60day',
       'Dewpoint_60day', 'Precip_60day','Latitude','Longitude','week',
       'intensity','Spray','Species', 'Block', 'Trap']

nummoscols=['Tmax_x', 'Tmin_x', 'Tavg_x', 'Depart_x', 'DewPoint_x',
       'WetBulb_x', 'Heat_x', 'Cool_x', u'Sunrise_x', 'Sunset_x',
       'Depth_x', 'SnowFall_x', 'PrecipTotal_x', 'StnPressure_x',
       'SeaLevel_x', 'ResultSpeed_x', 'ResultDir_x', 'AvgSpeed_x',
       'T_20day',
       'Dewpoint_20day', 'Precip_20day', 'T_40day',
       'Dewpoint_40day', 'Precip_40day','T_60day',
       'Dewpoint_60day', 'Precip_60day','Latitude','Longitude',
'week','year','intensity','Spray','Species', 'Block', 'Trap']

# Merge with weather data
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train = train.drop(['Date','Date2'], axis = 1)
test = test.drop(['Date','Date2'], axis = 1)

# Convert categorical data to numbers
lbl = sklearn.preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)
lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)
lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)

#Convert to Float
train['Tavg_x'] = train['Tavg_x'].astype(float)
test['Tavg_x'] = test['Tavg_x'].astype(float)
train['Tavg_y'] = train['Tavg_y'].astype(float)
test['Tavg_y'] = test['Tavg_y'].astype(float)
train['Depart_x'] = train['Depart_x'].astype(float)
test['Depart_x'] = test['Depart_x'].astype(float)
train['WetBulb_x'] = train['WetBulb_x'].astype(float)
test['WetBulb_x'] = test['WetBulb_x'].astype(float)
train['WetBulb_y'] = train['WetBulb_y'].astype(float)
test['WetBulb_y'] = test['WetBulb_y'].astype(float)
train['Heat_x'] = train['Heat_x'].astype(float)
test['Heat_x'] = test['Heat_x'].astype(float)
train['Heat_y'] = train['Heat_y'].astype(float)
test['Heat_y'] = test['Heat_y'].astype(float)
train['Cool_x'] = train['Cool_x'].astype(float)
test['Cool_x'] = test['Cool_x'].astype(float)
train['Cool_y'] = train['Cool_y'].astype(float)
test['Cool_y'] = test['Cool_y'].astype(float)
train['Sunrise_x'] = train['Sunrise_x'].astype(float)
test['Sunrise_x'] = test['Sunrise_x'].astype(float)
train['Sunset_x'] = train['Sunset_x'].astype(float)
test['Sunset_x'] = test['Sunset_x'].astype(float)
train['Depth_x'] = train['Depth_x'].astype(float)
test['Depth_x'] = test['Depth_x'].astype(float)
train['SnowFall_x'] = train['SnowFall_x'].astype(float)
test['SnowFall_x'] = test['SnowFall_x'].astype(float)
train['StnPressure_x'] = train['StnPressure_x'].astype(float)
test['StnPressure_x'] = test['StnPressure_x'].astype(float)
train['StnPressure_y'] = train['StnPressure_y'].astype(float)
test['StnPressure_y'] = test['StnPressure_y'].astype(float)
train['SeaLevel_x'] = train['SeaLevel_x'].astype(float)
test['SeaLevel_x'] = test['SeaLevel_x'].astype(float)
train['SeaLevel_y'] = train['SeaLevel_y'].astype(float)
test['SeaLevel_y'] = test['SeaLevel_y'].astype(float)
train['PrecipTotal_x'] = train['PrecipTotal_x'].astype(float)
test['PrecipTotal_x'] = test['PrecipTotal_x'].astype(float)
train['PrecipTotal_y'] = train['PrecipTotal_y'].astype(float)
test['PrecipTotal_y'] = test['PrecipTotal_y'].astype(float)
train['AvgSpeed_x'] = train['AvgSpeed_x'].astype(float)
test['AvgSpeed_x'] = test['AvgSpeed_x'].astype(float)
train['AvgSpeed_y'] = train['AvgSpeed_y'].astype(float)
test['AvgSpeed_y'] = test['AvgSpeed_y'].astype(float)

## drop columns with -1s
#train = train.ix[:,(train != -1).any(axis=0)]
#test = test.ix[:,(test != -1).any(axis=0)]


## Create Test Folds
testidx = create_test_folds(train.year)

#%%
############################# Predict Mosquito presence and number###########

print "Moquito Prediction"
train['MosPresent']=(train['NumMosquitos']>=6)*1
rfecv2 = sklearn.feature_selection.RFECV(
    estimator=sklearn.linear_model.LogisticRegression(C=200), 
    step=1, cv=sklearn.cross_validation.PredefinedSplit(testidx),
    scoring=youdenJ)
    
rfecv2.fit(train.drop(['NumMosquitos','MosPresent'],axis=1),train.MosPresent)



test['MosPresent']=rfecv2.estimator_.predict(
            test[test.columns[rfecv2.support_]])
nummospipeline = sklearn.pipeline.Pipeline([
    ('selector', ItemSelector(key=nummoscols)),
    ('reg', sklearn.ensemble.RandomForestClassifier(n_estimators=3,
                                                    min_samples_split=2,
                                                    n_jobs=2))
    ])
param_grid = dict(reg__n_estimators=range(1,16,3),
                  reg__max_features=range(4,55,3),
                  reg__min_samples_split=range(2,7,4),
                  reg__min_samples_leaf=range(1,8,3),
                  reg__min_weight_fraction_leaf=[0]
                  )
#grid_search2 = sklearn.grid_search.GridSearchCV(
#    nummospipeline, n_jobs=1, param_grid=param_grid, verbose=100,
#    cv=sklearn.cross_validation.PredefinedSplit(testidx))
#grid_search2.fit(train.drop('NumMosquitos',axis=1),train.MosPresent)
#print grid_search2.best_params_
#test['MosPresent'] = pd.Series(grid_search2.best_estimator_.predict(test))
nummospipeline.fit(train.drop(['NumMosquitos'],axis=1),train.MosPresent)
test['NumMosquitos'] = pd.Series(nummospipeline.predict(test))
print "Main"
train=train.drop(['NumMosquitos'],axis=1)

#%%
#######################WNV Prediction#####################################


pipeline = sklearn.pipeline.Pipeline([
    # Use FeatureUnion to combine the features from subject and body
#    ('union', sklearn.pipeline.FeatureUnion(
#        transformer_list=[
#            # Pipeline for pulling features from the post's subject line
##            ('weather', sklearn.pipeline.Pipeline([
##                ('selector', ItemSelector(key=weathercols)),
##                ('cluster', sklearn.cluster.KMeans(n_clusters=20)),
##            ])),
#
#            # Pipeline for standard bag-of-words model for body
##            ('loc', sklearn.pipeline.Pipeline([
##                ('selector', ItemSelector(key=loccols)),
##                ('cluster', sklearn.cluster.KMeans(n_clusters=20)),
##            ])),
#
#            # Pipeline for pulling ad hoc features from post's body
##            ('pass', sklearn.pipeline.Pipeline([
##                ('selector', ItemSelector(key=passcols)),
##                ('filter',sklearn.feature_selection.SelectKBest(
##                score_func=sklearn.feature_selection.f_regression,k=10))
##            ])),
#            
#            
##            ('nmf', sklearn.pipeline.Pipeline([
##                ('selector', ItemSelector(key=passcols)),
##                ('decomp',sklearn.decomposition.TruncatedSVD(n_components=0))
##            ])),
#
#        ],
#
#    )),
#    ('selector', ItemSelector(key=passcols)),
    # Use a SVC classifier on the combined features
#    ('clf', sklearn.svm.SVC(kernel='rbf',C=100,gamma=0.1))
#    ('clf', sklearn.ensemble.AdaBoostClassifier(n_estimators=500))
#    ('clf', sklearn.neighbors.KNeighborsClassifier(n_neighbors=40))
    ('clf', sklearn.ensemble.RandomForestClassifier(n_estimators=7,
                min_samples_leaf=1,min_samples_split=8,verbose=0,n_jobs=2,))
])

FactAnal=sklearn.decomposition.PCA()
trainFact=FactAnal.fit_transform(train)
testFact=FactAnal.transform(test)

rfecv = sklearn.feature_selection.RFECV(
    estimator=sklearn.linear_model.LogisticRegression(C=200), 
    step=1, cv=sklearn.cross_validation.PredefinedSplit(testidx),
    scoring=youdenJ)
rfecv.fit(train, labels)

####### Result Mask 
#[u'Species', u'Latitude', u'Longitude', u'AddressAccuracy', u'day',
#       u'year', u'week', u'Long_int', u'Tmax_x', u'Tmin_x', u'Tavg_x',
#       u'Depart_x', u'DewPoint_x', u'WetBulb_x', u'Heat_x', u'Cool_x',
#       u'PrecipTotal_x', u'StnPressure_x', u'SeaLevel_x', u'ResultSpeed_x',
#       u'Tmin_y', u'Tavg_y', u'WetBulb_y', u'Cool_y', u'PrecipTotal_y',
#       u'StnPressure_y', u'SeaLevel_y', u'ResultSpeed_y', u'AvgSpeed_y',
#       u'T_20day', u'Dewpoint_20day', u'Precip_20day', u'Dewpoint_40day',
#       u'Precip_40day', u'T_60day', u'Dewpoint_60day', u'Precip_60day',
#       u'MosPresent']

pipeline = sklearn.pipeline.Pipeline([
    # Use FeatureUnion to combine the features from subject and body


    ('clf', sklearn.ensemble.GradientBoostingClassifier(
    n_estimators=100,min_samples_leaf=3,min_samples_split=3,verbose=0))
])


param_grid = dict(clf__n_estimators=range(100,207,200),
                  clf__max_features=range(2,shape(trainFact[:,rfecv.support_])[1],4),
                  clf__min_samples_split=range(1,4,3),
                  clf__min_samples_leaf=range(2,4,1),
                  clf__min_weight_fraction_leaf=[0],

                    )
#grid_search = sklearn.grid_search.GridSearchCV(
#    pipeline, n_jobs=1, param_grid=param_grid, verbose=100,
#    scoring=youdenJ,score_func=youdenJ,
#    cv=sklearn.cross_validation.PredefinedSplit(testidx))
#grid_search.fit(trainFact[:,rfecv.support_], labels)
#results1=([sklearn.metrics.confusion_matrix(labels,grid_search.best_estimator_.predict(train))])
#grid_search_results1=(grid_search.grid_scores_)
#kwargs=grid_search.best_params_
#pipeline.set_params(**kwargs)

pipeline.fit(train[train.columns[rfecv.support_]],labels)

predictions=(pipeline.predict_proba(test[train.columns[rfecv.support_]])[:,1]>=0.02)*1
predictionstrain=(pipeline.predict_proba(train[train.columns[rfecv.support_]])[:,1]>=0.02)*1


print Youdens_func(labels,predictionstrain)

# create predictions and submission file
sample['WnvPresent'] = predictions
sample.to_csv('testpredicts5.csv', index=False)

print sum(predictions)




#%%
##########################ROC Plots ###########################################
for yr in [2007,2009,2011,2013]:
    pipeline.fit(train[train.year!=yr][train.columns[rfecv.support_]],
                 list(compress(labels,(train.year!=yr).tolist())))
    predictionsval=(pipeline.predict_proba(train[train.year==yr][test.columns[rfecv.support_]]))[:,1]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        list(compress(labels,(train.year==yr).tolist())), predictionsval)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # Plot ROC curve
    #    pl.clf()
    pl.plot(fpr, tpr, label='%d ROC curve (area = %0.2f)' %(yr, roc_auc))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.show()

logreg=sklearn.linear_model.LogisticRegression(C=200)
pl.figure()
for yr in [2007,2009,2011,2013]:
    logreg.fit(train[train.year!=yr][train.columns[rfecv.support_]],
               list(compress(labels,(train.year!=yr).tolist())))
    predictionsval=(logreg.predict_proba(train[train.year==yr][test.columns[rfecv.support_]]))[:,1]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        list(compress(labels,(train.year==yr).tolist())), predictionsval)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # Plot ROC curve
    
    pl.plot(fpr, tpr, label='%d ROC curve (area = %0.2f)' %(yr, roc_auc))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.show()
    
    
logreg=sklearn.linear_model.LogisticRegression(C=200)
pl.figure()
for yr in [2007,2009,2011,2013]:
    logreg.fit(train[train.year!=yr][train.columns[rfecv2.support_]],
               train[train.year!=yr]['MosPresent'])
    predictionsval=(logreg.predict_proba(train[train.year==yr][test.columns[rfecv2.support_]]))[:,1]
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(train[train.year==yr]['MosPresent'], predictionsval)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    # Plot ROC curve
    
    pl.plot(fpr, tpr, label='%d ROC curve (area = %0.2f)' %(yr, roc_auc))
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.show()
    
#%%
############################## Histogram Plots #############################
for column in (train.columns):
    pl.hist([np.array(train[column][labels==0]),
             np.array(train[column][labels==1])],label=['No WNV','WNV'])
    pl.title(column)
    pl.legend(loc="upper right")
    pl.savefig("Figures/Hist"+column+".pdf",bbox_inches="tight")
    pl.close()