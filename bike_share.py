# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:00:42 2021

@author: indu
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score
from math import sqrt
from datetime import datetime
import matplotlib.pyplot as plt
import sdCov as sc

data = pd.read_csv('bike_sh_hour.csv')
#test = pd.read_csv('bike_sh_test.csv')
#train = pd.read_csv('bike_sh_train.csv')

#print(data.info())
#print(test)
#print(train)

#all_data_temp = pd.concat([train, test])
#print(all_data_temp.datetime)
#all_data_temp.info()

#all_data_temp['datetime'] = pd.to_datetime(all_data_temp.datetime)
#
#all_data_temp=all_data_temp.sort_values(by='datetime')
#
#print(all_data_temp.info())
#print(all_data_temp)

#all_data_temp.to_csv('all_data_bike2.csv', index=False)

all_bike_data = pd.read_csv('all_data_bike2.csv')

#all_bike_data['datetime'] = pd.to_datetime(all_bike_data.datetime)
#print(all_bike_data.info())

data = data.join(all_bike_data.datetime)
data['datetime'] = pd.to_datetime(data.datetime)
data=data.sort_values(by='datetime')
#print(data.head())
#print(data.info())

#F_data = data.drop(columns={'hr','dteday'}, axis=1)
F_data = data.set_index('datetime')
print('-------------printing final data------------------')
#print(F_data.head)
print(F_data.info())
#print(F_data.describe())
#print(F_data.columns.isnull().sum())

#print(F_data.nunique())




#print('------------------------------')
'''min-max normalization'''
for column in F_data.columns: 
    if F_data[column].dtype == 'object':
        continue
    F_data[column] = (F_data[column] - F_data[column].min()) / (F_data[column].max() - F_data[column].min())     
print(F_data)
#



print('------------- features and modeling ------------------')
target = 'cnt'
target2 = ['cnt']
datetime = 'datetime'

numerical_features = ['mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed']
categorical_features = ['season', 'holiday', 'weekday', 'workingday', 'weathersit',]

features = numerical_features + categorical_features
features2 = features + target2

X = F_data[features]
Y = F_data[target]

#print(X)
#print(Y)

print('------------- correlation w.r.t target------------------')
df = pd.DataFrame(F_data[features2])
df = df[df.columns[:]].corr()[target][:]
df=df.drop(target)
print(df)
plt.figure()
plt.title('Target correlation with features')
ax = df.plot.bar(stacked=True)
ax.legend(loc='best')
plt.xlabel('features')
plt.ylabel('corr. values')



'''PCA'''
#labl=['Diagnosis','Weight Status','Other problems','W_mean','N1_mean','N2_mean','N3_mean','REM_mean','AR','CA','Awake','OH','MH','OA','REM Aw','MChg','PLM','LM']
pdta = F_data[features]
#pdta = dtass.copy()
pca = PCA()
pca.fit(pdta)
x_pca = pca.transform(pdta)
#print(x_pca.shape)
#print(pdta.shape)
#print(pca.components_)

#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)

plt.figure()
plt.bar(range(1,len(pca.explained_variance_)+1),pca.explained_variance_ratio_,label='pca')
plt.ylabel('Percentage of variance explained')
plt.xlabel('Principal Components')
#plt.plot(range(1,len(pca.explained_variance_ )+1),
#         np.cumsum(pca.explained_variance_),
#         c='red',
#         label="Cumulative Explained Variance")
plt.legend(loc='best')

#pdta23 = dta23.copy()
#pca23 = PCA(n_components=2)
#pca23.fit(pdta23)
#x_pca23 = pca23.transform(pdta)
#print(x_pca23.shape)
#print(pdta23.shape)
#print(pca23.components_)





#
#mea=[]
##for i in range(500):
#x_train, x_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2,
#                                                      random_state=0)
#model = RandomForestRegressor(n_estimators=50, random_state = 0, n_jobs=2)
#model.fit(x_train, y_train)
#
#
#preds = model.predict(x_valid)
#mea.append(mean_absolute_error(preds, y_valid))
#print(mea)
#print(preds[:10])
#print(y_valid.head(10))
#print(F_data.tail())





''' for the 1st mont data is collected '''

print('for the 1st mont data is collected')
reference = F_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current_m = F_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']
production_w1 = F_data.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00']
production_w2 = F_data.loc['2011-02-08 00:00:00':'2011-02-14 23:00:00']
production_w3 = F_data.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00']
production_w4 = F_data.loc['2011-02-21 00:00:00':'2011-02-28 23:00:00']

production = [production_w1 , production_w2 , production_w3]

print(reference[features])
#print(production[features])
#

#import lime
#import lime.lime_tabular
import shap
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor
from math import sqrt
#    
col=['datetime','error_val']
xdat=pd.DataFrame(columns=col)
xdat = xdat.set_index('datetime')

def score_model(xdat,strr, estimators, X_t, y_t, X_v, y_v):
    model4 = RandomForestRegressor(n_estimators=estimators, random_state = 0, n_jobs=2)
    model4.fit(X_t, y_t)
#    model4 = XGBRegressor().fit(X_t, y_t)
    preds = model4.predict(X_v)
    print(preds[:10])
    mae = mean_absolute_error(y_v, preds)
#    val_er = (preds-y_v)**2
    rmse =  sqrt(mean_squared_error(y_v, preds))
#    mse_m = mean_squared_error(y_t, y_v)
    mse_m = ((y_t.mean() - y_v)**2).mean()
    mape = mean_absolute_percentage_error(preds,y_v)
    r2=r2_score(y_v,preds)
    cov=[]
    cov.append(sc.getCoV(preds))
#    error_model=XGBRegressor()
#    error_model.fit(X_v, val_er)
#    mean = model4.predict(X_v)
#    st_dev = error_model.predict(X_v)
#    print(st_dev)
#    print(mean)
    print('mae = ' , mae)
    print('rmse = ', rmse)
    print('mape = ', mape)
    print('MSE when predicting mean : ', mse_m)
    print('r2 score = ', r2)
    print('accuracy = ',round(100-mape.mean(),2),'%')
    print('cov for prediction values = ', cov)
    
#
#    explainer = lime.lime_tabular.LimeTabularExplainer(X_t, feature_names=features, categorical_features = categorical_features, class_names=[target], random_state=0, verbose = True, mode = 'regression')
#    i=25
#    exp=explainer.explain_instance(X_t[i], preds, num_features=6, num_sample=100 )
#    print(exp)

#    roc_auc = roc_auc_score(y_v, preds)
    #    aps = average_precision_score(y_v, preds)
        
    print('------------- correlation w.r.t target------------------')
    fea = X_t.join(y_t)
    fea2 = X_v.join(y_v)
    df = pd.DataFrame(fea[features2])
    df = df[df.columns[:]].corr()[target][:]
    df=df.drop(target)
    df2 = pd.DataFrame(fea2[features2])
    df2 = df2[df2.columns[:]].corr()[target][:]
    df2=df2.drop(target)
#    df2 = df2.dropna(axis=0)
#    for f in range(len(features2)):
#        if df2[f].isnull():
#            df2=df2.drop(f)
#        if df[f] == 'NaN':
#            df=df.drop(df[f])
#            df2=df2.drop(df2[f])
    
#    print(df)
#    print(df2)
    
#    df3 = pd.DataFrame({'speed':df,'life:'df2})
#    df3 = df3[df3.columns[:]].corr()[target][:]
#    print(df3)
    plt.figure()
#    res=11
#    print(df[df.columns[:]])
#    print(res)
#    index=np.asarray([i for i in range(len(features))])
#    for i in range(len(fea[features])):
#        index.append(i)
#    width = 0.30
    plt.title('Target correation of refrance and '+strr)
#    ax=plt.plot()
    ax=df.plot.barh(rot=0,color='red', stacked=True)
    ax2 = df2.plot.barh(rot=0,color='blue', stacked=True)
#    ax.legend(['refrence'], loc='best')
    ax2.legend(['refrence',strr], loc='best')
#    plt.xticks(index)
    plt.ylabel('features')
    plt.xlabel('corr. values')
    plt.savefig(strr+'_barh')
    '''----------------------hitogram--------------------------'''
    err={'error_val':preds-y_v}
    error=pd.DataFrame(err)
    plt.figure()
    error.hist(bins=30)
    plt.title('histogram of prediction error')
    plt.xlabel('prediction error values')
    plt.ylabel('frquency')
    plt.savefig(strr+'_error_hist')
    
    '''-------------------scatter--------------------------'''
    plt.figure()
    plt.scatter(x=range(0,y_v.size), y=y_v,cmap='plasma',label='actual')
    plt.scatter(x=range(0,preds.size), y=preds,cmap='plasma',label='prectited')
    plt.legend(loc='best')    
    plt.title('actual VS prediction')
    plt.xlabel('observation')
    plt.ylabel('actual/preds values')
    plt.savefig(strr+'_scatter')
    
    
    plt.figure()
    plt.scatter(y_v,preds,cmap='plasma')
#    plt.scatter(x=range(0,preds.size), y=preds,cmap='plasma',label='prectited')
#    plt.legend(loc='best')    
    plt.title('actual VS prediction')
    plt.xlabel('actual values')
    plt.ylabel('prediction values')
    plt.savefig(strr+'_scatter')
#    
    
    
#    plt.figure(figsize=(30,30))
#    sns.heatmap(X_t, cmap="plasma", annot = True)



#    xdat=pd.DataFrame()
#    print(xdat)
#    z_val = ((preds-y_v)-error['error_val'].mean())/(error['error_val'].std())
#    error['error_val'] = 
    frame=[abs(error)]
#    xdat=xdat
#    xdat=pd.concat(frame)
    xdat=xdat.append(frame)
#    print(frame)
#    print(xdat)
#    plt.figure()
#    plt.plot(xdat)
#    plt.xticks(rotation=90)
    print('-------------------------------end-------------------------------')

    return mse_m,frame

#model0 = RandomForestRegressor(n_estimators=500, random_state = 0, n_jobs=2)
#model0.fit(reference[features], reference[target])

#x_train, x_valid, y_train, y_valid = train_test_split(reference[features], reference[target], train_size=0.8, test_size=0.2, random_state=0)
#model2 = RandomForestRegressor(n_estimators=50, random_state = 0, n_jobs=2)
#model2.fit(x_train, y_train)


'''shap'''
plt.figure()
#modelS= RandomForestRegressor(n_estimators=1000, random_state = 0, n_jobs=2).fit(reference[features], reference[target])
modelS = XGBRegressor().fit(reference[features], reference[target])
explainer = shap.Explainer(modelS)
shap_values = explainer(reference[features])
#print(shap_values)
shap.plots.waterfall(shap_values[1])
#shap.initjs()
#shap.force_plot(explainer.expected_value[1], shap_values[1], reference[features])
#shap.plots.force(shap_values[1])

'''prdiction'''
print('-----reference prediction----')
#ref_prediction0 = model0.predict(reference[features])
#mea0 = mean_absolute_error(ref_prediction0, reference[target])

listi=1000
strr = 'refrance'
#listi=[50,100,500,1000]
#for i in range(0, len(listi)):
mse,frame=score_model(xdat,strr, listi,reference[features], reference[target], reference[features], reference[target])
#print("Model %d MAE: %f" % (i+1, mae))
#print("Model %d RSME: %f" % (i+1, rmse))
#print("Model %d ROC_AUC: %f" % (i+1, roc_auc))
#print("Model %d APS: %f" % (i+1, aps))
#print(exp)

#ref_prediction2 = model2.predict(x_valid)
#mea2 = mean_absolute_error(ref_prediction2, y_valid)

#print(mea0)
#print(mea2)
#print(ref_prediction0[:10])
#print(ref_prediction2[:10])


xdat=xdat.append(frame)
print(xdat)

print('-----production prediction----')

print('---------------week : +1-------------------')
strr = 'week 1'
#prod_prediction0 = model0.predict(production_w1[features])
#mea_prod0 = mean_absolute_error(prod_prediction0, production_w1[target])
#print('mea_prod:', mea_prod0)
#print(prod_prediction0[:10])

#prod_prediction1 = model2.predict(production_w1[features])
#mea_prod1 = mean_absolute_error(prod_prediction1, production_w1[target])
#print('mea_prod:', mea_prod0)
mse,frame=score_model(xdat,strr, listi,reference[features], reference[target], production_w1[features], production_w1[target])
#print(prod_prediction1[:10])
#print(production_w1[target].head(10))

xdat=xdat.append(frame)
print(xdat)

print('--------------week : +2-------------------')
strr = 'week 2'
#prod_prediction2 = model2.predict(production_w2[features])
#mea_prod = mean_absolute_error(prod_prediction2, production_w2[target])
#print('mea_prod:', mea_prod)
#print(prod_prediction2[:10])
mse,frame=score_model(xdat,strr, listi,reference[features], reference[target], production_w2[features], production_w2[target])
#print(production_w2[target].head(10))

xdat=xdat.append(frame)


print('-------------week : +3------------------')
strr = 'week 3'
#prod_prediction3 = model2.predict(production_w3[features])
#mea_prod = mean_absolute_error(prod_prediction3, production_w3[target])
mse,frame=score_model(xdat,strr, listi,reference[features], reference[target], production_w3[features], production_w3[target])
#print('mea_prod:', mea_prod)
#print(prod_prediction3[:10])
#print(production_w3[target].head(10))


xdat=xdat.append(frame)


print('----------------week : +4----------------')
strr = 'week 4'
#prod_prediction4 = model2.predict(production_w4[features])
#mea_prod = mean_absolute_error(prod_prediction4, production_w4[target])
mse,frame=score_model(xdat,strr, listi,reference[features], reference[target], production_w4[features], production_w4[target])
#print('mea_prod:', mea_prod)
#print(prod_prediction4[:10])
#print(production_w4[target].head(10))

#
#
#
#print(xdat)

xdat=xdat.append(frame)
#xdat=xdat.sort_index(axis=0)
print(xdat)
plt.figure()
plt.plot(xdat)
plt.xticks(rotation=90)
plt.xlabel('date')
plt.ylabel('error vals')
plt.title('no. of errors against date/time')



#import lime
#import lime.lime_tabular
#
#explainer = lime.lime_tabular.LimeTabularExplainer(X_t, feature_names=data.feature_names, categorical_features = categorical_features, class_name=target, verbose = True, mode = 'regression')
#i=25
#exp=explainer.explain_instance(x_t[i], preds, num_features=numerical_features.count(),num_sample=100 )
#
#




