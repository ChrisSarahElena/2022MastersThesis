#!/usr/bin/env python
# coding: utf-8

# In[12]:


pip install pandas


# In[13]:


pip install xlrd


# In[14]:


import pandas as pd


# In[15]:


df = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_TRAIN.xlsx")


df_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Oversampled_Original_TRAIN.xlsx")

df_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_TEST.xlsx")

print(df_train)


# In[16]:


pip install scikit-learn


# In[17]:


print(df_train)


# In[18]:


pip install xgboost


# In[19]:


pip install xlwt


# In[20]:


import xlwt


# In[21]:


from xgboost import XGBRegressor


# In[22]:


from sklearn.metrics import mean_squared_error


# In[23]:


import numpy as np


# In[24]:


print(df_train)


# In[ ]:


# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = df
# split data into X and y

X = df[df.columns.difference(['IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]

y = df['IsDeceased'].values
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
print(plot_importance)
pyplot.rcParams["figure.figsize"] = (20,20)
pyplot.rcParams["figure.dpi"] = 400
pyplot.show()


# In[ ]:





# In[296]:


# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = df
# split data into X and y

X = df[df.columns.difference(['IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]

y = df['IsTotal'].values
# fit model no training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
print(plot_importance)
pyplot.rcParams["figure.figsize"] = (20,20)
pyplot.rcParams["figure.dpi"] = 400
pyplot.show()


# In[ ]:


'Date','MunicipalityCode','Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
'Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',


# In[343]:


pip install shap


# In[351]:


# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns


X_train = df[df.columns.difference(['Date','IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]

y_train = df['IsTotal'].values

X_test = df_test[df_test.columns.difference(['Date','IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]

y_test = df_test['IsTotal'].values


xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)



# In[352]:


def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
   df = pd.DataFrame.from_dict({'IsDeceased':[IsDeceased],
                                'IsTotal': [IsTotal]})
   predict = xgbr.predict_proba(df)[0]
   return {model.classes_[i]: predict[i] for i in range(3)}


# In[353]:


# Installing and importing Gradio
get_ipython().system('pip install gradio')
import gradio as gr
IsDeceased = gr.inputs.Slider(minimum=0, maximum=10, default=5, label="Deceased"
)
IsTotal = gr.inputs.Slider(minimum=0, maximum=10, default=5, label="IsTotal")
gr.Interface(predict_flower, ['IsDeceased','IsTotal'], "label", live=True, interpretation="default").launch(debug=True)


# In[349]:


importance <- xgbr.importance(feature_names, model = bst)


# In[28]:



# use feature importance for feature selection
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
# load data
dataset = df_train
# split data into X and y

X_train = df[df.columns.difference(['IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]

y_train = df['IsDeceased'].values


X_test = df_test[df_test.columns.difference(['IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]

y_test = df_test['IsDeceased'].values


# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data and evaluate

# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    
    print(thresh)
    
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    
    report = classification_report(y_test,y_pred)
    print("Thresh= {} , n= {}\n {}" .format(thresh,select_X_train.shape[1], report))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(selection_model.get_booster().get_score(importance_type='gain'))
    

   
    


# In[464]:


print (df_train.columns)


# In[411]:


print (y_pred)

import pandas as pd
CSV = pd.DataFrame({
    "Prediction": y_pred
})

CSV.to_csv("prediction.csv", index=False)

print(CSV)


# In[415]:


print(thresholds)


# In[378]:


print(y_pred)


# In[302]:


print(thresholds)


# In[336]:


#Overall data

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

y_train = df_train['IsDeceased'].values
y_test = df_test['IsDeceased'].values


X_train = df_train[['Population'
]].values

X_test = df_test[['Population'
]].values


xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[337]:


X_train = df_train[['Population','x_Lat_Lon','IncomeExtremelypoor1991','IncomeExtremelypoor2010','y_Lat_Lon',
'Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature','One_Month_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
'One_Month_Prior_Maximum_Hourly_Precipitation','Two_Months_Prior_Average_temperature_bulbo_seco','MunicipalityCode','IncomeExtremelypoor2000',
'Event_Code','z_Lat_Lon','Two_Months_Prior_Standard_Deviation_Precipitation','Two_Months_Prior_Maximum_Hourly_Precipitation',
'Two_Months_Prior_Daily_Average_Precipitation','One_Month_Prior_Maximum_Daily_Precipitation','One_Month_Prior_Min_Distance_(km)',
'One_Month_Prior_Daily_Average_Precipitation','Longitude','Two_Months_Prior_Maximum_Hourly_Regular_Temperature','One_Month_Prior_Average_temperature_bulbo_seco',
'IDH_M_Longevity2010']].values

X_test = df_test[['Population','x_Lat_Lon','IncomeExtremelypoor1991','IncomeExtremelypoor2010','y_Lat_Lon',
'Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature','One_Month_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
'One_Month_Prior_Maximum_Hourly_Precipitation','Two_Months_Prior_Average_temperature_bulbo_seco','MunicipalityCode','IncomeExtremelypoor2000',
'Event_Code','z_Lat_Lon','Two_Months_Prior_Standard_Deviation_Precipitation','Two_Months_Prior_Maximum_Hourly_Precipitation',
'Two_Months_Prior_Daily_Average_Precipitation','One_Month_Prior_Maximum_Daily_Precipitation','One_Month_Prior_Min_Distance_(km)',
'One_Month_Prior_Daily_Average_Precipitation','Longitude','Two_Months_Prior_Maximum_Hourly_Regular_Temperature','One_Month_Prior_Average_temperature_bulbo_seco',
'IDH_M_Longevity2010']].values


# In[338]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[332]:


# ROC Curve
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix

model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[259]:


#Now we will do the same code, but with a smaller portion of the dataset, to adjust for distances
#split the datasets into the distances and train/test


zeroto3k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-3km.xlsx")
zeroto35k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-35km.xlsx")
zeroto85k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-85km.xlsx")

zeroto3k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-3km.xlsx")
zeroto35k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-35km.xlsx")
zeroto85k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-85km.xlsx")


# In[260]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

y_train = zeroto3k_train['IsDeceased'].values
y_test = zeroto3k_test['IsDeceased'].values


X_train = zeroto3k_train[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values



X_test = zeroto3k_test[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values

xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[261]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[262]:


#Plot ROC Curve

model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[263]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

#Set up y train and test for each

y_train = zeroto35k_train['IsDeceased'].values
y_test = zeroto35k_test['IsDeceased'].values

X_train = zeroto35k_train[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values

X_test = zeroto35k_test[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values


xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[264]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[265]:


model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[266]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

#Set up y train and test for each



y_train = zeroto85k_train['IsDeceased'].values
y_test = zeroto85k_test['IsDeceased'].values

#set up x train and x test for each


X_train = zeroto85k_train[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values

X_test = zeroto85k_test[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values


xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[267]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[268]:


model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[269]:


#Full Dataset for Total with Other Variable Setup

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

y_train = df_train['IsTotal'].values
y_test = df_test['IsTotal'].values


X_train = df_train[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values

X_test = df_test[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values



xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[270]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[271]:


model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[272]:


#0 to 3k with other variable set up - Total

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

y_train = zeroto3k_train['IsTotal'].values
y_test = zeroto3k_test['IsTotal'].values


X_train = zeroto3k_train[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values

X_test = zeroto3k_test[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values


xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[273]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[274]:


model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[237]:


#0 to 35k with other variable setup 

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

#Set up y train and test for each

y_train = zeroto35k_train['IsTotal'].values
y_test = zeroto35k_test['IsTotal'].values

X_train = zeroto35k_train[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values

X_test = zeroto35k_test[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values


xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[238]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[239]:


model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[240]:


#0 to 85k with other variable setup

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

knn_r_acc = []

#Set up y train and test for each

y_train = zeroto85k_train['IsTotal'].values
y_test = zeroto85k_test['IsTotal'].values

#set up x train and x test for each


X_train = zeroto85k_train[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values

X_test = zeroto85k_test[['Latitude', 'Longitude', 'Extremelypoor1991', 'Poor2010', 'One_Month_Prior_Total_Monthly_Precipitation']].values


xgbr = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)

mse = mean_squared_error(y_test, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))


predictions = [round(value) for value in ypred]
probpredictions = [(value) for value in ypred]


# In[241]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, ypred)


# In[242]:


model_xg = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10).fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, roc_curve

auc_xg = roc_auc_score(y_test, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test, probs_xg)

plt.figure(figsize=(12, 7))
plt.plot(fpr_xg, tpr_xg, label=f'AUC (XGBoost) = {auc_xg:.2f}')
#plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.fill_between(fpr_xg,tpr_xg, facecolor='lightblue',  alpha=0.7)
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


# In[ ]:





# In[ ]:




