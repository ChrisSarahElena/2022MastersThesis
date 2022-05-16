#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install xlrd


# In[3]:


import pandas as pd
df_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_Dataset_No_Weather_TRAIN.xlsx")

df_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_Dataset_No_Weather_TEST.xlsx")


# In[4]:


pip install scikit-learn


# In[5]:


pip install xgboost


# In[6]:


pip install xlwt


# In[7]:


import xlwt


# In[8]:


from xgboost import XGBRegressor


# In[9]:


from sklearn.model_selection import cross_val_score, KFold


# In[10]:


from sklearn.metrics import mean_squared_error


# In[11]:


import numpy as np


# In[12]:


from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

seed = 40
test_size = 0.2

knn_r_acc = []

y_train = df_train['Total'].values
y_test = df_test['Total'].values


X_train = df_train[['Event_Code', 'Longitude', 'MesoregionCode', 'MicroregionCode','MunicipalityCode', 'isSummer'






]].values



X_test = df_test[['Event_Code', 'Longitude', 'MesoregionCode', 'MicroregionCode','MunicipalityCode', 'isSummer'




]].values
    
xgbr = XGBRegressor(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)





# In[13]:


import numpy as np
import pandas as pd


# In[14]:


#WAPE and RMSE

wape = np.sum(abs(y_test- ypred)) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, ypred)
print("RMSE: %.2f" % (mse**(1/2.0)))


# In[15]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  ypred

n = len(y_test)
sum = 0

# for loop for iteration
for i in range(n):
    sum += abs(actual[i] - calculated[i])

error = sum/n

# display
print("Mean absolute error : " + str(error))

#Now create Mean absolute error naive

sp = 1

y_pred_naive = y_test[:-sp]

# mean absolute error of naive seasonal prediction
mae_naive = np.mean(np.abs(y_test[sp:] - y_pred_naive))

print("MAE Naive : " + str(mae_naive))

MASE = str(error/mae_naive)

print("Mean Absolute Scaled Error : " + str(error/mae_naive))


# In[16]:


NRMSE = (mse**(1/2.0))/np.mean(y_test)


# In[17]:


print(NRMSE)


# In[18]:


#Now we will do the same code, but with a smaller portion of the dataset, to adjust for distances
#split the datasets into the distances and train/test


zeroto3k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-3km.xlsx")
zeroto35k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-35km.xlsx")
zeroto85k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-85km.xlsx")

zeroto3k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-3km.xlsx")
zeroto35k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-35km.xlsx")
zeroto85k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-85km.xlsx")


# In[19]:


y_train = zeroto3k_train['Total'].values
y_test = zeroto3k_test['Total'].values


X_train = zeroto3k_train[['Extremelypoor2010', 'Verypoor1991', 'Poor2010','IncomePoor1991'




]].values



X_test = zeroto3k_test[['Extremelypoor2010', 'Verypoor1991', 'Poor2010','IncomePoor1991'




]].values


# In[20]:


#Run model on 0-3k

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

seed = 40
test_size = 0.2

knn_r_acc = []

y_train = zeroto3k_train['Deceased'].values
y_test = zeroto3k_test['Deceased'].values


X_train = zeroto3k_train[['Event_Code', 'MunicipalityCode', 'Poor1991', 'Two_Months_Prior_Station_Longitude'


]].values

X_test = zeroto3k_test[['Event_Code', 'MunicipalityCode', 'Poor1991', 'Two_Months_Prior_Station_Longitude'


]].values
    
xgbr = XGBRegressor(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)


# In[21]:


#WAPE and RMSE

wape = np.sum(abs(y_test- ypred)) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, ypred)
print("RMSE: %.2f" % (mse**(1/2.0)))


# In[22]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  ypred

n = len(y_test)
sum = 0

# for loop for iteration
for i in range(n):
    sum += abs(actual[i] - calculated[i])

error = sum/n

# display
print("Mean absolute error : " + str(error))

#Now create Mean absolute error naive

sp = 1

y_pred_naive = y_test[:-sp]

# mean absolute error of naive seasonal prediction
mae_naive = np.mean(np.abs(y_test[sp:] - y_pred_naive))

print("MAE Naive : " + str(mae_naive))

MASE = str(error/mae_naive)

print("Mean Absolute Scaled Error : " + str(error/mae_naive))


# In[23]:


NRMSE = (mse**(1/2.0))/np.mean(y_test)
print(NRMSE)


# In[24]:


#Run Model on 0-35k
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

seed = 40
test_size = 0.2

knn_r_acc = []

y_train = zeroto35k_train['Deceased'].values
y_test = zeroto35k_test['Deceased'].values


X_train = zeroto35k_train[['Event_Code', 'MunicipalityCode', 'Poor1991', 'Two_Months_Prior_Station_Longitude'



]].values

X_test = zeroto35k_test[['Event_Code', 'MunicipalityCode', 'Poor1991', 'Two_Months_Prior_Station_Longitude'



]].values
    
xgbr = XGBRegressor(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)


# In[25]:


#WAPE and RMSE

wape = np.sum(abs(y_test- ypred)) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, ypred)
print("RMSE: %.2f" % (mse**(1/2.0)))


# In[26]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  ypred

n = len(y_test)
sum = 0

# for loop for iteration
for i in range(n):
    sum += abs(actual[i] - calculated[i])

error = sum/n

# display
print("Mean absolute error : " + str(error))

#Now create Mean absolute error naive

sp = 1

y_pred_naive = y_test[:-sp]

# mean absolute error of naive seasonal prediction
mae_naive = np.mean(np.abs(y_test[sp:] - y_pred_naive))

print("MAE Naive : " + str(mae_naive))

MASE = str(error/mae_naive)

print("Mean Absolute Scaled Error : " + str(error/mae_naive))


# In[27]:


NRMSE = (mse**(1/2.0))/np.mean(y_test)
print(NRMSE)


# In[28]:


#Run Model on 0-85k
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

seed = 40
test_size = 0.2

knn_r_acc = []

y_train = zeroto85k_train['Deceased'].values
y_test = zeroto85k_test['Deceased'].values


X_train = zeroto85k_train[['Event_Code', 'MunicipalityCode', 'Poor1991', 'Two_Months_Prior_Station_Longitude'


]].values

X_test = zeroto85k_test[['Event_Code', 'MunicipalityCode', 'Poor1991', 'Two_Months_Prior_Station_Longitude'




]].values
    
xgbr = XGBRegressor(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10) 
print(xgbr)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)  
print("Training score: ", score)

ypred = xgbr.predict(X_test)


# In[29]:


#WAPE and RMSE

wape = np.sum(abs(y_test- ypred)) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, ypred)
print("RMSE: %.2f" % (mse**(1/2.0)))


# In[30]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  ypred

n = len(y_test)
sum = 0

# for loop for iteration
for i in range(n):
    sum += abs(actual[i] - calculated[i])

error = sum/n

# display
print("Mean absolute error : " + str(error))

#Now create Mean absolute error naive

sp = 1

y_pred_naive = y_test[:-sp]

# mean absolute error of naive seasonal prediction
mae_naive = np.mean(np.abs(y_test[sp:] - y_pred_naive))

print("MAE Naive : " + str(mae_naive))

MASE = str(error/mae_naive)

print("Mean Absolute Scaled Error : " + str(error/mae_naive))


# In[31]:


NRMSE = (mse**(1/2.0))/np.mean(y_test)


# In[32]:


print(NRMSE)


# In[ ]:




