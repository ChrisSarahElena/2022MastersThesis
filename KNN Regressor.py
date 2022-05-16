#!/usr/bin/env python
# coding: utf-8

# In[146]:


pip install pandas


# In[147]:


pip install xlrd


# In[148]:


import pandas as pd


# In[149]:


df_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_Dataset_No_Weather_TRAIN.xlsx")

df_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_Dataset_No_Weather_TEST.xlsx")


# In[150]:


pip install scikit-learn


# In[151]:


print(df_train)


# In[152]:


df_train.isnull().sum()


# In[153]:


y_train = df_train['Total'].values
y_test = df_test['Total'].values


X_train = df_train[['Event_Code', 'Longitude', 'MesoregionCode', 'MicroregionCode','MunicipalityCode', 'isSummer'



]].values

X_test = df_test[['Event_Code', 'Longitude', 'MesoregionCode', 'MicroregionCode','MunicipalityCode', 'isSummer'


]].values


# In[154]:


import numpy as np


# In[155]:


from sklearn.neighbors import KNeighborsRegressor


# In[156]:


#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
x_test = pd.DataFrame(x_test_scaled)


# In[157]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt


# In[158]:


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[159]:


pip install matplotlib


# In[160]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9, 10, 11,12,13,14,15,16,17,18,19,20]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
model.best_params_


# In[161]:


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve
curve.plot()


# In[162]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9, 10, 11,12,13,14,15,16,17,18,19,20]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
model.best_params_


# In[163]:


reg = KNeighborsRegressor(n_neighbors = 20)


# In[164]:


#Fitting the model
reg.fit(X_train, y_train)


# In[165]:


from sklearn.metrics import mean_squared_error as mse


# In[166]:


#Predicting over the test set
test_predict = reg.predict(X_test)
k = mse(test_predict, y_test)
print('Test MSE  ',k)


# In[167]:


knn.fit(X_train,y_train)
test_score = knn.score(X_test,y_test)
train_score = knn.score(X_train,y_train)


# In[168]:


print(test_score)


# In[169]:


print(train_score)

pred=reg.predict(X_test)
print("Accuracy={}%".format((np.sum(y_test==pred)/y_test.shape[0])*100))


# In[170]:


# Create an array
knn_r_acc = []


# In[171]:


knn_r_acc.append((1, test_score ,train_score,"Accuracy={}%".format((np.sum(y_test==pred)/y_test.shape[0])*100)))


# In[172]:


print(knn_r_acc)


# In[173]:


print(pred)


# In[174]:


#WAPE and RMSE

wape = np.sum(abs((np.array(y_test) - pred))) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, pred)
print("RMSE: %.2f" % (mse**(1/2.0)))




# In[175]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  pred

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

NRMSE = (mse**(1/2.0))/np.mean(y_test)
print(NRMSE)


# In[ ]:





# In[31]:


#Now we will do the same code, but with a smaller portion of the dataset, to adjust for distances
#split the datasets into the distances and train/test

zeroto3k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-3km.xlsx")
zeroto35k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-35km.xlsx")
zeroto85k_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_Oversampled_TrainSet_0-85km.xlsx")

zeroto3k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-3km.xlsx")
zeroto35k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-35km.xlsx")
zeroto85k_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\ORIGINAL_TestSet_0-85km.xlsx")


# In[32]:


#Set up y train and test for each
y_train_zeroto3k = zeroto3k_train['Total'].values
y_test_zeroto3k = zeroto3k_test['Total'].values

y_train_zeroto35k = zeroto35k_train['Total'].values
y_test_zeroto35k = zeroto35k_test['Total'].values

y_train_zeroto85k = zeroto85k_train['Total'].values
y_test_zeroto85k = zeroto85k_test['Total'].values

#set up x train and x test for each

zeroto3k_X_train = zeroto3k_train[['Event_Code', 'MesoregionCode', 'MunicipalityCode', 'isSummer'




]].values


zeroto3k_X_test = zeroto3k_test[['Event_Code', 'MesoregionCode', 'MunicipalityCode', 'isSummer'



]].values

zeroto35k_X_train = zeroto35k_train[['Event_Code', 'MesoregionCode', 'MunicipalityCode', 'isSummer'

]].values

zeroto35k_X_test = zeroto35k_test[['Event_Code', 'MesoregionCode', 'MunicipalityCode', 'isSummer'

]].values

zeroto85k_X_train = zeroto85k_train[['Event_Code', 'MesoregionCode', 'MunicipalityCode', 'isSummer'

]].values

zeroto85k_X_test = zeroto85k_test[['Event_Code', 'MesoregionCode', 'MunicipalityCode', 'isSummer'


]].values


# In[33]:


from sklearn.neighbors import KNeighborsRegressor

#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_zeroto3k_scaled = scaler.fit_transform(zeroto3k_X_train)
x_train_zeroto3k = pd.DataFrame(x_train_zeroto3k_scaled)

x_test_zeroto3k_scaled = scaler.fit_transform(zeroto3k_X_test)
x_test_zeroto3k = pd.DataFrame(x_test_zeroto3k_scaled)


# In[34]:


#test values for 0-3k

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train_zeroto3k, y_train_zeroto3k)  #fit the model
    pred=model.predict(x_test_zeroto3k) #make prediction on test set
    error = sqrt(mean_squared_error(y_test_zeroto3k,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    
   


# In[35]:


#Graph the k values
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve
curve.plot()


# In[36]:



from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9, 10, 11,12,13,14,15,16,17,18,19,20]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train_zeroto3k, y_train_zeroto3k)
model.best_params_


# In[37]:



#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt


# In[38]:


x_test = x_test_zeroto3k 
x_train = x_train_zeroto3k
y_test = y_test_zeroto3k
y_train = y_train_zeroto3k


# In[39]:


reg = KNeighborsRegressor(n_neighbors = 8)

#Fitting the model
reg.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error as mse


# In[40]:


#Predicting over the test set
test_predict = reg.predict(x_test)
k = mse(test_predict, y_test)
print('Test MSE  ',k)
knn = neighbors.KNeighborsRegressor()
knn.fit(x_train,y_train)
test_score = knn.score(x_test,y_test)
train_score = knn.score(x_train,y_train)

print(test_score)

print(train_score)

pred=reg.predict(x_test)
print("Accuracy={}%".format((np.sum(y_test==pred)/y_test.shape[0])*100))


# In[41]:


#WAPE and RMSE

wape = np.sum(abs((np.array(y_test) - pred))) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, pred)
print("RMSE: %.2f" % (mse**(1/2.0)))


# In[42]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  pred

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

NRMSE = (mse**(1/2.0))/np.mean(y_test)
print(NRMSE)


# In[43]:


#Repeat for 0-35
from sklearn.neighbors import KNeighborsRegressor

#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_zeroto35k_scaled = scaler.fit_transform(zeroto35k_X_train)
x_train_zeroto35k = pd.DataFrame(x_train_zeroto35k_scaled)

x_test_zeroto35k_scaled = scaler.fit_transform(zeroto35k_X_test)
x_test_zeroto35k = pd.DataFrame(x_test_zeroto35k_scaled)


# In[44]:


#test values for 0-35k

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train_zeroto35k, y_train_zeroto35k)  #fit the model
    pred=model.predict(x_test_zeroto35k) #make prediction on test set
    error = sqrt(mean_squared_error(y_test_zeroto35k,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    
   


# In[45]:


#Graph the k values
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve
curve.plot()


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9, 10, 11,12,13,14,15,16,17,18,19,20]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train_zeroto35k, y_train_zeroto35k)
model.best_params_


# In[46]:



#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt

x_test = x_test_zeroto35k 
x_train = x_train_zeroto35k
y_test = y_test_zeroto35k
y_train = y_train_zeroto35k

reg = KNeighborsRegressor(n_neighbors = 19)

#Fitting the model
reg.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error as mse

#Predicting over the test set
test_predict = reg.predict(x_test)
k = mse(test_predict, y_test)
print('Test MSE  ',k)
knn = neighbors.KNeighborsRegressor()
knn.fit(x_train,y_train)
test_score = knn.score(x_test,y_test)
train_score = knn.score(x_train,y_train)

print(test_score)

print(train_score)

pred=reg.predict(x_test)
print("Accuracy={}%".format((np.sum(y_test==pred)/y_test.shape[0])*100))


# In[47]:


#WAPE and RMSE

wape = np.sum(abs((np.array(y_test) - pred))) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, pred)
print("RMSE: %.2f" % (mse**(1/2.0)))


# In[48]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  pred

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

NRMSE = (mse**(1/2.0))/np.mean(y_test)
print(NRMSE)


# In[49]:


#Repeat for 0-85
from sklearn.neighbors import KNeighborsRegressor

#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_zeroto85k_scaled = scaler.fit_transform(zeroto85k_X_train)
x_train_zeroto85k = pd.DataFrame(x_train_zeroto85k_scaled)

x_test_zeroto85k_scaled = scaler.fit_transform(zeroto85k_X_test)
x_test_zeroto85k = pd.DataFrame(x_test_zeroto85k_scaled)



# In[50]:


#test values for 0-85k

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train_zeroto85k, y_train_zeroto85k)  #fit the model
    pred=model.predict(x_test_zeroto85k) #make prediction on test set
    error = sqrt(mean_squared_error(y_test_zeroto85k,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[51]:


#Graph the k values
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve
curve.plot()


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9, 10, 11,12,13,14,15,16,17,18,19,20]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train_zeroto85k, y_train_zeroto85k)
model.best_params_


# In[52]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt


# In[53]:



#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt

x_test = x_test_zeroto85k 
x_train = x_train_zeroto85k
y_test = y_test_zeroto85k
y_train = y_train_zeroto85k

reg = KNeighborsRegressor(n_neighbors = 20)

#Fitting the model
reg.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error as mse

#Predicting over the test set
test_predict = reg.predict(x_test)
k = mse(test_predict, y_test)
print('Test MSE  ',k)
knn = neighbors.KNeighborsRegressor()
knn.fit(x_train,y_train)
test_score = knn.score(x_test,y_test)
train_score = knn.score(x_train,y_train)

print(test_score)

print(train_score)

pred=reg.predict(x_test)
print("Accuracy={}%".format((np.sum(y_test==pred)/y_test.shape[0])*100))


# In[54]:


#WAPE and RMSE

wape = np.sum(abs((np.array(y_test) - pred))) / np.array(y_test).sum()

print(wape)

mse = mean_squared_error(y_test, pred)
print("RMSE: %.2f" % (mse**(1/2.0)))


# In[55]:


# Python program for calculating Mean Absolute Error

# consider a list of integers for actual
actual = y_test

# consider a list of integers for actual
calculated =  pred

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

NRMSE = (mse**(1/2.0))/np.mean(y_test)
print(NRMSE)

