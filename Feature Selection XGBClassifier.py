#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install xlrd


# In[3]:


pip install -U scikit-learn


# In[4]:


import pandas as pd


# In[5]:



df_train = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_Dataset_No_Weather_TRAIN.xlsx")

df_test = pd.read_excel("C:\\Users\\Sarah\\Desktop\\Thesis\\TestTrainSets\\Original_Dataset_No_Weather_TEST.xlsx")



# In[6]:


X_train = df_train[df_train.columns.difference(['IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]


# In[7]:


print(X_train.columns)


# In[8]:



df_x = pd.DataFrame(X_train, columns=['0-35_km', '0-3_km', '0-85_km', 'Date', 'Event_Code',
       'Extremelypoor1991', 'Extremelypoor2000', 'Extremelypoor2010',
       'IDH_M_2000', 'IDH_M_2010', 'IDH_M_Education2000',
       'IDH_M_Education2010', 'IDH_M_Income2000', 'IDH_M_Income2010',
       'IDH_M_Longevity2000', 'IDH_M_Longevity2010', 'IncomeExtremelypoor1991',
       'IncomeExtremelypoor2000', 'IncomeExtremelypoor2010', 'IncomePoor1991',
       'IncomePoor2000', 'IncomePoor2010', 'IncomeVerypoor1991',
       'IncomeVerypoor2000', 'IncomeVerypoor2010', 'Latitude', 'Longitude',
       'MesoregionCode', 'MicroregionCode', 'Month_Int', 'Month_cos',
       'Month_sin', 'MunicipalityCode', 'One_Month_Prior_Altitude',
       'One_Month_Prior_Average_temperature_bulbo_seco',
       'One_Month_Prior_Daily_Average_Precipitation',
       'One_Month_Prior_Maximum_Daily_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Maximum_Hourly_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Regular_Temperature',
       'One_Month_Prior_Min_Distance_(km)', 'One_Month_Prior_Min_Distance_(m)',
       'One_Month_Prior_Minimum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Minimum_Hourly_Regular_Temperature',
       'One_Month_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'One_Month_Prior_Standard_Deviation_Precipitation',
       'One_Month_Prior_Station_Latitude', 'One_Month_Prior_Station_Longitude',
       'One_Month_Prior_Total_Monthly_Precipitation',
       'One_Month_Prior_x_Lat_Lon', 'One_Month_Prior_y_Lat_Lon',
       'One_Month_Prior_z_Lat_Lon', 'Poor1991', 'Poor2000', 'Poor2010',
       'Population', 'Prior_Month', 'StateCode', 'Train_Set',
       'Two_Months_Prior', 'Two_Months_Prior_Altitude',
       'Two_Months_Prior_Average_temperature_bulbo_seco',
       'Two_Months_Prior_Daily_Average_Precipitation',
       'Two_Months_Prior_Maximum_Daily_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Maximum_Hourly_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Min_Distance_(km)',
       'Two_Months_Prior_Min_Distance_(m)',
       'Two_Months_Prior_Minimum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Minimum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'Two_Months_Prior_Standard_Deviation_Precipitation',
       'Two_Months_Prior_Station_Latitude',
       'Two_Months_Prior_Station_Longitude',
       'Two_Months_Prior_Total_Monthly_Precipitation',
       'Two_Months_Prior_x_Lat_Lon', 'Two_Months_Prior_y_Lat_Lon',
       'Two_Months_Prior_z_Lat_Lon', 'Type_Code', 'Verypoor1991',
       'Verypoor2000', 'Verypoor2010', 'Year', 'YearMonth', 'isAutumn',
       'isSpring', 'isSummer', 'isWinter', 'x_Lat_Lon', 'y_Lat_Lon',
       'z_Lat_Lon'])


# In[9]:


print(df_x)


# In[10]:



X_train = df_x.loc[:, ['0-35_km', '0-3_km', '0-85_km', 'Date', 'Event_Code',
       'Extremelypoor1991', 'Extremelypoor2000', 'Extremelypoor2010',
       'IDH_M_2000', 'IDH_M_2010', 'IDH_M_Education2000',
       'IDH_M_Education2010', 'IDH_M_Income2000', 'IDH_M_Income2010',
       'IDH_M_Longevity2000', 'IDH_M_Longevity2010', 'IncomeExtremelypoor1991',
       'IncomeExtremelypoor2000', 'IncomeExtremelypoor2010', 'IncomePoor1991',
       'IncomePoor2000', 'IncomePoor2010', 'IncomeVerypoor1991',
       'IncomeVerypoor2000', 'IncomeVerypoor2010', 'Latitude', 'Longitude',
       'MesoregionCode', 'MicroregionCode', 'Month_Int', 'Month_cos',
       'Month_sin', 'MunicipalityCode', 'One_Month_Prior_Altitude',
       'One_Month_Prior_Average_temperature_bulbo_seco',
       'One_Month_Prior_Daily_Average_Precipitation',
       'One_Month_Prior_Maximum_Daily_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Maximum_Hourly_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Regular_Temperature',
       'One_Month_Prior_Min_Distance_(km)', 'One_Month_Prior_Min_Distance_(m)',
       'One_Month_Prior_Minimum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Minimum_Hourly_Regular_Temperature',
       'One_Month_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'One_Month_Prior_Standard_Deviation_Precipitation',
       'One_Month_Prior_Station_Latitude', 'One_Month_Prior_Station_Longitude',
       'One_Month_Prior_Total_Monthly_Precipitation',
       'One_Month_Prior_x_Lat_Lon', 'One_Month_Prior_y_Lat_Lon',
       'One_Month_Prior_z_Lat_Lon', 'Poor1991', 'Poor2000', 'Poor2010',
       'Population', 'Prior_Month', 'StateCode', 'Train_Set',
       'Two_Months_Prior', 'Two_Months_Prior_Altitude',
       'Two_Months_Prior_Average_temperature_bulbo_seco',
       'Two_Months_Prior_Daily_Average_Precipitation',
       'Two_Months_Prior_Maximum_Daily_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Maximum_Hourly_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Min_Distance_(km)',
       'Two_Months_Prior_Min_Distance_(m)',
       'Two_Months_Prior_Minimum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Minimum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'Two_Months_Prior_Standard_Deviation_Precipitation',
       'Two_Months_Prior_Station_Latitude',
       'Two_Months_Prior_Station_Longitude',
       'Two_Months_Prior_Total_Monthly_Precipitation',
       'Two_Months_Prior_x_Lat_Lon', 'Two_Months_Prior_y_Lat_Lon',
       'Two_Months_Prior_z_Lat_Lon', 'Type_Code', 'Verypoor1991',
       'Verypoor2000', 'Verypoor2010', 'Year', 'YearMonth', 'isAutumn',
       'isSpring', 'isSummer', 'isWinter', 'x_Lat_Lon', 'y_Lat_Lon',
       'z_Lat_Lon']]


# In[11]:


print(X_train)


# In[12]:


X_test = df_test[df_test.columns.difference(['IsDeceased','IsTotal','Deceased','Sick','Injured','Displaced','Homeless','MissingPeople','Other','Total'])]


# In[13]:



df_xtest = pd.DataFrame(X_test, columns=['0-35_km', '0-3_km', '0-85_km', 'Date', 'Event_Code',
       'Extremelypoor1991', 'Extremelypoor2000', 'Extremelypoor2010',
       'IDH_M_2000', 'IDH_M_2010', 'IDH_M_Education2000',
       'IDH_M_Education2010', 'IDH_M_Income2000', 'IDH_M_Income2010',
       'IDH_M_Longevity2000', 'IDH_M_Longevity2010', 'IncomeExtremelypoor1991',
       'IncomeExtremelypoor2000', 'IncomeExtremelypoor2010', 'IncomePoor1991',
       'IncomePoor2000', 'IncomePoor2010', 'IncomeVerypoor1991',
       'IncomeVerypoor2000', 'IncomeVerypoor2010', 'Latitude', 'Longitude',
       'MesoregionCode', 'MicroregionCode', 'Month_Int', 'Month_cos',
       'Month_sin', 'MunicipalityCode', 'One_Month_Prior_Altitude',
       'One_Month_Prior_Average_temperature_bulbo_seco',
       'One_Month_Prior_Daily_Average_Precipitation',
       'One_Month_Prior_Maximum_Daily_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Maximum_Hourly_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Regular_Temperature',
       'One_Month_Prior_Min_Distance_(km)', 'One_Month_Prior_Min_Distance_(m)',
       'One_Month_Prior_Minimum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Minimum_Hourly_Regular_Temperature',
       'One_Month_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'One_Month_Prior_Standard_Deviation_Precipitation',
       'One_Month_Prior_Station_Latitude', 'One_Month_Prior_Station_Longitude',
       'One_Month_Prior_Total_Monthly_Precipitation',
       'One_Month_Prior_x_Lat_Lon', 'One_Month_Prior_y_Lat_Lon',
       'One_Month_Prior_z_Lat_Lon', 'Poor1991', 'Poor2000', 'Poor2010',
       'Population', 'Prior_Month', 'StateCode', 'Train_Set',
       'Two_Months_Prior', 'Two_Months_Prior_Altitude',
       'Two_Months_Prior_Average_temperature_bulbo_seco',
       'Two_Months_Prior_Daily_Average_Precipitation',
       'Two_Months_Prior_Maximum_Daily_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Maximum_Hourly_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Min_Distance_(km)',
       'Two_Months_Prior_Min_Distance_(m)',
       'Two_Months_Prior_Minimum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Minimum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'Two_Months_Prior_Standard_Deviation_Precipitation',
       'Two_Months_Prior_Station_Latitude',
       'Two_Months_Prior_Station_Longitude',
       'Two_Months_Prior_Total_Monthly_Precipitation',
       'Two_Months_Prior_x_Lat_Lon', 'Two_Months_Prior_y_Lat_Lon',
       'Two_Months_Prior_z_Lat_Lon', 'Type_Code', 'Verypoor1991',
       'Verypoor2000', 'Verypoor2010', 'Year', 'YearMonth', 'isAutumn',
       'isSpring', 'isSummer', 'isWinter', 'x_Lat_Lon', 'y_Lat_Lon',
       'z_Lat_Lon'])


# In[14]:



X_test = df_xtest.loc[:, ['0-35_km', '0-3_km', '0-85_km', 'Date', 'Event_Code',
       'Extremelypoor1991', 'Extremelypoor2000', 'Extremelypoor2010',
       'IDH_M_2000', 'IDH_M_2010', 'IDH_M_Education2000',
       'IDH_M_Education2010', 'IDH_M_Income2000', 'IDH_M_Income2010',
       'IDH_M_Longevity2000', 'IDH_M_Longevity2010', 'IncomeExtremelypoor1991',
       'IncomeExtremelypoor2000', 'IncomeExtremelypoor2010', 'IncomePoor1991',
       'IncomePoor2000', 'IncomePoor2010', 'IncomeVerypoor1991',
       'IncomeVerypoor2000', 'IncomeVerypoor2010', 'Latitude', 'Longitude',
       'MesoregionCode', 'MicroregionCode', 'Month_Int', 'Month_cos',
       'Month_sin', 'MunicipalityCode', 'One_Month_Prior_Altitude',
       'One_Month_Prior_Average_temperature_bulbo_seco',
       'One_Month_Prior_Daily_Average_Precipitation',
       'One_Month_Prior_Maximum_Daily_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Maximum_Hourly_Precipitation',
       'One_Month_Prior_Maximum_Hourly_Regular_Temperature',
       'One_Month_Prior_Min_Distance_(km)', 'One_Month_Prior_Min_Distance_(m)',
       'One_Month_Prior_Minimum_Hourly_Bulbo_Seco',
       'One_Month_Prior_Minimum_Hourly_Regular_Temperature',
       'One_Month_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'One_Month_Prior_Standard_Deviation_Precipitation',
       'One_Month_Prior_Station_Latitude', 'One_Month_Prior_Station_Longitude',
       'One_Month_Prior_Total_Monthly_Precipitation',
       'One_Month_Prior_x_Lat_Lon', 'One_Month_Prior_y_Lat_Lon',
       'One_Month_Prior_z_Lat_Lon', 'Poor1991', 'Poor2000', 'Poor2010',
       'Population', 'Prior_Month', 'StateCode', 'Train_Set',
       'Two_Months_Prior', 'Two_Months_Prior_Altitude',
       'Two_Months_Prior_Average_temperature_bulbo_seco',
       'Two_Months_Prior_Daily_Average_Precipitation',
       'Two_Months_Prior_Maximum_Daily_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Maximum_Hourly_Precipitation',
       'Two_Months_Prior_Maximum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Min_Distance_(km)',
       'Two_Months_Prior_Min_Distance_(m)',
       'Two_Months_Prior_Minimum_Hourly_Bulbo_Seco',
       'Two_Months_Prior_Minimum_Hourly_Regular_Temperature',
       'Two_Months_Prior_Standard_Deviation_Bulbo_Seco_Daily_Temperature',
       'Two_Months_Prior_Standard_Deviation_Precipitation',
       'Two_Months_Prior_Station_Latitude',
       'Two_Months_Prior_Station_Longitude',
       'Two_Months_Prior_Total_Monthly_Precipitation',
       'Two_Months_Prior_x_Lat_Lon', 'Two_Months_Prior_y_Lat_Lon',
       'Two_Months_Prior_z_Lat_Lon', 'Type_Code', 'Verypoor1991',
       'Verypoor2000', 'Verypoor2010', 'Year', 'YearMonth', 'isAutumn',
       'isSpring', 'isSummer', 'isWinter', 'x_Lat_Lon', 'y_Lat_Lon',
       'z_Lat_Lon']]


# In[15]:


from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# load data

# split data into X and y

y_train = df_train['IsTotal'].values

y_test = df_test['IsTotal'].values


# fit model on all training data
model = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10)
model.fit(X_train, y_train)
# make predictions for test data and evaluate
ypred = model.predict(X_test)
predictions = [round(value) for value in ypred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(verbosity=0, booster = "gbtree", eval_metric = "mae", num_boost_round = 999, early_stopping_rounds=10)
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
    
    col_index = selection.get_support()
    print([col_index])
    print(X_train.columns[col_index])
    print(X_train.columns[col_index] == True)

