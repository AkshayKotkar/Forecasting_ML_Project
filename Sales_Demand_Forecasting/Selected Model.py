#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm

from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pickle


# In[2]:


data = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Ai Varient\Demand Forecasting\train (2).csv")


# In[3]:


data['date'] = pd.to_datetime(data['date'])


# In[4]:


data


# In[5]:


store_sales = data.groupby('store_nbr')['sales'].sum()
plt.figure(figsize=(10,6))
plt.bar(store_sales.index, store_sales.values, color ='red')
plt.title('Total Sales by Store Number')
plt.xlabel('Store Number')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


# In[6]:


colors = ['blue', 'green', 'red', 'purple', 'orange','pink','yellow']
data['days'] = data['date'].dt.day_name()
weekdays_sales = data.groupby(data['days'])['sales'].sum()
weekdays_sales = weekdays_sales.sort_values(ascending=False)
plt.figure(figsize=(10,6))
plt.bar(weekdays_sales.index, weekdays_sales.values, color=colors)
plt.title('Sales by Days')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.show()


# In[7]:


data['month'] = data['date'].dt.month
month_sales = data.groupby(data['month'])['sales'].sum()
month_sales = month_sales.sort_values(ascending=False)
plt.figure(figsize=(10,6))
plt.bar(month_sales.index, month_sales.values, color=colors)
plt.title('Sales by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()


# In[8]:


data


# In[9]:


data1 = data.set_index('date')


# In[10]:


data1_month = data1.resample('M').sum()
plt.figure(figsize=(15,8))
data1_month['sales'].plot()


# In[11]:


data1_week = data1.resample('W').sum()
plt.figure(figsize=(15,8))
data1_week['sales'].plot()


# In[12]:


smax_data = data.drop(data.iloc[:,[0,2,3,5,6,7]], axis=1)
smax_data = smax_data.set_index('date')
smax_data


# In[13]:


weekly_data = smax_data.resample('W').sum()//100
weekly_data['sales'].plot()

plt.figure(figsize=(20,10))
decompose_ts_add = seasonal_decompose(weekly_data.sales,period=12) #period 12 is 
decompose_ts_add.plot()
plt.show()


# In[14]:


monthly_data = smax_data.resample('M').sum()
monthly_data['sales'].plot()

plt.figure(figsize=(20,10))
decompose_ts_add = seasonal_decompose(monthly_data.sales,period=12) #period 12 is 
decompose_ts_add.plot()
plt.show()


# In[15]:


weekly_data


# In[16]:


weekly_data = weekly_data.iloc[:241]
weekly_data


# In[17]:


train, test= train_test_split(weekly_data, train_size=0.8, shuffle=False)


# In[18]:


plot_acf(train.diff().dropna())
plot_pacf(train.diff().dropna())
plt.show()


# In[19]:


sarimax_model = SARIMAX(train, order=(3,1,3), seasonal_order=(3,1,3,19))
sarimax_fit = sarimax_model.fit()
sarimax_fit.summary()


# In[20]:


predict_sarima = sarimax_fit.predict(start=len(train['sales']), end=len(train['sales'])+len(test['sales'])-1)
predict_sarima.index=test.index


# In[21]:


plt.figure(figsize=(12,5))
plt.plot(train['sales'], label='Train')
plt.plot(test['sales'], label='Test')
plt.plot(predict_sarima, label='Predict_Sarima')
plt.title('SARIMAX Predicted Model')
plt.legend()
plt.show()

rmse_of_SARIMAX = mean_squared_error(y_true= test, y_pred= predict_sarima)
mape_of_SARIMAX = mean_absolute_percentage_error(y_true= test, y_pred=predict_sarima)
print('Rmse of Auto Arima :',rmse_of_SARIMAX)
print('Mape of Auto Arima :',mape_of_SARIMAX)


# In[22]:


future_dates = pd.date_range(start=test['sales'].index[-1], periods=30, freq='W')
weekly_forecast = sarimax_fit.predict(start=len(train['sales'])+len(test['sales']), end=len(train['sales'])+len(test['sales'])+30, dynamic=True)
weekly_forecast


# In[23]:


# Plot the predicted values
plt.figure(figsize=(12,5))
plt.plot(train.index, train["sales"], label='Train',color="black")
plt.plot(test.index, test["sales"], label='Test',color="orange")
plt.plot(predict_sarima, label='Predict')
plt.title('SARIMAX 30 Week Forecast Values')

plt.plot(weekly_forecast, label='Forecast',color="Green")
plt.legend()
plt.show()


# In[24]:


pickle.dump(sarimax_fit,open('Deploy.pkl','wb'))

