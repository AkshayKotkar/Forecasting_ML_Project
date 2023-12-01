#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels as sm

from statsmodels.tsa.stattools import adfuller,kpss
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pylab import rcParams

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM # Long Short Term Memory

import pickle


# In[2]:


data = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Ai Varient\Demand Forecasting\train (2).csv")


# # EDA

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['days'] = data['date'].dt.day_name()


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()


# In[11]:


data.duplicated().sum()


# In[12]:


data.corr()


# # Visualization

# In[13]:


sns.heatmap(data.corr(), annot = True)


# In[14]:


store_sales = data.groupby('store_nbr')['sales'].sum()


# In[15]:


plt.figure(figsize=(10,6))
plt.bar(store_sales.index, store_sales.values, color ='red')

plt.title('Total Sales by Store Number')
plt.xlabel('Store Number')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()


# In[16]:


family_sales = data.groupby('family')['sales'].sum()
family_sales = family_sales.sort_values(ascending=False)


# In[17]:


plt.figure(figsize=(10,6))
colors = ['blue', 'green', 'red', 'purple', 'orange','pink','yellow']
plt.bar(family_sales.index, family_sales.values, color =colors)

plt.title('Total Sales by Products')
plt.xlabel('Products')
plt.ylabel('Total Sales ')
plt.xticks(rotation=90)
plt.show()


# In[18]:


family_sales = data.groupby('family')['onpromotion'].sum()
family_sales = family_sales.sort_values(ascending=False)

plt.figure(figsize=(10,6))
colors = ['blue', 'green', 'red', 'purple', 'orange','pink','yellow']
plt.bar(family_sales.index, family_sales.values, color =colors)

plt.title('Total Promotion by Products')
plt.xlabel('Products')
plt.ylabel('Total Promotion ')
plt.xticks(rotation=90)
plt.show()


# In[19]:


store_promote = data.groupby('store_nbr')['onpromotion'].sum()


# In[20]:


plt.figure(figsize=(10,6))
plt.bar(store_promote.index, store_promote.values, color='pink')

plt.title('Total Promotion of Products')
plt.xlabel('Store Number')
plt.ylabel('No of Promotion Products')
plt.grid(True)
plt.show()


# In[21]:


weekdays_sales = data.groupby(data['days'])['sales'].sum()
weekdays_sales = weekdays_sales.sort_values(ascending=False)


# In[22]:


plt.figure(figsize=(10,6))
plt.bar(weekdays_sales.index, weekdays_sales.values, color=colors)

plt.title('Sales by Days')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.show()


# In[23]:


month_sales = data.groupby(data['month'])['sales'].sum()
month_sales = month_sales.sort_values(ascending=False)


# In[24]:


plt.figure(figsize=(10,6))
plt.bar(month_sales.index, month_sales.values, color=colors)

plt.title('Sales by Month')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()


# In[25]:


year_sales = data.groupby(data['year'])['sales'].sum()
year_sales = year_sales.sort_values(ascending=False)


# In[26]:


plt.figure(figsize=(10,6))
plt.bar(year_sales.index, year_sales.values, color=colors)

plt.title('Sales by Year')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()


# In[27]:


plt.figure(figsize=(10,6))
plt.scatter(x='sales', y='onpromotion', data=data)

plt.title('Compare with Sales and Promotion')
plt.xlabel('Sales')
plt.ylabel('Promotion')
plt.show()


# In[28]:


plt.figure(figsize=(10,5))
sns.lineplot(x='date',y='sales',data=data)
plt.title('Sales')

Time Series Forecasting Methods
# In[29]:


data.head()


# In[30]:


ts_data = data.drop(data.iloc[:,[0,2,3,5,6,7,8,]], axis=1)
ts_data.set_index('date')

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    critical_values = pd.Series(dftest[3], name='Critical Values')
    
    dfoutput = dfoutput.append(critical_values)
    
    print (dfoutput)

adf_test(ts_data.sales)
# In[31]:


def kpss_test(timeseries):
    print('result of KPSS test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'P_value', 'Lags Used'])
    
    critical_values = pd.Series(kpsstest[3], name='Critical Values')
    
    kpss_output = kpss_output.append(critical_values)
    
    print(kpss_output)

kpss_test(ts_data.sales)


# In[32]:


#Differencing
plt.figure(figsize=(15,8))
ts_data['sales_diff'] = ts_data['sales'] - ts_data['sales'].shift(1)
ts_data['sales_diff'].dropna().plot()


# In[33]:


plt.figure(figsize=(15,8))
sns.lineplot(x='date',y='sales_diff',data=ts_data)
plt.title('Sales')


# In[34]:


def kpss_test(timeseries):
    print('result of KPSS test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'P_value', 'Lags Used'])
    
    critical_values = pd.Series(kpsstest[3], name='Critical Values')
    
    kpss_output = kpss_output.append(critical_values)
    
    print(kpss_output)

kpss_test(ts_data.sales_diff.iloc[1:])


# In[35]:


ts_data


# In[36]:


df = ts_data.resample('W',on='date')['sales','sales_diff'].sum()/100
df = df.drop('sales_diff', axis=1)
df = df.iloc[:241]


# In[37]:


df


# In[38]:


def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    critical_values = pd.Series(dftest[3], name='Critical Values')
    
    dfoutput = dfoutput.append(critical_values)
    
    print (dfoutput)

adf_test(df.sales)


# In[39]:


plt.figure(figsize=(15,8))
df['sales'].plot()


# In[40]:


train, test= train_test_split(df, train_size=0.8, shuffle=False)


# In[41]:


train.shape, test.shape


# In[42]:


train


# In[43]:


test


# In[44]:


plt.figure(figsize=(12,6))
sns.lineplot(x='date',y='sales',data=train)
plt.title('Train Data')


# In[45]:


plt.figure(figsize=(12,6))
sns.lineplot(x='date',y='sales',data=test)
plt.title('Test Data')


# In[46]:


model1 = SimpleExpSmoothing(train).fit(smoothing_level=0.2)
pred1 = model1.forecast(49).rename('alpha = 0.2')

ax = df.plot( color='black', figsize=(18,8), legend=True)

pred1.plot(marker='+', ax=ax, color='red', legend=True)
model1.fittedvalues.plot(marker='+', ax=ax, color='red')


# In[47]:


rmse_of_SES_SL_2 = np.sqrt(mean_squared_error(y_true= test, y_pred= pred1))
mape_of_SES_SL_2 = mean_absolute_percentage_error(y_true= test, y_pred= pred1)
print('Rmse of Simple Exponential Smoothing of 0.2:',rmse_of_SES_SL_2)
print('Mape of Simple Exponential Smoothing of 0.2:',mape_of_SES_SL_2)


# In[48]:


add_model = ExponentialSmoothing(train, trend='add').fit()
add_pred = add_model.forecast(49).rename('Additive ES')

ax = df.plot( color='black', figsize=(18,8), legend=True)

add_pred.plot(marker='+', ax=ax, color='red', legend=True)
add_model.fittedvalues.plot(marker='+', ax=ax, color='red')


# In[49]:


mul_model = ExponentialSmoothing(train, trend='mul').fit()
mul_pred = mul_model.forecast(49).rename('Multiplicative')

ax = df.plot( color='black', figsize=(18,8), legend=True)

mul_pred.plot(marker='+', ax=ax, color='red', legend=True)
mul_model.fittedvalues.plot(marker='+', ax=ax, color='red')


# In[50]:


rmse_of_AES = np.sqrt(mean_squared_error(y_true= test, y_pred= add_pred))
mape_of_AES = mean_absolute_percentage_error(y_true= test, y_pred= add_pred)
print('Rmse of Additive Exponential Smoothing :',rmse_of_AES)
print('Mape of Additive Exponential Smoothing :',mape_of_AES)

rmse_of_MES = np.sqrt(mean_squared_error(y_true= test, y_pred= mul_pred))
mape_of_MES = mean_absolute_percentage_error(y_true= test, y_pred= mul_pred)
print('Rmse of multiplicative Exponential Smoothing :',rmse_of_MES)
print('Mape of multiplicative Exponential Smoothing :',mape_of_MES)


# In[51]:


additive_model = ExponentialSmoothing(train, trend='add', seasonal='add').fit()
additive_pred = additive_model.forecast(49).rename('Additive ES')

ax = df.plot( color='black', figsize=(18,8), legend=True)

additive_pred.plot(marker='+', ax=ax, color='red', legend=True)
additive_model.fittedvalues.plot(marker='+', ax=ax, color='red')


# In[52]:


addmulticative_model = ExponentialSmoothing(train, trend='add', seasonal='mul').fit()
addmulticative_pred = addmulticative_model.forecast(49).rename('Additive Multiplicative ES')

ax = df.plot( color='black', figsize=(18,8), legend=True)

addmulticative_pred.plot(marker='+', ax=ax, color='red', legend=True)
addmulticative_model.fittedvalues.plot(marker='+', ax=ax, color='red')


# In[53]:


multiplicative_model = ExponentialSmoothing(train, trend='mul', seasonal='mul').fit()
multiplicative_pred = multiplicative_model.forecast(49).rename('Multiplicative ES')

ax = df.plot( color='black', figsize=(18,8), legend=True)

multiplicative_pred.plot(marker='+', ax=ax, color='red', legend=True)
multiplicative_model.fittedvalues.plot(marker='+', ax=ax, color='red')


# In[54]:


multiaddtive_model = ExponentialSmoothing(train, trend='mul', seasonal='add').fit()
multiaddtive_pred = multiaddtive_model.forecast(49).rename('Multiplicative Additive ES')

ax = df.plot( color='black', figsize=(18,8), legend=True)

multiaddtive_pred.plot(marker='+', ax=ax, color='red', legend=True)
multiaddtive_model.fittedvalues.plot(marker='+', ax=ax, color='red')


# In[55]:


rmse_of_AAES = np.sqrt(mean_squared_error(y_true= test, y_pred= additive_pred))
mape_of_AAES = mean_absolute_percentage_error(y_true= test, y_pred= additive_pred)
print('Rmse of Both Additive Exponential Smoothing :',rmse_of_AAES)
print('Mape of Both Additive Exponential Smoothing :',mape_of_AAES)
print('')
rmse_of_AMES = np.sqrt(mean_squared_error(y_true= test, y_pred= addmulticative_pred ))
mape_of_AMES = mean_absolute_percentage_error(y_true= test, y_pred= addmulticative_pred)
print('Rmse of Additive Multiplicative Exponential Smoothing :',rmse_of_AMES)
print('Mape of Additive Multiplicative Exponential Smoothing :',mape_of_AMES)
print('')
rmse_of_MMES = np.sqrt(mean_squared_error(y_true= test, y_pred= multiplicative_pred))
mape_of_MMES = mean_absolute_percentage_error(y_true= test, y_pred= multiplicative_pred)
print('Rmse of Both Multiplicative Exponential Smoothing :',rmse_of_MMES)
print('Mape of Both Multiplicative Exponential Smoothing :',mape_of_MMES)
print('')
rmse_of_MAES = np.sqrt(mean_squared_error(y_true= test, y_pred= multiaddtive_pred))
mape_of_MAES = mean_absolute_percentage_error(y_true= test, y_pred= multiaddtive_pred)
print('Rmse of Multiplicative Additive Exponential Smoothing :',rmse_of_MAES)
print('Mape of Multiplicative Additive Exponential Smoothing :',mape_of_MAES)


# In[56]:


Error = {'Models':pd.Series(['Simple_Exp_alpha_0.2','Double_Exp_Add','Double_Exp_Mul','Triple_Exp_add_add',
                             'Triple_Exp_add_mul','Triple_Exp_mul_mul','Triple_Exp_mul_add']),\
            'RMSE':pd.Series([rmse_of_SES_SL_2,rmse_of_AES,rmse_of_MES,rmse_of_AAES,rmse_of_AMES,rmse_of_MMES,rmse_of_MAES]),\
            'MAPE':pd.Series([mape_of_SES_SL_2,mape_of_AES,mape_of_MES,mape_of_AAES,mape_of_AMES,mape_of_MMES,mape_of_MAES])}
Check_best_model = pd.DataFrame(Error)
Check_best_model.sort_values(['MAPE'], inplace=True, ignore_index=True)
Check_best_model


# In[57]:


fig,(ax1, ax2, ax3) = plt.subplots(3)
plot_acf(df, ax=ax1)
plot_acf(df.diff().dropna(), ax=ax2)
plot_acf(df.diff().diff().dropna(), ax=ax3)
plt.show()


# In[58]:


plot_pacf(df.diff().dropna())
plot_acf(df.diff().dropna())
plt.show()


# In[59]:


model = auto_arima(train, order = (3,1,3))
model_fit = model.fit(train)
model_predict = model_fit.predict(49).rename("Auto Arima")

ax = df.plot( color='black', figsize=(12,6), legend=True)
model_predict.plot(marker='+', ax=ax, color='red', legend=True)
model_fit.summary()


# In[60]:


rmse_of_Auto_Arima = np.sqrt(mean_squared_error(y_true= test, y_pred= model_predict))
mape_of_Auto_Arima = mean_absolute_percentage_error(y_true= test, y_pred= model_predict)
print('Rmse of Auto Arima :',rmse_of_Auto_Arima)
print('Mape of Auto Arima :',mape_of_Auto_Arima)


# In[61]:


rcParams['figure.figsize'] = 18, 8
decomposition = seasonal_decompose(df, model='additive')
decomposition.plot()
plt.show()


# In[62]:


train


# In[63]:


sarimax_model = SARIMAX(train, order=(3,1,3), seasonal_order=(3,1,3,19))
sarimax_fit = sarimax_model.fit()
sarimax_fit.summary()


# In[64]:


df['forecast']=sarimax_fit.predict(start=192,end=241,dynamic=True)
df[['sales','forecast']].plot(figsize=(18,8))


# In[65]:


rmse_of_SARIMAX = np.sqrt(mean_squared_error(y_true= test, y_pred= df['forecast'].iloc[192:]))
mape_of_SARIMAX = mean_absolute_percentage_error(y_true= test, y_pred= df['forecast'].iloc[192:])
print('Rmse of Auto Arima :',rmse_of_SARIMAX)
print('Mape of Auto Arima :',mape_of_SARIMAX)


# In[66]:


df


# In[67]:


df = df.drop(['forecast'],axis = 1)
df


# In[68]:


df['sales_diff'] = df['sales'].shift(1)
df['sales_diff1'] = df['sales'].shift(2)
df = df.dropna()
df


# In[69]:


X = df.iloc[:,1:].values


# In[70]:


X


# In[71]:


y = df.iloc[:,0].values


# In[72]:


y


# In[73]:


X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)


# In[74]:


X_train.shape


# In[75]:


X_test.shape


# In[76]:


regressor = LinearRegression()
regressor_fit = regressor.fit(X_train, y_train)
regressor_pred = regressor_fit.predict(X_test)


# In[77]:


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o')
plt.plot(regressor_pred, label='Predicted', marker='x')
plt.xlabel("Data Point")
plt.ylabel("Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()


# In[78]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, regressor_pred, alpha=0.5)

p1 = max(max(regressor_pred), max(y_test))
p2 = min(min(regressor_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Regression: Actual vs Predicted Values")
plt.grid(True)
plt.show()


# In[79]:


Coefficient_regression = regressor_fit.coef_
rmse_of_regression = np.sqrt(mean_squared_error(y_true= y_test, y_pred= regressor_pred))
mape_of_regression = mean_absolute_percentage_error(y_true= y_test, y_pred= regressor_pred)
print('Coefficient of regression :', Coefficient_regression)
print('Rmse of regression :',rmse_of_regression)
print('Mape of regression :',mape_of_regression)


# In[80]:


knn_model = KNeighborsRegressor(n_neighbors=5,p=2,metric='minkowski')
knn_fit = knn_model.fit(X_train, y_train)
knn_pred = knn_fit.predict(X_test)


# In[81]:


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o')
plt.plot(knn_pred, label='Predicted', marker='x')
plt.xlabel("Data Point")
plt.ylabel("Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()


# In[82]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, knn_pred, alpha=0.5)

p1 = max(max(knn_pred), max(y_test))
p2 = min(min(knn_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("KNN Regression: Actual vs Predicted Values")
plt.grid(True)
plt.show()


# In[83]:


rmse_of_knn = np.sqrt(mean_squared_error(y_true= y_test, y_pred= knn_pred))
mape_of_knn = mean_absolute_percentage_error(y_true= y_test, y_pred= knn_pred)
print('Rmse of knn :',rmse_of_knn)
print('Mape of knn :',mape_of_knn)


# In[84]:


n_estimators_range = [5,10,15,20,25,30,35,40,45,50]

cv_scores_mean = []
cv_scores_std = []

#Cross_Validation
for n_estimators in n_estimators_range:
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=0.1, max_depth=3, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error')
    cv_scores_mean.append(-cv_scores.mean())
    cv_scores_std.append(cv_scores.std())

plt.figure(figsize=(10, 5))
plt.errorbar(n_estimators_range, cv_scores_mean, yerr=cv_scores_std, marker='o', linestyle='-', color='b')
plt.title('Gradient Boosting Learning Curve')
plt.xlabel('n_estimators')
plt.ylabel('Negative Mean Squared Error (RMSE)')
plt.grid(True)


# In[85]:


gbs = GradientBoostingRegressor(n_estimators=20,learning_rate=0.5,max_depth=2,random_state=42)
gbs_fit = gbs.fit(X_train, y_train)
gbs_pred = gbs_fit.predict(X_test)


# In[86]:


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o')
plt.plot(gbs_pred, label='Predicted', marker='x')
plt.xlabel("Data Point")
plt.ylabel("Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()


# In[87]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, gbs_pred, alpha=0.5)

p1 = max(max(gbs_pred), max(y_test))
p2 = min(min(gbs_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("GBS Regression: Actual vs Predicted Values")
plt.grid(True)
plt.show()


# In[88]:


rmse_of_gbs = np.sqrt(mean_squared_error(y_true= y_test, y_pred= gbs_pred))
mape_of_gbs = mean_absolute_percentage_error(y_true= y_test, y_pred= gbs_pred)
print('Rmse of Gradient Boosting Regressor :',rmse_of_gbs)
print('Mape of Gradient Boosting Regressor :',mape_of_gbs)


# In[89]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[90]:


trainX


# In[91]:


lstm = Sequential()
lstm.add(LSTM(24, activation='relu'))
lstm.add(Dense(1))
lstm.compile(optimizer='adam', loss='mape')
lstm_fit = lstm.fit(trainX, y_train, epochs=50,  batch_size=1, verbose=2)


# In[92]:


lstm_pred = lstm.predict(testX)


# In[93]:


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o')
plt.plot(lstm_pred, label='Predicted', marker='x')
plt.xlabel("Data Point")
plt.ylabel("Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()


# In[94]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, lstm_pred, alpha=0.5)

p1 = max(max(lstm_pred), max(y_test))
p2 = min(min(lstm_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("GBS Regression: Actual vs Predicted Values")
plt.grid(True)
plt.show()


# In[95]:


rmse_of_lstm = np.sqrt(mean_squared_error(y_true= y_test, y_pred= lstm_pred))
mape_of_lstm = mean_absolute_percentage_error(y_true= y_test, y_pred= lstm_pred)
print('Rmse of LSTM :',rmse_of_lstm)
print('Mape of LSTM :',mape_of_lstm)


# In[96]:


Error = {'Models':pd.Series(['Auto Arima','SARIMAX','Linear Regression','Gradient Boosting Regressor','LSTM']),            'RMSE':pd.Series([rmse_of_Auto_Arima,rmse_of_SARIMAX,rmse_of_regression,rmse_of_gbs,rmse_of_lstm]),            'MAPE':pd.Series([mape_of_Auto_Arima,mape_of_SARIMAX,mape_of_regression,mape_of_gbs,mape_of_lstm])}
Check_best_model = pd.DataFrame(Error)
Check_best_model.sort_values(['MAPE'], inplace=True, ignore_index=True)
Check_best_model

