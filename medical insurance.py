#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[4]:


#load dataset
insurance= pd.read_csv('medical_cost_insurance.csv')


# In[5]:


insurance.head()


# In[10]:


insurance.isnull().sum()


# In[12]:


insurance.duplicated()


# In[15]:


# Encode catergorical variables
le = LabelEncoder()
insurance['sex'] = le.fit_transform(insurance['sex'])
insurance['smoker'] = le.fit_transform(insurance['smoker'])
insurance['region'] = le.fit_transform(insurance['region'])


# In[17]:


# separate features x and y
X = insurance.drop('charges', axis=1)
Y = insurance['charges']


# In[18]:


# split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[19]:


# standardize feautures using standardscaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[20]:


# create a linear regression model
model = LinearRegression()


# In[21]:


# Train the model
model.fit(X_train_scaled, Y_train)


# In[22]:


# Make predictions on the test set
Y_pred = model.predict(X_test_scaled)


# In[23]:


# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)


# In[24]:


print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[26]:


# print predicted values
print('\nsample of Actual vs predicted values:')
result_insurance = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print(result_insurance.head())



# In[ ]:




