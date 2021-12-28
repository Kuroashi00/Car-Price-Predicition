#!/usr/bin/env python
# coding: utf-8

# 

# ## Libraries and Modules 

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm
sns.set()


# ## Dataset

# In[25]:


dataset = pd.read_csv('dataset/Car_sales.csv')
dataset.head()


# ## Variable exploration through descriptive statistics 

# In[26]:


dataset.describe(include='all')


# In[27]:


data = dataset.drop(['Model'],axis=1)


# In[32]:


data.isnull().sum()


# ## Drop all missing value

# In[35]:


data_no_mv = data.dropna(axis=0)
data_no_mv.isnull().sum()


# ## Display the probability distribution function (PDF) of a variable

# In[38]:


sns.displot(data_no_mv['Price_in_thousands'])
# There is outliers present to this graph, the outliers situated around higher prices(right side of the graph)


# In[43]:


# I just removing by 0.5% or 1% of the problematic samples
q = data_no_mv['Price_in_thousands'].quantile(0.93)
data_1 = data_no_mv[data_no_mv['Price_in_thousands']<q]
sns.displot(data_1['Price_in_thousands'])


# In[45]:


sns.displot(data_no_mv['Engine_size'])


# In[50]:


q = data_1['Engine_size'].quantile(0.99)
data_2 = data_1[data_1['Engine_size']<q]
sns.displot(data_2['Engine_size'])


# In[52]:


sns.displot(data_no_mv['Horsepower'])


# In[53]:


q = data_2['Horsepower'].quantile(0.999)
data_3 = data_2[data_2['Horsepower']<q]
sns.displot(data_3['Horsepower'])


# In[57]:


sns.displot(data_no_mv['Power_perf_factor'])


# In[55]:


q = data_3['Power_perf_factor'].quantile(0.99)
data_4 = data_3[data_3['Power_perf_factor']<q]
sns.displot(data_4['Power_perf_factor'])


# In[56]:


data_4.describe()
# We can see that there are no outliers on some samples


# In[168]:


data_cleaned = data_4.reset_index(drop=True)
# we reset the index, a new column will be created containing the old index
data_4


# ## Checking the OLS assumptions 

# In[106]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Engine_size'],data_cleaned['Price_in_thousands'])
ax1.set_title('Price and Engine_size')
ax2.scatter(data_cleaned['Horsepower'],data_cleaned['Price_in_thousands'])
ax2.set_title('Price and Horsepower')
ax3.scatter(data_cleaned['Power_perf_factor'],data_cleaned['Price_in_thousands'])
ax3.set_title('Price and Power_perf_factor')

plt.show()


# In[189]:


# Determine that 'Price' is exponentially distributed
# A good transformation in that case is a log transformation
# Let's transform 'Price_in_thousands' with a log transformation
log_price = np.log(data_cleaned['Price_in_thousands'])

# Then we add it to our data frame
data_cleaned['log_price'] = log_price


# In[190]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Engine_size'],data_cleaned['log_price'])
ax1.set_title('Log Price and Engine_size')
ax2.scatter(data_cleaned['Horsepower'],data_cleaned['log_price'])
ax2.set_title('Log Price and Horsepower')
ax3.scatter(data_cleaned['Power_perf_factor'],data_cleaned['log_price'])
ax3.set_title('Log Price and Power_perf_factor')

plt.show()

# The relationships show a more clear linear relationship
# This is some good linear regression material


# In[193]:


data_cleaned
#data_cleaned.drop(['Latest_Launch', 'Price_in_thousands'],axis=1)


# In[194]:


data_cleaned.columns.values


# ## Create dummy variable

# In[195]:


# To include the categorical data in the regression, let's create dummies
dummy = pd.get_dummies(data_cleaned, drop_first=True)
data_with_dummy = dummy.drop(['Price_in_thousands'], axis=1)


# In[196]:


data_with_dummy.columns.values


# In[200]:


cols = ['log_price', 'Sales_in_thousands', '__year_resale_value', 'Engine_size',
       'Horsepower', 'Wheelbase', 'Width', 'Length', 'Curb_weight',
       'Fuel_capacity', 'Fuel_efficiency', 'Power_perf_factor',
       'Manufacturer_Audi', 'Manufacturer_BMW', 'Manufacturer_Buick',
       'Manufacturer_Cadillac', 'Manufacturer_Chevrolet',
       'Manufacturer_Chrysler', 'Manufacturer_Dodge', 'Manufacturer_Ford',
       'Manufacturer_Honda', 'Manufacturer_Hyundai',
       'Manufacturer_Infiniti', 'Manufacturer_Jeep', 'Manufacturer_Lexus',
       'Manufacturer_Lincoln', 'Manufacturer_Mercedes-B',
       'Manufacturer_Mercury', 'Manufacturer_Mitsubishi',
       'Manufacturer_Nissan', 'Manufacturer_Oldsmobile',
       'Manufacturer_Plymouth', 'Manufacturer_Pontiac',
       'Manufacturer_Porsche', 'Manufacturer_Saturn',
       'Manufacturer_Toyota', 'Manufacturer_Volkswagen',
       'Vehicle_type_Passenger']


# In[ ]:





# In[145]:





# In[ ]:





# In[203]:


data_preprocessed = data_with_dummy[cols]
data_preprocessed.head()


# ## Linear regression model

# In[204]:


# Declare targets and inputs

targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)


# In[205]:


# Scale the data

scaler = StandardScaler()
scaler.fit(inputs)

# store inputs/features in a new variable
inputs_scaled = scaler.transform(inputs)


# In[206]:


# Train Test Split
# Split the variables with an 75-25 split and some random state
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.25, random_state=365)

# Create the regression
reg = LinearRegression() 
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)


# In[207]:


# Store outputs in y_hat as this is the 'theoretical' name of the predictions
y_hat = reg.predict(x_train)


# In[208]:


# Compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_hat)

# name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.show()


# In[209]:


# Plot the PDF of the residuals and check for anomalies
sns.displot(y_train - y_hat)
plt.title("Residuals PDF", size=18)

# we notice that there are many negative residuals
# given the definition of the residuals (y_train - y_hat), negative values imply
# that y_hat (predictions) are much higher than y_train (the targets)


# In[210]:


# Find the R-squared of the model
reg.score(x_train,y_train)


# In[211]:


# Find the bias (intercept) of the regression
reg.intercept_


# In[212]:


# Find the weights (coefficients) of the regression
reg.coef_


# In[213]:


# Regression summary where we can compare them with one-another
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[214]:


data_cleaned['Manufacturer'].unique()


# ## Testing

# In[217]:


# test inputs are 'x_test', while the outputs: 'y_test' 
y_hat_test = reg.predict(x_test)


# In[220]:


# Create a scatter plot with the test targets and the test predictions
plt.scatter(y_test, y_hat_test)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.show()


# In[221]:


# manually check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[222]:


# include the test targets in that data frame
df_pf['Target'] = np.exp(y_test)
df_pf

# There are a lot of missing values


# In[223]:


# After displaying y_test, we find what the issue is
# The old indexes are preserved (recall earlier in that code we made a note on that)
# The code was: data_cleaned = data_4.reset_index(drop=True)

# Therefore, to get a proper result, we must reset the index and drop the old indexing
y_test = y_test.reset_index(drop=True)

# Check the result
y_test.head()


# In[224]:


# overwrite the 'Target' column with the appropriate values

df_pf['Target'] = np.exp(y_test)
df_pf


# In[225]:


# calculate the difference between the targets and the predictions
# the residual that already plotted
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# Since OLS is basically an algorithm which minimizes the total sum of squared errors (residuals)


# In[226]:


# we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[227]:


# To see all rows
pd.options.display.max_rows = 999

# make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])


# In[ ]:




