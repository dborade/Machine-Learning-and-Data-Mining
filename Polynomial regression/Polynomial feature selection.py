#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages
import time
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read in data from a data file to data_df in DateFrame format


data_df= pd.read_csv("poly_data.csv")

#verify the dataframe is imported correctly 
print(data_df.head(6))


# ## EDA

# In[3]:


#joint plot (or scatter plot) of X1 and y
sns.jointplot(data_df['X1'], data_df['y'])


# In[4]:


#joint plot (or scatter plot) of X2 and y
sns.jointplot(data_df['X2'], data_df['y'])


# In[5]:


#joint plot (or scatter plot) of X1 and X2
sns.jointplot(data_df['X1'], data_df['X2'])


# ### Based on observing the above 3 diagrams and the p-values displayed, we found both X1 and X2 have close correlation with y. X1 and X2 are independent from each other. 

# ### 3. Split the Data

# In[6]:


# split the data into training and testing datasets
# the percentage of training data is 75%

#split point 
percentage_for_training = 0.75
n_samples = data_df.shape[0]


number_of_training_data = int(n_samples*percentage_for_training)



#create training and testing datasets
train_df  = data_df[0:number_of_training_data]
test_df = data_df[number_of_training_data:]
print(train_df.shape)
print(test_df.shape)


# ### 4. Create Polynomial Features

# In[7]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#set the degree to 3
#for degree = 3, we will generate 9 features. 
polynomial_features = PolynomialFeatures(degree=3)


# In[8]:


X_poly = polynomial_features.fit_transform(data_df[['X1','X2']])

#extract X for training and testing from the data frames
X_train = X_poly[0:number_of_training_data]
X_test = X_poly[number_of_training_data:]


# ### 5. Create and Train a Linear Regression Model

# In[9]:


# mse() calculates mean square error of a model on given X and y
def mse(X, y, model):
    return  ((y-model.predict(X))**2).sum()/y.shape[0]


# In[10]:


# use all the features to train the linear model 
lm = LinearRegression()
lm.fit(X_train, train_df['y'])
train_mse = mse(X_train, train_df['y'], lm)
print("Training Data Set's MSE is: \t", train_mse)
test_mse = mse(X_test, test_df['y'], lm)
print("Testing Data Set's MSE is : \t", test_mse)


# ### 6. Use Lasso in Linear Regression to Penalize Large Number of Features

# In[11]:


#import lasso
#lasso is controlled by a parameter alpha.
#by fine tuning this parameter, we can control the number of features

from sklearn.linear_model import Lasso
#Train the model, try different alpha values.
Lasso_model = Lasso(alpha=0.15,normalize=True, max_iter=1e5, )
Lasso_model.fit(X_train, train_df['y'])


# In[12]:


#see the trained parameters. Zero means the feature can be removed from the model
Lasso_model.coef_


# In[13]:


#let's see the train_mse and test_mse from Lasso when 
#alpha = 0.15

train_mse = mse(X_train, train_df['y'], Lasso_model)
print("Training Data Set's MSE is: \t", train_mse)
test_mse = mse(X_test, test_df['y'], Lasso_model)
print("Testing Data Set's MSE is : \t", test_mse)


# In[14]:


#let's try a large range of values for alpha first
#create 50 alphas from 100 to 0.00001 in logspace
alphas = np.logspace(2, -5, base=10, num=50)
alphas


# In[15]:


#use arrays to keep track of the MSE of each alpha used. 
train_mse_array =[]
test_mse_array=[]

#try each alpha
for alpha in alphas:
    
    #create Lasso model using alpha
    Lasso_model = Lasso(alpha=alpha,normalize=True, max_iter=1e5, )
    Lasso_model.fit(X_train, train_df['y'])
    
    #Calculate MSEs of train and test datasets 
    train_mse = mse(X_train, train_df['y'], Lasso_model)
    test_mse = mse(X_test, test_df['y'], Lasso_model)
    
    #add the MSEs to the arrays
    train_mse_array.append(train_mse)
    test_mse_array.append(test_mse)
    


# In[16]:


#plot the MSEs based on alpha values
#blue line is for training data
#red line is for the testing data
plt.plot(np.log10(alphas), train_mse_array)
plt.plot(np.log10(alphas), test_mse_array, color='r')


# ### There is something interesting between 0 and 1 in the above diagram. 0 mean 10^0=1 While 1 means 10^1 = 10  so, we will look closely within this range to find the optimal alpha value

# In[30]:


# We can try a smaller search space now (a line space between 1 and 10)
alphas = np.linspace(1, 10, 1000)
train_mse_array =[]
test_mse_array=[]
print(alphas)


# In[31]:


earlier_dif = 5000

#try each alpha
for alpha in alphas:
    
    #create Lasso model using alpha
    Lasso_model = Lasso(alpha=alpha,normalize=True, max_iter=1e5, )
    Lasso_model.fit(X_train, train_df['y'])
    
    #Calculate MSEs of train and test datasets 
    train_mse = mse(X_train, train_df['y'], Lasso_model)
    test_mse = mse(X_test, test_df['y'], Lasso_model)
    
    dif = abs(train_mse-test_mse)
    if(dif < earlier_dif):
        earlier_dif = dif
        best_train_mse = train_mse
        best_test_mse = test_mse
        best_alpha = alpha   
    
    #add the MSEs to the arrays
    train_mse_array.append(train_mse)
    test_mse_array.append(test_mse)
    
#Print best alpha, train_mse and test_mse
print("Train  MSE is", best_train_mse)
print("Test MSE is", best_test_mse)
print("The optimal alpha is", best_alpha)
    


# In[34]:


#Plotting MSE's
plt.plot(alphas, train_mse_array)
plt.plot(alphas, test_mse_array, color='r')


# ### By observing a smaller range of alpha, we can clearly see how the MSEs change as we change the model and features. Use the diagram to explain the trends of the two lines and summarize what you learned so far. 

# In[36]:


#Fitting lasso on the best alpha
Lasso_model = Lasso(alpha=3.5585585585585586,normalize=True, max_iter=1e5, )
Lasso_model.fit(X_train, train_df['y'])
Lasso_model.coef_


# In[38]:


## type your code here to describe the above diagram and what you learned 
## so far about feature and model selection ( about 200 words )
print("Alpha value for training data is increasing while for test data it is decreasing, the point at which both intersect can give us the optimal value of alpha fro our model selection. As we found the best alpha we could find the coefficients which are closer to true model as shown above. Although our model would not be too perfect but we can a chieve the closest fit to true model based on MSE values. ")


# In[ ]:




