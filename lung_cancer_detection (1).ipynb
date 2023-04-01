#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# # import data

# In[53]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[54]:


train_data = pd.read_csv(r'C:\Users\popo\Pictures\survey lung cancer.csv')
train_data.head()


# In[55]:


pd.set_option('display.max_row', None)
pd.set_option('display.max_colum', None)
print(train_data.describe())


# In[56]:


print(train_data.describe(include=["O"]))
# print(train_data.describe(include=["O"]))


# In[57]:


print(train_data.info())


# In[58]:


GENDER_mapping = {"M":1,"F":0}
train_data['GENDER'] = train_data['GENDER'].map(GENDER_mapping)

LUNG_CANCER_mapping = {"YES":1,"NO":0}
train_data['LUNG_CANCER'] = train_data['LUNG_CANCER'].map(LUNG_CANCER_mapping)
print(train_data.info())


# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import  StandardScaler
# from sklearn.model_selection import train_test_split # data split
# 
# sc = StandardScaler()
# 
# x = train_data.iloc[:,:-1]
# y = train_data.iloc[:,-1]
# 
# x = sc.fit_transform(x)
# 
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

# # logistic regression
# 

# In[62]:


lr = LogisticRegression()

lr.fit(x_train,y_train)
print("Logistic Score : ",lr.score(x_test,y_test))


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(lr,x_test,y_test)
plt.show()


# # Train with Tree Decision

# In[67]:


from sklearn import tree
dtree =tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)
print("decission tree score:",dtree.score(x_test, y_test))

plot_confusion_matrix(dtree,x_test,y_test)
plt.show()

print(lr.coef_)


# # Train with Polynomial Features

# In[68]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)
lr.fit(x_train_poly,y_train)
print('Polynomial Score = ', lr.score(x_test_poly,y_test))


# # Train with MLPClassifier

# In[69]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(25,11,7,5,3,)) # Random state number is a random method, 42 is most popular
mlp.fit(x_train,y_train)
print('MLPClassifier Score = ',mlp.score(x_test,y_test)) 


# In[ ]:




