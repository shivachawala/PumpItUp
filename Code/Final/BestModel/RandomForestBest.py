
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# read data set
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")

data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)
data_values.head()
#Train-Test split: 75%-25%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.25, random_state=42)


# In[2]:


from sklearn.ensemble import RandomForestClassifier
randomFrstclassifier = RandomForestClassifier(n_jobs=-1, max_features='sqrt', oob_score = True,n_estimators=500,max_depth=30,min_samples_leaf=2) 
randomFrstclassifier.fit(X_train, y_train)


# In[3]:


from sklearn.metrics import confusion_matrix
predClf = randomFrstclassifier.predict(X_train)
trainConfusionMtrx = confusion_matrix(y_train, predClf)
trainConfusionMtrx
randomFrstclassifier.score(X_train, y_train)


# In[4]:


predClfTest = randomFrstclassifier.predict(X_test)
testConfusionMtrx = confusion_matrix(y_test, predClfTest)
print("Confusion Matrix: \n",testConfusionMtrx)


# In[5]:


#Classification report
print("Classification Report:\n",classification_report(y_test, predClfTest))


# In[6]:


print("Accuracy:",randomFrstclassifier.score(X_test, y_test))


# In[7]:


#To avoid overfitting use kfold cross validation
from sklearn import model_selection
k = 10

kFold = model_selection.KFold(n_splits=k, random_state=7)
randomFrstclassifier = RandomForestClassifier(n_jobs=-1, max_features='sqrt', oob_score = True,n_estimators=500,max_depth=30,min_samples_leaf=2) 
accuracy = model_selection.cross_val_score(randomFrstclassifier, data_values, data_labels, cv=kFold)
print("Accuracy with 10fold Cross Valid:",accuracy.mean())

