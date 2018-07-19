
# coding: utf-8

# In[2]:


# Best achieved SVM Learning Classification
import pandas as pd
import numpy as np
import sys
# Load dataset
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#Splitting the dataset in to train and test, splitting percentage taken is 33%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.33, random_state=10)


# In[3]:


#Our best SVC classifier
svmclf = SVC(C=1.0, kernel='sigmoid', degree=3, gamma = 0.1,random_state=10, max_iter=10000)
svmclf.fit(X_train, y_train)


# In[4]:


#Predicting the test data set on our best classifier
predclf = svmclf.predict(X_test)
#Confusion matrix
predclfmatrix = confusion_matrix(y_test, predclf)
print("Confusion Matrix: \n",predclfmatrix)


# In[5]:


#Classification report
print("Classification Report:\n",classification_report(y_test, predclf))


# In[6]:


#Accuracy achieved
print("Accuracy:",svmclf.score(X_test, y_test))


# In[8]:


#To avoid overfitting use kfold cross validation
from sklearn import model_selection
k = 10

kFold = model_selection.KFold(n_splits=k, random_state=7)
svmclf = SVC(C=1.0, kernel='sigmoid', degree=3, gamma = 0.1,random_state=10, max_iter=10000)
accuracy = model_selection.cross_val_score(svmclf, data_values, data_labels, cv=kFold)
print("Accuracy with 10fold Cross Valid:",accuracy.mean())

