
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
#Train-Test split: 67%-33%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.33, random_state=42)


# In[2]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2',max_iter=2000, random_state=10)
logreg.fit(X_train,y_train)


# In[6]:


from sklearn.metrics import confusion_matrix
logreg_predict = logreg.predict(X_train)
trainConfusionMtrx = confusion_matrix(y_train, logreg_predict)
trainConfusionMtrx
logreg.score(X_train, y_train)


# In[7]:


logreg_predict = logreg.predict(X_test)
testConfusionMtrx = confusion_matrix(y_test, logreg_predict)
print("Confusion Matrix: \n",testConfusionMtrx)


# In[8]:


#Classification report
print("Classification Report:\n",classification_report(y_test, logreg_predict))


# In[9]:


logreg.score(X_test, y_test)


# In[11]:


#To avoid overfitting use kfold cross validation
from sklearn import model_selection
k = 10
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2',max_iter=2000, random_state=10)
kFold = model_selection.KFold(n_splits=10, random_state=10)
LR_accuracy = model_selection.cross_val_score(logreg, data_values, data_labels, cv=kFold)
print("Accuracy with 10fold Cross Valid:",LR_accuracy.mean())

