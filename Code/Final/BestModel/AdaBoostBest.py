
# coding: utf-8

# In[1]:


# AdaBoost Classification
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Read data set
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)
#Train-Test split: 75%-25%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.25, random_state=42)


# In[2]:


from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
Adaclf = AdaBoostClassifier(algorithm='SAMME.R',learning_rate=0.01,n_estimators=600,
                                base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=40,
                                min_samples_leaf=1, min_samples_split=2,
                                splitter='best'))
Adaclf.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
predClf = Adaclf.predict(X_train)
trainConfusionMtrx = confusion_matrix(y_train, predClf)
trainConfusionMtrx


# In[3]:


AdaClfPred = Adaclf.predict(X_test)
testConfusionMtrx = confusion_matrix(y_test, AdaClfPred)
print("Confusion Matrix: \n",testConfusionMtrx)


# In[4]:


#Classification report
print("Classification Report:\n",classification_report(y_test, AdaClfPred))


# In[5]:


print("Accuracy:",Adaclf.score(X_test, y_test))


# In[6]:


#To avoid overfitting use kfold cross validation
k = 10

kFold = model_selection.KFold(n_splits=k, random_state=7)
Adaclf = AdaBoostClassifier(algorithm='SAMME.R',learning_rate=0.1,n_estimators=130,
                                base_estimator=DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=5,
                                min_samples_leaf=1, min_samples_split=2,
                                splitter='best'))
accuracy = model_selection.cross_val_score(Adaclf, data_values, data_labels, cv=kFold)
print("Accuracy with 10fold Cross Valid:",accuracy.mean())

