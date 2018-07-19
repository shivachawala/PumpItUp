
# coding: utf-8

# In[1]:


# Deep Learning Classification
import pandas as pd 
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Read dataset
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)
data_values.head()
#Train-Test split: 75%-25%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.25, random_state=42)


# In[2]:


MlpClf = MLPClassifier(solver='adam',activation='relu',learning_rate='constant',learning_rate_init=0.01,alpha=0.0001,hidden_layer_sizes=(100))
MlpClf.fit(X_train, y_train)
print("Accuracy:",MlpClf.score(X_test, y_test))


# In[4]:


from sklearn.metrics import confusion_matrix
MlpClfPred = MlpClf.predict(X_test)
testConfusionMtrx = confusion_matrix(y_test, MlpClfPred)
print("Confusion Matrix: \n",testConfusionMtrx)


# In[5]:


#Classification report
print("Classification Report:\n",classification_report(y_test, MlpClfPred))


# In[6]:


#To avoid overfitting use kfold cross validation
from sklearn import model_selection
k = 10

kFold = model_selection.KFold(n_splits=k, random_state=7)
MlpClf = MLPClassifier(solver='adam',activation='relu',learning_rate='constant',learning_rate_init=0.01,alpha=0.0001,
                          hidden_layer_sizes=(100))
accuracy = model_selection.cross_val_score(MlpClf, data_values, data_labels, cv=kFold)
print("Accuracy with 10fold Cross Valid:",accuracy.mean())

