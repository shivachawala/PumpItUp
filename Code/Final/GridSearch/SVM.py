
# coding: utf-8

# In[ ]:


#[GridSearch] SVM Learning Classification
import pandas as pd
import numpy as np
import sys
# Read dataset
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

#Splitting the dataset in to train and test, splitting percentage taken is 25%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.25, random_state=42)


# In[ ]:


#[Model]: SVM 
models ={"SVM":SVC()}
#[Grid Search]: Combination of features based on our trails which are best suit for this model 
parameters = {"SVM":{"C":[1,10,100],
                     "kernel":('sigmoid', 'rbf'),
                     "gamma":(0.01,0.1,0.5,1),
                     "max_iter":[2000,5000,10000],
                     "random_state":[10]}}
classifier = ["SVM"]
#Running Grid Search on the parameters mentioned above
for c in classifier:
    SvmClf = GridSearchCV(models[c],parameters[c],cv=5)
    SvmClf = SvmClf.fit(X_train,y_train)
    score = SvmClf.score(X_test,y_test)
    prediction = SvmClf.predict(X_test)
    print("Accuracy using ",c," classifier is: ",score)
    print("-------------------------------------------")
    print("Below is the confusion Matrix for ",c )
    print(metrics.confusion_matrix(y_test,prediction))
    print("-------------------------------------------")
    print("Classification Report for ",c," is below")
    print(classification_report(prediction, y_test))
    print("-------------------------------------------")


# In[ ]:


SvmClf.best_params_
SvmClf.best_estimator_
SvmClf.best_score_


# In[ ]:


score = SvmClf.score(X_test, y_test)
print(score)

