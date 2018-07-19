
# coding: utf-8

# In[ ]:


#[GridSearch] RandomForest Classifier
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Read data
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)
#Train-Test split: 75%-25%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.25, random_state=42)


# In[ ]:


from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
parameters={ "n_estimators" : [60,100,500],
           "max_depth" : [5, 10, 15, 20, 25, 30],
           "min_samples_leaf" : [2, 4, 6, 8, 10]}
classifier = [RandomForestClassifier(n_jobs=-1, max_features='sqrt', oob_score = True)]

from sklearn import metrics
for c in classifier:
    RandomForestClf = GridSearchCV(c, param_grid=parameters, cv= 10)
    RandomForestClf.fit(X_train, y_train)
    score = RandomForestClf.score(X_test,y_test)
    prediction = RandomForestClf.predict(X_test)
    print("Accuracy using ",c," classifier is: ",score)
    print("-------------------------------------------")
    print("Below is the confusion Matrix for ",c )
    print(metrics.confusion_matrix(y_test,prediction))
    print("-------------------------------------------")
    print("Classification Report for ",c," is below")
    print(classification_report(prediction, y_test))
    print("-------------------------------------------")


# In[ ]:


RandomForestClf.best_params_
RandomForestClf.best_estimator_
RandomForestClf.best_score_


# In[ ]:


score = RandomForestClf.score(X_test, y_test)
print(score)

