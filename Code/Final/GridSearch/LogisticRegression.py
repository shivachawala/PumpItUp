
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
# Load and read the training data
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#Train-Test split: 67%-33%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.33, random_state=20)

from sklearn.model_selection import GridSearchCV
model = LogisticRegression()
parameters = { "penalty":['l2'],
               # "dual": [True,False],
                "random_state": [10, 20],
                "max_iter":[500, 1000, 2000],
                "multi_class":['multinomial'],
                "solver":('newton-cg','lbfgs','saga')
            }
classifier = ["LogisticRegression"]
for c in classifier:
    LRclf = GridSearchCV(c, parameters)
    LRclf.fit(X_train,y_train)
    score = LRclf.score(X_test,y_test)
    prediction = LRclf.predict(X_test)
    print("Accuracy using ",c," classifier is: ",score)
    print("-------------------------------------------")
    print("Below is the confusion Matrix for ",c )
    print(metrics.confusion_matrix(y_test,prediction))
    print("-------------------------------------------")
    print("Classification Report for ",c," is below")
    print(classification_report(prediction, y_test))
    print("-------------------------------------------")


# In[ ]:


LRclf.best_params_
LRclf.best_estimator_
LRclf.best_score_


# In[ ]:


score = LRclf.score(X_test, y_test)
print(score)

