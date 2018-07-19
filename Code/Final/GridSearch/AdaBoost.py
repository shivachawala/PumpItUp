
# coding: utf-8

# In[1]:


#[GridSearch] AdaBoost Classification
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# read the training_data set
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)
#Train-Test Split : 75%-25%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.25, random_state=42)


# In[ ]:


from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
parameters={'base_estimator': [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4),
                               ExtraTreeClassifier(max_depth=4)],
            'learning_rate': [0.01, 0.1, 0.5, 1.],
            'n_estimators': [5, 10, 15, 20, 30, 40, 50, 75, 100, 125],
            'algorithm': ['SAMME', 'SAMME.R']}
model = AdaBoostClassifier()

AdaBoostClf = GridSearchCV(model,param_grid=parameters)
AdaBoostClf.fit(X_train, y_train)
score = AdaBoostClf.score(X_test,y_test)
prediction = AdaBoostClf.predict(X_test)
print("Accuracy using ",AdaBoostClf," classifier is: ",score)
print("-------------------------------------------")
print("Below is the confusion Matrix for ",AdaBoostClf )
print(metrics.confusion_matrix(y_test,prediction))
print("-------------------------------------------")
print("Classification Report for ",c," is below")
print(classification_report(prediction, y_test))
print("-------------------------------------------")


# In[ ]:


AdaBoostClf.best_params_
AdaBoostClf.best_estimator_
AdaBoostClf.best_score_


# In[ ]:


#Accuracy 
score = AdaBoostClf.score(X_test, y_test)
print(score)

