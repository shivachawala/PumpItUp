
# coding: utf-8

# In[ ]:


#[GridSearch] Deep Learning Classification
import pandas as pd 
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# read the training_data set
data_values = pd.read_csv("../../../Datasets/train_values_processed.csv")
data_labels = data_values["status_group"]
data_values.drop(['status_group'], axis=1, inplace=True)
#Train-Test split: 75%-25%
X_train, X_test, y_train, y_test = train_test_split(data_values, data_labels, test_size=0.25, random_state=42)


# In[ ]:


from sklearn.grid_search import GridSearchCV
parameters={
'learning_rate':["constant"],
'learning_rate_init':[0.01,0.1,0.5],
'hidden_layer_sizes':[(100),(90,50),(100,60,20)],
'alpha': [0.001,0.01,0.1],
'activation':["logistic", "relu", "tanh"]
}
classifier = [MLPClassifier()]
for c in classifier:
    MlpClf = GridSearchCV(c,param_grid=parameters)
    MlpClf.fit(X_train, y_train)
    score = MlpClf.score(X_test,y_test)
    prediction = MlpClf.predict(X_test)
    print("Accuracy using ",c," classifier is: ",score)
    print("-------------------------------------------")
    print("Below is the confusion Matrix for ",c )
    print(metrics.confusion_matrix(y_test,prediction))
    print("-------------------------------------------")
    print("Classification Report for ",c," is below")
    print(classification_report(prediction, y_test))
    print("-------------------------------------------")


# In[ ]:


MlpClf.best_params_
MlpClf.best_estimator_
MlpClf.best_score_


# In[ ]:


score = MlpClf.score(X_test, y_test)
print(score)

