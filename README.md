# Pump It Up: Data Mining The Water Tables

CS 6375 Machine Learning Project
<br>

**Team members**
- Ankita Patil
- Abhilash Gudasi
- Shiva Chawala

<hr>

**Project Source:** <a href="https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/">DrivenData</a><br>

**Challenge Name:** Pump It Up: Data Mining the Water Table <br>

**Goal:** To predict the operating condition of waterpoints in Tanzania i.e. to determine whether the water pump is functional, non-functional or needs repair

<hr>

## Workflow

1.	Data Exploration
    - Univariate Analysis
    - Correlation Graph
2.	Pre-processing/Feature Engineering
3.	Algorithm Implementation
    -	Model training and parameter tuning using GridSearchCV 
      - With train-test split
      - With k-fold cross validation
4.	With the trained model, predict the accuracy on the test data 


## Algorithms Implemented

**Justification of selecting supervised machine learning algorithms** <br>
By exploring the datasets, we observe that for each instance, a label is provided. When data with label is provided, supervised machine learning algorithms can be applied.<br>
Following five algorithms are used in model creation for Pump It Up: Data Mining the Water Table dataset<br>
1.	Logistic Regression
2.	Support Vector Machine
3.	Adaboosting
4.	Neural Net
5.	Random Forest

<hr>

Folder structure as submitted :

```
Root(Pumpitup)
|
---Code--|
|	 |--Final--|
|		   |
|		   ---BestModel
|		   ---GridSearch
|		   ---Pre-process
|		   ---ROC
|
|
|
---Datasets--|
	     |
	     ---Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv
	     ---Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv
	     ---test_values_processed.csv
	     ---train_values_processed.csv
	     ---heights.csv
```


1. Root folder(Pumpitup) is divided in to two folder one for code and another for Datasets. <br>
  Code-->Final: <br>
		Inside Final folder: <br>
		 - BestModel contains the all five best model using different techniques we achieved in this project. <br>
		 - GridSearch contains our initial exploration to find the best model trying different parameters. <br>
		 - Pre-process contains one python file for generating the preprocessed data and one for generating the missing gps_heights(one of the attribute in the given dataset). The preprocessing python file will generate the processed csv file inside Dataset folder and the other python file will compute the missing gps_height values and generate heights.csv file inside Dataset folder.  <br>
		- ROC contains the python file for all best models output ROC curve generation code. <br>
  
  **Datasets:** This folder contains the dataset(values and labels) we got from DataDriven competition website for PumpitUp problem. This also contains the preprocessed datasets.


2. To run the code:
	If the same folder structure is maintained as mentioned above <br>
	- **Preprocessing:** Since we are displaying various plots in this, you cannot run it in command line. So recommended to run the file in Jupyter using the PumpItUpPreprocessing.ipynb file.

	 	
	Running Best Models:
	Go inside BestModel folder and run the below command:
	
		Syntax: python Modelname.py

		Ex: python AdaBoostBest.py
	    	    python DeepLearningBest.py
	    	    python LogisticRegressionBest.py
	    	    python RandomForestBest.py
	    	    python SVMBest.py

	After running above command in command line you will see <br>
	    -->Confusion matrix,<br>
		-->Classification report, <br>
		-->Accuracy of the model using train-test split and <br>
		-->Accuracy of the model using kfold cross validation outputs. <br>
 
