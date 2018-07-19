Extra days : 02

Pre-requirement on machine:
- Python 3.6
- Scikit-learn, pandas packages installed

Executing and running the project:

Folder structure as submitted :

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


1.Root folder(Pumpitup) is divided in to two folder one for code and another for Datasets.
  Code-->Final:
		Inside Final folder: 
		i) BestModel contains the all five best model using different techniques we achieved in this project.
		ii)GridSearch contains our initial exploration to find the best model trying different parameters.
		iii)Pre-process contains one python file for generating the preprocessed data and one for generating the missing gps_heights(one of the attribute in 		the given dataset). The preprocessing python file will generate the processed csv file inside Dataset folder and the other python file will compute the 		missing gps_height values and generate heights.csv file inside Dataset folder.  
		iv)ROC contains the python file for all best models output ROC curve generation code.
  
  Datasets: This folder contains the dataset(values and labels) we got from DataDriven competition website for PumpitUp problem.
	    This also contains the preprocessed datasets.


2. To run the code:
	If the same folder structure is maintained as mentioned above
	Preprocessing:Since we are displays plots in this, you cannot run it in command line.
	              So recommended to run it in Jupyter using the PumpItUpPreprocessing.ipynb file.

	 	
	Running Best Models:
	Go inside BestModel folder and run the below command:
	
		Syntax: python Modelname.py

		Ex: python AdaBoostBest.py
	    	    python DeepLearningBest.py
	    	    python LogisticRegressionBest.py
	    	    python RandomForestBest.py
	    	    python SVMBest.py

		After running above command in command line you will see 
			-->Confusion matrix,
			-->Classification report, 
			-->Accuracy of the model using train-test split and 
			-->Accuracy of the model using kfold cross validation outputs. 
 
	
