# Customer-conversion-prediction
The following project is to build a machine learning model to predict the customer conversion based on diffrent feature. Various model have been applied to see which performs the best. the evaluation of the model is done on AUROC sore only. At the end feature importance has also been found that contributes the most.

Note: The sample file and problem statement are attach along with the py and ipynb file.

The data is proceesed in the following sequence:
            1. import libraries
            2. import data
            3. Clean Data
            4. EDA
            5. Encode
            6. Split
            7. Balance
            8. Scale
            9. Modeling and Evaluation
            10. Feature Importance
            
3. The data did not had any missing value and was not needed to be dealt with. The data did however contained outliers that were dealth with iqr technique. The csv file also had some duplicates record that were droped in the cleaning process.
4. EDA was performed and various plot were build to understan the data better. It was found that was highly imbalnced also dur column had linear decsion boundry but no other feature showed such segregation. It was also visible that our data was not highly correlated.
5. All the nominal data was encoded with one hot encoding and the target variable was encoded with label encoding.
6. As the data was highly imbalnced the data was split and balanced using smoteen libariy. Scaling was done to ensure standardizations.
7. Further modeling was done for classification model and feature importance was extracted.

It was found that logistic regression and Ensembl technique were the best model and had an auroc score of 91%. 
