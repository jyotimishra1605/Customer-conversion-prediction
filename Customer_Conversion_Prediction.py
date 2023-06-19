#!/usr/bin/env python
# coding: utf-8

# # Customer Conversion Prediction

# ### 1. Importing important libraries
import matplotlib.pyplot as plt
import numpy as np
# Importing the important libraries
import pandas as pd
# Importing libraries for plotting graph
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
# Importing Libaries for modeling and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# Importing libraries for pre-porocessing of our data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ### 2. Importing the dataset and displaying the summary
# Importing my traing dataset and getting an intial look on how data looks
act_data = pd.read_csv("C:\\Users\\ADMIN\\Downloads\\train.csv")
act_data = act_data.copy()

act_data
act_data.shape
act_data.describe()

# ### 3. Clean Data
# ##### 3.1 Finding missing Values
# Finding missing values in our data
act_data.isnull().sum()
# There are no missing values in our dataset

# ##### 3.2 Finding and deleting duplicate rows
# Finding duplicate values and droping them
act_data = act_data.drop_duplicates()
act_data.shape

# ##### 3.3 Removing outliers from our numerical data
act_data.describe()
# Finding and clipping our data based on outliers using iqr technique
for i in act_data.select_dtypes(include=['int64', 'float64']):
    iqr = act_data[i].quantile(0.75) - act_data[i].quantile(0.25)
    upper_threshold = act_data[i].quantile(0.75) + (1.5 * iqr)  # q3 + 1.5iqr
    lower_threshold = act_data[i].quantile(0.25) - (1.5 * iqr)  # q1 - 1.5iqr
    act_data = act_data.copy()
    act_data[i] = act_data[i].clip(lower_threshold, upper_threshold)
act_data.describe()

# ##### 3.4 Handling invalid data
# Frequency of unique elements in catogorical column
cat_cols = act_data.select_dtypes(include=object).columns.tolist()
(pd.DataFrame(act_data[cat_cols].melt(var_name='column', value_name='value').value_counts()).rename(
    columns={0: 'counts'}).sort_values(by=['column', 'counts']))
# As we can see in education_qual and job column their are few unknown elements which can be replaced with mode
# Replacing unknown values in education_qual and job columns with mode
act_data = act_data.copy()
act_data['education_qual'] = act_data['education_qual'].replace('unknown', act_data['education_qual'].mode()[0])
act_data['job'] = act_data['job'].replace('unknown', act_data['job'].mode()[0])
# Also call_type and prev_outcome column have unknown values but their frequency is higher and can be treated as seprated element itself
# Renaming the unknown values so as to better analyse and does not coincide with each other
act_data['call_type'] = act_data['call_type'].replace('unknown', 'unknown_call_type')
act_data['prev_outcome'] = act_data['prev_outcome'].replace('unknown', 'unknown_prev_outcome')
# Frequency of unique elements in catogorical column after renaming
cat_cols = act_data.select_dtypes(include=object).columns.tolist()
(pd.DataFrame(act_data[cat_cols].melt(var_name='column', value_name='value').value_counts()).rename(
    columns={0: 'counts'}).sort_values(by=['column', 'counts']))

# ### 4. Exploratory data analysis (EDA)
# Feature vs Target Plot
fig, axes = plt.subplots(2, 5, figsize=(16, 8), sharey=True)
fig.suptitle('Feature vs Target')
xc = 0
yc = 0
for i in act_data.columns[:-1]:
    sns.scatterplot(data=act_data, x=i, y='y', hue='y', ax=axes[xc, yc])
    yc = yc + 1
    if yc == 5:
        yc = 0
        xc = 1

# Catogorical data count plot
fig, axes = plt.subplots(3, 3, figsize=(25, 20), sharey=False)
fig.suptitle('Feature vs Target')
xc = 0
yc = 0
for i in act_data.select_dtypes(include=['object'], exclude=['int64', 'float64']).columns:
    sns.countplot(x=i, data=act_data, hue='y', ax=axes[xc, yc])
    yc = yc + 1
    if yc == 3:
        yc = 0
        xc = xc + 1

# Feature vs Feature (only numeric)
sns.pairplot(act_data, hue='y')

"""
for i in act_data.select_dtypes(include=['object'], exclude=['int64', 'float64']).columns[:-1]:
    for j in act_data.select_dtypes(include=['object'], exclude=['int64', 'float64']).columns[:-1]:
        plt.figure(figsize=(20,5))
        plt.subplot(121)
        sns.countplot(data=act_data, x=i, hue=j)
"""

# ### 5. Encoding our categorical data
for i in act_data.select_dtypes(include=['object'], exclude=['int64', 'float64']).columns[:-1]:
    # Get one hot encoding of job column
    one_hot = pd.get_dummies(act_data[i])
    # Drop column B as it is now encoded
    act_data = act_data.drop(i, axis=1)
    # Join the encoded df
    act_data = act_data.join(one_hot)
# Get label encoding for y column
act_data["y"] = act_data["y"].map({"yes": 1, "no": 0})  # encoding binary class data (run only once)

# ### 6. Split the dataset in train and test
# Spliting the dat into train and test
x = act_data[
    ['age', 'day', 'dur', 'num_calls', 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
     'self-employed', 'services', 'student', 'technician', 'unemployed', 'divorced', 'married', 'single', 'primary',
     'secondary', 'tertiary', 'cellular', 'telephone', 'unknown_call_type', 'apr', 'aug', 'dec', 'feb', 'jan', 'jul',
     'jun', 'mar', 'may', 'nov', 'oct', 'sep', 'failure', 'other', 'success', 'unknown_prev_outcome']].values

y = act_data[['y']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# ### 7. Balancing the dataset
# Balancing the dataset
smt = SMOTEENN(sampling_strategy='all')
x_train, y_train = smt.fit_resample(x_train, y_train)

# ### 8. Standardize the dataset
# Standarize the dataset before fitting it into the model
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
act_data
act_data.describe()


# ### 9. Model, Loss, Learning, and Evaluation
class ClassificationModel:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_test = x_test
        self.x_train = x_train
        self.y_train = y_train
        self.y_test = y_test

    def log_reg_model(self):
        log_reg = LogisticRegression()  # initialise the model
        log_reg.fit(self.x_train, self.y_train)  # training the data
        y_pred = log_reg.predict_proba(self.x_test)  # Predicting
        roc = roc_auc_score(self.y_test, y_pred[:, 1])  # Evaluation
        return roc

    def knn_model(self):
        # Finding the best value for K hyper parameter based on higest cv score
        '''
        khp = 0
        hcv = 0
        for i in [1,2,3,4,5,6,7,8,9,10]:
            knn = KNeighborsClassifier(i) #initialising the model
            knn.fit(x_train,y_train) # training the model
            if np.mean(cross_val_score(knn, x_train, y_train, cv=10, scoring = "roc_auc")) > hcv:
                hcv = np.mean(cross_val_score(knn, x_train, y_train, cv=10, scoring = "roc_auc"))
                khp = i
            else:
                break
        '''
        # Input the kbest K value and fit the model
        knn = KNeighborsClassifier(6)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        roc = roc_auc_score(y_test, y_pred)

        return roc

    def dec_tree_model(self):
        dt = DecisionTreeClassifier(max_depth=9)
        dt.fit(self.x_train, self.y_train)
        y_pred = dt.predict(self.x_test)
        roc = roc_auc_score(self.y_test, y_pred)
        return roc

    def ens_model(self):
        model1 = LogisticRegression(random_state=1)
        model2 = tree.DecisionTreeClassifier(max_depth=9, random_state=1)
        model3 = KNeighborsClassifier(6)
        model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('knn', model3)], voting='soft')
        model.fit(self.x_train, self.y_train)
        model.predict(self.x_test)
        y_pred = model.predict_proba(self.x_test)
        roc = roc_auc_score(self.y_test, y_pred[:, 1])
        return roc

    def rf_model(self):
        rf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features='sqrt')
        rf.fit(self.x_train, self.y_train)
        y_pred = rf.predict(self.x_test)
        roc = roc_auc_score(self.y_test, y_pred)
        return roc

    def xg_model(self):
        model = XGBClassifier(learning_rate=0.5, n_estimators=100, verbosity=None)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        roc = roc_auc_score(y_test, y_pred)
        return roc


clsmod = ClassificationModel(x_train, x_test, y_train, y_test)  # Intialzie the class
temp_dict = {}
knn_score = clsmod.knn_model()
temp_dict['KNN'] = knn_score
log_reg_score = clsmod.log_reg_model()
temp_dict['Logestic Regression'] = log_reg_score
dec_score = clsmod.dec_tree_model()
temp_dict['Decision Tree'] = dec_score
ens_score = clsmod.ens_model()
temp_dict['Ensemble'] = ens_score
rf_score = clsmod.rf_model()
temp_dict['Random Forest'] = rf_score
xg_score = clsmod.xg_model()
temp_dict['XGBoost'] = xg_score
df = pd.DataFrame.from_dict(temp_dict, orient='index', columns=['AUROC Score'])


# ### 10. Feature Importance
# Finding feature imporatance using random forest classifier
rf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features='sqrt')
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
roc = roc_auc_score(y_test, y_pred)
importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = act_data.columns[1:]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[sorted_indices[f]],
                            importances[sorted_indices[f]]))
