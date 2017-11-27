
# coding: utf-8

# # Topics To Cover
# 1)Read a CSV Data file with index

# # Import all  necessary libraries

# In[1]:

import pandas as pd


# # Read a csv file: assign variable "Data_csv"

# In[2]:

#Read a csv file: assign variable "Data_csv"

Data_csv = pd.read_csv("E:/DataAnalyticsCourse/GittHub/Datasets/Loan_prediction/train.csv")
#Data_csv = pd.read_csv("url")
Data_csv.head()


# In[5]:

#Suppose if you dont have(want) columns name
Data_csv1 = pd.read_csv("E:/DataAnalyticsCourse/GittHub/Datasets/Loan_prediction/train.csv",header=None)
Data_csv1.head()


# In[10]:

#Suppose if you want to assign new column name as per your wish:
#Assign all col name in a variable
col_name = ["Loan_ID","Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area","Loan_Status"]
Data_csv2 = pd.read_csv("E:/DataAnalyticsCourse/GittHub/Datasets/Loan_prediction/train.csv",names =col_name)
Data_csv2.head()


# In[11]:

#now you can delete the first row of the table
Data_csv2 = Data_csv2.drop(Data_csv2.index[0])
Data_csv2.head()


# In[12]:

#Use first column as Index
Data_csv3 = pd.read_csv("E:/DataAnalyticsCourse/GittHub/Datasets/Loan_prediction/train.csv",index_col="Loan_ID")
Data_csv3.head()


# # Read a Text File

# In[17]:

columns =['State','Account_Len','Area','Ph_No.','Int_Plan','Vmail_Plan','messgs',
            'tot_day_mins','tot_day_calls','tot_day_chrgs','tot_evening_mins',
            'tot_evening_calls','tot_evening_chrgs','tot_ngt_mins','tot_ngt_calls',
            'tot_ngt_chrgs','tot_int_mins','tot_int_calls','tot_int_chrgs',
'cust_calls_made','churn_status']

Data_txt = pd.read_csv("E:/DataAnalyticsCourse/GittHub/Datasets/Telecom_churn/telecom_churn_data.txt",names=columns)
Data_txt.head()


# In[19]:

Data_txt.describe()


# In[20]:

Data_txt.info()


# In[ ]:

#Observations:Int_Plan and Vmail_Plan


# In[28]:

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.figsize"]=(10,5)


# # Data Visualization

# In[29]:

sns.pairplot(Data_txt,hue="churn_status")


# In[42]:

sns.distplot(Data_txt.tot_day_chrgs)
plt.show()
sns.distplot(Data_txt.tot_evening_chrgs)
plt.show()
sns.distplot(Data_txt.tot_ngt_chrgs)
plt.show()
sns.distplot(Data_txt.tot_int_chrgs)
plt.show()


# In[45]:

#checking outliers
sns.boxplot(x="State",y="tot_int_chrgs",hue ="churn_status",data=Data_txt)


# # Data Cleaning

# In[ ]:

# Assuming Area and Phone no. is irrelavent as of now, so we can drop it
#Int_Plan,Vmail_Plan and churn_status is objective, we can change it to boolian


# In[55]:

Data_txt["Int_Plan"] = np.where(Data_txt.Int_Plan =="yes",1,0)
Data_txt["Vmail_Plan"] = np.where(Data_txt.Vmail_Plan =="yes",1,0)
Data_txt["churn_status"] = np.where(Data_txt.churn_status =="True",1,0)


# In[73]:

Data_txt.head()


# In[74]:

#Data_txt = Data_txt.drop(["Ph_No."], axis=1)
Data_txt = Data_txt.drop(["State"], axis=1)


# In[105]:

Data_txt.tail()


# In[118]:

from sklearn.model_selection import train_test_split
X=Data_txt.drop(["churn_status"], axis=1)
y=Data_txt.churn_status
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.8)


# In[117]:

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Logistic Regression
# An erreo i am geeting because in the data set one or more label value is only one class, for analysis we need atleast two class
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
logreg_pred = logreg.predict(X_test)
# In[126]:

# here the lable value is onlt one class ie Zero
y_train.head()


# In[128]:

# we have to we have to identify the unique values in that label and clean the data
Data_txt["Int_Plan"].unique()


# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.figsize"]=(10,5)

columns =['State','Account_Len','Area','Ph_No.','Int_Plan','Vmail_Plan','messgs',
            'tot_day_mins','tot_day_calls','tot_day_chrgs','tot_evening_mins',
            'tot_evening_calls','tot_evening_chrgs','tot_ngt_mins','tot_ngt_calls',
            'tot_ngt_chrgs','tot_int_mins','tot_int_calls','tot_int_chrgs',
'cust_calls_made','churn_status']

Data_txt2 = pd.read_csv("E:/DataAnalyticsCourse/GittHub/Datasets/Telecom_churn/telecom_churn_data.txt",names=columns)


# In[162]:

Data_txt2.Int_Plan.unique()


# In[171]:

#Here we found there is a space prifix in the value
# lets same try for the Vmail_Plan
print("Vmail_Plan:",Data_txt2.Vmail_Plan.unique())
print("churn_status: ",Data_txt2.churn_status.unique())
# we have found the same issue


# In[2]:

# short out the issue

Data_txt2["Int_Plan"] = np.where(Data_txt2.Int_Plan ==" yes",1,0)
Data_txt2["Vmail_Plan"] = np.where(Data_txt2.Vmail_Plan ==" yes",1,0)
Data_txt2["churn_status"] = np.where(Data_txt2.churn_status ==" True.",1,0)


# In[3]:

# Now check the unique values
print(Data_txt2.Int_Plan.unique())
print(Data_txt2.Vmail_Plan.unique())
print(Data_txt2.churn_status.unique())


# In[4]:

Data_txt2.head()


# In[9]:

plt.rcParams['figure.figsize']=(15,6)
sns.countplot(x="State", data=Data_txt2)


# In[12]:

sns.barplot(x="State",y="churn_status" ,data=Data_txt2)


# In[13]:

feature_cols =['Account_Len','Area','Int_Plan','Vmail_Plan','messgs',
            'tot_day_mins','tot_day_calls','tot_day_chrgs','tot_evening_mins',
            'tot_evening_calls','tot_evening_chrgs','tot_ngt_mins','tot_ngt_calls',
            'tot_ngt_chrgs','tot_int_mins','tot_int_calls','tot_int_chrgs',
'cust_calls_made']

X = Data_txt2[feature_cols]
y = Data_txt2.churn_status


# In[22]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(X,y,test_size=0.2,random_state=45)


# In[23]:

print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# # # Perform Logistic Regression

# In[26]:

# Perform Logistic Regression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(xtrain,ytrain)
logreg_pred = logreg.predict(xtest)


# In[28]:

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[30]:

# check the model Score and accuracy

Model_log = round(logreg.score(xtrain,ytrain)*100,2)
print("Model Score : " ,Model_log)
Acc_log = accuracy_score(logreg_pred,ytest)
print("Acc_Score : ", Acc_log)


# In[33]:

#confusion Matrix
from sklearn import metrics
cnf_metrix = (metrics.confusion_matrix(ytest,logreg_pred))
cmap = sns.cubehelix_palette(50, hue=0.5, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(cnf_metrix,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[34]:

# Cross Validation

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy') #fitting logistic regression to whole data with 5 fold
print("Log:",scores)
print("Log:" ,scores.mean())


# # Perform Decision Tree

# In[44]:

from sklearn import tree
tree = tree.DecisionTreeClassifier(criterion="entropy",max_depth=3)

tree.fit(xtrain,ytrain)
tree_Pred = tree.predict(xtest).astype(int)



Model_tree = round(tree.score(xtrain,ytrain)*100,2)
print("Model Score : " ,Model_tree)
Acc_tree = accuracy_score(tree_Pred,ytest,normalize=True)
print("Acc_Score : ", Acc_tree)

#Confusion Metrix 

from sklearn import metrics
cnf_metrix = (metrics.confusion_matrix(ytest,tree_Pred))
cmap = sns.cubehelix_palette(50, hue=0.5, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(cnf_metrix,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Cross-Validation
scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy') #fitting Decision Tree to whole data with 5 fold
print("Log:",scores)
print("Log:" ,scores.mean())


# # Perform Random Forest Tree

# In[47]:

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth = 3, min_samples_split=2, n_estimators = 200, random_state = 1)

forest.fit(xtrain,ytrain)
forest_Pred = forest.predict(xtest).astype(int)

Model_forest = round(forest.score(xtrain,ytrain)*100,2)
print("Model Score : " ,Model_forest)
Acc_forest = metrics.accuracy_score(forest_Pred,ytest,normalize=True)
print("Acc_Score : ", Acc_forest)

#Confusion Metrix 

from sklearn import metrics
cnf_metrix = (metrics.confusion_matrix(ytest,forest_Pred))
cmap = sns.cubehelix_palette(50, hue=0.5, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(cnf_metrix,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')


#Cross Validation
# Cross-Validation
scores = cross_val_score(forest, X, y, cv=5, scoring='accuracy') #fitting Decision Tree to whole data with 5 fold
print("Log:",scores)
print("Log:" ,scores.mean())


# # PCA

# In[105]:

from sklearn.decomposition import PCA 


# In[ ]:



