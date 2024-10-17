# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.     
2.Find the null values and count them.   
3.Count number of left values.     
4.From sklearn import LabelEncoder to convert string values to numerical values.          
5.From sklearn.model_selection import train_test_split.          
6.Assign the train dataset and test dataset.        
7.From sklearn.tree import DecisionTreeClassifier.       
8.Use criteria as entropy.        
9.From sklearn import metrics.      
10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ezhil Nevedha.K
RegisterNumber:21222323055
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data
![369763367-36a090d7-6842-404b-abc5-a92e40d6a065](https://github.com/user-attachments/assets/cae7cc3b-247c-44a2-aa97-ab9df1f428bf)

### Accuracy
![369763543-5deef1d4-9426-4bdc-bc78-d4715385e90f](https://github.com/user-attachments/assets/e00746b8-8346-408e-b45e-04345ebbbfe3)

### Prediction
![369763662-7196c860-d971-473d-bc44-1eb5c561a53f](https://github.com/user-attachments/assets/d825c07a-5739-48be-a306-39d97438cd0b)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
