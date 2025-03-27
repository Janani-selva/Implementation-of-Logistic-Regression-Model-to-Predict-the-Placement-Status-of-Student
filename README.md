## Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.
```

## Program:
```
Developed by: Janani s
RegisterNumber:  212224230103
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('/content/Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```
## Output:
## Placement Data
![image](https://github.com/user-attachments/assets/24e8f9cb-1ef8-4a18-b48e-7c3fe78444f0)

## Checking the null() function
![image](https://github.com/user-attachments/assets/fa68579c-024f-499e-90ed-2b88e2e2c2b1)

## Print Value
![image](https://github.com/user-attachments/assets/fca46381-8c90-4251-93da-d7eda4442e06)

## Y-Prediction Value
![image](https://github.com/user-attachments/assets/17e87006-8f1d-4537-8e5c-ab6fe84021de)

## Confusion array
![image](https://github.com/user-attachments/assets/9fd0e251-840e-4562-94a3-3cadda38c3fe)

## Classification report
![image](https://github.com/user-attachments/assets/df62e8bc-70f0-4263-adb1-1c01f657517b)

## Prediction of LR
![image](https://github.com/user-attachments/assets/6ae6a337-1bba-415b-9062-a865eadddf70)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
