# EXNO:4-DS

### NAME : THARUN SRIDHAR 
### REGISTER NO : 212223230230

# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

**DEVELOPED BY : THARUN SRIDHAR**
**REGISTER NO : 212223230230**

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/133cedbe-5e31-4b2b-895c-025cb31fdf04)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/2a558f2c-d1c2-4e1f-bd45-1bdc9846e475)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/9a6cb486-4179-4e64-9aef-2ae355d0e77c)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/dc38f2c6-48fd-4d45-a525-4a653e9dc612)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/26c4901b-a19b-4c5f-b43d-413908978102)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/7e21aa53-6f48-47d7-b7e8-2385a3bc3f35)

```
data2
```
![image](https://github.com/user-attachments/assets/a135cb99-2eeb-4ee9-87c1-3068807d7b18)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/92d4fb06-f5da-41ed-a158-8873843a70b1)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/9780c644-8067-40c7-82b8-56d4003d4e93)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/aa2c6cc7-15a4-4dd6-8161-4eb3d60763bb)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/5db5e758-3628-439b-a71d-49163e856b5d)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/85a95845-f5af-4ff5-bb7a-6c2eecbd03e0)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/333b2fd3-ec9b-4965-9730-8e307df5dfaf)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/9e81818c-0ec3-4000-bf4c-0315e03e67ba)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/be3d1e0c-971a-4c96-b86b-f9f22fabb78a)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/17f771ff-dc24-4ec1-9adb-624eef7954fc)

```
data.shape
```
![image](https://github.com/user-attachments/assets/6f551615-fbc7-4de6-858f-5573b29f6849)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/0f541beb-8c94-46ca-b693-0ba2ab96407c)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/2d954a7f-69e7-4ccf-be31-6108aedddab4)

```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/72810461-9734-4e9a-9189-47dd7f4e4253)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/cc8974b4-72af-4c85-88cc-3c82d0f33d0b)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/4b60729b-5dd9-4c57-a90f-ab20a0f73bab)


# RESULT:
       # INCLUDE YOUR RESULT HERE
