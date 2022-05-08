import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
calories = pd.read_csv('16_Calories_Burnt_Prediction\calories.csv ')
calories.head()
exercise_data = pd.read_csv('16_Calories_Burnt_Prediction\exercise.csv ')
exercise_data.head()
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
calories_data.head()
calories_data.shape
calories_data.info()
calories_data.isnull().sum()
calories_data.describe()
sns.set()
sns.countplot(calories_data['Gender'])
sns.distplot(calories_data['Age'])
sns.distplot(calories_data['Height'])
sns.distplot(calories_data['Weight'])
correlation = calories_data.corr()
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
calories_data.head()
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = XGBRegressor()
model.fit(X_train, Y_train)
test_data_prediction = model.predict(X_test)
print(test_data_prediction)
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)
# input_data = (0,68,190.0,94.0,29.0,105.0,40.8)
input_data = (1,20,166.0,60.0,14.0,94.0,40.3)
# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(f"The Calories burnt are {float(prediction)}")