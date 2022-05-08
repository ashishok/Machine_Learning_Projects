import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
titanic_data = pd.read_csv('15_Titanic_Survival_Prediction/train.csv')
titanic_data.head()
titanic_data.shape
titanic_data.info()
titanic_data.isnull().sum()
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])
titanic_data['Embarked'].fillna(
    titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.isnull().sum()
titanic_data.describe()
titanic_data['Survived'].value_counts()
sns.set()
sns.countplot('Survived', data=titanic_data)
titanic_data['Sex'].value_counts()
sns.countplot('Sex', data=titanic_data)
sns.countplot('Sex', hue='Survived', data=titanic_data)
sns.countplot('Pclass', data=titanic_data)
sns.countplot('Pclass', hue='Survived', data=titanic_data)
titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {
                     'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
titanic_data.head()
X = titanic_data.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
X = X.values
Y = titanic_data['Survived']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
# input_data = (3,0,22,1,0,7.25,0)
input_data = (1, 1, 38, 1, 0, 71.2833, 1)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
    print("Survived")

else:
    print("Not Survived")
