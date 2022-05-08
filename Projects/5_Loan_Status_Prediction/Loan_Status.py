import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
loan_dataset = pd.read_csv('5_Loan_Status_Prediction\Loan.csv')
type(loan_dataset)
loan_dataset.head()
loan_dataset.shape  
loan_dataset.describe()
loan_dataset.isnull().sum()
loan_dataset = loan_dataset.dropna()
loan_dataset.isnull().sum()
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
loan_dataset.head()
loan_dataset['Dependents'].value_counts()
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
loan_dataset['Dependents'].value_counts()
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
loan_dataset.replace({'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0}, 'Self_Employed': {'No': 0, 'Yes': 1},
                      'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}, 'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
loan_dataset.head(20)
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']
X = X.values
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuray)
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data : ', test_data_accuray)
input_data = (1, 1, 0,	0,	0,	2600,	1911.0,	116.0,	360.0,	0.0,	1)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = classifier.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
    print('Loan Rejected')
else:
    print('Loan Accepted')
