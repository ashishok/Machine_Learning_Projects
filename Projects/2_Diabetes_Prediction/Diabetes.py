import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('2_Diabetes_Prediction\diabetes.csv')
# printing the first five rows of dataset
diabetes_dataset.head()
# number of rows and coulmns in this datset
diabetes_dataset.shape
# getting the statistical measures of the data
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
# Separating the and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
scalar = StandardScaler()
X = X.values
scalar.fit(X)
standarized_data = scalar.transform(X)
print(standarized_data)
X = standarized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
# trianing the support vector machine Classifier
classifier.fit(X_train, Y_train)
#Accuracy score on the training data 

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('accuracy score for the training data : ',training_data_accuracy)
#Accuracy score on the test data 

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('accuracy score on the test data is : ', test_data_accuracy)
input_data = (4,110,92,0,0,37.6,0.191,30)

# changing the input_data as numpy array

input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standarize the input data 
std_data = scalar.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)
if prediction[0] == 0:
    print("The Person is not Dibetic")
else:
    print("The Person is Dibetic")