import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('10_Credit_Card_Fruad_Detection\credit_card_data.csv')
credit_card_data.head()
credit_card_data.tail()
credit_card_data.info()
credit_card_data.isnull().sum()
credit_card_data['Class'].value_counts()
# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)
legit.Amount.describe()
fraud.Amount.describe()
credit_card_data.groupby('Class').mean()
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.head()
new_dataset.tail()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()
X = new_dataset.drop(columns='Class', axis=1)
X = X.values
Y = new_dataset['Class']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression()
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
input_data = (172792, -0.533412522, -0.189733337, 0.703337367, -0.50627124, -0.012545679, -0.649616686, 1.577006254, -0.414650408, 0.486179505, -0.915426649, -1.040458335, -0.031513054, -0.188092901, -
              0.08431647, 0.041333455, -0.302620086, -0.660376645, 0.167429934, -0.256116871, 0.382948105, 0.261057331, 0.643078438, 0.376777014, 0.008797379, -0.473648704, -0.818267121, -0.002415309, 0.013648914, 217)
# input_data = (169966,-3.113831607,0.585864172,-5.399730211,1.817092473,-0.840618466,-2.943547791,-2.20800192,1.058732677,-1.63233335,-5.245983838,1.933519537,-5.030464797,-1.127454575,-6.416627976,0.141237234,-2.549498236,-4.614717069,-1.478137941,-0.035480366,0.30627074,0.583275999,-0.269208638,-0.456107773,-0.18365913,-0.328167759,0.60611581,0.88487554,-0.253700319,245)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0] == 0):
    print("Credit card is Valid.")
else:
    print("Fraudential Credit Card.")
