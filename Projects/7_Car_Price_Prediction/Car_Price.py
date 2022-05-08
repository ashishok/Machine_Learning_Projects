import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
car_dataset = pd.read_csv('7_Car_Price_Prediction\car_data.csv')
car_dataset.head()
car_dataset.shape
car_dataset.info()
car_dataset.isnull().sum()
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())
car_dataset.replace(
    {'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace(
    {'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace(
    {'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
car_dataset.head()
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
X = X.values
Y = car_dataset['Selling_Price']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=2)
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)
training_data_prediction = lass_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
test_data_prediction = lass_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)
plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
# input_data = (2014 ,5.59,27000,0,0,0,0)
input_data = (2011, 4.15, 5200, 0, 0, 0, 0)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = lin_reg_model.predict(input_data_reshaped)
print(prediction)

# input_data = (2014 ,5.59,27000,0,0,0,0)
input_data = (2011, 4.15, 5200, 0, 0, 0, 0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = lass_reg_model.predict(input_data_reshaped)
print(prediction)
