import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.datasets 
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor
from sklearn import metrics
house_price_dataset = sklearn.datasets.load_boston()
print(house_price_dataset)
#loading the datset to a pandas Dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)
# Print first five rows of our DataFrame
house_price_dataframe.head()
# add the Target column to the DataFrame 
house_price_dataframe["Price"] = house_price_dataset.target
house_price_dataframe.head()
#checking the number of rows and columns in the dataFrame
house_price_dataframe.shape
#check for missing values 
house_price_dataframe.isnull().sum()
# Statistical measure of the dataset 
house_price_dataframe.describe()
correlation = house_price_dataframe.corr()
# constructinf the heapmapto understand the correlation
plt.figure(figsize=(12,12))
sns.heatmap(correlation,cbar = True, square = True, fmt = '.1f', annot= True, annot_kws = {'size':10}, cmap='Blues')
X = house_price_dataframe.drop('Price', axis=1)
Y = house_price_dataframe['Price']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape, X_test.shape)
# loading the Model
model = XGBRegressor()
# training the model with X_train
model.fit(X_train,Y_train)
# accuracy for prediction on the training data 
training_data_prediction = model.predict(X_train)
print(training_data_prediction)
# R square error
score_1 = metrics.r2_score(Y_train,training_data_prediction)

# mean absolute error
score_2 = metrics.mean_absolute_error(Y_train,training_data_prediction)

print('R square error :', score_1)
print('Mean Absolute Error :', score_2)
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Prices")
plt.show()
# accuracy for prediction on the training data 
test_data_prediction = model.predict(X_test)
# R square error
score_1 = metrics.r2_score(Y_test,test_data_prediction)

# mean absolute error
score_2 = metrics.mean_absolute_error(Y_test,test_data_prediction)

print('R square error :', score_1)
print('Mean Absolute Error :', score_2)
input_data = (0.02729,	0.0,	7.07,	0.0,	0.469, 7.185,	61.1,
              4.9671,	2.0, 242.0,	17.8, 392.83,	4.03)

# changing the input_data as numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standarize the input data

prediction = model.predict(input_data_reshaped)
print(prediction)