import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv('8_Gold_Price_Prediction\gld_price_data.csv')
gold_data.head()
gold_data.tail()
gold_data.shape
gold_data.info()
gold_data.isnull().sum()
gold_data.describe()
correlation = gold_data.corr()
# constructing a heatmap to understand the correlatiom
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')
print(correlation['GLD'])
sns.histplot(gold_data['GLD'],color='green',kde=True, stat="density", linewidth=0)
X = gold_data.drop(['Date','GLD'],axis=1)
X = X.values
Y = gold_data['GLD']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)
# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
input_data = (1447.160034,78.370003,15.2850,1.474491)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = regressor.predict(input_data_reshaped)
print("The Price of the Gold is --> ", +float(prediction)) 