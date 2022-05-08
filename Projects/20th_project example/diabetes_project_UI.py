import numpy as np
import pickle

loaded_model = pickle.load(
    open('E:/Projects/ML/20_Deploy_ML_Model_using_streamlit/trained_model.sav', 'rb'))
# input_data = (2, 197, 70, 45, 543, 30.5, 0.158, 53)
input_data = (1,89,66,23,94,28.1,0.167,21)

# changing the input_data as numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The Person is not Dibetic")
else:
    print("The Person is Dibetic")
