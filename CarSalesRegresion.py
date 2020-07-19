# Section 0 : Import Library
# -----------------------------------------------------

print('Mengimport Library...\n')
# Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Split Data
from sklearn.model_selection import train_test_split

# Machine Learning Model
from sklearn.linear_model import Ridge

# Evaluasi Model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# Section 1 : Persiapan Data
# -----------------------------------------------------

# Kelola Data
dataframe = pd.read_csv('data/car-sales-extended.csv')
print('Original Dataframe')
print(dataframe.head())

# Split data dan harga
data = dataframe.drop('Price',axis=1)
price = dataframe['Price']

# Encoding Data
categorical_features = ['Make','Colour','Doors']
encoder = OneHotEncoder()
transformer = ColumnTransformer([('encoder',encoder,categorical_features)],remainder='passthrough')
transformed_data = transformer.fit_transform(data)
encoded_data = pd.get_dummies(data)
print('\nDataframe dengan Encoding')
print(encoded_data.head())

# Split data train dan test
x_train,x_test,y_train,y_test = train_test_split(encoded_data,price,test_size=0.2)


# Section 2 : Membangun Machine Learning Model
# -----------------------------------------------------

# Build Machine Learning Model
model = Ridge()
model.fit(x_train,y_train)

# Membuat Prediksi
prediction = model.predict(x_test)

# Evaluasi Model
def evaluateModel(y_true,y_pred):
    print('Your Evaluation Metrics')
    print(f'Mean Absolute Error (MAE) : {mean_absolute_error(y_true,y_pred)}')
    print(f'Root Mean Squared Error (RMSE) : {mean_squared_error(y_true,y_pred)}')
    print(f'R-Squared Score (r2) : {r2_score(y_true,y_pred)}')

print('')
evaluateModel(y_test,prediction)


# Section 3 : Visualisasi Data
# -----------------------------------------------------

def visualizeModel(y_true,y_pred):
    plt.figure(figsize=(20,5))
    plt.plot(range(len(y_pred)),prediction,label='Prediction')
    plt.plot(range(len(y_test)),y_test,label='Actual')
    mean_pre = np.mean(y_pred)
    plt.plot(range(len(y_pred)),[mean_pre for i in range(len(y_pred))],label='Mean of Prediction')
    plt.legend()
    plt.show()

visualizeModel(y_test,prediction)