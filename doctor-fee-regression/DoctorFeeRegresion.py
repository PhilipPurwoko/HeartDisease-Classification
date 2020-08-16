# Section 0 : Import Library
# -----------------------------------------------------

print('Mengimport Library...\n')
# Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Encoding Data
from sklearn.preprocessing import LabelEncoder

# Split Data
from sklearn.model_selection import train_test_split

# Machine Learning Model
from sklearn.linear_model import LinearRegression

# Evaluasi Model
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# Section 1 : Persiapan Data
# -----------------------------------------------------

# Import Dataframe
dataframe = pd.read_excel('data/doctor-fee.xlsx')
dataframe = dataframe.drop('Miscellaneous_Info',axis=1)
print('Dataframe pada kondisi awal')
print(dataframe.head())

# Overview Dataframe
def describe(dataframe):
    for col in list(dataframe):
        print(f'{col} : {len(dataframe[col].unique())} Unique Values, {len([i for i in dataframe[col] if type(i) == float])} NaN')

describe(dataframe)

# Mengisi Data Kosong Dengan Nilai Rata-Rata
for_mean = []
for rate in dataframe['Rating']:
    try:
        for_mean.append(int(rate[:-1]))
    except TypeError:
        pass

rating = []
for rate in dataframe['Rating']:
    try:
        rating.append(float(rate[:-1]))
    except TypeError:
        rating.append(np.mean(for_mean))

dataframe = dataframe.drop('Rating',axis=1)
dataframe['Rating'] = rating
print('Dataframe Setelah Koreksi Data')
print(dataframe.head())

# Konversi Data Menjadi Numerik
experience = []
for exp in dataframe['Experience']:
    exp = int(exp[:-17])
    experience.append(exp)

dataframe = dataframe.drop('Experience',axis=1)
dataframe['Experience'] = experience
print('Dataframe setelah konversi menjadi data numerik')
print(dataframe.head())

# Menghapus Data Dengan Nilai Kosong
dataframe.dropna(subset = ['Place'], inplace=True)
print('Dataframe Setelah Menghapus Data Kosong')
print(dataframe.head())

# Encoding Data
encoder = LabelEncoder()
for col in ['Qualification','Place','Profile']:
    dataframe[col] = encoder.fit_transform(dataframe[col])
print('Data Setelah Encoding')
describe(dataframe)
print(dataframe.head())

# Split Data
data = dataframe.drop('Fees',axis=1)
fees = dataframe['Fees']
x_train,x_test,y_train,y_test = train_test_split(data,fees,test_size=0.2)


# Section 2 : Membangun Machine Learning Model
# -----------------------------------------------------

# Membuat Model
model = LinearRegression()
model.fit(x_train,y_train)

# Membuat Prediksi
prediction = model.predict(x_test)


# Section 3 : Evaluasi Machine Learning Model
# -----------------------------------------------------

def evaluateModel(y_true,y_pred):
    print('Your Evaluation Metrics')
    print(f'Mean Absolute Error (MAE) : {mean_absolute_error(y_true,y_pred)}')
    print(f'Root Mean Squared Error (RMSE) : {mean_squared_error(y_true,y_pred)}')
    print(f'R-Squared Score (r2) : {r2_score(y_true,y_pred)}')

evaluateModel(y_test,prediction)


# Section 4 : Visualisasi Model
# -----------------------------------------------------

def visualize(y_true,y_pred,title=None):
    plt.figure(figsize=(20,5))
    plt.title(title)
    plt.plot(range(len(y_true)),y_true,label='True Value')
    plt.plot(range(len(y_pred)),y_pred,label='Prediction')
    plt.legend()
    plt.show()

visualize(y_test,prediction)