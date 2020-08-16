# Section 0 : Import Library
# -----------------------------------------------------

# Data Analysis
import numpy as np
import pandas as pd

# Encoding
from sklearn.preprocessing import LabelEncoder

# Split Data
from sklearn.model_selection import train_test_split

# Machine Learning Model
from sklearn.svm import SVC

# Evalutation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix


# Section 1 : Persiapan Data
# -----------------------------------------------------

# Import Dataframe
dataframe = pd.read_csv('data/mushroom.csv')
print('Dataframe dalam kondisi awal')
print(dataframe.head())

# Encoding dataframe
label_encoder = LabelEncoder()
for i in dataframe:
    dataframe[i] = label_encoder.fit_transform(dataframe[i])

print('Dataframe setelah encoding data')
print(dataframe.head())

# Split Data
data = dataframe.drop('class',axis=1)
label = dataframe['class']
x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.2)


# Section 2 : Membangun Machine Learning Model
# -----------------------------------------------------

# Melatih Model
model = SVC(gamma='auto')
model.fit(x_train,y_train)

# Membuat Prediksi
prediction = model.predict(x_test)

# Evaluasi Model
print(f'Akurasi Model : {accuracy_score(y_test,prediction)}')