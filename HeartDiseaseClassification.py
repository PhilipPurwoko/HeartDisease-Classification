# Section 0 : Import Library
# -----------------------------------------------------

print('Mengimport Library...\n')
# Data Analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Split Data
from sklearn.model_selection import train_test_split

# Machine Learning Model
from sklearn.ensemble import RandomForestClassifier


# Section 1 : Persiapan Data
# -----------------------------------------------------

# Kelola Data
dataframe = pd.read_csv('data/heart_dataset.csv')
print('Dataframe')
print(dataframe.head())

# Split Data
data = dataframe.drop('target',axis=1)
label = dataframe.target
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size=0.2)

# Data Overview
for x in ['x_train','x_test','y_train','y_test']:
    print(f'Size of {x} : {len(eval(x))}')


# Section 2 : Membangun Machine Learning Model
# -----------------------------------------------------

# Membangun Model
model = RandomForestClassifier(n_estimators=20)
model.fit(x_train,y_train)

# Membuat Prediksi
prediction = model.predict(x_test)

# Evaluasi Model
print('Akurasi Model : {model.score(x_test,y_test)}')