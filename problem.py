import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

## Loading the data ##
data = pd.read_csv("heart_failure.csv")
print(data.info())
print(Counter(data["death_event"]))
 
y = data["death_event"]
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']] #features

## Data Preprocessing ##
x = pd.get_dummies(x)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)

## Prepare labels for classification ##
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.fit_transform(Y_test.astype(str))
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

## Design the model ##
model = Sequential()
input = model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dense(256, activation = "relu"))
model.add(Dense(16, activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

## Train and evaluate the model ##
#model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose=1)
#loss, acc = model.evaluate(X_test, Y_test, verbose = 0)


