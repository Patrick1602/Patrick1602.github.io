import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score


# DATA LOADING #
dataset = pd.read_csv("admissions_data.csv")
dataset = dataset.drop(columns = ["Serial No."])
#print(dataset.head())
#print(dataset.columns)
#print(dataset.describe())
#print(dataset.dtypes)
labels = dataset.iloc[:,-1]
features = dataset.iloc[:,0:-1]

# DATA PREPROCESSING #
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# DESIGNING THE MODEL #
my_model = Sequential()
input = InputLayer(input_shape = (features.shape[1], )) # CREATION OF INPUT LAYER #
my_model.add(input) # ADDING INPUT LAYER #
my_model.add(Dense(256, activation = "relu")) # HIDDEN LAYER WITH 256 NEURONS #
my_model.add(Dense(1)) # ADDING OUTPUT LAYER #
print(my_model.summary())

# INITIALIZING THE OPTIMIZER AND COMPILING THE MODEL #
opt = Adam(learning_rate = 0.01) # OPTIMIZER: ADAM #
my_model.compile(loss = "mse", metrics = ["mae"], optimizer = opt)
early_stop = EarlyStopping(monitor = 'val_loss',mode='min',patience = 40)

# FIT AND EVALUATE THE MODEL #
history = my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 3, verbose = 1, validation_split = 0.2, callbacks = [early_stop])
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)
print(res_mse, res_mae)

predicted_values = my_model.predict(features_test_scaled) 
print(r2_score(labels_test, predicted_values)) 

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
# PLOT LOSS AND VAL_LOSS OVER EACH EPOCH #
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# KEEP PLOTS FROM OVERLAPPING # 
fig.tight_layout()
fig.savefig('static/images/my_plots.png')

