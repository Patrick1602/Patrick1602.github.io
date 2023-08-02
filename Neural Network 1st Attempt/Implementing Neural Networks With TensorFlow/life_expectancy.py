# IMPORT #
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# DATA LOADING #
dataset = pd.read_csv("life_expectancy.csv")
#print(dataset.head())
#print(dataset.describe())
dataset = dataset.drop(["Country"], axis = 1)
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:-1]

# DATA PREPROCESSING # 
features = pd.get_dummies(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 23)
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

# BUILDING THE MODEL #
my_model = Sequential()
input = InputLayer(input_shape = (features.shape[1], ))
my_model.add(input)
my_model.add(Dense(64, activation = "relu"))
my_model.add(Dense(1))
print(my_model.summary())

# INITIALIZING THE OPTIMIZER AND COMPILING THE MODEL #
opt = Adam(learning_rate = 0.01)
my_model.compile(loss = "mse", metrics = ["mae"], optimizer = opt)

# FIT AND EVALUATE THE MODEL #
my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 1)
res_mse, res_mae = my_model.evaluate(features_test_scaled, lables_test, verbose = 0)

print("Final Loss (RMSE):", res_mse)
print("Final Metric (MAE):", res_mae)








