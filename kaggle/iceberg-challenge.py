from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import pandas as pd


# load model from weight_path
def get_model(weight_path):
    model = Sequential()
    #Conv Layer 1
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Dropout(0.2))
    #Conv Layer 2
    model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Dropout(0.2))
    #Conv Layer 3
    model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Dropout(0.2))
    #Conv Layer 4
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Dropout(0.2))
    # Flatten
    model.add(Flatten())
    # Dense Layer 1
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # Dense Layer 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    #Sigmoid Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.load_weights(weight_path)
    return model


# process data
def process_data(data_path):
    # load data
    df = pd.read_json(data_path)
    # concentrate band 1 and band 2 into a numpy array
    X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    X = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
    return X


# predict 
def predict():
    weight_path = "C:\\e\\dev\\iceberg-classifier-challenge\\project\\saved_models\\weights.best.improved.hdf5"
    data_path = "C:\\e\\dev\\capstones\\data\\test\\test.json"
    model = get_model(weight_path)
    X_test = process_data(data_path)
    predicted_test=model.predict_proba(X_test)
    return predicted_test


# load data
data_path = "C:\\e\\dev\\capstones\\data\\test\\test.json"
df = pd.read_json(data_path)
prediction = predict()

# write prediction to file
filename = "submission.txt"
file = open(filename, "a")
for index, row in df.iterrows():
   file.writelines(str(row['id'])+","+str(round(prediction[index][0],2))+"\n")
