from keras.models import Sequential
import numpy as np
import pandas as pd


# load model from weight_path
def get_model(weight_path):
    model = Sequential()
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

print(predict())