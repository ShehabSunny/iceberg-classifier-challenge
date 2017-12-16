from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
import numpy as np
import pandas as pd


# load model from weight_path
def get_model(weight_path_):
    model = Sequential()
    # Conv Layer 1
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Dropout(0.2))
    # Conv Layer 2
    model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Dropout(0.2))
    # Conv Layer 3
    model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Dropout(0.2))
    # Conv Layer 4
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
    # Sigmoid Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.load_weights(weight_path_)
    return model


# process data
def process_data(data_set_path):
    # load data
    df_data = pd.read_json(data_set_path)
    # concentrate band 1 and band 2 into a numpy array
    x_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_data["band_1"]])
    x_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_data["band_2"]])
    x = np.concatenate([x_band_1[:, :, :, np.newaxis],
                        x_band_2[:, :, :, np.newaxis],
                        ((x_band_1+x_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
    return x


# predict 
def predict():
    model = get_model(weight_path)
    x_test = process_data(data_path)
    predicted_test = model.predict_proba(x_test)
    return predicted_test


if __name__ == "__main__":
    # data paths
    data_path = "C:\\e\\dev\\capstones\\data\\test\\test.json"
    weight_path = "C:\\e\\dev\\iceberg-classifier-challenge\\project\\saved_models\\weights.best.improved.hdf5"
    submission_filename = "submission.txt"

    # load data
    df = pd.read_json(data_path)
    # predict
    prediction = predict()

    # write prediction to file
    file = open(submission_filename, "a")
    file.writelines("id,is_iceberg\n")
    for index, row in df.iterrows():
        file.writelines(str(row['id'])+","+str(round(prediction[index][0], 2))+"\n")
