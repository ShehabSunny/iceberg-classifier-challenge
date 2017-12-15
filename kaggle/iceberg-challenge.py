import keras


# load model from weight_path
def get_model(weight_path):
    model = Sequential()
    model.load_weights(weight_path)
    return model


