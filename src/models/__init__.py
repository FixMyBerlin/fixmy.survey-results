import numpy as np
from tensorflow.keras.layers import Dense, Softmax, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from src.models.predict_model import ordinal_predict
import numpy as np

def nn_ordinal_labels(y):
    size = np.max(y)
    output = np.zeros((len(y), int(size)))
    for i, value in enumerate(y):
        for j in range(int(value)):
            output[i,j] = 1
    return output

#def score_model(estimator, x, y): # TODO made negative, 

#    y_pred = estimator.predict(x.to_numpy())
def score_model(y_test, y_pred, **kwargs):
    y_pred = ordinal_predict(y_pred)
    y_test = ordinal_predict(y_test)
    return -mean_absolute_error(y_test, y_pred)  
