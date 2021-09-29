

# train connection.py

'''
-----------------------------------------------------
This model calculate the label of connection.
Multiple Class Calibration Model --> Using MLP Model

Object is 'Inlet Temperature at the next time'
Input Variables are Flowrate, Outlet Temperature at the present time
but, if you use Inlet Temperautre, that means operation in current time, too.
-----------------------------------------------------
'''

'==============================================================================================================================='

import sys
import numpy as np
from pandas import DataFrame, Series
import pandas as pd 
import matplotlib.pyplot as plt
import csv

import warnings
warnings.filterwarnings('ignore')

import os
import random as rn
import tensorflow as tf
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.utils.np_utils import to_categorical

'--------------------------------------------------'

# GPU Computing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Set to -1 if CPU should be used CPU = -1 , GPU = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
cpus = tf.config.experimental.list_physical_devices('CPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
elif cpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        logical_cpus= tf.config.experimental.list_logical_devices('CPU')
        print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


'--------------------------------------------------'

# Read Data
Train = pd.read_csv('C:\\System_Air\\Input\\Train\\TRNSYS_2016.csv')
Predict = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
Encoded_Train = pd.read_csv('C:\\System_Air\\Input\\Train\\Admatrix_Train.csv')
Encoded_Predict = pd.read_csv('C:\\System_Air\\Input\\Test\\Admatrix_Predict.csv')

'--------------------------------------------------'
    

# Input Variable from Train Data
feature_names = ['Week', 'Timepermin', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Secondary_Load']
                # Flowrate input variables = 7
                # Outlet Temperature input variables = 8 + Return Header Temperature     
                # Inlet Temperauture input variables = 7
                # Out of Flowrate and Temperature = 5
x_data = Train[feature_names]

# y_train data by encoded 1x49 vector
y_train = Train['Label']
Encoded_Train = to_categorical(y_train)

# MLP Model Classification
# Training Step
seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def Classifiation_MLP():
    model = tf.keras.models.Sequential()  
    model.add(Dense(100, input_dim = 5, activation = 'sigmoid'))
    model.add(Dense(60, activation = 'sigmoid'))
    model.add(Dense(31, activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

es = EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.001, verbose=1)
mc = ModelCheckpoint('C:\\System_Air\\best_model\\connection\\Case01.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only=True)

MLP_Model = Classifiation_MLP()
history = MLP_Model.fit(x_data, Encoded_Train, epochs=1000, batch_size=50000, validation_split=0.2, verbose=1, callbacks=[es, mc])


# Graph
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('Case01.png')


# Prediction data (NEW Dataset)
x_predict = Predict[feature_names]
x_predict = np.array(x_predict)

y_predict = Predict['Label']
Encoded_Predict = to_categorical(y_predict)


# Save the data
Result_Predict = MLP_Model.predict(x_predict)
Result_Predict = Result_Predict.reshape(122400, 31)
Result_Predict = np.rint(Result_Predict)
Prediction_Label = np.argmax(Result_Predict, axis=1)


# Evaluate
loss, acc = MLP_Model.evaluate(x_predict, Result_Predict, batch_size=128)

# Erorr
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

def MAE(x, y):
    return mean_absolute_error(x, y)

Result_R2 = r2_score(y_predict, Prediction_Label)
Result_RMSE = RMSE(y_predict, Prediction_Label)
Result_MAE = MAE(y_predict, Prediction_Label)
Result_MBE = np.mean(y_predict-Prediction_Label)

Result_Estimation = np.array([['Case01', loss, acc, Result_RMSE, Result_R2]])
Result_Predict = pd.DataFrame(Result_Predict)
Encoded_Train = pd.DataFrame(Encoded_Train)
Encoded_Predict = pd.DataFrame(Encoded_Predict)
Prediction = pd.DataFrame(Prediction_Label)

Result_Predict.to_csv('Result_Predict_Case01.csv', index=False)
Encoded_Train.to_csv('Encoded_Train_Case01.csv', index=False)
Encoded_Predict.to_csv('Encoded_Predict_Case01.csv', index=False)
Prediction.to_csv('Prediction_Label_Case01.csv', index=False)

with open('Result_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Case01', loss, acc, Result_R2, Result_RMSE, Result_MAE, Result_MBE])


'====================================================================================='

