

# train hex.py

'''
-----------------------------------------------------
hex's grey-box model: based on Counter Flow type
-----------------------------------------------------
'''

'==============================================================================================================================='

import sys
import numpy as np
from pandas import DataFrame, Series
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import csv

import warnings
warnings.filterwarnings('ignore')

import os
import random as rn
import tensorflow as tf
import keras
from tensorflow.keras import optimizers, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from physic_hex import *

'--------------------------------------------------'

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

# Read Data by pickle
with open('Train.bin', 'rb') as file:
   Train = pickle.load(file)

with open('Predict.bin', 'rb') as file:
   Predict = pickle.load(file)

'--------------------------------------------------'


SITemp_train = Train['HEX01_SITemp']
LITemp_train = Train['HEX01_LITemp']
SOTemp_train = Train['HEX01_SOTemp']
LOTemp_train = Train['HEX01_LOTemp']
SOFlow_train = Train['HEX01_SFlow']
LOFlow_train = Train['HEX01_LFlow']

SITemp_train = np.array(SITemp_train).astype('float16')
LITemp_train = np.array(LITemp_train).astype('float16')
SOTemp_train = np.array(SOTemp_train).astype('float16')
LOTemp_train = np.array(LOTemp_train).astype('float16')
SOFlow_train = np.array(SOFlow_train).astype('float16')
LOFlow_train = np.array(LOFlow_train).astype('float16')


# Input Variable from Predict Data
SITemp_predict = Predict['HEX01_SITemp']
LITemp_predict = Predict['HEX01_LITemp']
SOTemp_predict = Predict['HEX01_SOTemp']
LOTemp_predict = Predict['HEX01_LOTemp']
SOFlow_predict = Predict['HEX01_SFlow']
LOFlow_predict = Predict['HEX01_LFlow']

SITemp_predict = np.array(SITemp_predict).astype('float16')
LITemp_predict = np.array(LITemp_predict).astype('float16')
SOTemp_predict = np.array(SOTemp_predict).astype('float16')
LOTemp_predict = np.array(LOTemp_predict).astype('float16')
SOFlow_predict = np.array(SOFlow_predict).astype('float16')
LOFlow_predict = np.array(LOFlow_predict).astype('float16')

'''
KA01 = []
KA02 = []

for i, (a, b, c, d, e, f) in enumerate(zip(SITemp_train, LITemp_train, SOTemp_train, LOTemp_train, SOFlow_train, LOFlow_train)):
    HEX_KA01 = HEX_KA(a, b, c, d, e, f)
    KA01.append(HEX_KA01)

for i, (a, b, c, d, e, f) in enumerate(zip(SITemp_predict, LITemp_predict, SOTemp_predict, LOTemp_predict, SOFlow_predict, LOFlow_predict)):
    HEX_KA02 = HEX_KA(a, b, c, d, e, f)
    KA02.append(HEX_KA02)

Train['HEX_KA_python'] = KA01

submission = pd.read_csv('C:\\System_Air\\Input\\Train\\TRNSYS_2016.csv')
submission["HEX_KA"] = KA01
submission.to_csv('C:\\System_Air\\Input\\Train\\HEX_parameter.csv', index = False)

submission = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
submission["HEX_KA"] = KA02
submission.to_csv('C:\\System_Air\\Input\\Test\\HEX_parameter.csv', index = False)
'''

'--------------------------------------------------'

input_shape = 4
output_shape = 1

# KFold setting
kfold=5
num_val_samples = len(Train) // kfold

'-------------------------------------------------'

# X_train data
InputVariable = ['HEX01_SITemp', 'HEX01_LITemp', 'HEX01_SFlow', 'HEX01_LFlow'] 

x_train = Train[InputVariable]
x_train = np.array(x_train).astype('float16')

# y_train data --> HEX Parameters 'KA'
KA_Para = Train['HEX_KA']

##### Training Step #####
seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

all_cv_score_01=[]

for i in range(kfold):

    print('Fold calculating', i)

    val_data_01 = x_train[i*num_val_samples: (i+1)*num_val_samples]
    val_target_01 = KA_Para[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_01 = np.concatenate([x_train[:i*num_val_samples], x_train[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_01 = np.concatenate([KA_Para[:i*num_val_samples], KA_Para[(i+1)*num_val_samples:]], axis=0)
    
    def keras_model_01():
        model = tf.keras.models.Sequential()  
        model.add(Dense(200, input_dim = input_shape, kernel_initializer = 'random_uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

        model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam",  metrics=['accuracy'])
        return model

    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_HEX_KA.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    best_model_y01 = keras_model_01()
    best_model_y01.fit(partial_train_data_01, partial_train_targets_01, epochs=30, batch_size=9000)
    best_model_y01.save('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_HEX_KA.h5')
    val_mse_01, val_mae_01 = best_model_y01.evaluate(val_data_01, val_target_01, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_01.append(val_mae_01)


x_predict_01 = Predict[InputVariable]
x_predict_01 = np.array(x_predict_01)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_HEX_KA.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
y01_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_HEX_KA.h5')
history = y01_Model.fit(x_train, KA_Para, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])

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
plt.savefig('HEX_KA.png')

y01_Predict = y01_Model.predict(x_predict_01)
y01_Predict = y01_Predict.reshape(122400,)


# ================= Predict Step ===================== #

# Input Variable from Predict Data
SITemp_predict = Predict['HEX01_SITemp']
LITemp_predict = Predict['HEX01_LITemp']
SOTemp_predict = Predict['HEX01_SOTemp']
LOTemp_predict = Predict['HEX01_LOTemp']
SOFlow_predict = Predict['HEX01_SFlow']
LOFlow_predict = Predict['HEX01_LFlow']

SITemp_predict = np.array(SITemp_predict).astype('float16')
LITemp_predict = np.array(LITemp_predict).astype('float16')
SOTemp_predict = np.array(SOTemp_predict).astype('float16')
LOTemp_predict = np.array(LOTemp_predict).astype('float16')
SOFlow_predict = np.array(SOFlow_predict).astype('float16')
LOFlow_predict = np.array(LOFlow_predict).astype('float16')

# Input parameter to Reserve formula
# Source side Temperature
HEX_SOWT = []
for i, (a, b, c, d, e) in enumerate(zip(SITemp_predict, LITemp_predict, SOFlow_predict, LOFlow_predict, y01_Predict)):
    HEX_S_OT = HEX_SOT(a, b, c, d, e)
    HEX_SOWT.append(HEX_S_OT)

# Load side Temperature
HEX_LWOT = []
for i, (a, b, c, d, e) in enumerate(zip(SITemp_predict, LITemp_predict, SOFlow_predict, LOFlow_predict, y01_Predict)):
    HEX_L_OT = HEX_LOT(a, b, c, d, e)
    HEX_LWOT.append(HEX_L_OT)


# ================ Save the Result =================== #


submission = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
submission["HEX_KA_Predict"] = y01_Predict
submission["HEX_SOTemp_Predict"] = HEX_SOWT
submission["HEX_LOTemp_Predict"] = HEX_LWOT
submission.to_csv('HEX_prediction.csv', index = False)


predict_KA_raw = Predict['HEX_KA']

# Function
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

def MAE(x, y):
    return mean_absolute_error(x, y)

# Evaluate
loss, acc = y01_Model.evaluate(x_predict_01, y01_Predict, batch_size=128)
Result_KA_R2 = r2_score(predict_KA_raw, y01_Predict)
Result_KA_RMSE = RMSE(predict_KA_raw, y01_Predict)
Result_KA_MAE = MAE(predict_KA_raw, y01_Predict)
Result_KA_MBE = np.mean(predict_KA_raw-y01_Predict)

# Save the File
with open('Result_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['KA', loss, acc, Result_KA_R2, Result_KA_RMSE, Result_KA_MAE, Result_KA_MBE])


# Outlet Temperature
Result_SOWT_R2 = r2_score(SOTemp_predict, HEX_SOWT)
Result_SOWT_RMSE = RMSE(SOTemp_predict, HEX_SOWT)
Result_SOWT_MAE = MAE(SOTemp_predict, HEX_SOWT)
Result_SOWT_MBE = np.mean(SOTemp_predict-HEX_SOWT)

Result_LOWT_R2 = r2_score(LOTemp_predict, HEX_LWOT)
Result_LOWT_RMSE = RMSE(LOTemp_predict, HEX_LWOT)
Result_LOWT_MAE = MAE(LOTemp_predict, HEX_LWOT)
Result_LOWT_MBE = np.mean(LOTemp_predict-HEX_LWOT)

# Save the File
with open('ResultTempHEX_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['SOWT', Result_SOWT_R2, Result_SOWT_RMSE, Result_SOWT_MAE, Result_SOWT_MBE])
    writer.writerow(['LOWT', Result_LOWT_R2, Result_LOWT_RMSE, Result_LOWT_MAE, Result_LOWT_MBE])


'==============================================================================================================================='
