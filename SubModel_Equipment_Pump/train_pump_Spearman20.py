

# train pump.py

'''
-----------------------------------------------------
pump's grey-box model: based on INV(Inverter) pump

Pump Case
Case02 : inlet water temperature / flow
Case02 : inlet water temperature / flow / Control signal(INV)
Case03 : inlet water temperature / flow / Onoff signal

-----------------------------------------------------
'''

'==============================================================================================================================='

import sys
import numpy as np
from pandas import DataFrame, Series
import pandas as pd 
import csv
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import os
import random as rn
from math import sqrt
import math
from physic_pump import *

import tensorflow as tf
import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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


'''
# X_train data
HOWT_train = Train['RH01_Temp']

P01_Ftrain = Train['R01Pump_Flow']
P01_Ptrain = Train['R01Pump_Power']

P02_Ftrain = Train['R02Pump_Flow']
P02_Ptrain = Train['R02Pump_Power']

P03_WTtrain = Train['ST01_LOTemp']
P03_Ftrain = Train['ST01Pump_Flow']
P03_Ptrain = Train['ST01Pump_Power']

P04_Ftrain = Train['HEX01Pump_Flow']
P04_Ptrain = Train['HEX01Pump_Power']

P05_WTtrain = Train['SH01_Temp']
P05_Ftrain = Train['2ndPump_Flow']
P05_Ptrain = Train['2ndPump_Power']


for i, (a, b, c) in enumerate(zip(P01_Ptrain, P01_Ftrain, HOWT_train)):
    if i == 0:
        p01 = para(a, b, c)
        p01_Para = np.array(p01)
    else:
        p01 = para(a, b, c)
        p01_Para = np.append(p01_Para, p01)

for i, (a, b, c) in enumerate(zip(P02_Ptrain, P02_Ftrain, HOWT_train)):
    if i == 0:
        p02 = para(a, b, c)
        p02_Para = np.array(p02)
    else:
        p02 = para(a, b, c)
        p02_Para = np.append(p02_Para, p02)

for i, (a, b, c) in enumerate(zip(P03_Ptrain, P03_Ftrain, P03_WTtrain)):
    if i == 0:
        p03 = para(a, b, c)
        p03_Para = np.array(p03)
    else:
        p03 = para(a, b, c)
        p03_Para = np.append(p03_Para, p03)

for i, (a, b, c) in enumerate(zip(P04_Ptrain, P04_Ftrain, HOWT_train)):
    if i == 0:
        p04 = para(a, b, c)
        p04_Para = np.array(p04)
    else:
        p04 = para(a, b, c)
        p04_Para = np.append(p04_Para, p04)

for i, (a, b, c) in enumerate(zip(P05_Ptrain, P05_Ftrain, P05_WTtrain)):
    if i == 0:
        p05 = para(a, b, c)
        p05_Para = np.array(p05)
    else:
        p05 = para(a, b, c)
        p05_Para = np.append(p05_Para, p05)
'''

'--------------------------------------------------'

output_shape = 1

# KFold setting
kfold=5
num_val_samples = len(Train) // kfold

'--------------------------------------------------'

feature_names = ['Timepermin', 'Week', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Secondary_Load', 'Tnak_Storage_rate',
                    'R01_ITemp', 'R02_ITemp', 'HEX01_SITemp', 'HEX01_LITemp', 'ST01_SITemp', 'ST01_LITemp', 'SH01_Temp',
                    'R01Pump_Flow', 'R02Pump_Flow', 'ST01Pump_Flow', 'HEX01Pump_Flow', '2ndPump_Flow', 'RH01_Temp']

x_data = Train[feature_names] # Total 18 in Flow Model

'--------------------------------------------------'
# Pump 01 <-- 
'--------------------------------------------------'

# y_train
p01_Para = Train['p01_Parameter']

# Selected Feature
corr_spear_p01_para = x_data.corrwith(Train['p01_Parameter'], method='spearman')  # spearman Correlation
corr_spear_p01_para = abs(corr_spear_p01_para)
indices_01 = np.argsort(corr_spear_p01_para)[::-1]

Feature_01 = []
Correlation_feature_01 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 5, 
                            feature_names[indices_01[f]], corr_spear_p01_para[indices_01[f]]))

    Feature_01.append(feature_names[indices_01[f]])
    Correlation_feature_01.append(corr_spear_p01_para[indices_01[f]])


# Selected Feature by rule
X_selected_01 = 4

feature_result_01=[]
for f in range(X_selected_01):
    print("%2d) %-*s %f" % (f + 1, 5, 
                            feature_names[indices_01[f]], corr_spear_p01_para[indices_01[f]]))
    feature_result_01.append(feature_names[indices_01[f]])

print(feature_result_01)

input_shape_01 = X_selected_01
X_trian_01 = Train[feature_result_01]


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

    val_data_01 = X_trian_01[i*num_val_samples: (i+1)*num_val_samples]
    val_target_01 = p01_Para[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_01 = np.concatenate([X_trian_01[:i*num_val_samples], X_trian_01[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_01 = np.concatenate([p01_Para[:i*num_val_samples], p01_Para[(i+1)*num_val_samples:]], axis=0)
    
    def keras_model_01():
        model = tf.keras.models.Sequential()  
        model.add(Dense(200, input_dim = input_shape_01, kernel_initializer = 'random_uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

        model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam",  metrics=['accuracy'])
        return model

    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p01_parameter.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    best_model_y01 = keras_model_01()
    best_model_y01.fit(partial_train_data_01, partial_train_targets_01, epochs=30, batch_size=9000)
    best_model_y01.save('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p01_parameter.h5')
    val_mse_01, val_mae_01 = best_model_y01.evaluate(val_data_01, val_target_01, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_01.append(val_mae_01)


x_predict_01 = Predict[feature_result_01]
x_predict_01 = np.array(x_predict_01)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p01_parameter.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
y01_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p01_parameter.h5')
history01 = y01_Model.fit(X_trian_01, p01_Para, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


# Graph
plt.figure()
plt.subplot(1,2,1)
plt.plot(history01.history['loss'], label='loss')
plt.plot(history01.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.subplot(1,2,2)
plt.plot(history01.history['accuracy'], label='accuracy')
plt.plot(history01.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('Case02_y01.png')


P01_ParaPred = y01_Model.predict(x_predict_01)
P01_ParaPred = P01_ParaPred.reshape(122400,)


'--------------------------------------------------'
# Pump 02 <-- 
'--------------------------------------------------'

# y_train
p02_Para = Train['p02_Parameter']

# Selected Feature
corr_spear_p02_para = x_data.corrwith(Train['p02_Parameter'], method='spearman')
corr_spear_p02_para = abs(corr_spear_p02_para)
indices_02 = np.argsort(corr_spear_p02_para)[::-1]

Feature_02 = []
Correlation_feature_02 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_02[f]], corr_spear_p02_para[indices_02[f]]))

    Feature_02.append(feature_names[indices_02[f]])
    Correlation_feature_02.append(corr_spear_p02_para[indices_02[f]])


# Selected Feature by rule
X_selected_02 = 4

feature_result_02=[]
for f in range(X_selected_02):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_02[f]], corr_spear_p02_para[indices_02[f]]))
    feature_result_02.append(feature_names[indices_02[f]])

print(feature_result_02)

input_shape_02 = X_selected_02
X_trian_02 = Train[feature_result_02]


##### Training Step #####
seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

all_cv_score_02=[]

for i in range(kfold):

    print('Fold calculating', i)

    val_data_02 = X_trian_02[i*num_val_samples: (i+1)*num_val_samples]
    val_target_02 = p02_Para[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_02 = np.concatenate([X_trian_02[:i*num_val_samples], X_trian_02[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_02 = np.concatenate([p02_Para[:i*num_val_samples], p02_Para[(i+1)*num_val_samples:]], axis=0)
    
    def keras_model_02():
        model = tf.keras.models.Sequential()  
        model.add(Dense(200, input_dim = input_shape_02, kernel_initializer = 'random_uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

        model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam",  metrics=['accuracy'])
        return model

    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p02_parameter.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    best_model_y02 = keras_model_02()
    best_model_y02.fit(partial_train_data_02, partial_train_targets_02, epochs=30, batch_size=9000)
    best_model_y02.save('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p02_parameter.h5')
    val_mse_02, val_mae_02 = best_model_y02.evaluate(val_data_02, val_target_02, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_02.append(val_mae_02)


x_predict_02 = Predict[feature_result_02]
x_predict_02 = np.array(x_predict_02)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p02_parameter.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
y02_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p02_parameter.h5')
history02 = y02_Model.fit(X_trian_02, p02_Para, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


# Graph
plt.figure()
plt.subplot(1,2,1)
plt.plot(history02.history['loss'], label='loss')
plt.plot(history02.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.subplot(1,2,2)
plt.plot(history02.history['accuracy'], label='accuracy')
plt.plot(history02.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('Case02_y02.png')


P02_ParaPred = y02_Model.predict(x_predict_02)
P02_ParaPred = P02_ParaPred.reshape(122400,)


'--------------------------------------------------'
# Pump 03 <-- 
'--------------------------------------------------'

# y_train
p03_Para = Train['p03_Parameter']

# Selected Feature
corr_spear_p03_para = x_data.corrwith(Train['p03_Parameter'], method='spearman')
corr_spear_p03_para = abs(corr_spear_p03_para)
indices_03 = np.argsort(corr_spear_p03_para)[::-1]

Feature_03 = []
Correlation_feature_03 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_03[f]], corr_spear_p03_para[indices_03[f]]))

    Feature_03.append(feature_names[indices_03[f]])
    Correlation_feature_03.append(corr_spear_p03_para[indices_03[f]])


# Selected Feature by rule
X_selected_03 = 4

feature_result_03=[]
for f in range(X_selected_03):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_03[f]], corr_spear_p03_para[indices_03[f]]))
    feature_result_03.append(feature_names[indices_03[f]])

print(feature_result_03)

input_shape_03 = X_selected_03
X_trian_03 = Train[feature_result_03]


##### Training Step #####
seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

all_cv_score_03=[]

for i in range(kfold):

    print('Fold calculating', i)

    val_data_03 = X_trian_03[i*num_val_samples: (i+1)*num_val_samples]
    val_target_03 = p03_Para[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_03 = np.concatenate([X_trian_03[:i*num_val_samples], X_trian_03[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_03 = np.concatenate([p03_Para[:i*num_val_samples], p03_Para[(i+1)*num_val_samples:]], axis=0)
    
    def keras_model_03():
        model = tf.keras.models.Sequential()  
        model.add(Dense(200, input_dim = input_shape_03, kernel_initializer = 'random_uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

        model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam",  metrics=['accuracy'])
        return model

    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.003, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p03_parameter.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    best_model_y03 = keras_model_03()
    best_model_y03.fit(partial_train_data_03, partial_train_targets_03, epochs=30, batch_size=9000)
    best_model_y03.save('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p03_parameter.h5')
    val_mse_03, val_mae_03 = best_model_y03.evaluate(val_data_03, val_target_03, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_03.append(val_mae_03)


x_predict_03 = Predict[feature_result_03]
x_predict_03 = np.array(x_predict_03)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p03_parameter.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
y03_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p03_parameter.h5')
history03 = y03_Model.fit(X_trian_03, p03_Para, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


# Graph
plt.figure()
plt.subplot(1,2,1)
plt.plot(history03.history['loss'], label='loss')
plt.plot(history03.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.subplot(1,2,2)
plt.plot(history03.history['accuracy'], label='accuracy')
plt.plot(history03.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('Case02_y03.png')


P03_ParaPred = y03_Model.predict(x_predict_03)
P03_ParaPred = P03_ParaPred.reshape(122400,)


'--------------------------------------------------'
# Pump 04 <-- 
'--------------------------------------------------'

# y_train
p04_Para = Train['p04_Parameter']

# Selected Feature
corr_spear_p04_para = x_data.corrwith(Train['p04_Parameter'], method='spearman')
corr_spear_p04_para = abs(corr_spear_p04_para)
indices_04 = np.argsort(corr_spear_p04_para)[::-1]

Feature_04 = []
Correlation_feature_04 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_04[f]], corr_spear_p04_para[indices_04[f]]))

    Feature_04.append(feature_names[indices_04[f]])
    Correlation_feature_04.append(corr_spear_p04_para[indices_04[f]])


# Selected Feature by rule
X_selected_04 = 4

feature_result_04=[]
for f in range(X_selected_04):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_04[f]], corr_spear_p04_para[indices_04[f]]))
    feature_result_04.append(feature_names[indices_04[f]])

print(feature_result_04)

input_shape_04 = X_selected_04
X_trian_04 = Train[feature_result_04]


##### Training Step #####
seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

all_cv_score_04=[]

for i in range(kfold):

    print('Fold calculating', i)

    val_data_04 = X_trian_04[i*num_val_samples: (i+1)*num_val_samples]
    val_target_04 = p04_Para[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_04 = np.concatenate([X_trian_04[:i*num_val_samples], X_trian_04[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_04 = np.concatenate([p04_Para[:i*num_val_samples], p04_Para[(i+1)*num_val_samples:]], axis=0)
    
    def keras_model_04():
        model = tf.keras.models.Sequential()  
        model.add(Dense(200, input_dim = input_shape_04, kernel_initializer = 'random_uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

        model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam",  metrics=['accuracy'])
        return model

    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p04_parameter.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    best_model_y04 = keras_model_04()
    best_model_y04.fit(partial_train_data_04, partial_train_targets_04, epochs=30, batch_size=9000)
    best_model_y04.save('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p04_parameter.h5')
    val_mse_04, val_mae_04 = best_model_y04.evaluate(val_data_04, val_target_04, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_04.append(val_mae_04)


x_predict_04 = Predict[feature_result_04]
x_predict_04 = np.array(x_predict_04)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p04_parameter.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
y04_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p04_parameter.h5')
history04 = y04_Model.fit(X_trian_04, p04_Para, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


# Graph
plt.figure()
plt.subplot(1,2,1)
plt.plot(history04.history['loss'], label='loss')
plt.plot(history04.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.subplot(1,2,2)
plt.plot(history04.history['accuracy'], label='accuracy')
plt.plot(history04.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('Case02_y04.png')


P04_ParaPred = y04_Model.predict(x_predict_04)
P04_ParaPred = P04_ParaPred.reshape(122400,)


'--------------------------------------------------'
# Pump 05 <-- 
'--------------------------------------------------'

# y_train
p05_Para = Train['p05_Parameter']

# Selected Feature
corr_spear_p05_para = x_data.corrwith(Train['p05_Parameter'], method='spearman')
corr_spear_p05_para = abs(corr_spear_p05_para)
indices_05 = np.argsort(corr_spear_p05_para)[::-1]

Feature_05 = []
Correlation_feature_05 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_05[f]], corr_spear_p05_para[indices_05[f]]))

    Feature_05.append(feature_names[indices_05[f]])
    Correlation_feature_05.append(corr_spear_p05_para[indices_05[f]])


# Selected Feature by rule
X_selected_05 = 4

feature_result_05=[]
for f in range(X_selected_05):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_05[f]], corr_spear_p05_para[indices_05[f]]))
    feature_result_05.append(feature_names[indices_05[f]])

print(feature_result_05)

input_shape_05 = X_selected_05
X_trian_05 = Train[feature_result_05]


##### Training Step #####
seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

all_cv_score_05=[]

for i in range(kfold):

    print('Fold calculating', i)

    val_data_05 = X_trian_05[i*num_val_samples: (i+1)*num_val_samples]
    val_target_05 = p05_Para[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_05 = np.concatenate([X_trian_05[:i*num_val_samples], X_trian_05[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_05 = np.concatenate([p05_Para[:i*num_val_samples], p05_Para[(i+1)*num_val_samples:]], axis=0)
    
    def keras_model_05():
        model = tf.keras.models.Sequential()  
        model.add(Dense(200, input_dim = input_shape_05, kernel_initializer = 'random_uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

        model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam",  metrics=['accuracy'])
        return model


    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p05_parameter.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    best_model_y05 = keras_model_05()
    best_model_y05.fit(partial_train_data_05, partial_train_targets_05, epochs=30, batch_size=9000)
    best_model_y05.save('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p05_parameter.h5')
    val_mse_05, val_mae_05 = best_model_y05.evaluate(val_data_05, val_target_05, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_05.append(val_mae_05)


x_predict_05 = Predict[feature_result_05]
x_predict_05 = np.array(x_predict_05)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p05_parameter.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
y05_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Spearman20_p05_parameter.h5')
history05 = y05_Model.fit(X_trian_05, p05_Para, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


# Graph
plt.figure()
plt.subplot(1,2,1)
plt.plot(history05.history['loss'], label='loss')
plt.plot(history05.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.subplot(1,2,2)
plt.plot(history05.history['accuracy'], label='accuracy')
plt.plot(history05.history['val_accuracy'], label='val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('Case02_y05.png')


P05_ParaPred = y05_Model.predict(x_predict_05)
P05_ParaPred = P05_ParaPred.reshape(122400,)


'--------------------------------------------------'

'''
# X_pred data
P01_Ppred = Predict['R01Pump_Power'].astype('float16')
P02_Ppred = Predict['R02Pump_Power'].astype('float16')
P03_Ppred = Predict['ST01Pump_Power'].astype('float16')
P04_Ppred = Predict['HEX01Pump_Power'].astype('float16')
P05_Ppred = Predict['2ndPump_Power'].astype('float16')


# Calculate Pump's Parameters
# P01
for i, (a, b, c) in enumerate(zip(P01_Ppred, P01_Fpred, HOWT_pred)):
    if i == 0:
        p01 = para(a, b, c)
        p01_ParaCal = np.array(p01)
    else:
        p01 = para(a, b, c)
        p01_ParaCal = np.append(p01_ParaCal, p01)

# P02
for i, (a, b, c) in enumerate(zip(P02_Ppred, P02_Fpred, HOWT_pred)):
    if i == 0:
        p02 = para(a, b, c)
        p02_ParaCal = np.array(p02)
    else:
        p02 = para(a, b, c)
        p02_ParaCal = np.append(p02_ParaCal, p02)

# P03
for i, (a, b, c) in enumerate(zip(P03_Ppred, P03_Fpred, P03_WTpred)):
    if i == 0:
        p03 = para(a, b, c)
        p03_ParaCal = np.array(p03)
    else:
        p03 = para(a, b, c)
        p03_ParaCal = np.append(p03_ParaCal, p03)

# P04
for i, (a, b, c) in enumerate(zip(P04_Ppred, P04_Fpred, HOWT_pred)):
    if i == 0:
        p04 = para(a, b, c)
        p04_ParaCal = np.array(p04)
    else:
        p04 = para(a, b, c)
        p04_ParaCal = np.append(p04_ParaCal, p04)

# P05
for i, (a, b, c) in enumerate(zip(P05_Ppred, P05_Fpred, P05_WTpred)):
    if i == 0:
        p05 = para(a, b, c)
        p05_ParaCal = np.array(p05)
    else:
        p05 = para(a, b, c)
        p05_ParaCal = np.append(p05_ParaCal, p05)
'''

# ================ Reverse Calculate ================= #

# Input Variable from Predict Data
HOWT_pred = Predict['RH01_Temp'].astype('float16')
P01_Fpred = Predict['R01Pump_Flow'].astype('float16')
P02_Fpred = Predict['R02Pump_Flow'].astype('float16')
P03_Fpred = Predict['ST01Pump_Flow'].astype('float16')
P03_WTpred = Predict['ST01_LOTemp'].astype('float16')
P04_Fpred = Predict['HEX01Pump_Flow'].astype('float16')
P05_WTpred = Predict['SH01_Temp'].astype('float16')
P05_Fpred = Predict['2ndPump_Flow'].astype('float16')


# Calculate Pump's Power
# P01
for i, (a, b, c) in enumerate(zip(P01_ParaPred, HOWT_pred, P01_Fpred)):
    if i == 0:
        p01 = pump_power(a, b, c)
        p01_power = np.array(p01)
    else:
        p01 = pump_power(a, b, c)
        p01_power = np.append(p01_power, p01)

# P02
for i, (a, b, c) in enumerate(zip(P02_ParaPred, HOWT_pred, P02_Fpred)):
    if i == 0:
        p02 = pump_power(a, b, c)
        p02_power = np.array(p02)
    else:
        p02 = pump_power(a, b, c)
        p02_power = np.append(p02_power, p02)

# P03
for i, (a, b, c) in enumerate(zip(P03_ParaPred, P03_WTpred, P03_Fpred)):
    if i == 0:
        p03 = pump_power(a, b, c)
        p03_power = np.array(p03)
    else:
        p03 = pump_power(a, b, c)
        p03_power = np.append(p03_power, p03)

# P04
for i, (a, b, c) in enumerate(zip(P04_ParaPred, HOWT_pred, P04_Fpred)):
    if i == 0:
        p04 = pump_power(a, b, c)
        p04_power = np.array(p04)
    else:
        p04 = pump_power(a, b, c)
        p04_power = np.append(p04_power, p04)

# P05
for i, (a, b, c) in enumerate(zip(P05_ParaPred, P05_WTpred, P05_Fpred)):
    if i == 0:
        p05 = pump_power(a, b, c)
        p05_power = np.array(p05)
    else:
        p05 = pump_power(a, b, c)
        p05_power = np.append(p05_power, p05)


# ================ Save the Result =================== #


submission = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
submission["R01Pump_Para_Pred"] = P01_ParaPred
submission["R02Pump_Para_Pred"] = P02_ParaPred
submission["TankPump_Para_Pred"] = P03_ParaPred 
submission["HEXPump_Para_Pred"] = P04_ParaPred
submission["2ndPump_Para_Pred"] = P05_ParaPred
submission["R01Pump_power_Pred"] = p01_power
submission["R02Pump_power_Pred"] = p02_power
submission["TankPump_power_Pred"] = p03_power 
submission["HEXPump_power_Pred"] = p04_power
submission["2ndPump_power_Pred"] = p05_power
submission.to_csv('Spearman20_Correlation.csv', index = False)


p01_Parameter_raw = Predict['p01_Parameter']
p02_Parameter_raw = Predict['p02_Parameter']
p03_Parameter_raw = Predict['p03_Parameter']
p04_Parameter_raw = Predict['p04_Parameter']
p05_Parameter_raw = Predict['p05_Parameter']


# Function
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

def MAE(x, y):
    return mean_absolute_error(x, y)

# Evaluate
loss_y01, acc_y01 = y01_Model.evaluate(x_predict_01, P01_ParaPred, batch_size=128)
loss_y02, acc_y02 = y02_Model.evaluate(x_predict_02, P02_ParaPred, batch_size=128)
loss_y03, acc_y03 = y03_Model.evaluate(x_predict_03, P03_ParaPred, batch_size=128)
loss_y04, acc_y04 = y04_Model.evaluate(x_predict_04, P04_ParaPred, batch_size=128)
loss_y05, acc_y05 = y05_Model.evaluate(x_predict_05, P05_ParaPred, batch_size=128)

Result_R2_y01 = r2_score(p01_Parameter_raw, P01_ParaPred)
Result_R2_y02 = r2_score(p02_Parameter_raw, P02_ParaPred)
Result_R2_y03 = r2_score(p03_Parameter_raw, P03_ParaPred)
Result_R2_y04 = r2_score(p04_Parameter_raw, P04_ParaPred)
Result_R2_y05 = r2_score(p05_Parameter_raw, P05_ParaPred)

Result_RMSE_y01 = RMSE(p01_Parameter_raw, P01_ParaPred)
Result_RMSE_y02 = RMSE(p02_Parameter_raw, P02_ParaPred)
Result_RMSE_y03 = RMSE(p03_Parameter_raw, P03_ParaPred)
Result_RMSE_y04 = RMSE(p04_Parameter_raw, P04_ParaPred)
Result_RMSE_y05 = RMSE(p05_Parameter_raw, P05_ParaPred)

Result_MAE_y01 = MAE(p01_Parameter_raw, P01_ParaPred)
Result_MAE_y02 = MAE(p02_Parameter_raw, P02_ParaPred)
Result_MAE_y03 = MAE(p03_Parameter_raw, P03_ParaPred)
Result_MAE_y04 = MAE(p04_Parameter_raw, P04_ParaPred)
Result_MAE_y05 = MAE(p05_Parameter_raw, P05_ParaPred)

Result_MBE_y01 = np.mean(p01_Parameter_raw-P01_ParaPred)
Result_MBE_y02 = np.mean(p02_Parameter_raw-P02_ParaPred)
Result_MBE_y03 = np.mean(p03_Parameter_raw-P03_ParaPred)
Result_MBE_y04 = np.mean(p04_Parameter_raw-P04_ParaPred)
Result_MBE_y05 = np.mean(p05_Parameter_raw-P05_ParaPred)


# Save the File
with open('Result_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Spearman_Case02_P01', loss_y01, acc_y01, Result_R2_y01, Result_RMSE_y01, Result_MAE_y01, Result_MBE_y01])
    writer.writerow(['Spearman_Case02_P02', loss_y02, acc_y02, Result_R2_y02, Result_RMSE_y02, Result_MAE_y02, Result_MBE_y02])
    writer.writerow(['Spearman_Case02_P03', loss_y03, acc_y03, Result_R2_y03, Result_RMSE_y03, Result_MAE_y03, Result_MBE_y03])
    writer.writerow(['Spearman_Case02_P04', loss_y04, acc_y04, Result_R2_y04, Result_RMSE_y04, Result_MAE_y04, Result_MBE_y04])
    writer.writerow(['Spearman_Case02_P05', loss_y05, acc_y05, Result_R2_y05, Result_RMSE_y05, Result_MAE_y05, Result_MBE_y05])


# About Power
P01_Ppred = Predict['R01Pump_Power']
P02_Ppred = Predict['R02Pump_Power']
P03_Ppred = Predict['ST01Pump_Power']
P04_Ppred = Predict['HEX01Pump_Power']
P05_Ppred = Predict['2ndPump_Power']


ResultPower_R2_y01 = r2_score(P01_Ppred, p01_power)
ResultPower_R2_y02 = r2_score(P02_Ppred, p02_power)
ResultPower_R2_y03 = r2_score(P03_Ppred, p03_power)
ResultPower_R2_y04 = r2_score(P04_Ppred, p04_power)
ResultPower_R2_y05 = r2_score(P05_Ppred, p05_power)

ResultPower_RMSE_y01 = RMSE(P01_Ppred, p01_power)
ResultPower_RMSE_y02 = RMSE(P02_Ppred, p02_power)
ResultPower_RMSE_y03 = RMSE(P03_Ppred, p03_power)
ResultPower_RMSE_y04 = RMSE(P04_Ppred, p04_power)
ResultPower_RMSE_y05 = RMSE(P05_Ppred, p05_power)

ResultPower_MAE_y01 = MAE(P01_Ppred, p01_power)
ResultPower_MAE_y02 = MAE(P02_Ppred, p02_power)
ResultPower_MAE_y03 = MAE(P03_Ppred, p03_power)
ResultPower_MAE_y04 = MAE(P04_Ppred, p04_power)
ResultPower_MAE_y05 = MAE(P05_Ppred, p05_power)

ResultPower_MBE_y01 = np.mean(P01_Ppred-p01_power)
ResultPower_MBE_y02 = np.mean(P02_Ppred-p02_power)
ResultPower_MBE_y03 = np.mean(P03_Ppred-p03_power)
ResultPower_MBE_y04 = np.mean(P04_Ppred-p04_power)
ResultPower_MBE_y05 = np.mean(P05_Ppred-p05_power)


# Save the File
with open('ResultPower_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Spearman_Power_Case02_P01', ResultPower_R2_y01, ResultPower_RMSE_y01, ResultPower_MAE_y01, ResultPower_MBE_y01])
    writer.writerow(['Spearman_Power_Case02_P02', ResultPower_R2_y02, ResultPower_RMSE_y02, ResultPower_MAE_y02, ResultPower_MBE_y02])
    writer.writerow(['Spearman_Power_Case02_P03', ResultPower_R2_y03, ResultPower_RMSE_y03, ResultPower_MAE_y03, ResultPower_MBE_y03])
    writer.writerow(['Spearman_Power_Case02_P04', ResultPower_R2_y04, ResultPower_RMSE_y04, ResultPower_MAE_y04, ResultPower_MBE_y04])
    writer.writerow(['Spearman_Power_Case02_P05', ResultPower_R2_y05, ResultPower_RMSE_y05, ResultPower_MAE_y05, ResultPower_MBE_y05])


'==============================================================================================================================='
