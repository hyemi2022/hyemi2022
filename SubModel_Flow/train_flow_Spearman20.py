

# train flow.py

'''
-----------------------------------------------------
flow : black box model
-----------------------------------------------------
'''

'==============================================================================================================================='

import sys
import numpy as np
from pandas import DataFrame, Series
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn import preprocessing

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

# Setting Data
with open('Train.bin', 'rb') as file:
   Train = pickle.load(file)

with open('Predict.bin', 'rb') as file:
   Predict = pickle.load(file)

'--------------------------------------------------'

output_shape = 1

# KFold setting
kfold=5
num_val_samples = len(Train) // kfold

'--------------------------------------------------'

feature_names = ['Timepermin', 'Week', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Secondary_Load', 'Tnak_Storage_rate',
                    'R01_ITemp', 'R02_ITemp', 'HEX01_SITemp', 'HEX01_LITemp', 'ST01_SITemp', 'ST01_LITemp', 'SH01_Temp', 'RH01_Temp']                               
# 'R01_OTemp_1bef', 'R02_OTemp_1bef', 'ST01_LOTemp_1bef', 'ST01_SOTemp_1bef', 'HEX01_SOTemp_1bef', 'HEX01_LOTemp_1bef', 'RH01_Temp_1bef'                     

x_data = Train[feature_names] # Total 20 in Flow Model

'--------------------------------------------------'

# y01 : Pump01 Flow
Pump01_flow = Train['R01Pump_Flow']

# Selected Feature
corr_spear_flow01 = x_data.corrwith(Train['R01Pump_Flow'], method='spearman')  # Spearman Correlation
corr_spear_flow01 = abs(corr_spear_flow01)
indices_01 = np.argsort(corr_spear_flow01)[::-1]

Feature_01 = []
Correlation_feature_01 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,  
                            feature_names[indices_01[f]], corr_spear_flow01[indices_01[f]]))

    Feature_01.append(feature_names[indices_01[f]])
    Correlation_feature_01.append(corr_spear_flow01[indices_01[f]])

# Selected Feature by rule
X_selected_01 = 3

feature_result_01=[]
for f in range(X_selected_01):
    print("%2d) %-*s %f" % (f + 1, 30,  
                            feature_names[indices_01[f]], corr_spear_flow01[indices_01[f]]))
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
    val_target_01 = Pump01_flow[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_01 = np.concatenate([X_trian_01[:i*num_val_samples], X_trian_01[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_01 = np.concatenate([Pump01_flow[:i*num_val_samples], Pump01_flow[(i+1)*num_val_samples:]], axis=0)
    
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
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y01.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    best_model_Flow_y01 = keras_model_01()
    best_model_Flow_y01.fit(partial_train_data_01, partial_train_targets_01, epochs=30, batch_size=9000)
    best_model_Flow_y01.save('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y01.h5')
    val_mse_01, val_mae_01 = best_model_Flow_y01.evaluate(val_data_01, val_target_01, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_01.append(val_mae_01)


x_predict_01 = Predict[feature_result_01]
x_predict_01 = np.array(x_predict_01)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y01.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
Flow_y01_Model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y01.h5')
history01 = Flow_y01_Model.fit(X_trian_01, Pump01_flow, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])

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
plt.savefig('Spearman_Case02_y01.png')

Flow_y01_Predict = Flow_y01_Model.predict(x_predict_01)
Flow_y01_Predict = Flow_y01_Predict.reshape(122400,)


'--------------------------------------------------'

# y02 : Pump02 Flow
Pump02_flow = Train['R02Pump_Flow']

# Selected Feature
corr_spear_flow02 = x_data.corrwith(Train['R02Pump_Flow'], method='spearman')
corr_spear_flow02 = abs(corr_spear_flow02)
indices_02 = np.argsort(corr_spear_flow02)[::-1]

Feature_02 = []
Correlation_feature_02 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_02[f]], corr_spear_flow02[indices_02[f]]))

    Feature_02.append(feature_names[indices_02[f]])
    Correlation_feature_02.append(corr_spear_flow02[indices_02[f]])

# Selected Feature by rule
X_selected_02 = 3

feature_result_02=[]
for f in range(X_selected_02):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_02[f]], corr_spear_flow02[indices_02[f]]))
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
    val_target_02 = Pump02_flow[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_02 = np.concatenate([X_trian_02[:i*num_val_samples], X_trian_02[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_02 = np.concatenate([Pump02_flow[:i*num_val_samples], Pump02_flow[(i+1)*num_val_samples:]], axis=0)
    
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
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y02.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    best_model_Flow_y02 = keras_model_02()
    best_model_Flow_y02.fit(partial_train_data_02, partial_train_targets_02, epochs=30, batch_size=9000)
    best_model_Flow_y02.save('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y02.h5')
    val_mse_02, val_mae_02 = best_model_Flow_y02.evaluate(val_data_02, val_target_02, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_02.append(val_mae_02)


x_predict_02 = Predict[feature_result_02]
x_predict_02 = np.array(x_predict_02)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y02.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
Flow_y02_Model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y02.h5')
history02 = Flow_y02_Model.fit(X_trian_02, Pump02_flow, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])

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
plt.savefig('Spearman_Case02_y02.png')

Flow_y02_Predict = Flow_y02_Model.predict(x_predict_02)
Flow_y02_Predict = Flow_y02_Predict.reshape(122400,)


'--------------------------------------------------'

# y03 : Pump03 Flow
Pump03_flow = Train['ST01Pump_Flow']

# Selected Feature
corr_spear_flow03 = x_data.corrwith(Train['ST01Pump_Flow'], method='spearman')
corr_spear_flow03 = abs(corr_spear_flow03)
indices_03 = np.argsort(corr_spear_flow03)[::-1]

Feature_03 = []
Correlation_feature_03 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_03[f]], corr_spear_flow03[indices_03[f]]))

    Feature_03.append(feature_names[indices_03[f]])
    Correlation_feature_03.append(corr_spear_flow03[indices_03[f]])

# Selected Feature by rule
X_selected_03 = 3

feature_result_03=[]
for f in range(X_selected_03):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_03[f]], corr_spear_flow03[indices_03[f]]))
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
    val_target_03 = Pump03_flow[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_03 = np.concatenate([X_trian_03[:i*num_val_samples], X_trian_03[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_03 = np.concatenate([Pump03_flow[:i*num_val_samples], Pump03_flow[(i+1)*num_val_samples:]], axis=0)
    
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

    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y03.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    best_model_Flow_y03 = keras_model_03()
    best_model_Flow_y03.fit(partial_train_data_03, partial_train_targets_03, epochs=30, batch_size=9000)
    best_model_Flow_y03.save('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y03.h5')
    val_mse_03, val_mae_03 = best_model_Flow_y03.evaluate(val_data_03, val_target_03, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_03.append(val_mae_03)


x_predict_03 = Predict[feature_result_03]
x_predict_03 = np.array(x_predict_03)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y03.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
Flow_y03_Model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y03.h5')
history03 = Flow_y03_Model.fit(X_trian_03, Pump03_flow, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])

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
plt.savefig('Spearman_Case02_y03.png')

Flow_y03_Predict = Flow_y03_Model.predict(x_predict_03)
Flow_y03_Predict = Flow_y03_Predict.reshape(122400,)


'--------------------------------------------------'

# y04 : Pump04 Flow
Pump04_flow = Train['HEX01Pump_Flow']

# Selected Feature
corr_spear_flow04 = x_data.corrwith(Train['HEX01Pump_Flow'], method='spearman')
corr_spear_flow04 = abs(corr_spear_flow04)
indices_04 = np.argsort(corr_spear_flow04)[::-1]

Feature_04 = []
Correlation_feature_04 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_04[f]], corr_spear_flow04[indices_04[f]]))

    Feature_04.append(feature_names[indices_04[f]])
    Correlation_feature_04.append(corr_spear_flow04[indices_04[f]])

# Selected Feature by rule
X_selected_04 = 3 

feature_result_04=[]
for f in range(X_selected_04):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_04[f]], corr_spear_flow04[indices_04[f]]))
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
    val_target_04 = Pump04_flow[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_04 = np.concatenate([X_trian_04[:i*num_val_samples], X_trian_04[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_04 = np.concatenate([Pump04_flow[:i*num_val_samples], Pump04_flow[(i+1)*num_val_samples:]], axis=0)
    
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
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y04.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    best_model_Flow_y04 = keras_model_04()
    best_model_Flow_y04.fit(partial_train_data_04, partial_train_targets_04, epochs=30, batch_size=9000)
    best_model_Flow_y04.save('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y04.h5')
    val_mse_04, val_mae_04 = best_model_Flow_y04.evaluate(val_data_04, val_target_04, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_04.append(val_mae_04)


x_predict_04 = Predict[feature_result_04]
x_predict_04 = np.array(x_predict_04)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y04.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
Flow_y04_Model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y04.h5')
history04 = Flow_y04_Model.fit(X_trian_04, Pump04_flow, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])

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
plt.savefig('Spearman_Case02_y04.png')

Flow_y04_Predict = Flow_y04_Model.predict(x_predict_04)
Flow_y04_Predict = Flow_y04_Predict.reshape(122400,)


'--------------------------------------------------'

# y05 : Pump05 Flow
Pump05_flow = Train['2ndPump_Flow']

# Selected Feature
corr_spear_flow05 = x_data.corrwith(Train['2ndPump_Flow'], method='spearman')
corr_spear_flow05 = abs(corr_spear_flow05)
indices_05 = np.argsort(corr_spear_flow05)[::-1]

Feature_05 = []
Correlation_feature_05 = []
for f in range(x_data.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_05[f]], corr_spear_flow05[indices_05[f]]))

    Feature_05.append(feature_names[indices_05[f]])
    Correlation_feature_05.append(corr_spear_flow05[indices_05[f]])

# Selected Feature by rule
X_selected_05 = 3

feature_result_05=[]
for f in range(X_selected_05):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_05[f]], corr_spear_flow05[indices_05[f]]))
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
    val_target_05 = Pump05_flow[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_05 = np.concatenate([X_trian_05[:i*num_val_samples], X_trian_05[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_05 = np.concatenate([Pump05_flow[:i*num_val_samples], Pump05_flow[(i+1)*num_val_samples:]], axis=0)
    
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
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y05.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    best_model_Flow_y05 = keras_model_05()
    best_model_Flow_y05.fit(partial_train_data_05, partial_train_targets_05, epochs=30, batch_size=9000)
    best_model_Flow_y05.save('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y05.h5')
    val_mse_05, val_mae_05 = best_model_Flow_y05.evaluate(val_data_05, val_target_05, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_05.append(val_mae_05)


x_predict_05 = Predict[feature_result_05]
x_predict_05 = np.array(x_predict_05)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y05.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
Flow_y05_Model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Spearman20_flow_y05.h5')
history05 = Flow_y05_Model.fit(X_trian_05, Pump05_flow, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])

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
plt.savefig('Spearman_Case02_y05.png')

Flow_y05_Predict = Flow_y05_Model.predict(x_predict_05)
Flow_y05_Predict = Flow_y05_Predict.reshape(122400,)


# ================ Save the Result =================== #

submission = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
submission["Pump01_flow_Pred"] = Flow_y01_Predict
submission["Pump02_flow_Pred"] = Flow_y02_Predict
submission["Pump03_flow_Pred"] = Flow_y03_Predict
submission["Pump04_flow_Pred"] = Flow_y04_Predict
submission["Pump05_flow_Pred"] = Flow_y05_Predict
submission.to_csv('Spearman20_Correlation.csv', index = False)


# Function
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

def MAE(x, y):
    return mean_absolute_error(x, y)

# Evaluate
loss_y01, acc_y01 = Flow_y01_Model.evaluate(x_predict_01, Flow_y01_Predict, batch_size=128)
loss_y02, acc_y02 = Flow_y02_Model.evaluate(x_predict_02, Flow_y02_Predict, batch_size=128)
loss_y03, acc_y03 = Flow_y03_Model.evaluate(x_predict_03, Flow_y03_Predict, batch_size=128)
loss_y04, acc_y04 = Flow_y04_Model.evaluate(x_predict_04, Flow_y04_Predict, batch_size=128)
loss_y05, acc_y05 = Flow_y05_Model.evaluate(x_predict_05, Flow_y05_Predict, batch_size=128)

# Erorr
y_predict_01 = Predict['R01Pump_Flow']
y_predict_02 = Predict['R02Pump_Flow']
y_predict_03 = Predict['ST01Pump_Flow']
y_predict_04 = Predict['HEX01Pump_Flow']
y_predict_05 = Predict['2ndPump_Flow']

Result_R2_y01 = r2_score(y_predict_01, Flow_y01_Predict)
Result_R2_y02 = r2_score(y_predict_02, Flow_y02_Predict)
Result_R2_y03 = r2_score(y_predict_03, Flow_y03_Predict)
Result_R2_y04 = r2_score(y_predict_04, Flow_y04_Predict)
Result_R2_y05 = r2_score(y_predict_05, Flow_y05_Predict)

Result_RMSE_y01 = RMSE(y_predict_01, Flow_y01_Predict)
Result_RMSE_y02 = RMSE(y_predict_02, Flow_y02_Predict)
Result_RMSE_y03 = RMSE(y_predict_03, Flow_y03_Predict)
Result_RMSE_y04 = RMSE(y_predict_04, Flow_y04_Predict)
Result_RMSE_y05 = RMSE(y_predict_05, Flow_y05_Predict)

Result_MAE_y01 = MAE(y_predict_01, Flow_y01_Predict)
Result_MAE_y02 = MAE(y_predict_02, Flow_y02_Predict)
Result_MAE_y03 = MAE(y_predict_03, Flow_y03_Predict)
Result_MAE_y04 = MAE(y_predict_04, Flow_y04_Predict)
Result_MAE_y05 = MAE(y_predict_05, Flow_y05_Predict)

Result_MBE_y01 = np.mean(y_predict_01-Flow_y01_Predict)
Result_MBE_y02 = np.mean(y_predict_02-Flow_y02_Predict)
Result_MBE_y03 = np.mean(y_predict_03-Flow_y03_Predict)
Result_MBE_y04 = np.mean(y_predict_04-Flow_y04_Predict)
Result_MBE_y05 = np.mean(y_predict_05-Flow_y05_Predict)


# Save the File
with open('Result_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Spearman_Case02_y01', loss_y01, acc_y01, Result_R2_y01, Result_RMSE_y01, Result_MAE_y01, Result_MBE_y01])
    writer.writerow(['Spearman_Case02_y02', loss_y02, acc_y02, Result_R2_y02, Result_RMSE_y02, Result_MAE_y02, Result_MBE_y02])
    writer.writerow(['Spearman_Case02_y03', loss_y03, acc_y03, Result_R2_y03, Result_RMSE_y03, Result_MAE_y03, Result_MBE_y03])
    writer.writerow(['Spearman_Case02_y04', loss_y04, acc_y04, Result_R2_y04, Result_RMSE_y04, Result_MAE_y04, Result_MBE_y04])
    writer.writerow(['Spearman_Case02_y05', loss_y05, acc_y05, Result_R2_y05, Result_RMSE_y05, Result_MAE_y05, Result_MBE_y05])


'==============================================================================================================================='
