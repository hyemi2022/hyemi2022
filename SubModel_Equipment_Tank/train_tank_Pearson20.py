

# train Tank.py

'''
-----------------------------------------------------
Tank's grey-box model: based on Vertical
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
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

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

# Input Variable from Train Data
SITemp_train = Train['ST01_SITemp']
LITemp_train = Train['ST01_LITemp']
SOTemp_train = Train['ST01_SOTemp']
LOTemp_train = Train['ST01_LOTemp']
SOFlow_train = Train['ST01_Sflow']
LOFlow_train = Train['ST01_Lflow']


# ================= Training Step ===================== #

output_shape = 1

# KFold setting
kfold=5
num_val_samples = len(Train) // kfold

# train data setting
feature_names = ['Timepermin', 'Week', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Secondary_Load', 'Tnak_Storage_rate',
                'R01_ITemp', 'R02_ITemp', 'HEX01_SITemp', 'HEX01_LITemp', 'ST01_SITemp', 'ST01_LITemp', 'SH01_Temp',
                'R01Pump_Flow', 'R02Pump_Flow', 'ST01Pump_Flow', 'HEX01Pump_Flow', '2ndPump_Flow', 'RH01_Temp'] 

x_data = Train[feature_names]

'--------------------------------------------------'

# y_train data --> Tank Parameters
UAs_Para = Train['total_Uas']

# Selected Feature
corr_pears_tank_para = x_data.corrwith(Train['total_Uas'], method='pearson')
corr_pears_tank_para = abs(corr_pears_tank_para)
indices_01 = np.argsort(corr_pears_tank_para)[::-1]


Feature_01 = []
Correlation_feature_01 = []
for f in range(x_data.shape[1]):
   print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_01[f]], corr_pears_tank_para[indices_01[f]]))

   Feature_01.append(feature_names[indices_01[f]])
   Correlation_feature_01.append(corr_pears_tank_para[indices_01[f]])


# Save to the Pickle Importance Dataset
with open("corr_pears_tank_para.pickle", "wb") as f:
    pickle.dump(corr_pears_tank_para, f)
    

# Selected Feature by rule
X_selected_01 = 4

feature_result_01=[]
for f in range(X_selected_01):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feature_names[indices_01[f]], corr_pears_tank_para[indices_01[f]]))
    feature_result_01.append(feature_names[indices_01[f]])

print(feature_result_01)

input_shape_01 = X_selected_01
X_trian_01 = Train[feature_result_01]


# MLP Model Auto tune
seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.random.set_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


all_cv_score_01=[]

for i in range(kfold):

    print('Fold calculating', i)

    val_data_01 = X_trian_01[i*num_val_samples: (i+1)*num_val_samples]
    val_target_01 = UAs_Para[i*num_val_samples: (i+1)*num_val_samples]

    partial_train_data_01 = np.concatenate([X_trian_01[:i*num_val_samples], X_trian_01[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets_01 = np.concatenate([UAs_Para[:i*num_val_samples], UAs_Para[(i+1)*num_val_samples:]], axis=0)
    
    def keras_model_01():
        model = tf.keras.models.Sequential()  
        model.add(Dense(300, input_dim = input_shape_01, kernel_initializer = 'random_uniform', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(200, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

        model.compile(loss = 'mean_squared_logarithmic_error', optimizer ="adam",  metrics=['accuracy'])
        return model

    es = EarlyStopping(monitor = 'val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
    mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Pearson20_Tank_parameter.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    best_model_y01 = keras_model_01()
    best_model_y01.fit(partial_train_data_01, partial_train_targets_01, epochs=30, batch_size=9000)
    best_model_y01.save('C:\\System_Air\\best_model\\physics_parameter\\Pearson20_Tank_parameter.h5')
    val_mse_01, val_mae_01 = best_model_y01.evaluate(val_data_01, val_target_01, batch_size=2500, callbacks=[es], verbose=1)
    all_cv_score_01.append(val_mae_01)


x_predict_01 = Predict[feature_result_01]
x_predict_01 = np.array(x_predict_01)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Pearson20_Tank_parameter.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
y01_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Pearson20_Tank_parameter.h5')
history = y01_Model.fit(X_trian_01, UAs_Para, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


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
plt.savefig('Pearson_Case02.png')


Tank_Para_Predict = y01_Model.predict(x_predict_01)
Tank_Para_Predict = Tank_Para_Predict.reshape(122400,)

Predict['Tank_Para_Predict'] = Tank_Para_Predict


# ================ Reverse Calculate ================= #

# Initial Temperature of each layer
Node1_1st = Predict['Node1_temp'].loc[0]
Node2_1st = Predict['Node2_temp'].loc[0]
Node3_1st = Predict['Node3_temp'].loc[0]
Node4_1st = Predict['Node4_temp'].loc[0]
Node5_1st = Predict['Node5_temp'].loc[0]
Node6_1st = Predict['Node6_temp'].loc[0]
Node7_1st = Predict['Node7_temp'].loc[0]
Node8_1st = Predict['Node8_temp'].loc[0]
Node9_1st = Predict['Node9_temp'].loc[0]
Node10_1st = Predict['Node10_temp'].loc[0]


SITemp = Predict['ST01_SITemp']
LITemp = Predict['ST01_LITemp']
Tank_SF = Predict['ST01_Sflow']
Tank_LF = Predict['ST01_Lflow']
Ta = Predict['Outdoor_Temperature'] 

Node1_result = []
Node2_result = []
Node3_result = []
Node4_result = []
Node5_result = []
Node6_result = []
Node7_result = []
Node8_result = []
Node9_result = []
Node10_result = []


for i in range(len(SITemp_train)):

    # Computing the temperature of each layer
    # Parameter
    # Z = 0.3                             # Total High = 3m, Each layer = 0.3m
    # Aq = 400                            # Area of bottom = 600m^2
    # w = 0.6*1.163                       # Effective vertical heat conductivity of water
    # UAs = Tank_Para_Predict/-100        # Heat loss with Outdoor temperature
    # UAs = UAs[0, 0]
    v = 120                               # Volume of Each layer
    delt = 1/60                           # delta temperautre

    SITemp = Predict['ST01_SITemp'].iloc[i]
    LITemp = Predict['ST01_LITemp'].iloc[i]
    Tank_SF = Predict['ST01_Sflow'].iloc[i]
    Tank_LF = Predict['ST01_Lflow'].iloc[i]
    Ta = Predict['Outdoor_Temperature'].iloc[i]
    UAs = Predict['Tank_Para_Predict'].iloc[i]/10          # kJ/hk 

    if i == 0:
        if Tank_SF !=0 and Tank_LF ==0:     # Charging Time
            
            Node10_2nd = SITemp-((SITemp-Node10_1st)*math.exp(-Tank_SF*delt/v))
            loss_node10 = UAs*(Node10_1st-Ta)*delt      # kJ/h
            dt_node10 = loss_node10/(v*1000*4.2)        # k
            Node10_2nd = Node10_2nd + dt_node10         # k

            Node9_2nd = Node10_2nd-((Node10_2nd-Node9_1st)*math.exp(-Tank_SF*delt/v))
            loss_node9 = UAs*(Node9_1st-Ta)*delt        # kJ/h
            dt_node9 = loss_node9/(v*1000*4.2)          # k
            Node9_2nd = Node9_2nd + dt_node9            # k

            Node8_2nd = Node9_2nd-((Node9_2nd-Node8_1st)*math.exp(-Tank_SF*delt/v))
            loss_node8 = UAs*(Node8_1st-Ta)*delt        # kJ/h
            dt_node8 = loss_node8/(v*1000*4.2)          # k
            Node8_2nd = Node8_2nd + dt_node8            # k
            
            Node7_2nd = Node8_2nd-((Node8_2nd-Node7_1st)*math.exp(-Tank_SF*delt/v))
            loss_node7 = UAs*(Node7_1st-Ta)*delt        # kJ/h
            dt_node7 = loss_node7/(v*1000*4.2)          # k
            Node7_2nd = Node7_2nd + dt_node7            # k

            Node6_2nd = Node7_2nd-((Node7_2nd-Node6_1st)*math.exp(-Tank_SF*delt/v))
            loss_node6 = UAs*(Node6_1st-Ta)*delt        # kJ/h
            dt_node6 = loss_node6/(v*1000*4.2)          # k
            Node6_2nd = Node6_2nd + dt_node6            # k

            Node5_2nd = Node6_2nd-((Node6_2nd-Node5_1st)*math.exp(-Tank_SF*delt/v))
            loss_node5 = UAs*(Node5_1st-Ta)*delt        # kJ/h
            dt_node5 = loss_node5/(v*1000*4.2)          # k
            Node5_2nd = Node5_2nd + dt_node5            # k

            Node4_2nd = Node5_2nd-((Node5_2nd-Node4_1st)*math.exp(-Tank_SF*delt/v))
            loss_node4 = UAs*(Node4_1st-Ta)*delt        # kJ/h
            dt_node4 = loss_node4/(v*1000*4.2)          # k
            Node4_2nd = Node4_2nd + dt_node4            # k

            Node3_2nd = Node4_2nd-((Node4_2nd-Node3_1st)*math.exp(-Tank_SF*delt/v))
            loss_node3 = UAs*(Node3_1st-Ta)*delt        # kJ/h
            dt_node3 = loss_node3/(v*1000*4.2)          # k
            Node3_2nd = Node3_2nd + dt_node3            # k

            Node2_2nd = Node3_2nd-((Node3_2nd-Node2_1st)*math.exp(-Tank_SF*delt/v))
            loss_node2 = UAs*(Node2_1st-Ta)*delt        # kJ/h
            dt_node2 = loss_node2/(v*1000*4.2)          # k
            Node2_2nd = Node2_2nd + dt_node2            # k

            Node1_2nd = Node2_2nd-((Node2_2nd-Node1_1st)*math.exp(-Tank_SF*delt/v))
            loss_node1 = UAs*(Node1_1st-Ta)*delt        # kJ/h
            dt_node1 = loss_node1/(v*1000*4.2)          # k
            Node1_2nd = Node1_2nd + dt_node1            # k

        elif Tank_LF != 0 and Tank_SF ==0:      # Discharging Time

            Node1_2nd = LITemp-((LITemp-Node1_1st)*math.exp(-Tank_LF*delt/v))
            loss_node1 = UAs*(Node1_1st-Ta)*delt        # kJ/h
            dt_node1 = loss_node1/(v*1000*4.2)          # k
            Node1_2nd = Node1_2nd + dt_node1            # k

            Node2_2nd = Node1_2nd-((Node1_2nd-Node2_1st)*math.exp(-Tank_LF*delt/v))
            loss_node2 = UAs*(Node2_1st-Ta)*delt        # kJ/h
            dt_node2 = loss_node2/(v*1000*4.2)          # k
            Node2_2nd = Node2_2nd + dt_node2            # k

            Node3_2nd = Node2_2nd-((Node2_2nd-Node3_1st)*math.exp(-Tank_LF*delt/v))
            loss_node3 = UAs*(Node3_1st-Ta)*delt        # kJ/h
            dt_node3 = loss_node3/(v*1000*4.2)          # k
            Node3_2nd = Node3_2nd + dt_node3            # k

            Node4_2nd = Node3_2nd-((Node3_2nd-Node4_1st)*math.exp(-Tank_LF*delt/v))
            loss_node4 = UAs*(Node4_1st-Ta)*delt        # kJ/h
            dt_node4 = loss_node4/(v*1000*4.2)          # k
            Node4_2nd = Node4_2nd + dt_node4            # k

            Node5_2nd = Node4_2nd-((Node4_2nd-Node5_1st)*math.exp(-Tank_LF*delt/v))
            loss_node5 = UAs*(Node5_1st-Ta)*delt        # kJ/h
            dt_node5 = loss_node5/(v*1000*4.2)          # k
            Node5_2nd = Node5_2nd + dt_node5            # k

            Node6_2nd = Node5_2nd-((Node5_2nd-Node6_1st)*math.exp(-Tank_LF*delt/v))
            loss_node6 = UAs*(Node6_1st-Ta)*delt        # kJ/h
            dt_node6 = loss_node6/(v*1000*4.2)          # k
            Node6_2nd = Node6_2nd + dt_node6            # k

            Node7_2nd = Node6_2nd-((Node6_2nd-Node7_1st)*math.exp(-Tank_LF*delt/v))
            loss_node7 = UAs*(Node7_1st-Ta)*delt        # kJ/h
            dt_node7 = loss_node7/(v*1000*4.2)          # k
            Node7_2nd = Node7_2nd + dt_node7            # k

            Node8_2nd = Node7_2nd-((Node7_2nd-Node8_1st)*math.exp(-Tank_LF*delt/v))
            loss_node8 = UAs*(Node8_1st-Ta)*delt        # kJ/h
            dt_node8 = loss_node8/(v*1000*4.2)          # k
            Node8_2nd = Node8_2nd + dt_node8            # k

            Node9_2nd = Node8_2nd-((Node8_2nd-Node9_1st)*math.exp(-Tank_LF*delt/v))
            loss_node9 = UAs*(Node9_1st-Ta)*delt        # kJ/h
            dt_node9 = loss_node9/(v*1000*4.2)          # k
            Node9_2nd = Node9_2nd + dt_node9            # k

            Node10_2nd = Node9_2nd-((Node9_2nd-Node10_1st)*math.exp(-Tank_LF*delt/v))
            loss_node10 = UAs*(Node10_1st-Ta)*delt      # kJ/h
            dt_node10 = loss_node10/(v*1000*4.2)        # k
            Node10_2nd = Node10_2nd + dt_node10         # k

        else:

            loss_node10 = UAs*(Node10_1st-Ta)*delt      # kJ/h
            dt_node10 = loss_node10/(v*1000*4.2)        # k
            Node10_2nd = Node10_2nd + dt_node10         # k

            loss_node9 = UAs*(Node9_1st-Ta)*delt        # kJ/h
            dt_node9 = loss_node9/(v*1000*4.2)          # k
            Node9_2nd = Node9_2nd + dt_node9            # k

            loss_node8 = UAs*(Node8_1st-Ta)*delt        # kJ/h
            dt_node8 = loss_node8/(v*1000*4.2)          # k
            Node8_2nd = Node8_2nd + dt_node8            # k
            
            loss_node7 = UAs*(Node7_1st-Ta)*delt        # kJ/h
            dt_node7 = loss_node7/(v*1000*4.2)          # k
            Node7_2nd = Node7_2nd + dt_node7            # k

            loss_node6 = UAs*(Node6_1st-Ta)*delt        # kJ/h
            dt_node6 = loss_node6/(v*1000*4.2)          # k
            Node6_2nd = Node6_2nd + dt_node6            # k

            loss_node5 = UAs*(Node5_1st-Ta)*delt        # kJ/h
            dt_node5 = loss_node5/(v*1000*4.2)          # k
            Node5_2nd = Node5_2nd + dt_node5            # k

            loss_node4 = UAs*(Node4_1st-Ta)*delt        # kJ/h
            dt_node4 = loss_node4/(v*1000*4.2)          # k
            Node4_2nd = Node4_2nd + dt_node4            # k

            loss_node3 = UAs*(Node3_1st-Ta)*delt        # kJ/h
            dt_node3 = loss_node3/(v*1000*4.2)          # k
            Node3_2nd = Node3_2nd + dt_node3            # k

            loss_node2 = UAs*(Node2_1st-Ta)*delt        # kJ/h
            dt_node2 = loss_node2/(v*1000*4.2)          # k
            Node2_2nd = Node2_2nd + dt_node2            # k

            loss_node1 = UAs*(Node1_1st-Ta)*delt        # kJ/h
            dt_node1 = loss_node1/(v*1000*4.2)          # k
            Node1_2nd = Node1_2nd + dt_node1            # k

    else:   # Second time 
        if Tank_SF !=0 and Tank_LF ==0:     # Charging Time

            Node10_3rd = SITemp-((SITemp-Node10_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node10_2nd = UAs*(Node10_2nd-Ta)*delt          # kJ/h
            dt_node10_2nd = loss_node10_2nd/(v*1000*4.2)        # k
            Node10_3rd = Node10_3rd + dt_node10_2nd             # k

            Node9_3rd = Node10_3rd-((Node10_3rd-Node9_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node9_2nd = UAs*(Node9_2nd-Ta)*delt            # kJ/h
            dt_node9_2nd = loss_node9_2nd/(v*1000*4.2)          # k
            Node9_3rd = Node9_3rd + dt_node9_2nd                # k

            Node8_3rd = Node9_3rd-((Node9_3rd-Node8_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node8_2nd = UAs*(Node8_2nd-Ta)*delt            # kJ/h
            dt_node8_2nd = loss_node8_2nd/(v*1000*4.2)          # k
            Node8_3rd = Node8_3rd + dt_node8_2nd                # k

            Node7_3rd = Node8_3rd-((Node8_3rd-Node7_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node7_2nd = UAs*(Node7_2nd-Ta)*delt            # kJ/h
            dt_node7_2nd = loss_node7_2nd/(v*1000*4.2)          # k
            Node7_3rd = Node7_3rd + dt_node7_2nd                # k
                    
            Node6_3rd = Node7_3rd-((Node7_3rd-Node6_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node6_2nd = UAs*(Node6_2nd-Ta)*delt            # kJ/h
            dt_node6_2nd = loss_node6_2nd/(v*1000*4.2)          # k
            Node6_3rd = Node6_3rd + dt_node6_2nd                # k

            Node5_3rd = Node6_3rd-((Node6_3rd-Node5_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node5_2nd = UAs*(Node5_2nd-Ta)*delt            # kJ/h
            dt_node5_2nd = loss_node5_2nd/(v*1000*4.2)          # k
            Node5_3rd = Node5_3rd + dt_node5_2nd                # k

            Node4_3rd = Node5_3rd-((Node5_3rd-Node4_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node4_2nd = UAs*(Node4_2nd-Ta)*delt            # kJ/h
            dt_node4_2nd = loss_node4_2nd/(v*1000*4.2)          # k
            Node4_3rd = Node4_3rd + dt_node4_2nd                # k

            Node3_3rd = Node4_3rd-((Node4_3rd-Node3_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node3_2nd = UAs*(Node3_2nd-Ta)*delt            # kJ/h
            dt_node3_2nd = loss_node3_2nd/(v*1000*4.2)          # k
            Node3_3rd = Node3_3rd + dt_node3_2nd                # k

            Node2_3rd = Node3_3rd-((Node3_3rd-Node2_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node2_2nd = UAs*(Node2_2nd-Ta)*delt            # kJ/h
            dt_node2_2nd = loss_node2_2nd/(v*1000*4.2)          # k
            Node2_3rd = Node2_3rd + dt_node2_2nd                # k

            Node1_3rd = Node2_3rd-((Node2_3rd-Node1_2nd)*math.exp(-Tank_SF*delt/v))
            loss_node1_2nd = UAs*(Node1_2nd-Ta)*delt            # kJ/h
            dt_node1_2nd = loss_node1_2nd/(v*1000*4.2)          # k
            Node1_3rd = Node1_3rd + dt_node1_2nd                # k

        elif Tank_LF != 0 and Tank_SF ==0:      # Discharging Time

            Node1_3rd = LITemp-((LITemp-Node1_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node1_2nd = UAs*(Node1_2nd-Ta)*delt            # kJ/h
            dt_node1_2nd = loss_node1_2nd/(v*1000*4.2)          # k
            Node1_3rd = Node1_3rd + dt_node1_2nd                # k

            Node2_3rd = Node1_3rd-((Node1_3rd-Node2_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node2_2nd = UAs*(Node2_2nd-Ta)*delt            # kJ/h
            dt_node2_2nd = loss_node2_2nd/(v*1000*4.2)          # k
            Node2_3rd = Node2_3rd + dt_node2_2nd                # k

            Node3_3rd = Node2_3rd-((Node2_3rd-Node3_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node3_2nd = UAs*(Node3_2nd-Ta)*delt            # kJ/h
            dt_node3_2nd = loss_node3_2nd/(v*1000*4.2)          # k
            Node3_3rd = Node3_3rd + dt_node3_2nd                # k
                    
            Node4_3rd = Node3_3rd-((Node3_3rd-Node4_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node4_2nd = UAs*(Node4_2nd-Ta)*delt            # kJ/h
            dt_node4_2nd = loss_node4_2nd/(v*1000*4.2)          # k
            Node4_3rd = Node4_3rd + dt_node4_2nd                # k

            Node5_3rd = Node4_3rd-((Node4_3rd-Node5_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node5_2nd = UAs*(Node5_2nd-Ta)*delt            # kJ/h
            dt_node5_2nd = loss_node5_2nd/(v*1000*4.2)          # k
            Node5_3rd = Node5_3rd + dt_node5_2nd                # k

            Node6_3rd = Node5_3rd-((Node5_3rd-Node6_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node6_2nd = UAs*(Node6_2nd-Ta)*delt            # kJ/h
            dt_node6_2nd = loss_node6_2nd/(v*1000*4.2)          # k
            Node6_3rd = Node6_3rd + dt_node6_2nd                # k

            Node7_3rd = Node6_3rd-((Node6_3rd-Node7_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node7_2nd = UAs*(Node7_2nd-Ta)*delt            # kJ/h
            dt_node7_2nd = loss_node7_2nd/(v*1000*4.2)          # k
            Node7_3rd = Node7_3rd + dt_node7_2nd                # k

            Node8_3rd = Node7_3rd-((Node7_3rd-Node8_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node8_2nd = UAs*(Node8_2nd-Ta)*delt            # kJ/h
            dt_node8_2nd = loss_node8_2nd/(v*1000*4.2)          # k
            Node8_3rd = Node8_3rd + dt_node8_2nd                # k

            Node9_3rd = Node8_3rd-((Node8_3rd-Node9_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node9_2nd = UAs*(Node9_2nd-Ta)*delt            # kJ/h
            dt_node9_2nd = loss_node9_2nd/(v*1000*4.2)          # k
            Node9_3rd = Node9_3rd + dt_node9_2nd                # k

            Node10_3rd = Node9_3rd-((Node9_3rd-Node10_2nd)*math.exp(-Tank_LF*delt/v))
            loss_node10_2nd = UAs*(Node10_2nd-Ta)*delt          # kJ/h
            dt_node10_2nd = loss_node10_2nd/(v*1000*4.2)        # k
            Node10_3rd = Node10_3rd + dt_node10_2nd             # k

        else:

            loss_node10_2nd = UAs*(Node10_2nd-Ta)*delt          # kJ/h
            dt_node10_2nd = loss_node10_2nd/(v*1000*4.2)        # k
            Node10_3rd = Node10_3rd + dt_node10_2nd             # k

            loss_node9_2nd = UAs*(Node9_2nd-Ta)*delt            # kJ/h
            dt_node9_2nd = loss_node9_2nd/(v*1000*4.2)          # k
            Node9_3rd = Node9_3rd + dt_node9_2nd                # k

            loss_node8_2nd = UAs*(Node8_2nd-Ta)*delt            # kJ/h
            dt_node8_2nd = loss_node8_2nd/(v*1000*4.2)          # k
            Node8_3rd = Node8_3rd + dt_node8_2nd                # k

            loss_node7_2nd = UAs*(Node7_2nd-Ta)*delt            # kJ/h
            dt_node7_2nd = loss_node7_2nd/(v*1000*4.2)          # k
            Node7_3rd = Node7_3rd + dt_node7_2nd                # k
                    
            loss_node6_2nd = UAs*(Node6_2nd-Ta)*delt            # kJ/h
            dt_node6_2nd = loss_node6_2nd/(v*1000*4.2)          # k
            Node6_3rd = Node6_3rd + dt_node6_2nd                # k

            loss_node5_2nd = UAs*(Node5_2nd-Ta)*delt            # kJ/h
            dt_node5_2nd = loss_node5_2nd/(v*1000*4.2)          # k
            Node5_3rd = Node5_3rd + dt_node5_2nd                # k

            loss_node4_2nd = UAs*(Node4_2nd-Ta)*delt            # kJ/h
            dt_node4_2nd = loss_node4_2nd/(v*1000*4.2)          # k
            Node4_3rd = Node4_3rd + dt_node4_2nd                # k

            loss_node3_2nd = UAs*(Node3_2nd-Ta)*delt            # kJ/h
            dt_node3_2nd = loss_node3_2nd/(v*1000*4.2)          # k
            Node3_3rd = Node3_3rd + dt_node3_2nd                # k

            loss_node2_2nd = UAs*(Node2_2nd-Ta)*delt            # kJ/h
            dt_node2_2nd = loss_node2_2nd/(v*1000*4.2)          # k
            Node2_3rd = Node2_3rd + dt_node2_2nd                # k

            loss_node1_2nd = UAs*(Node1_2nd-Ta)*delt            # kJ/h
            dt_node1_2nd = loss_node1_2nd/(v*1000*4.2)          # k
            Node1_3rd = Node1_3rd + dt_node1_2nd                # k
    
        Node1_2nd = Node1_3rd
        Node2_2nd = Node2_3rd
        Node3_2nd = Node3_3rd
        Node4_2nd = Node4_3rd
        Node5_2nd = Node5_3rd
        Node6_2nd = Node6_3rd
        Node7_2nd = Node7_3rd
        Node8_2nd = Node8_3rd
        Node9_2nd = Node9_3rd
        Node10_2nd = Node10_3rd

    Node1_result.append(Node1_2nd)
    Node2_result.append(Node2_2nd)
    Node3_result.append(Node3_2nd)
    Node4_result.append(Node4_2nd)
    Node5_result.append(Node5_2nd)
    Node6_result.append(Node6_2nd)
    Node7_result.append(Node7_2nd)
    Node8_result.append(Node8_2nd)
    Node9_result.append(Node9_2nd)
    Node10_result.append(Node10_2nd)


# ================ Save the Result =================== #


submission = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
submission["Tank_UAs_Predict"] = Tank_Para_Predict
submission["Result_Node1"] = Node1_result
submission["Result_Node2"] = Node2_result
submission["Result_Node3"] = Node3_result
submission["Result_Node4"] = Node4_result
submission["Result_Node5"] = Node5_result
submission["Result_Node6"] = Node6_result
submission["Result_Node7"] = Node7_result
submission["Result_Node8"] = Node8_result
submission["Result_Node9"] = Node9_result
submission["Result_Node10"] = Node10_result
submission.to_csv('Pearson20_Correlation.csv', index = False)


# Function
predict_raw = Predict['total_Uas']

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

def MAE(x, y):
    return mean_absolute_error(x, y)

# Evaluate
loss, acc = y01_Model.evaluate(x_predict_01, Tank_Para_Predict, batch_size=128)
Result_Para_R2 = r2_score(predict_raw, Tank_Para_Predict)
Result_Para_RMSE = RMSE(predict_raw, Tank_Para_Predict)
Result_Para_MAE = MAE(predict_raw, predict_raw)
Result_Para_MBE = np.mean(predict_raw-predict_raw)

# Save the File
with open('Result_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Pearson_Case02_Tank_US', loss, acc, Result_Para_R2, Result_Para_RMSE, Result_Para_MAE, Result_Para_MBE])


# Outlet Temperature
Layer01_raw = Predict['Node1_temp']
Layer10_raw = Predict['Node10_temp']

Result_Layer01_R2 = r2_score(Layer01_raw, Node1_result)
Result_Layer01_RMSE = RMSE(Layer01_raw, Node1_result)
Result_Layer01_MAE = MAE(Layer01_raw, Node1_result)
Result_Layer01_MBE = np.mean(Layer01_raw-Node1_result)

Result_Layer10_R2 = r2_score(Layer10_raw, Node10_result)
Result_Layer10_RMSE = RMSE(Layer10_raw, Node10_result)
Result_Layer10_MAE = MAE(Layer10_raw, Node10_result)
Result_Layer10_MBE = np.mean(Layer10_raw-Node10_result)

# Save the File
with open('ResultTempTank_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Pearson_Case02_Layer01', Result_Layer01_R2, Result_Layer01_RMSE, Result_Layer01_MAE, Result_Layer01_MBE])
    writer.writerow(['Pearson_Case02_Layer10', Result_Layer10_R2, Result_Layer10_RMSE, Result_Layer10_MAE, Result_Layer10_MBE])


'==============================================================================================================================='

