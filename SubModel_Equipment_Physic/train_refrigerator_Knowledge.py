

# train_refrigerator.py

'''
-----------------------------------------------------
refrigerator's hybrid model

Refrigerator Case
Case01 : inlet water / flow / Outdoor Air
Case02 : inlet water / flow / Outdoor Air / Setting temperature
Case03 : inlet water / flow / Outdoor Air / Onoff signal
Case04 : inlet water / flow / Outdoor Air / 2nd load

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

import warnings
warnings.filterwarnings('ignore')

import os
import random as rn
from math import sqrt
from physic_refrigerator import *

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

# =============== Calcluate Parameter ================= #

# Calculate Parameter - Train

# Data Setting
# Train
Rc01_LOWT = Train['R01_OTemp']
Rc01_LIWT = Train['R01_ITemp']
Rc01_LWF = Train['R01_Flow']  
Rc01_SIWT = Train['Outdoor_Temperature']

Rc01_LOWT = np.array(Rc01_LOWT).astype('float16')
Rc01_LIWT = np.array(Rc01_LIWT).astype('float16')
Rc01_LWF = np.array(Rc01_LWF).astype('float16')
Rc01_SIWT = np.array(Rc01_SIWT).astype('float16')

Rc02_LOWT = Train['R02_OTemp']
Rc02_LIWT = Train['R02_ITemp']
Rc02_LWF = Train['R02_Flow']
Rc02_SIWT = Train['Outdoor_Temperature']

Rc02_LOWT = np.array(Rc02_LOWT).astype('float16')
Rc02_LIWT = np.array(Rc02_LIWT).astype('float16')
Rc02_LWF = np.array(Rc02_LWF).astype('float16')
Rc02_SIWT = np.array(Rc02_SIWT).astype('float16')


# Predict
Rc01_LOWT_Pred = Predict['R01_OTemp']
Rc01_LIWT_Pred = Predict['R01_ITemp']
Rc01_LWF_Pred = Predict['R01_Flow']
Rc01_SIWT_Pred = Predict['Outdoor_Temperature']

Rc02_LOWT_Pred = Predict['R02_OTemp']
Rc02_LIWT_Pred = Predict['R02_ITemp']
Rc02_LWF_Pred = Predict['R02_Flow']
Rc02_SIWT_Pred = Predict['Outdoor_Temperature']


# Catalogue 
Cata05 = pd.read_csv('C:\\System_Air\\Catalogue\\\\TRNSYS_CATA_444.csv')
Cata06 = pd.read_csv('C:\\System_Air\\Catalogue\\\\TRNSYS_CATA_556.csv')
Cata07 = pd.read_csv('C:\\System_Air\\Catalogue\\\\TRNSYS_CATA_667.csv')
Cata08 = pd.read_csv('C:\\System_Air\\Catalogue\\\\TRNSYS_CATA_778.csv')
Cata09 = pd.read_csv('C:\\System_Air\\Catalogue\\\\TRNSYS_CATA_889.csv')
Cata10 = pd.read_csv('C:\\System_Air\\Catalogue\\\\TRNSYS_CATA_100.csv')


'''
# 1st Parameter : Outlet Chilled Water Temperature
# Rc01
Rc01_LOWT_Train_Para= []
for i, (a, b, c, d) in enumerate(zip(Rc01_LOWT, Rc01_LIWT, Rc01_SIWT, Rc01_LWF)):
   LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
   Rc01_LOWT_Train_Para.append(LWOT_Para)

Rc02_LOWT_Train_Para = []
# Rc02
for i, (a, b, c, d) in enumerate(zip(Rc02_LOWT, Rc02_LIWT, Rc02_SIWT, Rc02_LWF)):

   LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
   Rc02_LOWT_Train_Para.append(LWOT_Para)

Rc01_TrainCapa=[]
# Compute Capacity of Chiller
# Rc01
for i, (a, b, c) in enumerate(zip(Rc01_LIWT, Rc01_LOWT, Rc01_LWF)):
   Capacity_cool = CoolingCapa(a, b, c)
   Rc01_TrainCapa.append(Capacity_cool)

Rc02_TrainCapa=[]
# Rc02
for i, (a, b, c) in enumerate(zip(Rc02_LIWT, Rc02_LOWT, Rc02_LWF)):
      Capacity_cool = CoolingCapa(a, b, c)
      Rc02_TrainCapa.append(Capacity_cool)


# Setting Train Data
Train['Rc01_LOWT_Parameter'] = Rc01_LOWT_Train_Para
Train['Rc02_LOWT_Parameter'] = Rc02_LOWT_Train_Para
Train['Rc01_Capacity'] = Rc01_TrainCapa
Train['Rc02_Capacity'] = Rc02_TrainCapa


'--------------------------------------------------'

# 2nd Parameter : COP - Capacity - Source side inlet temperature

# Rc01 - COP Parameters
Rc01_Settemp = Train['R01_SetTemp']
Rc01_InterCOPTrain = []
Rc01_COPParaTrain = []

for i, Rc01_Settemp in enumerate(Rc01_Settemp):
   
   Rc01_Settemp = Train['R01_SetTemp'].loc[i]

   if Rc01_Settemp == 5:
      x_Rc01 = Train['Outdoor_Temperature'].loc[i]
      y_Rc01 = Train['Rc01_Capacity'].loc[i]
      z_Rc01 = Train['R01_COP'].loc[i]

      Cata_SIT = Cata05['Air_Temp']
      Cata_Capa = Cata05['Capacity']
      Cata_COP = Cata05['COP']

      Rc01_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaTrain = COP_para(z_Rc01, Rc01_Inter_COPTrain)

   elif Rc01_Settemp == 6:
      x_Rc01 = Train['Outdoor_Temperature'].loc[i]
      y_Rc01 = Train['Rc01_Capacity'].loc[i]
      z_Rc01 = Train['R01_COP'].loc[i]

      Cata_SIT = Cata06['Air_Temp']
      Cata_Capa = Cata06['Capacity']
      Cata_COP = Cata06['COP']

      Rc01_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaTrain = COP_para(z_Rc01, Rc01_Inter_COPTrain)

   elif Rc01_Settemp == 7:
      x_Rc01 = Train['Outdoor_Temperature'].loc[i]
      y_Rc01 = Train['Rc01_Capacity'].loc[i]
      z_Rc01 = Train['R01_COP'].loc[i]

      Cata_SIT = Cata07['Air_Temp']
      Cata_Capa = Cata07['Capacity']
      Cata_COP = Cata07['COP']

      Rc01_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaTrain = COP_para(z_Rc01, Rc01_Inter_COPTrain)

   elif Rc01_Settemp == 8:
      x_Rc01 = Train['Outdoor_Temperature'].loc[i]
      y_Rc01 = Train['Rc01_Capacity'].loc[i]
      z_Rc01 = Train['R01_COP'].loc[i]

      Cata_SIT = Cata08['Air_Temp']
      Cata_Capa = Cata08['Capacity']
      Cata_COP = Cata08['COP']

      Rc01_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaTrain = COP_para(z_Rc01, Rc01_Inter_COPTrain)

   elif Rc01_Settemp == 9:
      x_Rc01 = Train['Outdoor_Temperature'].loc[i]
      y_Rc01 = Train['Rc01_Capacity'].loc[i]
      z_Rc01 = Train['R01_COP'].loc[i]

      Cata_SIT = Cata09['Air_Temp']
      Cata_Capa = Cata09['Capacity']
      Cata_COP = Cata09['COP']

      Rc01_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaTrain = COP_para(z_Rc01, Rc01_Inter_COPTrain)

   elif Rc01_Settemp == 10:
      x_Rc01 = Train['Outdoor_Temperature'].loc[i]
      y_Rc01 = Train['Rc01_Capacity'].loc[i]
      z_Rc01 = Train['R01_COP'].loc[i]

      Cata_SIT = Cata10['Air_Temp']
      Cata_Capa = Cata10['Capacity']
      Cata_COP = Cata10['COP']

      Rc01_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaTrain = COP_para(z_Rc01, Rc01_Inter_COPTrain)

   Rc01_InterCOPTrain.append(Rc01_Inter_COPTrain)
   Rc01_COPParaTrain.append(Rc01_COP_ParaTrain)


# Rc02 - COP Parameters
Rc02_Settemp = Train['R02_SetTemp']
Rc02_InterCOPTrain = []
Rc02_COPParaTrain = []

for i, Rc02_Settemp in enumerate(Rc02_Settemp):
   
   Rc02_Settemp = Train['R02_SetTemp'].loc[i]

   if Rc02_Settemp == 5:
      x_Rc02 = Train['Outdoor_Temperature'].loc[i]
      y_Rc02 = Train['Rc02_Capacity'].loc[i]
      z_Rc02 = Train['R02_COP'].loc[i]

      Cata_SIT = Cata05['Air_Temp']
      Cata_Capa = Cata05['Capacity']
      Cata_COP = Cata05['COP']

      Rc02_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaTrain = COP_para(z_Rc02, Rc02_Inter_COPTrain)

   elif Rc02_Settemp == 6:
      x_Rc02 = Train['Outdoor_Temperature'].loc[i]
      y_Rc02 = Train['Rc02_Capacity'].loc[i]
      z_Rc02 = Train['R02_COP'].loc[i]

      Cata_SIT = Cata06['Air_Temp']
      Cata_Capa = Cata06['Capacity']
      Cata_COP = Cata06['COP']

      Rc02_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaTrain = COP_para(z_Rc02, Rc02_Inter_COPTrain)

   elif Rc02_Settemp == 7:
      x_Rc02 = Train['Outdoor_Temperature'].loc[i]
      y_Rc02 = Train['Rc02_Capacity'].loc[i]
      z_Rc02 = Train['R02_COP'].loc[i]

      Cata_SIT = Cata07['Air_Temp']
      Cata_Capa = Cata07['Capacity']
      Cata_COP = Cata07['COP']

      Rc02_Inter_COP = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaTrain = COP_para(z_Rc02, Rc02_Inter_COPTrain)

   elif Rc02_Settemp == 8:
      x_Rc02 = Train['Outdoor_Temperature'].loc[i]
      y_Rc02 = Train['Rc02_Capacity'].loc[i]
      z_Rc02 = Train['R02_COP'].loc[i]

      Cata_SIT = Cata08['Air_Temp']
      Cata_Capa = Cata08['Capacity']
      Cata_COP = Cata08['COP']

      Rc02_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaTrain = COP_para(z_Rc02, Rc02_Inter_COPTrain)

   elif Rc02_Settemp == 9:
      x_Rc02 = Train['Outdoor_Temperature'].loc[i]
      y_Rc02 = Train['Rc02_Capacity'].loc[i]
      z_Rc02 = Train['R02_COP'].loc[i]

      Cata_SIT = Cata09['Air_Temp']
      Cata_Capa = Cata09['Capacity']
      Cata_COP = Cata09['COP']

      Rc02_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaTrain = COP_para(z_Rc02, Rc02_Inter_COPTrain)

   elif Rc02_Settemp == 10:
      x_Rc02 = Train['Outdoor_Temperature'].loc[i]
      y_Rc02 = Train['Rc02_Capacity'].loc[i]
      z_Rc02 = Train['R02_COP'].loc[i]

      Cata_SIT = Cata10['Air_Temp']
      Cata_Capa = Cata10['Capacity']
      Cata_COP = Cata10['COP']

      Rc02_Inter_COPTrain = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaTrain = COP_para(z_Rc02, Rc02_Inter_COPTrain)

   Rc02_InterCOPTrain.append(Rc02_Inter_COPTrain)
   Rc02_COPParaTrain.append(Rc02_COP_ParaTrain)
      

Train['Rc01_COP_Parameter'] = Rc01_COPParaTrain
Train['Rc02_COP_Parameter'] = Rc02_COPParaTrain
Train['Rc01_Inter_COP'] = Rc01_InterCOPTrain
Train['Rc02_Inter_COP'] = Rc02_InterCOPTrain

'--------------------------------------------------'

# Saved the Parameter to Train CSV
submission = pd.read_csv('c:\\System_Air\\Input\\Train\\TRNSYS_2016.csv')
submission["Rc01_Capacity"] = Rc01_TrainCapa
submission["Rc02_Capacity"] = Rc02_TrainCapa
submission["Rc01_LOWT_Parameter"] = Rc01_LOWT_Train_Para
submission["Rc02_LOWT_Parameter"] = Rc02_LOWT_Train_Para
submission["Rc01_Inter_COP"] = Rc01_InterCOPTrain
submission["Rc02_Inter_COP"] = Rc02_InterCOPTrain
submission["Rc01_COP_Parameter"] = Rc01_COPParaTrain
submission["Rc02_COP_Parameter"] = Rc02_COPParaTrain
submission.to_csv('c:\\System_Air\\Input\\Train\\Chiller_Parameters.csv', index = False)

'--------------------------------------------------'

# Calculate Parameter - Predict

# 1st Parameter : Outlet Chilled Water Temperature
# Rc01
for i, (a, b, c, d) in enumerate(zip(Rc01_LOWT_Pred, Rc01_LIWT_Pred, Rc01_SIWT_Pred, Rc01_LWF_Pred)):
   if i == 0:
      LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
      Rc01_LOWT_Pred_Para = np.array(LWOT_Para)
   else:
      LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
      Rc01_LOWT_Pred_Para = np.append(Rc01_LOWT_Pred_Para, LWOT_Para)

# Rc02
for i, (a, b, c, d) in enumerate(zip(Rc02_LOWT_Pred, Rc02_LIWT_Pred, Rc02_SIWT_Pred, Rc02_LWF_Pred)):
   if i == 0:
      LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
      Rc02_LOWT_Pred_Para = np.array(LWOT_Para)
   else:
      LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
      Rc02_LOWT_Pred_Para = np.append(Rc02_LOWT_Pred_Para, LWOT_Para)


# Compute Capacity of Chiller
# Rc01
for i, (a, b, c) in enumerate(zip(Rc01_LIWT_Pred, Rc01_LOWT_Pred, Rc01_LWF_Pred)):
   if i == 0:
      Capacity_cool = CoolingCapa(a, b, c)
      Rc01_PredCapa = np.array(Capacity_cool)
   else:
      Capacity_cool = CoolingCapa(a, b, c)
      Rc01_PredCapa = np.append(Rc01_PredCapa, Capacity_cool)

# Rc02
for i, (a, b, c) in enumerate(zip(Rc02_LIWT_Pred, Rc02_LOWT_Pred, Rc02_LWF_Pred)):
   if i == 0:
      Capacity_cool = CoolingCapa(a, b, c)
      Rc02_PredCapa = np.array(Capacity_cool)
   else:
      Capacity_cool = CoolingCapa(a, b, c)
      Rc02_PredCapa = np.append(Rc02_PredCapa, Capacity_cool)


'--------------------------------------------------'

# Setting Train Data
Predict['Rc01_LOWT_Parameter'] = Rc01_LOWT_Pred_Para
Predict['Rc02_LOWT_Parameter'] = Rc02_LOWT_Pred_Para
Predict['Rc01_Capacity'] = Rc02_PredCapa
Predict['Rc02_Capacity'] = Rc02_PredCapa

'--------------------------------------------------'

# 2nd Parameter : COP - Capacity - Source side inlet temperature

# Input Variable from Predict Data
# Rc01
Rc01_Settemp_Predict = Predict['R01_SetTemp']
Rc01_InterCOPPred = []
Rc01_COPParaPred = []

# Interpolate COP of Predict period
for i, Rc01_Settemp_Predict in enumerate(Rc01_Settemp_Predict):
   
   Rc01_Settemp_Predict = Predict['R01_SetTemp'].loc[i]

   if Rc01_Settemp_Predict == 5:
      x_Rc01 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc01 = Predict['Rc01_Capacity'].loc[i]
      z_Rc01 = Predict['R01_COP'].loc[i]

      Cata_SIT = Cata05['Air_Temp']
      Cata_Capa = Cata05['Capacity']
      Cata_COP = Cata05['COP']

      Rc01_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaPred = COP_para(z_Rc01, Rc01_Inter_COPPred)

   elif Rc01_Settemp_Predict == 6:
      x_Rc01 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc01 = Predict['Rc01_Capacity'].loc[i]
      z_Rc01 = Predict['R01_COP'].loc[i]

      Cata_SIT = Cata06['Air_Temp']
      Cata_Capa = Cata06['Capacity']
      Cata_COP = Cata06['COP']

      Rc01_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaPred = COP_para(z_Rc01, Rc01_Inter_COPPred)

   elif Rc01_Settemp_Predict == 7:
      x_Rc01 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc01 = Predict['Rc01_Capacity'].loc[i]
      z_Rc01 = Predict['R01_COP'].loc[i]

      Cata_SIT = Cata07['Air_Temp']
      Cata_Capa = Cata07['Capacity']
      Cata_COP = Cata07['COP']

      Rc01_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaPred = COP_para(z_Rc01, Rc01_Inter_COPPred)

   elif Rc01_Settemp_Predict == 8:
      x_Rc01 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc01 = Predict['Rc01_Capacity'].loc[i]
      z_Rc01 = Predict['R01_COP'].loc[i]

      Cata_SIT = Cata08['Air_Temp']
      Cata_Capa = Cata08['Capacity']
      Cata_COP = Cata08['COP']

      Rc01_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaPred = COP_para(z_Rc01, Rc01_Inter_COPPred)

   elif Rc01_Settemp_Predict == 9:
      x_Rc01 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc01 = Predict['Rc01_Capacity'].loc[i]
      z_Rc01 = Predict['R01_COP'].loc[i]

      Cata_SIT = Cata09['Air_Temp']
      Cata_Capa = Cata09['Capacity']
      Cata_COP = Cata09['COP']

      Rc01_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaPred = COP_para(z_Rc01, Rc01_Inter_COPPred)

   elif Rc01_Settemp_Predict == 10:
      x_Rc01 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc01 = Predict['Rc01_Capacity'].loc[i]
      z_Rc01 = Predict['R01_COP'].loc[i]

      Cata_SIT = Cata10['Air_Temp']
      Cata_Capa = Cata10['Capacity']
      Cata_COP = Cata10['COP']

      Rc01_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc01, y_Rc01)
      Rc01_COP_ParaPred = COP_para(z_Rc01, Rc01_Inter_COPPred)

   Rc01_InterCOPPred.append(Rc01_Inter_COPPred)
   Rc01_COPParaPred.append(Rc01_COP_ParaPred)

print(Rc01_InterCOPPred, Rc01_COPParaPred) 


# Rc02
Rc02_Settemp_Predict = Predict['R02_SetTemp']
Rc02_InterCOPPred = []
Rc02_COPParaPred = []

# Interpolate COP of Predict period
for i, Rc02_Settemp_Predict in enumerate(Rc02_Settemp_Predict):
   
   Rc02_Settemp_Predict = Predict['R02_SetTemp'].loc[i]

   if Rc02_Settemp_Predict == 5:
      x_Rc02 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc02 = Predict['Rc02_Capacity'].loc[i]
      z_Rc02 = Predict['R02_COP'].loc[i]

      Cata_SIT = Cata05['Air_Temp']
      Cata_Capa = Cata05['Capacity']
      Cata_COP = Cata05['COP']

      Rc02_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaPred = COP_para(z_Rc02, Rc02_Inter_COPPred)

   elif Rc02_Settemp_Predict == 6:
      x_Rc02 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc02 = Predict['Rc02_Capacity'].loc[i]
      z_Rc02 = Predict['R02_COP'].loc[i]

      Cata_SIT = Cata06['Air_Temp']
      Cata_Capa = Cata06['Capacity']
      Cata_COP = Cata06['COP']

      Rc02_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaPred = COP_para(z_Rc02, Rc02_Inter_COPPred)

   elif Rc02_Settemp_Predict == 7:
      x_Rc02 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc02 = Predict['Rc02_Capacity'].loc[i]
      z_Rc02 = Predict['R02_COP'].loc[i]

      Cata_SIT = Cata07['Air_Temp']
      Cata_Capa = Cata07['Capacity']
      Cata_COP = Cata07['COP']

      Rc02_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaPred = COP_para(z_Rc02, Rc02_Inter_COPPred)

   elif Rc02_Settemp_Predict == 8:
      x_Rc02 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc02 = Predict['Rc02_Capacity'].loc[i]
      z_Rc02 = Predict['R02_COP'].loc[i]

      Cata_SIT = Cata08['Air_Temp']
      Cata_Capa = Cata08['Capacity']
      Cata_COP = Cata08['COP']

      Rc02_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaPred = COP_para(z_Rc02, Rc02_Inter_COPPred)

   elif Rc02_Settemp_Predict == 9:
      x_Rc02 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc02 = Predict['Rc02_Capacity'].loc[i]
      z_Rc02 = Predict['R02_COP'].loc[i]

      Cata_SIT = Cata09['Air_Temp']
      Cata_Capa = Cata09['Capacity']
      Cata_COP = Cata09['COP']

      Rc02_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaPred = COP_para(z_Rc02, Rc02_Inter_COPPred)

   elif Rc02_Settemp_Predict == 10:
      x_Rc02 = Predict['Outdoor_Temperature'].loc[i]
      y_Rc02 = Predict['Rc02_Capacity'].loc[i]
      z_Rc02 = Predict['R02_COP'].loc[i]

      Cata_SIT = Cata10['Air_Temp']
      Cata_Capa = Cata10['Capacity']
      Cata_COP = Cata10['COP']

      Rc02_Inter_COPPred = InterCOP(Coef_C, Cata_SIT, Cata_Capa, Cata_COP, x_Rc02, y_Rc02)
      Rc02_COP_ParaPred = COP_para(z_Rc02, Rc02_Inter_COPPred)

   Rc02_InterCOPPred.append(Rc02_Inter_COPPred)
   Rc02_COPParaPred.append(Rc02_COP_ParaPred)


'--------------------------------------------------'

# Saved the Parameter to Predict CSV
submission = pd.read_csv('c:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
submission["Rc01_Capacity"] = Rc01_PredCapa
submission["Rc02_Capacity"] = Rc02_PredCapa
submission["Rc01_LOWT_Parameter"] = Rc01_LOWT_Pred_Para
submission["Rc02_LOWT_Parameter"] = Rc02_LOWT_Pred_Para
submission["Rc01_Inter_COP"] = Rc01_InterCOPPred
submission["Rc02_Inter_COP"] = Rc02_InterCOPPred
submission["Rc01_COP_Parameter"] = Rc01_COPParaPred
submission["Rc02_COP_Parameter"] = Rc02_COPParaPred
submission.to_csv('c:\\System_Air\\Input\\Test\\Chiller_Parameters.csv', index = False)
'''


# ================= Training Step ===================== #

input_shape = 3
output_shape = 1

# KFold setting
kfold=5
num_val_samples = len(Train) // kfold

'--------------------------------------------------'

# Equipment No.01
Inputvari01 = ['R01_ITemp', 'Outdoor_Temperature', 'R01_Flow']
Rc01_x = Train[Inputvari01]
Rc01_x = np.array(Rc01_x).astype('float16')

# Outlet Temp. parameter
Rc01_ytrain_LOWT = Train['Rc01_LOWT_Parameter']
Rc01_ytrain_LOWT = np.array(Rc01_ytrain_LOWT).astype('float16')


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

   val_data_01 = Rc01_x[i*num_val_samples: (i+1)*num_val_samples]
   val_target_01 = Rc01_ytrain_LOWT[i*num_val_samples: (i+1)*num_val_samples]

   partial_train_data_01 = np.concatenate([Rc01_x[:i*num_val_samples], Rc01_x[(i+1)*num_val_samples:]], axis=0)
   partial_train_targets_01 = np.concatenate([Rc01_ytrain_LOWT[:i*num_val_samples], Rc01_ytrain_LOWT[(i+1)*num_val_samples:]], axis=0)
    
   def keras_model_R01_para01():
      model = tf.keras.models.Sequential()  
      model.add(Dense(300, input_dim = input_shape, kernel_initializer = 'random_uniform', activation = 'relu'))
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
   mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para01.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

   best_model_y01 = keras_model_R01_para01()
   best_model_y01.fit(partial_train_data_01, partial_train_targets_01, epochs=30, batch_size=9000)
   best_model_y01.save('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para01.h5')
   val_mse_01, val_mae_01 = best_model_y01.evaluate(val_data_01, val_target_01, batch_size=2500, callbacks=[es], verbose=1)
   all_cv_score_01.append(val_mae_01)


x_predict_R01_Para01 = Predict[Inputvari01]
x_predict_R01_Para01 = np.array(x_predict_R01_Para01)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para01.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
R01_para01_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para01.h5')
history01 = R01_para01_Model.fit(Rc01_x, Rc01_ytrain_LOWT, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


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
plt.savefig('R01_para01.png')

R01_Para01_Pred = R01_para01_Model.predict(x_predict_R01_Para01)
R01_Para01_Pred = R01_Para01_Pred.reshape(122400,)


'--------------------------------------------------'

# COP parameter
Rc01_ytrain_COP = Train['Rc01_COP_Parameter']
Rc01_ytrain_COP = np.array(Rc01_ytrain_COP).astype('float16')


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

   val_data_02 = Rc01_x[i*num_val_samples: (i+1)*num_val_samples]
   val_target_02 = Rc01_ytrain_COP[i*num_val_samples: (i+1)*num_val_samples]

   partial_train_data_02 = np.concatenate([Rc01_x[:i*num_val_samples], Rc01_x[(i+1)*num_val_samples:]], axis=0)
   partial_train_targets_02 = np.concatenate([Rc01_ytrain_COP[:i*num_val_samples], Rc01_ytrain_COP[(i+1)*num_val_samples:]], axis=0)
    
   def keras_model_R01_para02():
      model = tf.keras.models.Sequential()  
      model.add(Dense(300, input_dim = input_shape, kernel_initializer = 'random_uniform', activation = 'relu'))
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
   mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para02.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

   best_model_y02 = keras_model_R01_para02()
   best_model_y02.fit(partial_train_data_02, partial_train_targets_02, epochs=30, batch_size=9000)
   best_model_y02.save('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para02.h5')
   val_mse_02, val_mae_02 = best_model_y02.evaluate(val_data_02, val_target_02, batch_size=2500, callbacks=[es], verbose=1)
   all_cv_score_02.append(val_mae_02)


x_predict_R01_para02 = Predict[Inputvari01]
x_predict_R01_para02 = np.array(x_predict_R01_para02)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para02.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
R01_para02_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para02.h5')
history02 = R01_para02_Model.fit(Rc01_x, Rc01_ytrain_COP, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


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
plt.savefig('R01_para02.png')


R01_para02_Pred = R01_para02_Model.predict(x_predict_R01_para02)
R01_para02_Pred = R01_para02_Pred.reshape(122400,)


'--------------------------------------------------'

# Equipment No.02
Inputvari02 = ['R02_ITemp', 'Outdoor_Temperature', 'R02_Flow']
Rc02_x = Train[Inputvari02]
Rc02_x = np.array(Rc02_x).astype('float16')

Rc02_ytrain_LOWT = Train['Rc02_LOWT_Parameter']
Rc02_ytrain_LOWT = np.array(Rc02_ytrain_LOWT).astype('float16')


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

   val_data_03 = Rc02_x[i*num_val_samples: (i+1)*num_val_samples]
   val_target_03 = Rc02_ytrain_LOWT[i*num_val_samples: (i+1)*num_val_samples]

   partial_train_data_03 = np.concatenate([Rc02_x[:i*num_val_samples], Rc02_x[(i+1)*num_val_samples:]], axis=0)
   partial_train_targets_03 = np.concatenate([Rc02_ytrain_LOWT[:i*num_val_samples], Rc02_ytrain_LOWT[(i+1)*num_val_samples:]], axis=0)
   
   def keras_model_R02_para01():
      model = tf.keras.models.Sequential()  
      model.add(Dense(300, input_dim = input_shape, kernel_initializer = 'random_uniform', activation = 'relu'))
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
   mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para01.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

   best_model_y03 = keras_model_R02_para01()
   best_model_y03.fit(partial_train_data_03, partial_train_targets_03, epochs=30, batch_size=9000)
   best_model_y03.save('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para01.h5')
   val_mse_03, val_mae_03 = best_model_y03.evaluate(val_data_03, val_target_03, batch_size=2500, callbacks=[es], verbose=1)
   all_cv_score_03.append(val_mae_03)


x_predict_R02_Para01 = Predict[Inputvari02]
x_predict_R02_Para01 = np.array(x_predict_R02_Para01)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para01.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
R02_para01_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para01.h5')
history03 = R02_para01_Model.fit(Rc02_x, Rc02_ytrain_LOWT, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


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
plt.xlabel('ephistory03och')
plt.ylabel('accuracy')
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2, fontsize=10)
plt.tight_layout()
plt.savefig('R02_para01.png')


R02_Para01_Pred = R02_para01_Model.predict(x_predict_R02_Para01)
R02_Para01_Pred = R02_Para01_Pred.reshape(122400,)


'--------------------------------------------------'

# COP parameter
Rc02_ytrain_COP = Train['Rc02_COP_Parameter']
Rc02_ytrain_COP = np.array(Rc02_ytrain_COP).astype('float16')


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

   val_data_04 = Rc02_x[i*num_val_samples: (i+1)*num_val_samples]
   val_target_04 = Rc02_ytrain_COP[i*num_val_samples: (i+1)*num_val_samples]

   partial_train_data_04 = np.concatenate([Rc02_x[:i*num_val_samples], Rc02_x[(i+1)*num_val_samples:]], axis=0)
   partial_train_targets_04 = np.concatenate([Rc02_ytrain_COP[:i*num_val_samples], Rc02_ytrain_COP[(i+1)*num_val_samples:]], axis=0)
    
   def keras_model_R02_para02():
      model = tf.keras.models.Sequential()  
      model.add(Dense(300, input_dim = input_shape, kernel_initializer = 'random_uniform', activation = 'relu'))
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
   mc = ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para02.h5', 
                            monitor='val_loss', mode='min', verbose=1, save_best_only=True)

   best_model_y04 = keras_model_R02_para02()
   best_model_y04.fit(partial_train_data_04, partial_train_targets_04, epochs=30, batch_size=9000)
   best_model_y04.save('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para02.h5')
   val_mse_04, val_mae_04 = best_model_y04.evaluate(val_data_04, val_target_04, batch_size=2500, callbacks=[es], verbose=1)
   all_cv_score_04.append(val_mae_04)


x_predict_R02_para02 = Predict[Inputvari02]
x_predict_R02_para02 = np.array(x_predict_R02_para02)


#### best Fitting Model ####
es=EarlyStopping(monitor='val_loss', patience=30, mode= 'min', min_delta=0.001, verbose=1)
mc=ModelCheckpoint('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para02.h5', 
                        monitor='val_loss', mode='min', verbose=1, save_best_only = True)
R02_para02_Model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para02.h5')
history04 = R02_para02_Model.fit(Rc02_x, Rc02_ytrain_COP, epochs=1000, batch_size=12240, validation_split=0.2, verbose=1, callbacks=[es, mc])


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
plt.savefig('R02_para02.png')


R02_para02_Pred = R02_para02_Model.predict(x_predict_R02_para02)
R02_para02_Pred = R02_para02_Pred.reshape(122400,)


# ================ Reverse Calculate ================= #

# Predict Data Read
Rc01_InterCOPPred = Predict['Rc01_Inter_COP']
Rc01_InterCOPPred = np.array(Rc01_InterCOPPred).astype('float16')

Rc02_InterCOPPred = Predict['Rc02_Inter_COP']
Rc02_InterCOPPred = np.array(Rc02_InterCOPPred).astype('float16')


# Predict Chilled Water Outlet Temperature
# Rc 01
Rc01_LOWT_Cal = []
for i, (a, b, c, d) in enumerate(zip(Rc01_LIWT_Pred, R01_Para01_Pred, Rc01_SIWT_Pred, Rc01_LWF_Pred)):
   if i == 0:
      if d == 0:
         OLWT_Cal = a
      else:
         OLWT_Cal =  a + (1 / (c * d * 1.162)) + ((1 / (a * c * d * 1.162)) * -10000 * b)
   else:
      if d == 0:
         OLWT_Cal = a
      else:
         OLWT_Cal =  a + (1 / (c * d * 1.162)) + ((1 / (a * c * d * 1.162)) * -10000 * b)

   Rc01_LOWT_Cal.append(OLWT_Cal)


# Rc02
Rc02_LOWT_Cal = []
for i, (a, b, c, d) in enumerate(zip(Rc02_LIWT_Pred, R02_Para01_Pred, Rc02_SIWT_Pred, Rc02_LWF_Pred)):
   if i == 0:
      if d == 0:
         OLWT_Cal = a
      else:
         OLWT_Cal =  a + (1 / (c * d * 1.162)) + ((1 / (a * c * d * 1.162)) * (-10000 * b))
   else:
      if d == 0:
         OLWT_Cal = a
      else:
         OLWT_Cal =  a + (1 / (c * d * 1.162)) + ((1 / (a * c * d * 1.162)) * (-10000 * b))

   Rc02_LOWT_Cal.append(OLWT_Cal)


# Predict Cooling Capacity
# Rc01
Rc01_Capa_Cal = []
for i, (a, b, c) in enumerate(zip(Rc01_LIWT_Pred, Rc01_LOWT_Cal, Rc01_LWF_Pred)):

   Capacity_cool = CoolingCapa(a, b, c)
   Rc01_Capa_Cal.append(Capacity_cool)

# Rc02
Rc02_Capa_Cal = []
for i, (a, b, c) in enumerate(zip(Rc02_LIWT_Pred, Rc02_LOWT_Cal, Rc02_LWF_Pred)):

   Capacity_cool = CoolingCapa(a, b, c)
   Rc02_Capa_Cal.append(Capacity_cool)


# Predict COP
# Rc01
Rc01_COP_Preict =[]
for i, (a, b) in enumerate(zip(R01_para02_Pred, Rc01_InterCOPPred)):
   if i == 0:
      COP = b * (a / 10)
   else:
      COP = b * (a / 10)

   Rc01_COP_Preict.append(COP)


# Rc02
Rc02_COP_Preict = []
for i, (a, b) in enumerate(zip(R02_para02_Pred, Rc02_InterCOPPred)):
   if i == 0:
      COP = b * (a / 10)
   else:
      COP = b * (a / 10)

   Rc02_COP_Preict.append(COP)


# Predict Power
# Rc01
Rc01_Power_Pred = []
for i, (a, b) in enumerate(zip(Rc01_Capa_Cal, Rc01_COP_Preict)):
   if b == 0:
      Power = 0
   else:
      Power = a / b
      
   Rc01_Power_Pred.append(Power)
 

# Rc02
Rc02_Power_Pred = []
for i, (a, b) in enumerate(zip(Rc02_Capa_Cal, Rc02_COP_Preict)):
   if b == 0:
      Power = 0
   else:
      Power = a / b
      
   Rc02_Power_Pred.append(Power)


# ================ Save the Result =================== #

submission = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')
submission["Rc01_LOWTPara_Pred"] = R01_Para01_Pred
submission["Rc01_COPPara_Pred"] = R01_para02_Pred
submission["Rc02_LOWTPara_Pred"] = R02_Para01_Pred
submission["Rc02_COPPara_Pred"] = R02_para02_Pred
submission["Rc01_LOWT_Pred"] = Rc01_LOWT_Cal
submission["Rc02_LOWT_Pred"] = Rc02_LOWT_Cal
submission["Rc01_Capacity_Pred"] = Rc01_Capa_Cal
submission["Rc02_Capacity_Pred"] = Rc02_Capa_Cal
submission["Rc01_COP_Pred"] = Rc01_COP_Preict
submission["Rc02_COP_Pred"] = Rc02_COP_Preict
submission["Rc01_Power_Pred"] = Rc01_Power_Pred
submission["Rc02_Power_Pred"] = Rc02_Power_Pred
submission.to_csv('refrigerator_prediction.csv', index = False)


Rc01_para01_raw = Predict['Rc01_LOWT_Parameter']
Rc01_para02_raw = Predict['Rc01_COP_Parameter']
Rc02_para01_raw = Predict['Rc02_LOWT_Parameter']
Rc02_para02_raw = Predict['Rc02_COP_Parameter']


# Function
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

def MAE(x, y):
    return mean_absolute_error(x, y)

# Evaluate
loss_y01, acc_y01 = R01_para01_Model.evaluate(x_predict_R01_Para01, R01_Para01_Pred, batch_size=128)
loss_y02, acc_y02 = R01_para02_Model.evaluate(x_predict_R01_para02, R01_para02_Pred, batch_size=128)
loss_y03, acc_y03 = R02_para01_Model.evaluate(x_predict_R02_Para01, R02_Para01_Pred, batch_size=128)
loss_y04, acc_y04 = R02_para02_Model.evaluate(x_predict_R02_para02, R02_para02_Pred, batch_size=128)

Result_R2_y01 = r2_score(Rc01_para01_raw, R01_Para01_Pred)
Result_R2_y02 = r2_score(Rc01_para02_raw, R01_para02_Pred)
Result_R2_y03 = r2_score(Rc02_para01_raw, R02_Para01_Pred)
Result_R2_y04 = r2_score(Rc02_para02_raw, R02_para02_Pred)

Result_RMSE_y01 = RMSE(Rc01_para01_raw, R01_Para01_Pred)
Result_RMSE_y02 = RMSE(Rc01_para02_raw, R01_para02_Pred)
Result_RMSE_y03 = RMSE(Rc02_para01_raw, R02_Para01_Pred)
Result_RMSE_y04 = RMSE(Rc02_para02_raw, R02_para02_Pred)

Result_MAE_y01 = MAE(Rc01_para01_raw, R01_Para01_Pred)
Result_MAE_y02 = MAE(Rc01_para02_raw, R01_para02_Pred)
Result_MAE_y03 = MAE(Rc02_para01_raw, R02_Para01_Pred)
Result_MAE_y04 = MAE(Rc02_para02_raw, R02_para02_Pred)

Result_MBE_y01 = np.mean(Rc01_para01_raw-R01_Para01_Pred)
Result_MBE_y02 = np.mean(Rc01_para02_raw-R01_para02_Pred)
Result_MBE_y03 = np.mean(Rc02_para01_raw-R02_Para01_Pred)
Result_MBE_y04 = np.mean(Rc02_para02_raw-R02_para02_Pred)


# Save the File
with open('Result_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Rc01_LOWT_Parameter', loss_y01, acc_y01, Result_R2_y01, Result_RMSE_y01, Result_MAE_y01, Result_MBE_y01])
    writer.writerow(['Rc01_COP_Parameter', loss_y02, acc_y02, Result_R2_y02, Result_RMSE_y02, Result_MAE_y01, Result_MBE_y02])
    writer.writerow(['Rc02_LOWT_Parameter', loss_y03, acc_y03, Result_R2_y03, Result_RMSE_y03, Result_MAE_y01, Result_MBE_y03])
    writer.writerow(['Rc02_COP_Parameter', loss_y04, acc_y04, Result_R2_y04, Result_RMSE_y04, Result_MAE_y01, Result_MBE_y04])


## Performance
Rc01_Power_raw = Predict['R01_Power']
Rc02_Power_raw = Predict['R02_Power']
Rc01_COP_raw = Predict['R01_COP']
Rc02_COP_raw = Predict['R02_COP']

ResultPerfor_R2_y01 = r2_score(Rc01_Power_raw, Rc01_COP_Preict)
ResultPerfor_R2_y02 = r2_score(Rc02_Power_raw, Rc02_COP_Preict)
ResultPerfor_R2_y03 = r2_score(Rc01_COP_raw, Rc01_Power_Pred)
ResultPerfor_R2_y04 = r2_score(Rc02_COP_raw, Rc02_Power_Pred)

ResultPerfor_RMSE_y01 = RMSE(Rc01_Power_raw, Rc01_COP_Preict)
ResultPerfor_RMSE_y02 = RMSE(Rc02_Power_raw, Rc02_COP_Preict)
ResultPerfor_RMSE_y03 = RMSE(Rc01_COP_raw, Rc01_Power_Pred)
ResultPerfor_RMSE_y04 = RMSE(Rc02_COP_raw, Rc02_Power_Pred)

ResultPerfor_MAE_y01 = MAE(Rc01_Power_raw, Rc01_COP_Preict)
ResultPerfor_MAE_y02 = MAE(Rc02_Power_raw, Rc02_COP_Preict)
ResultPerfor_MAE_y03 = MAE(Rc01_COP_raw, Rc01_Power_Pred)
ResultPerfor_MAE_y04 = MAE(Rc02_COP_raw, Rc02_Power_Pred)

ResultPerfor_MBE_y01 = np.mean(Rc01_Power_raw-Rc01_COP_Preict)
ResultPerfor_MBE_y02 = np.mean(Rc02_Power_raw-Rc02_COP_Preict)
ResultPerfor_MBE_y03 = np.mean(Rc01_COP_raw-Rc01_Power_Pred)
ResultPerfor_MBE_y04 = np.mean(Rc02_COP_raw-Rc02_Power_Pred)


# Save the File
with open('ResultPerfor_Estimation.csv', 'a', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['R01_Power', ResultPerfor_R2_y01, ResultPerfor_RMSE_y01, ResultPerfor_MAE_y01, ResultPerfor_MBE_y01])
    writer.writerow(['R02_Power', ResultPerfor_R2_y02, ResultPerfor_RMSE_y02, ResultPerfor_MAE_y02, ResultPerfor_MBE_y02])
    writer.writerow(['R01_COP', ResultPerfor_R2_y03, ResultPerfor_RMSE_y03, ResultPerfor_MAE_y03, ResultPerfor_MBE_y03])
    writer.writerow(['R02_COP', ResultPerfor_R2_y04, ResultPerfor_RMSE_y04, ResultPerfor_MAE_y04, ResultPerfor_MBE_y04])


'==============================================================================================================================='
