

# confirm the accuracy of ANN model : hyper-parameter


import warnings
warnings.filterwarnings('ignore')

import sys
import csv
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import time

import os
import random as rn
import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers.normalization import BatchNormalization

from sklearn.metrics import mean_squared_error
from math import sqrt

'----------------------------------------------------------------------------------------------------'

# read data

train = pd.read_csv('C:\Chillermodel\Preprocessing\Train\TR01_train.csv')
test = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')

# all same

BEMS_COP = test['TR01_COP']
Ytrain = train['TR01_COP']

'----------------------------------------------------------------------------------------------------'

'===================================='

# case 01 : 冷却水入口温度, 負荷率
## Variable 02

'===================================='

C01_start = time.time()

case01 = ['TR01_ISWT', 'TR01_PLR']       # 冷却水入口温度, 負荷率

Xtrain_01 = train[case01]
Xtest_01 = test[case01]


# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case01():
    model = Sequential()
    model.add(Dense(200, input_dim = 2, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C01 = Case01()
CO1R = C01.fit(Xtrain_01, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C01_COP = C01.predict(Xtest_01)

C01_RMSE = sqrt(mean_squared_error(BEMS_COP, C01_COP))

C01_End = time.time() - C01_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C01_COP
submission["RMSE"] = C01_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case01.csv', index = False)

'-----------------------------------------------------------------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C1_loss = CO1R.history['loss']
C1_val_loss = CO1R.history['val_loss']

C1_epochs = range(1, len(C1_loss) + 1)

plt.figure()
plt.plot(C1_epochs, C1_loss, 'g', label='Training loss')
plt.plot(C1_epochs, C1_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case01')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case01.png')

'-----------------------------------------------------------------------------------------------------------------------------'

'===================================='

# case 02 : 冷却水入口温度, 負荷率, 冷却水流量
## Variable 03

'===================================='

C02_start = time.time()

case02 = ['TR01_ISWT', 'TR01_PLR', 'TR01_SWF']     #　冷却水入口温度, 負荷率, 冷却水流量

Xtrain_02 = train[case02]
Xtest_02 = test[case02]

# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case02():
    model = Sequential()
    model.add(Dense(200, input_dim = 3, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C02 = Case02()
CO2R = C02.fit(Xtrain_02, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C02_COP = C02.predict(Xtest_02)

C02_RMSE = sqrt(mean_squared_error(BEMS_COP, C02_COP))

C02_End = time.time() - C02_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C02_COP
submission["RMSE"] = C02_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case02.csv', index = False)


'-------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C2_loss = CO2R.history['loss']
C2_val_loss = CO2R.history['val_loss']

C2_epochs = range(1, len(C2_loss) + 1)

plt.figure()
plt.plot(C2_epochs, C2_loss, 'g', label='Training loss')
plt.plot(C2_epochs, C2_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case02')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case02.png')

'-------------------------------------------------------------------'

'===================================='

# case 03 : 冷却水入口温度, 負荷率, 冷水入口温度
## Variable 03

'===================================='

C03_start = time.time()

case03 = ['TR01_ISWT', 'TR01_PLR', 'TR01_ILWT']     # 冷却水入口温度, 負荷率, 冷水入口温度

Xtrain_03 = train[case03]
Xtest_03 = test[case03]

# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case03():
    model = Sequential()
    model.add(Dense(200, input_dim = 3, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C03 = Case03()
CO3R = C03.fit(Xtrain_03, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C03_COP = C03.predict(Xtest_03)

C03_RMSE = sqrt(mean_squared_error(BEMS_COP, C03_COP))

C03_End = time.time() - C03_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C03_COP
submission["RMSE"] = C03_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case03.csv', index = False)

'-----------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C3_loss = CO3R.history['loss']
C3_val_loss = CO3R.history['val_loss']

C3_epochs = range(1, len(C3_loss) + 1)

plt.figure()
plt.plot(C3_epochs, C3_loss, 'g', label='Training loss')
plt.plot(C3_epochs, C3_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case03')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case03.png')

'-----------------------------------------------------------------------'

'===================================='

# case 04 :  冷却水入口温度, 負荷率, 冷水流量
## Variable 03

'===================================='

C04_start = time.time()

case04 = ['TR01_ISWT', 'TR01_PLR', 'TR01_LWF']       #  冷却水入口温度, 負荷率, 冷水流量

Xtrain_04 = train[case04]
Xtest_04 = test[case04]

# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case04():
    model = Sequential()
    model.add(Dense(200, input_dim = 3, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C04 = Case04()
CO4R = C04.fit(Xtrain_04, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C04_COP = C04.predict(Xtest_04)

C04_RMSE = sqrt(mean_squared_error(BEMS_COP, C04_COP))

C04_End = time.time() - C04_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C04_COP
submission["RMSE"] = C04_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case04.csv', index = False)

'-----------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C4_loss = CO4R.history['loss']
C4_val_loss = CO4R.history['val_loss']

C4_epochs = range(1, len(C4_loss) + 1)

plt.figure()
plt.plot(C4_epochs, C4_loss, 'g', label='Training loss')
plt.plot(C4_epochs, C4_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case04')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case04.png')

'-----------------------------------------------------------------------'

### ====================================================================================================================================

'===================================='

# case 05 : 冷却水入口温度, 負荷率, 冷却水流量, 冷水入口温度
## Variable 04

'===================================='

C05_start = time.time()

case05 = ['TR01_ISWT', 'TR01_PLR', 'TR01_SWF', 'TR01_ILWT']       # 冷却水入口温度, 負荷率, 冷却水流量, 冷水入口温度

Xtrain_05 = train[case05]
Xtest_05 = test[case05]

# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case05():
    model = Sequential()
    model.add(Dense(200, input_dim = 4, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C05 = Case05()
CO5R = C05.fit(Xtrain_05, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C05_COP = C05.predict(Xtest_05)

C05_RMSE = sqrt(mean_squared_error(BEMS_COP, C05_COP))

C05_End = time.time() - C05_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C05_COP
submission["RMSE"] = C05_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case05.csv', index = False)

'-----------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C5_loss = CO5R.history['loss']
C5_val_loss = CO5R.history['val_loss']

C5_epochs = range(1, len(C5_loss) + 1)

plt.figure()
plt.plot(C5_epochs, C5_loss, 'g', label='Training loss')
plt.plot(C5_epochs, C5_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case05')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case05.png')

'-----------------------------------------------------------------------'

'===================================='

# case 06 : 冷却水入口温度, 負荷率, 冷却水流量, 冷水流量
## Variable 04

'===================================='

C06_start = time.time()

case06 = ['TR01_ISWT', 'TR01_PLR', 'TR01_SWF', 'TR01_LWF']       # 冷却水入口温度, 負荷率, 冷却水流量, 冷水流量

Xtrain_06 = train[case06]
Xtest_06 = test[case06]

# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case06():
    model = Sequential()
    model.add(Dense(200, input_dim = 4, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C06 = Case06()
CO6R = C06.fit(Xtrain_06, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C06_COP = C06.predict(Xtest_06)

C06_RMSE = sqrt(mean_squared_error(BEMS_COP, C06_COP))

C06_End = time.time() - C06_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C06_COP
submission["RMSE"] = C06_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case06.csv', index = False)

'-----------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C6_loss = CO6R.history['loss']
C6_val_loss = CO6R.history['val_loss']

C6_epochs = range(1, len(C6_loss) + 1)

plt.figure()
plt.plot(C6_epochs, C6_loss, 'g', label='Training loss')
plt.plot(C6_epochs, C6_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case06')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case06.png')

'-----------------------------------------------------------------------'

'===================================='

# case 07 : 冷却水入口温度, 負荷率, 冷水入口温度, 冷水流量
## Variable 04

'===================================='

C07_start = time.time()

case07 = ['TR01_ISWT', 'TR01_PLR', 'TR01_ILWT', 'TR01_LWF']       # 冷却水入口温度, 負荷率, 冷水入口温度, 冷水流量

Xtrain_07 = train[case07]
Xtest_07 = test[case07]

# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case07():
    model = Sequential()
    model.add(Dense(200, input_dim = 4, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C07 = Case07()
CO7R = C07.fit(Xtrain_07, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C07_COP = C07.predict(Xtest_07)

C07_RMSE = sqrt(mean_squared_error(BEMS_COP, C07_COP))

C07_End = time.time() - C07_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C07_COP
submission["RMSE"] = C07_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case07.csv', index = False)

'-----------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C7_loss = CO7R.history['loss']
C7_val_loss = CO7R.history['val_loss']

C7_epochs = range(1, len(C7_loss) + 1)

plt.figure()
plt.plot(C7_epochs, C7_loss, 'g', label='Training loss')
plt.plot(C7_epochs, C7_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case07')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case07.png')

'-----------------------------------------------------------------------'

### ====================================================================================================================================

'===================================='

# case 08 : 冷却水入口温度, 負荷率, 冷却水流量, 冷水入口温度, 冷水流量
## Variable 05

'===================================='

C08_start = time.time()

case08 = ['TR01_ISWT', 'TR01_PLR', 'TR01_SWF', 'TR01_ILWT', 'TR01_LWF']       # 冷却水入口温度, 負荷率, 冷却水流量, 冷水入口温度, 冷水流量

Xtrain_08 = train[case08]
Xtest_08 = test[case08]

# MLP Model

seed_value = 1
np.random.seed(seed_value)
rn.seed(seed_value)
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads =1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def Case08():
    model = Sequential()
    model.add(Dense(200, input_dim = 5, kernel_initializer = 'random_uniform',  activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer = 'normal', activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer = 'normal', activation = 'relu'))

    model.compile(loss = 'mean_squared_logarithmic_error', optimizer = "adam")
    return model

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode= 'min', min_delta=0.0001, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

C08 = Case08()
CO8R = C08.fit(Xtrain_08, Ytrain, epochs = 1000, batch_size = 21196, validation_split = 0.2, verbose=1, callbacks = [es, mc])
C08_COP = C08.predict(Xtest_08)

C08_RMSE = sqrt(mean_squared_error(BEMS_COP, C08_COP))

C08_End = time.time() - C08_start

submission = pd.read_csv('C:\Chillermodel\Preprocessing\Test\TR01_test.csv')
submission["Pred_COP"] = C08_COP
submission["RMSE"] = C08_RMSE
submission.to_csv('C:\Chillermodel\ANNmodel(No)\TR1\Case08.csv', index = False)

'-----------------------------------------------------------------------'
# Result of training and Graph

import matplotlib.pyplot as plt

C8_loss = CO8R.history['loss']
C8_val_loss = CO8R.history['val_loss']

C8_epochs = range(1, len(C8_loss) + 1)

plt.figure()
plt.plot(C8_epochs, C8_loss, 'g', label='Training loss')
plt.plot(C8_epochs, C8_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of Case08')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig('C:\Chillermodel\ANNmodel(No)\TR1\Case08.png')

'-----------------------------------------------------------------------'

print(C01_End, C02_End, C03_End, C04_End, C05_End, C06_End, C07_End, C08_End)
