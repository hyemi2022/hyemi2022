

# predict whole system model

'''
-----------------------------------------------------
Predict the whole system based on training model

全体の予測を行う前、最後に機器ごとの学習を行った後
本格的な予測を行います。
ベストモデルを作る作業です。
-----------------------------------------------------
'''

import os
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd 
import csv
# from train_hex import *
# from train_pump import *
# from train_tank import *
# from train_refrigerator import *
from physic_hex import *
from physic_pump import *
from physic_refrigerator import *

import tensorflow as tf

'--------------------------------------------------'

### Call the all best model ###
# All Black
Rc01_PumpFlow_model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Case01_flow_y01.h5')
Rc02_PumpFlow_model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Case01_flow_y02.h5')
Tank_PumpFlow_model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Case01_flow_y03.h5')
HEX_PumpFlow_model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Case01_flow_y04.h5')
Load_PumpFlow_model = load_model('C:\\System_Air\\best_model\\IVS_Case\\Case01_flow_y05.h5')
Connection_model = load_model('C:\\System_Air\\best_model\\connection\\Case03.h5')

# Parameter Predict
Rc01_LOWTpara_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para01.h5')
Rc01_COPpara_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R01_para02.h5')
Rc02_LOWTpara_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para01.h5')
Rc02_COPpara_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_R02_para02.h5')
HEX_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_HEX_KA.h5')
Tank_para_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_Tank_parameter.h5')
P01_para_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_p01_parameter.h5')
P02_para_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_p02_parameter.h5')
P03_para_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_p03_parameter.h5')
P04_para_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_p04_parameter.h5')
P05_para_model = load_model('C:\\System_Air\\best_model\\physics_parameter\\Knowledge_p05_parameter.h5')


# Catalogue 
Cata05 = pd.read_csv('C:\\System_Air\\Catalogue\\TRNSYS_CATA_444.csv')
Cata06 = pd.read_csv('C:\\System_Air\\Catalogue\\TRNSYS_CATA_556.csv')
Cata07 = pd.read_csv('C:\\System_Air\\Catalogue\\TRNSYS_CATA_667.csv')
Cata08 = pd.read_csv('C:\\System_Air\\Catalogue\\TRNSYS_CATA_778.csv')
Cata09 = pd.read_csv('C:\\System_Air\\Catalogue\\TRNSYS_CATA_889.csv')
Cata10 = pd.read_csv('C:\\System_Air\\Catalogue\\TRNSYS_CATA_100.csv')

# Catalogue - Dataset
Cata05_SIT = Cata05['Air_Temp']
Cata05_PLR = Cata05['Capacity']
Cata05_COP = Cata05['COP']
Cata06_SIT = Cata06['Air_Temp']
Cata06_PLR = Cata06['Capacity']
Cata06_COP = Cata06['COP']
Cata07_SIT = Cata07['Air_Temp']
Cata07_PLR = Cata07['Capacity']
Cata07_COP = Cata07['COP']
Cata08_SIT = Cata08['Air_Temp']
Cata08_PLR = Cata08['Capacity']
Cata08_COP = Cata08['COP']
Cata09_SIT = Cata09['Air_Temp']
Cata09_PLR = Cata09['Capacity']
Cata09_COP = Cata09['COP']
Cata10_SIT = Cata10['Air_Temp']
Cata10_PLR = Cata10['Capacity']
Cata10_COP = Cata10['COP']

'--------------------------------------------------'

# Setting the Function for entire Computing
# Calculate Interpolate COP in System 
def Interpol_Rc(SettingTemp_Rc, x_Rc, y_Rc):
   
    if SettingTemp_Rc == 5:
        Rc_Inter_COP = InterCOP(Coef_C, Cata05_SIT, Cata05_PLR, Cata05_COP, x_Rc, y_Rc)
    elif SettingTemp_Rc == 6:
        Rc_Inter_COP = InterCOP(Coef_C, Cata06_SIT, Cata06_PLR, Cata06_COP, x_Rc, y_Rc)
    elif SettingTemp_Rc == 7:
        Rc_Inter_COP = InterCOP(Coef_C, Cata07_SIT, Cata07_PLR, Cata07_COP, x_Rc, y_Rc)
    elif SettingTemp_Rc == 8:
        Rc_Inter_COP = InterCOP(Coef_C, Cata08_SIT, Cata08_PLR, Cata08_COP, x_Rc, y_Rc)
    elif SettingTemp_Rc == 9:
        Rc_Inter_COP = InterCOP(Coef_C, Cata09_SIT, Cata09_PLR, Cata09_COP, x_Rc, y_Rc)
    elif SettingTemp_Rc == 10:
        Rc_Inter_COP = InterCOP(Coef_C, Cata10_SIT, Cata10_PLR, Cata10_COP, x_Rc, y_Rc)

    return Rc_Inter_COP

def ProFlow(A):
    if A < 1:
        D = 0
    else:
        D = A
    return D
    

   
#####################################  
####       Initial Predict       ####
#####################################


# Open the Predict Data
Predict = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017_oneday.csv') ## Actually Three days 21.07.29


# Initial Inlet Temperature
Load_2nd = Predict['Secondary_Load'].loc[0]
Tank_Storage_1st = Predict['Tnak_Storage_rate'].loc[0]
Rc01_LITemp = Predict['R01_ITemp'].loc[0] 
Rc02_LITemp = Predict['R02_ITemp'].loc[0]
HEX_LITemp = Predict['HEX01_LITemp'].loc[0]
HEX_SITemp = Predict['HEX01_SITemp'].loc[0]
Tank_SITemp = Predict['ST01_SITemp'].loc[0]
Tank_LITemp = Predict['ST01_LITemp'].loc[0]
RHeader_Temp = Predict['RH01_Temp'].loc[0]
SHeader_Temp = Predict['SH01_Temp'].loc[0]
Node10_1st = Predict['Node10_temp'].loc[0]
Node9_1st = Predict['Node9_temp'].loc[0]
Node8_1st = Predict['Node8_temp'].loc[0]
Node7_1st = Predict['Node7_temp'].loc[0]
Node6_1st = Predict['Node6_temp'].loc[0]
Node5_1st = Predict['Node5_temp'].loc[0]
Node4_1st = Predict['Node4_temp'].loc[0]
Node3_1st = Predict['Node3_temp'].loc[0]
Node2_1st = Predict['Node2_temp'].loc[0]
Node1_1st = Predict['Node1_temp'].loc[0]

P01_InTemp = Predict['RH01_Temp_1bef'].loc[0]
P02_InTemp = Predict['RH01_Temp_1bef'].loc[0]
P03_InTemp = Predict['ST01_LOTemp_1bef'].loc[0]
P04_InTemp = Predict['RH01_Temp_1bef'].loc[0]
P05_InTemp = Predict['SH01_Temp_1bef'].loc[0]

R01_OTemp_1st = Predict['R01_OTemp_1bef'].loc[0]
R02_OTemp_1st = Predict['R02_OTemp_1bef'].loc[0]
HEX01_LOTemp_1st = Predict['HEX01_LOTemp_1bef'].loc[0]
HEX01_SOTemp_1st = Predict['HEX01_SOTemp_1bef'].loc[0]

Time = Predict['Timepermin'].loc[0]
Week = Predict['Week'].loc[0]
Outdoor_Temp = Predict['Outdoor_Temperature'].loc[0]
Outdoor_Humidity = Predict['Outdoor_Humidity'].loc[0]
Rc01_SetTemp = Predict['R01_SetTemp'].loc[0]
Rc02_SetTemp = Predict['R02_SetTemp'].loc[0]

'--------------------------------------------------'


# flow 
    # combinate input data
Rc01_PumpFlow_input = (Week, Time, Outdoor_Temp, Outdoor_Humidity, Load_2nd)
Rc01_PumpFlow_input = np.asarray(Rc01_PumpFlow_input).reshape(1, 5)
Rc02_PumpFlow_input = (Week, Time, Outdoor_Temp, Outdoor_Humidity, Load_2nd)
Rc02_PumpFlow_input = np.asarray(Rc02_PumpFlow_input).reshape(1, 5)
Tank_PumpFlow_input = (Week, Time, Outdoor_Temp, Outdoor_Humidity, Load_2nd)
Tank_PumpFlow_input = np.asarray(Tank_PumpFlow_input).reshape(1, 5)
HEX_PumpFlow_input = (Week, Time, Outdoor_Temp, Outdoor_Humidity, Load_2nd)
HEX_PumpFlow_input = np.asarray(HEX_PumpFlow_input).reshape(1, 5)
Load_PumpFlow_input = (Week, Time, Outdoor_Temp, Outdoor_Humidity, Load_2nd)
Load_PumpFlow_input = np.asarray(Load_PumpFlow_input).reshape(1, 5)
    # prediction from ANN
Rc01_PumpFlow = Rc01_PumpFlow_model.predict(Rc01_PumpFlow_input)
Rc01_PumpFlow = Rc01_PumpFlow[0, 0]
Rc02_PumpFlow = Rc02_PumpFlow_model.predict(Rc02_PumpFlow_input)
Rc02_PumpFlow = Rc02_PumpFlow[0, 0]
Tank_PumpFlow = Tank_PumpFlow_model.predict(Tank_PumpFlow_input)
Tank_PumpFlow = Tank_PumpFlow[0, 0]
HEX_PumpFlow = HEX_PumpFlow_model.predict(HEX_PumpFlow_input)
HEX_PumpFlow = HEX_PumpFlow[0, 0]
Load_PumpFlow = Load_PumpFlow_model.predict(Load_PumpFlow_input)
Load_PumpFlow = Load_PumpFlow[0, 0]
Rc01_PumpFlow = ProFlow(Rc01_PumpFlow)
Rc02_PumpFlow = ProFlow(Rc02_PumpFlow)
Tank_PumpFlow = ProFlow(Tank_PumpFlow)
HEX_PumpFlow = ProFlow(HEX_PumpFlow)
Load_PumpFlow = ProFlow(Load_PumpFlow)


# Refrigerator
    # combinate input data
Rc01_para_input = (Rc01_LITemp, Outdoor_Temp, Rc01_PumpFlow)
Rc01_para_input = np.asarray(Rc01_para_input).reshape(1, 3)
Rc02_para_input = (Rc02_LITemp, Outdoor_Temp, Rc02_PumpFlow)
Rc02_para_input = np.asarray(Rc02_para_input).reshape(1, 3)

    # Computing Parameters from ANN
Rc01_LOWTpara = Rc01_LOWTpara_model.predict(Rc01_para_input)
Rc01_LOWTpara = Rc01_LOWTpara[0, 0]
Rc01_COPpara = Rc01_COPpara_model.predict(Rc01_para_input)
Rc01_COPpara = Rc01_COPpara[0, 0]
Rc02_LOWTpara = Rc02_LOWTpara_model.predict(Rc02_para_input)
Rc02_LOWTpara = Rc02_LOWTpara[0, 0]
Rc02_COPpara = Rc02_COPpara_model.predict(Rc02_para_input)
Rc02_COPpara = Rc02_COPpara[0, 0]


# HEX 
    # combinate HEX input data
HEX_input = (HEX_SITemp, HEX_LITemp, Tank_PumpFlow, HEX_PumpFlow)
HEX_input = np.asarray(HEX_input).reshape(1, 4)
    # Computing Parameters from ANN 
HEX_Para = HEX_model.predict(HEX_input)
HEX_Para = HEX_Para[0, 0]


# Tank
    # combinate Tank input data
Tank_para_input = (Tank_SITemp, Tank_LITemp, Rc02_PumpFlow, Tank_PumpFlow)
Tank_para_input = np.asarray(Tank_para_input).reshape(1, 4)

    # Computing Output from ANN
Tank_Para = Tank_para_model.predict(Tank_para_input)
Tank_Para = Tank_Para[0, 0]


# pump
    # combinate Pump input data
    # P01 <-- Return Header
Pump01_para_input = (P01_InTemp, Rc01_PumpFlow)
Pump01_para_input = np.asarray(Pump01_para_input).reshape(1, 2)
    # P02 <-- Return Header
Pump02_para_input = (P02_InTemp, Rc02_PumpFlow)
Pump02_para_input = np.asarray(Pump02_para_input).reshape(1, 2)
    # P03 <-- Storage Tank outlet to HEX 
Pump03_para_input = (P03_InTemp, Tank_PumpFlow)
Pump03_para_input = np.asarray(Pump03_para_input).reshape(1, 2)
    # P04 <-- Return Header
Pump04_para_input = (P04_InTemp, HEX_PumpFlow)
Pump04_para_input = np.asarray(Pump04_para_input).reshape(1, 2)
    # P05 <-- Send Header
Pump05_para_input = (P05_InTemp, Load_PumpFlow)
Pump05_para_input = np.asarray(Pump05_para_input).reshape(1, 2)
    # Computing Output from ANN --> patameter
Pump01_para = P01_para_model.predict(Pump01_para_input)
Pump01_para = Pump01_para[0, 0]
Pump02_para = P02_para_model.predict(Pump02_para_input)
Pump02_para = Pump02_para[0, 0]
Pump03_para = P03_para_model.predict(Pump03_para_input)
Pump03_para = Pump03_para[0, 0]
Pump04_para = P04_para_model.predict(Pump04_para_input)
Pump04_para = Pump04_para[0, 0]
Pump05_para = P05_para_model.predict(Pump05_para_input)
Pump05_para = Pump05_para[0, 0]


'--------------------------------------------------'


#### Reverse Physics Formula ####
# Rc01 Outlet(Cooling Water & Chilled Water) Temperature
    # (1) Chilled Out Temperature
Rc01_LOTemp = OLWT(Rc01_LITemp, Outdoor_Temp, Rc01_PumpFlow, Rc01_LOWTpara)

    # (2) Cooling Capacity
Rc01_Capacity = CoolingCapa(Rc01_LITemp, Rc01_LOTemp, Rc01_PumpFlow)

    # (3) Interpolate COP
        # Choose setting temperature
x_Rc01 = Outdoor_Temp
y_Rc01 = Rc01_Capacity
Rc01_Inter_COP = Interpol_Rc(Rc01_SetTemp, x_Rc01, y_Rc01)

    # (4) COP
Rc01_COP = Rc01_Inter_COP * (Rc01_COPpara / 10)


# Rc02 Outlet(Cooling Water & Chilled Water) Temperature
    # (1) Chilled Out Temperature
Rc02_LOTemp = OLWT(Rc02_LITemp, Outdoor_Temp, Rc02_PumpFlow, Rc02_LOWTpara)

    # (2) Cooling Capacity
Rc02_Capacity = CoolingCapa(Rc02_LITemp, Rc02_LOTemp, Rc02_PumpFlow)

    # (3) Interpolate COP
        # Choose setting temperature
x_Rc02 = Outdoor_Temp
y_Rc02 = Rc02_Capacity
Rc02_Inter_COP = Interpol_Rc(Rc02_SetTemp, x_Rc02, y_Rc02)

    # (4) COP
Rc02_COP = Rc02_Inter_COP * (Rc02_COPpara / 10)


# Tank Outlet Temperature (Source side and Load side)

    # (1) Temperature of each layer
v = 120                               # Volume of Each layer
delt = 1/60                           # delta temperautre
Tank_Para_in = Tank_Para/10

if Rc02_PumpFlow !=0 and Tank_PumpFlow ==0:     # Charging Time
    Node10_2nd = Tank_SITemp-((Tank_SITemp-Node10_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node10 = Tank_Para_in*(Node10_1st-Outdoor_Temp)*delt      # kJ/h
    dt_node10 = loss_node10/(v*1000*4.2)        # k
    Node10_2nd = Node10_2nd + dt_node10         # k

    Node9_2nd = Node10_2nd-((Node10_2nd-Node9_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node9 = Tank_Para_in*(Node9_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node9 = loss_node9/(v*1000*4.2)          # k
    Node9_2nd = Node9_2nd + dt_node9            # k

    Node8_2nd = Node9_2nd-((Node9_2nd-Node8_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node8 = Tank_Para_in*(Node8_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node8 = loss_node8/(v*1000*4.2)          # k
    Node8_2nd = Node8_2nd + dt_node8            # k
            
    Node7_2nd = Node8_2nd-((Node8_2nd-Node7_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node7 = Tank_Para_in*(Node7_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node7 = loss_node7/(v*1000*4.2)          # k
    Node7_2nd = Node7_2nd + dt_node7            # k

    Node6_2nd = Node7_2nd-((Node7_2nd-Node6_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node6 = Tank_Para_in*(Node6_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node6 = loss_node6/(v*1000*4.2)          # k
    Node6_2nd = Node6_2nd + dt_node6            # k

    Node5_2nd = Node6_2nd-((Node6_2nd-Node5_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node5 = Tank_Para_in*(Node5_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node5 = loss_node5/(v*1000*4.2)          # k
    Node5_2nd = Node5_2nd + dt_node5            # k

    Node4_2nd = Node5_2nd-((Node5_2nd-Node4_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node4 = Tank_Para_in*(Node4_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node4 = loss_node4/(v*1000*4.2)          # k
    Node4_2nd = Node4_2nd + dt_node4            # k

    Node3_2nd = Node4_2nd-((Node4_2nd-Node3_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node3 = Tank_Para_in*(Node3_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node3 = loss_node3/(v*1000*4.2)          # k
    Node3_2nd = Node3_2nd + dt_node3            # k

    Node2_2nd = Node3_2nd-((Node3_2nd-Node2_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node2 = Tank_Para_in*(Node2_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node2 = loss_node2/(v*1000*4.2)          # k
    Node2_2nd = Node2_2nd + dt_node2            # k

    Node1_2nd = Node2_2nd-((Node2_2nd-Node1_1st)*math.exp(-Rc02_PumpFlow*delt/v))
    loss_node1 = Tank_Para_in*(Node1_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node1 = loss_node1/(v*1000*4.2)          # k
    Node1_2nd = Node1_2nd + dt_node1            # k

elif Tank_PumpFlow != 0 and Rc02_PumpFlow ==0:      # Discharging Time

    Node1_2nd = Tank_LITemp-((Tank_LITemp-Node1_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node1 = Tank_Para_in*(Node1_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node1 = loss_node1/(v*1000*4.2)          # k
    Node1_2nd = Node1_2nd + dt_node1            # k

    Node2_2nd = Node1_2nd-((Node1_2nd-Node2_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node2 = Tank_Para_in*(Node2_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node2 = loss_node2/(v*1000*4.2)          # k
    Node2_2nd = Node2_2nd + dt_node2            # k

    Node3_2nd = Node2_2nd-((Node2_2nd-Node3_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node3 = Tank_Para_in*(Node3_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node3 = loss_node3/(v*1000*4.2)          # k
    Node3_2nd = Node3_2nd + dt_node3            # k

    Node4_2nd = Node3_2nd-((Node3_2nd-Node4_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node4 = Tank_Para_in*(Node4_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node4 = loss_node4/(v*1000*4.2)          # k
    Node4_2nd = Node4_2nd + dt_node4            # k

    Node5_2nd = Node4_2nd-((Node4_2nd-Node5_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node5 = Tank_Para_in*(Node5_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node5 = loss_node5/(v*1000*4.2)          # k
    Node5_2nd = Node5_2nd + dt_node5            # k

    Node6_2nd = Node5_2nd-((Node5_2nd-Node6_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node6 = Tank_Para_in*(Node6_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node6 = loss_node6/(v*1000*4.2)          # k
    Node6_2nd = Node6_2nd + dt_node6            # k

    Node7_2nd = Node6_2nd-((Node6_2nd-Node7_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node7 = Tank_Para_in*(Node7_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node7 = loss_node7/(v*1000*4.2)          # k
    Node7_2nd = Node7_2nd + dt_node7            # k

    Node8_2nd = Node7_2nd-((Node7_2nd-Node8_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node8 = Tank_Para_in*(Node8_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node8 = loss_node8/(v*1000*4.2)          # k
    Node8_2nd = Node8_2nd + dt_node8            # k

    Node9_2nd = Node8_2nd-((Node8_2nd-Node9_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node9 = Tank_Para_in*(Node9_1st-Outdoor_Temp)*delt        # kJ/h
    dt_node9 = loss_node9/(v*1000*4.2)          # k
    Node9_2nd = Node9_2nd + dt_node9            # k

    Node10_2nd = Node9_2nd-((Node9_2nd-Node10_1st)*math.exp(-Tank_PumpFlow*delt/v))
    loss_node10 = Tank_Para_in*(Node10_1st-Outdoor_Temp)*delt      # kJ/h
    dt_node10 = loss_node10/(v*1000*4.2)        # k
    Node10_2nd = Node10_2nd + dt_node10         # k


Tank_LOTemp = Node1_2nd
Tank_SOTemp = Node10_2nd

    # (2) Heat Storage
if Rc02_PumpFlow != 0 and Tank_PumpFlow == 0:
    Tank_Storage_2nd = (Tank_SITemp - Tank_SOTemp)*Rc02_PumpFlow*1000*4.184/3600
elif Tank_PumpFlow != 0 and Rc02_PumpFlow == 0:
    Tank_Storage_2nd = (Tank_LITemp - Tank_LOTemp)*Tank_PumpFlow*1000*4.184/3600
else:
    Tank_Storage_2nd = 0  


# HEX Outlet Temperature (Source side and Load side)
    # (1) HEX Source side Outlet Temperature
HEX_SOTemp = HEX_SOT(HEX_SITemp, HEX_LITemp, Tank_PumpFlow, HEX_PumpFlow, HEX_Para)

    # (2) HEX Load side Outlet Temperature
HEX_LOTemp = HEX_LOT(HEX_SITemp, HEX_LITemp, Tank_PumpFlow, HEX_PumpFlow, HEX_Para)


'--------------------------------------------------'

#### Performance or Power of Equipment ####
    # Rc01 & Rc02 Performance
Rc01_Power = Power(Rc01_Capacity, Rc01_COP)
Rc02_Power = Power(Rc02_Capacity, Rc02_COP)


# Pump Performance --> Power
    # P01~P05 Performance
P01_Power = pump_power(Pump01_para, P01_InTemp, Rc01_PumpFlow)
P02_Power = pump_power(Pump02_para, P02_InTemp, Rc02_PumpFlow)
P03_Power = pump_power(Pump03_para, P03_InTemp, Tank_PumpFlow)
P04_Power = pump_power(Pump04_para, P04_InTemp, HEX_PumpFlow)
P05_Power = pump_power(Pump05_para, P05_InTemp, Load_PumpFlow)


#####################################  
####  1st ---> 2nd Sending Data  ####
#####################################

# Inlet Temperature to next time by Connection model
    # combinate input data
Connectipon_input = (Rc01_PumpFlow, Rc02_PumpFlow, Tank_PumpFlow, HEX_PumpFlow, Rc02_PumpFlow, HEX_PumpFlow, Load_PumpFlow)
Connectipon_input = np.asarray(Connectipon_input).reshape(1, 7)

    # prediction from ANN
Label = Connection_model.predict(Connectipon_input)
Label = Label.reshape(1, 31)
Label = np.rint(Label)
Label = np.argmax(Label, axis=1)


# Reverse Model Graph to Temperature
if Label == 1:
    Tank_SITemp_2nd = Rc02_LOTemp
    Rc02_LITemp_2nd = Tank_SOTemp
    # OfF
    Rc01_LITemp_2nd = Rc01_LITemp
    HEX_LITemp_2nd = HEX_LITemp
    HEX_SITemp_2nd = HEX_SITemp
    Tank_LITemp_2nd = Tank_LITemp
    SHeader_Temp01 = SHeader_Temp

    SHeader_Temp_2nd = SHeader_Temp01
    
elif Label == 3:
    SHeader_Temp01 = Rc01_LOTemp 
    SHeader_Temp02 = Rc02_LOTemp
    Tank_LITemp_2nd = HEX_SOTemp
    HEX_SITemp_2nd = Tank_LOTemp 
    Rc01_LITemp_2nd = RHeader_Temp
    Rc02_LITemp_2nd = RHeader_Temp
    HEX_LITemp_2nd = RHeader_Temp

    SHeader_Temp_2nd = ((SHeader_Temp01 * Rc01_PumpFlow) + (SHeader_Temp02 * Rc02_PumpFlow)) / (Rc01_PumpFlow + Rc02_PumpFlow)

elif Label == 20:
    SHeader_Temp01 = Rc01_LOTemp
    Tank_LITemp_2nd = HEX_SOTemp
    SHeader_Temp02 = HEX_LOTemp
    HEX_SITemp_2nd = Tank_LOTemp
    Rc01_LITemp_2nd = RHeader_Temp
    HEX_LITemp_2nd = RHeader_Temp
    # Off
    Rc02_LITemp_2nd = Rc02_LITemp
    Tank_SITemp_2nd = Tank_SITemp

    SHeader_Temp_2nd = ((SHeader_Temp01 * Rc01_PumpFlow) + (SHeader_Temp02 * Rc02_PumpFlow)) / (Rc01_PumpFlow + Rc02_PumpFlow)
    
elif Label == 30:
    Tank_LITemp_2nd = HEX_SOTemp
    SHeader_Temp03 = HEX_LOTemp
    HEX_SITemp_2nd = Tank_LOTemp
    HEX_LITemp_2nd = RHeader_Temp
    # Off
    Rc01_LITemp_2nd = Rc01_LITemp
    Rc02_LITemp_2nd = Rc02_LITemp
    Tank_SITemp_2nd = Tank_SITemp

    SHeader_Temp_2nd = SHeader_Temp03

else:
    Rc01_LITemp_2nd = Rc01_LITemp
    Rc02_LITemp_2nd = Rc02_LITemp
    HEX_LITemp_2nd = HEX_LITemp
    HEX_SITemp_2nd = HEX_SITemp
    Tank_LITemp_2nd = Tank_LITemp
    Tank_SITemp_2nd = Tank_SITemp
    SHeader_Temp01 = SHeader_Temp

    SHeader_Temp_2nd = SHeader_Temp01


# Calculate the Return Header Temperature to Equipment 
if Load_PumpFlow == 0:
    RHeader_Temp_2nd = RHeader_Temp + 0.01
else:
    RHeader_Temp_2nd = SHeader_Temp_2nd + (Load_2nd / Load_PumpFlow * 1.162)

Rc01_LOTemp_2nd = Rc01_LOTemp
Rc02_LOTemp_2nd = Rc02_LOTemp
HEX_LOTemp_2nd = HEX_LOTemp
HEX_SOTemp_2nd = HEX_SOTemp

#####################################  
###  Start to 2nd Time Computing  ###
#####################################

Outdoor_Temperature = Predict['Outdoor_Temperature'].iloc[1:]
Outdoor_Temperature = np.array(Outdoor_Temperature.shape[0])

# Make to the list
RHeader_Temp_result = []

Rc01_PumpFlow_result = []
Rc02_PumpFlow_result = []
Tank_PumpFlow_result = []
HEX_PumpFlow_result = []
Load_PumpFlow_result = []

Rc01_connect_result = []
Rc02_connect_result = []
HEXS_connect_result = []
HEXL_connect_result = []

Rc01_LOWTpara_result = []
Rc01_COPpara_result = []
Rc02_LOWTpara_result = []
Rc02_COPpara_result = []

HEX_Para_result = []
Tank_Para_result = []

Rc01_LITemp_result = []
Rc02_LITemp_result = []
HEX_SITemp_result = []
HEX_LITemp_result = []
Tank_SITemp_result = []
Tank_LITemp_result = []

Rc01_LOTemp_result = [] 
Rc02_LOTemp_result = [] 
HEX_SOTemp_result = []
HEX_LOTemp_result = []
Tank_SOTemp_result = []
Tank_LOTemp_result = []
Tank_Storage_result = []

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

SHeader_Temp_result = []

Label_result = []


i = 1

# Start to Calculate 2nd Time
for i in range(Outdoor_Temperature) :


    Time_2nd = Predict['Timepermin'].iloc[i]
    Week_2nd = Predict['Week'].iloc[i]
    Load_2nd_2nd = Predict['Secondary_Load'].iloc[i]
    OutTemp_2nd = Predict['Outdoor_Temperature'].iloc[i]
    OutHum_2nd = Predict['Outdoor_Humidity'].iloc[i]
    Rc01_SetTemp_2nd = Predict['R01_SetTemp'].iloc[i]
    Rc02_SetTemp_2nd = Predict['R02_SetTemp'].iloc[i]


    # flow
    # combinate input data
    
    Rc01_PumpFlow_input_2nd = (Week_2nd, Time_2nd, OutTemp_2nd, OutHum_2nd, Load_2nd_2nd)
    Rc01_PumpFlow_input_2nd = np.asarray(Rc01_PumpFlow_input_2nd).astype(np.float32)
    Rc01_PumpFlow_input_2nd = Rc01_PumpFlow_input_2nd.reshape(1, 5)
    Rc02_PumpFlow_input_2nd = (Week_2nd, Time_2nd, OutTemp_2nd, OutHum_2nd, Load_2nd_2nd)
    Rc02_PumpFlow_input_2nd = np.asarray(Rc02_PumpFlow_input_2nd).astype(np.float32)
    Rc02_PumpFlow_input_2nd = Rc02_PumpFlow_input_2nd.reshape(1, 5)
    Tank_PumpFlow_input_2nd = (Week_2nd, Time_2nd, OutTemp_2nd, OutHum_2nd, Load_2nd_2nd)
    Tank_PumpFlow_input_2nd = np.asarray(Tank_PumpFlow_input_2nd).astype(np.float32)
    Tank_PumpFlow_input_2nd = Tank_PumpFlow_input_2nd.reshape(1, 5)
    HEX_PumpFlow_input_2nd = (Week_2nd, Time_2nd, OutTemp_2nd, OutHum_2nd, Load_2nd_2nd)
    HEX_PumpFlow_input_2nd = np.asarray(HEX_PumpFlow_input_2nd).astype(np.float32)
    HEX_PumpFlow_input_2nd = HEX_PumpFlow_input_2nd.reshape(1, 5)
    Load_PumpFlow_input_2nd = (Week_2nd, Time_2nd, OutTemp_2nd, OutHum_2nd, Load_2nd_2nd)
    Load_PumpFlow_input_2nd = np.asarray(Load_PumpFlow_input_2nd).astype(np.float32)
    Load_PumpFlow_input_2nd = Load_PumpFlow_input_2nd.reshape(1, 5)
    # Prediction by ANN
    Rc01_PumpFlow_2nd = Rc01_PumpFlow_model.predict(Rc01_PumpFlow_input_2nd)
    Rc01_PumpFlow_2nd = Rc01_PumpFlow_2nd[0, 0]
    Rc02_PumpFlow_2nd = Rc02_PumpFlow_model.predict(Rc02_PumpFlow_input_2nd)
    Rc02_PumpFlow_2nd = Rc02_PumpFlow_2nd[0, 0]
    Tank_PumpFlow_2nd = Tank_PumpFlow_model.predict(Tank_PumpFlow_input_2nd)
    Tank_PumpFlow_2nd = Tank_PumpFlow_2nd[0, 0]
    HEX_PumpFlow_2nd = HEX_PumpFlow_model.predict(HEX_PumpFlow_input_2nd)
    HEX_PumpFlow_2nd = HEX_PumpFlow_2nd[0, 0]
    Load_PumpFlow_2nd = Load_PumpFlow_model.predict(Load_PumpFlow_input_2nd)
    Load_PumpFlow_2nd = Load_PumpFlow_2nd[0, 0]


    # Refrigerator
    # combinate input data
    Rc01_para01_input_2nd = (Rc01_LITemp_2nd, OutTemp_2nd, Rc01_PumpFlow_2nd)
    Rc01_para01_input_2nd = np.asarray(Rc01_para01_input_2nd).reshape(1, 3)
    Rc01_para02_input_2nd = (Rc01_LITemp_2nd, OutTemp_2nd, Rc01_PumpFlow_2nd)
    Rc01_para02_input_2nd = np.asarray(Rc01_para02_input_2nd).reshape(1, 3)
    Rc02_para01_input_2nd = (Rc02_LITemp_2nd, OutTemp_2nd, Rc02_PumpFlow_2nd)
    Rc02_para01_input_2nd = np.asarray(Rc02_para01_input_2nd).reshape(1, 3)
    Rc02_para02_input_2nd = (Rc02_LITemp_2nd, OutTemp_2nd, Rc02_PumpFlow_2nd)
    Rc02_para02_input_2nd = np.asarray(Rc02_para02_input_2nd).reshape(1, 3)
    # Computing Parameters from ANN
    Rc01_LOWTpara_2nd = Rc01_LOWTpara_model.predict(Rc01_para01_input_2nd)
    Rc01_LOWTpara_2nd = Rc01_LOWTpara_2nd[0, 0]
    Rc01_COPpara_2nd = Rc01_COPpara_model.predict(Rc01_para01_input_2nd)
    Rc01_COPpara_2nd = Rc01_COPpara_2nd[0, 0]
    Rc02_LOWTpara_2nd = Rc02_LOWTpara_model.predict(Rc02_para01_input_2nd)
    Rc02_LOWTpara_2nd = Rc02_LOWTpara_2nd[0, 0]
    Rc02_COPpara_2nd = Rc02_COPpara_model.predict(Rc02_para02_input_2nd)
    Rc02_COPpara_2nd = Rc02_COPpara_2nd[0, 0]


    # Tank
    # combinate Tank input data
    Tank_para_input_2nd = (Tank_SITemp_2nd, Tank_LITemp_2nd, Rc02_PumpFlow_2nd, Tank_PumpFlow_2nd)
    Tank_para_input_2nd = np.asarray(Tank_para_input_2nd).astype(np.float16)
    Tank_para_input_2nd = Tank_para_input_2nd.reshape(1, 4)
    
    # Computing Parameters from ANN
    Tank_Para_2nd = Tank_para_model.predict(Tank_para_input_2nd)
    Tank_Para_2nd = Tank_Para_2nd[0, 0]


    # HEX 
    # combinate HEX input data
    HEX_input_2nd = (HEX_SITemp_2nd, HEX_LITemp_2nd, Tank_PumpFlow_2nd, HEX_PumpFlow_2nd)
    HEX_input_2nd = np.asarray(HEX_input_2nd).astype(np.float16)
    HEX_input_2nd = HEX_input_2nd.reshape(1, 4)
    # Computing Parameters from ANN 
    HEX_Para_2nd = HEX_model.predict(HEX_input_2nd)
    HEX_Para_2nd = HEX_Para_2nd[0, 0]


    '--------------------------------------------------'

    #### Reverse Physics Formula ####

    # Rc01 Outlet(Cooling Water & Chilled Water) Temperature
    Rc01_LOTemp_2nd = OLWT(Rc01_LITemp_2nd, OutTemp_2nd, Rc01_PumpFlow_2nd, Rc01_LOWTpara_2nd)
    # Rc02 Outlet(Cooling Water & Chilled Water) Temperature
    Rc02_LOTemp_2nd = OLWT(Rc02_LITemp_2nd, OutTemp_2nd, Rc02_PumpFlow_2nd, Rc02_LOWTpara_2nd)
    
    if Rc01_LOTemp_2nd < 0:
        Rc01_LOTemp_2nd = 7
    
    if Rc02_LOTemp_2nd < 0:
        Rc02_LOTemp_2nd = 6
    
    Tank_Para_in_2nd = Tank_Para_2nd/10
    
    # Tank Outlet Temperature (Source side and Load side)
    if Rc02_PumpFlow_2nd !=0 and Tank_PumpFlow_2nd ==0:     # Charging Time
        Node10_3rd = Tank_SITemp_2nd-((Tank_SITemp_2nd-Node10_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node10_2nd = Tank_Para_in_2nd*(Node10_2nd-OutTemp_2nd)*delt      # kJ/h
        dt_node10_2nd = loss_node10_2nd/(v*1000*4.2)        # k
        Node10_3rd = Node10_3rd + dt_node10_2nd             # k

        Node9_3rd = Node10_3rd-((Node10_3rd-Node9_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node9_2nd = Tank_Para_in_2nd*(Node9_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node9_2nd = loss_node9_2nd/(v*1000*4.2)          # k
        Node9_3rd = Node9_3rd + dt_node9_2nd                # k

        Node8_3rd = Node9_3rd-((Node9_3rd-Node8_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node8_2nd = Tank_Para_in_2nd*(Node8_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node8_2nd = loss_node8_2nd/(v*1000*4.2)          # k
        Node8_3rd = Node8_3rd + dt_node8_2nd                # k
            
        Node7_3rd = Node8_3rd-((Node8_3rd-Node7_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node7_2nd = Tank_Para_in_2nd*(Node7_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node7_2nd = loss_node7_2nd/(v*1000*4.2)          # k
        Node7_3rd = Node7_3rd + dt_node7_2nd                # k

        Node6_3rd = Node7_3rd-((Node7_3rd-Node6_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node6_2nd = Tank_Para_in_2nd*(Node6_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node6_2nd = loss_node6_2nd/(v*1000*4.2)          # k
        Node6_3rd = Node6_3rd + dt_node6_2nd            # k

        Node5_3rd = Node6_3rd-((Node6_3rd-Node5_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node5_2nd = Tank_Para_in_2nd*(Node5_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node5_2nd = loss_node5_2nd/(v*1000*4.2)          # k
        Node5_3rd = Node5_3rd + dt_node5_2nd            # k

        Node4_3rd = Node5_3rd-((Node5_3rd-Node4_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node4_2nd = Tank_Para_in_2nd*(Node4_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node4_2nd = loss_node4_2nd/(v*1000*4.2)          # k
        Node4_3rd = Node4_3rd + dt_node4_2nd            # k

        Node3_3rd = Node4_3rd-((Node4_3rd-Node3_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node3_2nd = Tank_Para_in_2nd*(Node3_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node3_2nd = loss_node3_2nd/(v*1000*4.2)          # k
        Node3_3rd = Node3_3rd + dt_node3_2nd            # k

        Node2_3rd = Node3_3rd-((Node3_3rd-Node2_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node2_2nd = Tank_Para_in_2nd*(Node2_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node2_2nd = loss_node2_2nd/(v*1000*4.2)          # k
        Node2_3rd = Node2_3rd + dt_node2_2nd            # k

        Node1_3rd = Node2_3rd-((Node2_3rd-Node1_2nd)*math.exp(-Rc02_PumpFlow_2nd*delt/v))
        loss_node1_2nd = Tank_Para_in_2nd*(Node1_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node1_2nd = loss_node1_2nd/(v*1000*4.2)          # k
        Node1_3rd = Node1_3rd + dt_node1_2nd            # k

    elif Tank_PumpFlow_2nd != 0 and Rc02_PumpFlow_2nd ==0:      # Discharging Time

        Node1_3rd = Tank_LITemp_2nd-((Tank_LITemp_2nd-Node1_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node1_2nd = Tank_Para_in_2nd*(Node1_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node1_2nd = loss_node1_2nd/(v*1000*4.2)          # k
        Node1_3rd = Node1_3rd + dt_node1_2nd            # k

        Node2_3rd = Node1_3rd-((Node1_3rd-Node2_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node2_2nd = Tank_Para_in_2nd*(Node2_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node2_2nd = loss_node2_2nd/(v*1000*4.2)          # k
        Node2_3rd = Node2_3rd + dt_node2_2nd            # k

        Node3_3rd = Node2_3rd-((Node2_3rd-Node3_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node3_2nd = Tank_Para_in_2nd*(Node3_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node3_2nd = loss_node3_2nd/(v*1000*4.2)          # k
        Node3_3rd = Node3_3rd + dt_node3_2nd            # k

        Node4_3rd = Node3_3rd-((Node3_3rd-Node4_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node4_2nd = Tank_Para_in_2nd*(Node4_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node4_2nd = loss_node4_2nd/(v*1000*4.2)          # k
        Node4_3rd = Node4_3rd + dt_node4_2nd            # k

        Node5_3rd = Node4_3rd-((Node4_3rd-Node5_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node5_2nd = Tank_Para_in_2nd*(Node5_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node5_2nd = loss_node5_2nd/(v*1000*4.2)          # k
        Node5_3rd = Node5_3rd + dt_node5_2nd            # k

        Node6_3rd = Node5_3rd-((Node5_3rd-Node6_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node6_2nd = Tank_Para_in_2nd*(Node6_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node6_2nd = loss_node6_2nd/(v*1000*4.2)          # k
        Node6_3rd = Node6_3rd + dt_node6_2nd            # k

        Node7_3rd = Node6_3rd-((Node6_3rd-Node7_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node7_2nd = Tank_Para_in_2nd*(Node7_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node7_2nd = loss_node7_2nd/(v*1000*4.2)          # k
        Node7_3rd = Node7_3rd + dt_node7_2nd            # k

        Node8_3rd = Node7_3rd-((Node7_3rd-Node8_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node8_2nd = Tank_Para_in_2nd*(Node8_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node8_2nd = loss_node8_2nd/(v*1000*4.2)          # k
        Node8_3rd = Node8_3rd + dt_node8_2nd            # k

        Node9_3rd = Node8_3rd-((Node8_3rd-Node9_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node9_2nd = Tank_Para_in_2nd*(Node9_2nd-OutTemp_2nd)*delt        # kJ/h
        dt_node9_2nd = loss_node9_2nd/(v*1000*4.2)          # k
        Node9_3rd = Node9_3rd + dt_node9_2nd            # k

        Node10_3rd = Node9_3rd-((Node9_3rd-Node10_2nd)*math.exp(-Tank_PumpFlow_2nd*delt/v))
        loss_node10_2nd = Tank_Para_in_2nd*(Node10_2nd-OutTemp_2nd)*delt      # kJ/h
        dt_node10_2nd = loss_node10_2nd/(v*1000*4.2)        # k
        Node10_3rd = Node10_3rd + dt_node10_2nd         # k

    Tank_LOTemp_2nd = Node1_3rd
    Tank_SOTemp_2nd = Node10_3rd

        # (2) Heat Storage
    if Rc02_PumpFlow_2nd != 0 and Tank_PumpFlow_2nd == 0:
        Tank_Storage_3rd = (Tank_SITemp_2nd-Tank_SOTemp_2nd)*Rc02_PumpFlow_2nd*1000*4.184/3600
    elif Tank_PumpFlow_2nd != 0 and Rc02_PumpFlow_2nd == 0:
        Tank_Storage_3rd = (Tank_LITemp_2nd-Tank_LOTemp_2nd)*Tank_PumpFlow_2nd*1000*4.184/3600
    else:
        Tank_Storage_3rd = 0  

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

    # HEX Outlet Temperature (Source side and Load side)
        # (1) HEX Source side Outlet Temperature
    HEX_SOTemp_2nd = HEX_SOT(HEX_SITemp_2nd, HEX_LITemp_2nd, Tank_PumpFlow_2nd, HEX_PumpFlow_2nd, HEX_Para_2nd)  
        # (2) HEX Load side Outlet Temperature
    HEX_LOTemp_2nd = HEX_LOT(HEX_SITemp_2nd, HEX_LITemp_2nd, Tank_PumpFlow_2nd, HEX_PumpFlow_2nd, HEX_Para_2nd)

    if HEX_SOTemp_2nd < 0:
        HEX_SOTemp_2nd = 12
    
    if HEX_LOTemp_2nd < 0:
        HEX_LOTemp_2nd = 7

    #####################################  
    ####    Sending Data 2nd --->    ####
    #####################################


    # Inlet Temperature
    # combinate input data
    Connection_input_2nd = (Rc01_PumpFlow, Rc02_PumpFlow, Tank_PumpFlow, HEX_PumpFlow, Rc02_PumpFlow, HEX_PumpFlow, Load_PumpFlow)
    Connection_input_2nd = np.asarray(Connection_input_2nd).reshape(1, 7)

    # prediction from ANN
    Label_2nd = Connection_model.predict(Connection_input_2nd)
    Label_2nd = Label_2nd.reshape(1, 31)
    Label_2nd = np.rint(Label_2nd)
    Label_2nd = np.argmax(Label_2nd, axis=1)
    Label_2nd = Label_2nd[0,]

    # Reverse Model Graph to Temperature
    if Label_2nd == 1:
        Tank_SITemp_3rd = Rc02_LOTemp_2nd
        Rc02_LITemp_3rd = Tank_SOTemp_2nd
        # OFF
        Rc01_LITemp_3rd = Rc01_LITemp_2nd
        HEX_LITemp_3rd = HEX_LITemp_2nd
        HEX_SITemp_3rd = HEX_SITemp_2nd
        Tank_LITemp_3rd = Tank_LITemp_2nd
        SHeader_Temp01_2nd = SHeader_Temp_2nd

        SHeader_Temp_3rd = SHeader_Temp01_2nd
    
    elif Label_2nd == 3:
        SHeader_Temp01_2nd = Rc01_LOTemp_2nd 
        SHeader_Temp02_2nd = Rc02_LOTemp_2nd
        Tank_LITemp_3rd = HEX_SOTemp_2nd
        HEX_SITemp_3rd = Tank_LOTemp_2nd 
        Rc01_LITemp_3rd = RHeader_Temp_2nd
        Rc02_LITemp_3rd = RHeader_Temp_2nd
        HEX_LITemp_3rd = RHeader_Temp_2nd

        SHeader_Temp_3rd = ((SHeader_Temp01_2nd * Rc01_PumpFlow_2nd) + (SHeader_Temp02_2nd * Rc02_PumpFlow_2nd)) / (Rc01_PumpFlow_2nd + Rc02_PumpFlow_2nd)

    elif Label_2nd == 20:
        SHeader_Temp01_2nd = Rc01_LOTemp_2nd
        Tank_LITemp_3rd = HEX_SOTemp_2nd
        SHeader_Temp02_2nd = HEX_LOTemp_2nd
        HEX_SITemp_3rd = Tank_LOTemp_2nd
        Rc01_LITemp_3rd = RHeader_Temp_2nd
        HEX_LITemp_3rd = RHeader_Temp_2nd
        # Off
        Rc02_LITemp_3rd = Rc02_LITemp_2nd
        Tank_SITemp_3rd = Tank_SITemp_2nd

        SHeader_Temp_3rd = ((SHeader_Temp01_2nd * Rc01_PumpFlow_2nd) + (SHeader_Temp02_2nd * Rc02_PumpFlow_2nd)) / (Rc01_PumpFlow_2nd + Rc02_PumpFlow_2nd)
    
    elif Label_2nd == 30:
        Tank_LITemp_3rd = HEX_SOTemp_2nd
        SHeader_Temp03_2nd = HEX_LOTemp_2nd
        HEX_SITemp_3rd = Tank_LOTemp_2nd
        HEX_LITemp_3rd = RHeader_Temp_2nd
        # Off
        Rc01_LITemp_3rd = Rc01_LITemp_2nd
        Rc02_LITemp_3rd = Rc02_LITemp_2nd
        Tank_SITemp_3rd = Tank_SITemp_2nd

        SHeader_Temp_3rd = SHeader_Temp03_2nd

    else:
        Rc01_LITemp_3rd = Rc01_LITemp_2nd
        Rc02_LITemp_3rd = Rc02_LITemp_2nd
        HEX_LITemp_3rd = HEX_LITemp_2nd
        HEX_SITemp_3rd = HEX_SITemp_2nd
        Tank_LITemp_3rd = Tank_LITemp_2nd
        Tank_SITemp_3rd = Tank_SITemp_2nd
        SHeader_Temp01_2nd = SHeader_Temp_2nd

        SHeader_Temp_3rd = SHeader_Temp01_2nd


    # Calculate the Return Header Temperature to Equipment 
    if Load_PumpFlow_2nd == 0:
        RHeader_Temp_3rd = SHeader_Temp_3rd
    else:
        RHeader_Temp_3rd = SHeader_Temp_3rd + (Load_2nd_2nd / Load_PumpFlow_2nd * 1.162)

    '''
    if RHeader_Temp_3rd > 14:
        RHeader_Temp_3rd = 12.5
    elif RHeader_Temp_3rd < 0:
        RHeader_Temp_3rd = SHeader_Temp_3rd
    '''

    Rc01_LITemp_2nd = Rc01_LITemp_3rd
    Rc02_LITemp_2nd = Rc02_LITemp_3rd
    HEX_SITemp_2nd = HEX_SITemp_3rd
    HEX_LITemp_2nd = HEX_LITemp_3rd
    Tank_SITemp_2nd = Tank_SITemp_3rd
    Tank_LITemp_2nd = Tank_LITemp_3rd
    RHeader_Temp_2nd = RHeader_Temp_3rd
    SHeader_Temp_2nd = SHeader_Temp_3rd
    Tank_Storage_2nd = Tank_Storage_3rd

    # Save to the List
    RHeader_Temp_result.append(RHeader_Temp_2nd)
    
    Rc01_PumpFlow_result.append(Rc01_PumpFlow_2nd)
    Rc02_PumpFlow_result.append(Rc02_PumpFlow_2nd)
    Tank_PumpFlow_result.append(Tank_PumpFlow_2nd)
    HEX_PumpFlow_result.append(HEX_PumpFlow_2nd)
    Load_PumpFlow_result.append(Load_PumpFlow_2nd)

    Rc01_LOWTpara_result.append(Rc01_LOWTpara_2nd)
    Rc01_COPpara_result.append(Rc01_COPpara_2nd)
    Rc02_LOWTpara_result.append(Rc02_LOWTpara_2nd)
    Rc02_COPpara_result.append(Rc02_COPpara_2nd)

    HEX_Para_result.append(HEX_Para_2nd)
    Tank_Para_result.append(Tank_Para_2nd)
    
    Rc01_LITemp_result.append(Rc01_LITemp_2nd)
    Rc02_LITemp_result.append(Rc02_LITemp_2nd)
    HEX_SITemp_result.append(HEX_SITemp_2nd)
    HEX_LITemp_result.append(HEX_LITemp_2nd)
    Tank_SITemp_result.append(Tank_SITemp_2nd)
    Tank_LITemp_result.append(Tank_LITemp_2nd)

    Rc01_LOTemp_result.append(Rc01_LOTemp_2nd)
    Rc02_LOTemp_result.append(Rc02_LOTemp_2nd)
    HEX_SOTemp_result.append(HEX_SOTemp_2nd)
    HEX_LOTemp_result.append(HEX_LOTemp_2nd)
    Tank_SOTemp_result.append(Tank_SOTemp_2nd)
    Tank_LOTemp_result.append(Tank_LOTemp_2nd)
    Tank_Storage_result.append(Tank_Storage_2nd) 

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

    SHeader_Temp_result.append(SHeader_Temp_2nd)

    Label_result.append(Label_2nd)

    i += 1

    print(i, '流量', Rc01_PumpFlow_2nd, Rc02_LOTemp_2nd, Tank_SITemp_2nd, Rc02_PumpFlow_2nd, Tank_PumpFlow_2nd, HEX_PumpFlow_2nd, Load_PumpFlow_2nd) 
    
    #, '接続', Tank_LoadFlow_2nd, Tank_DischargPara_2nd, Tank_LITemp_2nd,
           
    #print('熱交換器', HEX_Para_2nd, HEX_LOTemp_2nd, HEX_SOTemp_2nd, HEX_PumpFlow_2nd, '蓄熱槽', Tank_LoadFlow_2nd, Tank_DischargPara_2nd, Tank_LITemp_2nd,
    #        Tank_LOTemp_2nd, Tank_SourceFlow_2nd, Tank_ChargPara_2nd, Tank_SITemp_2nd, Tank_SOTemp_2nd)

#####################################  
####       Result Data Set       ####
#####################################


Result_Rc01_PumpFlow = np.concatenate((np.atleast_1d(Rc01_PumpFlow), Rc01_PumpFlow_result), axis = 0)
Result_Rc02_PumpFlow = np.concatenate((np.atleast_1d(Rc02_PumpFlow), Rc02_PumpFlow_result), axis = 0)
Result_Tank_PumpFlow = np.concatenate((np.atleast_1d(Tank_PumpFlow), Tank_PumpFlow_result), axis = 0)
Result_HEX_PumpFlow = np.concatenate((np.atleast_1d(HEX_PumpFlow), HEX_PumpFlow_result), axis = 0)
Result_Load_PumpFlow = np.concatenate((np.atleast_1d(Load_PumpFlow), Load_PumpFlow_result), axis = 0)

# Parameters
Result_Rc01_LOWTpara = np.concatenate((np.atleast_1d(Rc01_LOWTpara), Rc01_LOWTpara_result), axis = 0)
Result_Rc01_COPpara = np.concatenate((np.atleast_1d(Rc01_COPpara), Rc01_COPpara_result), axis = 0)
Result_Rc02_LOWTpara = np.concatenate((np.atleast_1d(Rc02_LOWTpara), Rc02_LOWTpara_result), axis = 0)
Result_Rc02_COPpara = np.concatenate((np.atleast_1d(Rc02_COPpara), Rc02_COPpara_result), axis = 0)

Result_HEX_Para = np.concatenate((np.atleast_1d(HEX_Para), HEX_Para_result), axis = 0)
Result_Tank_Para = np.concatenate((np.atleast_1d(Tank_Para), Tank_Para_result), axis = 0)

# Inlet Temperature
Result_Rc01_ChillIWT = np.concatenate((np.atleast_1d(Rc01_LITemp), Rc01_LITemp_result), axis = 0)
Result_Rc02_ChillIWT = np.concatenate((np.atleast_1d(Rc02_LITemp), Rc02_LITemp_result), axis = 0)
Result_HEX_SIWT = np.concatenate((np.atleast_1d(HEX_SITemp), HEX_SITemp_result), axis = 0)
Result_HEX_LIWT = np.concatenate((np.atleast_1d(HEX_LITemp), HEX_LITemp_result), axis = 0)
Result_Tank_SIWT = np.concatenate((np.atleast_1d(Tank_SITemp), Tank_SITemp_result), axis = 0)
Result_Tank_LIWT = np.concatenate((np.atleast_1d(Tank_LITemp), Tank_LITemp_result), axis = 0)
Result_Supply_Header = np.concatenate((np.atleast_1d(SHeader_Temp), SHeader_Temp_result), axis = 0)

# Outlet Temperature
Result_Rc01_ChillOWT = np.concatenate((np.atleast_1d(Rc01_LOTemp), Rc01_LOTemp_result), axis = 0)
Result_Rc02_ChillOWT = np.concatenate((np.atleast_1d(Rc02_LOTemp), Rc02_LOTemp_result), axis = 0)
Result_HEX_SOWT = np.concatenate((np.atleast_1d(HEX_SOTemp), HEX_SOTemp_result), axis = 0)
Result_HEX_LOWT = np.concatenate((np.atleast_1d(HEX_LOTemp), HEX_LOTemp_result), axis = 0)
Result_Tank_SOWT = np.concatenate((np.atleast_1d(Tank_SOTemp), Tank_SOTemp_result), axis = 0)
Result_Tank_LOWT = np.concatenate((np.atleast_1d(Tank_LOTemp), Tank_LOTemp_result), axis = 0)
Result_Tank_Storage = np.concatenate((np.atleast_1d(Tank_Storage_2nd), Tank_Storage_result), axis = 0)

Result_Node1_2nd = np.concatenate((np.atleast_1d(Node1_2nd), Node1_result), axis = 0)
Result_Node2_2nd = np.concatenate((np.atleast_1d(Node2_2nd), Node2_result), axis = 0)
Result_Node3_2nd = np.concatenate((np.atleast_1d(Node3_2nd), Node3_result), axis = 0)
Result_Node4_2nd = np.concatenate((np.atleast_1d(Node4_2nd), Node4_result), axis = 0)
Result_Node5_2nd = np.concatenate((np.atleast_1d(Node5_2nd), Node5_result), axis = 0)
Result_Node6_2nd = np.concatenate((np.atleast_1d(Node6_2nd), Node6_result), axis = 0)
Result_Node7_2nd = np.concatenate((np.atleast_1d(Node7_2nd), Node7_result), axis = 0)
Result_Node8_2nd = np.concatenate((np.atleast_1d(Node8_2nd), Node8_result), axis = 0)
Result_Node9_2nd = np.concatenate((np.atleast_1d(Node9_2nd), Node9_result), axis = 0)
Result_Node10_2nd = np.concatenate((np.atleast_1d(Node10_2nd), Node10_result), axis = 0)

Result_Return_Header = np.concatenate((np.atleast_1d(RHeader_Temp), RHeader_Temp_result), axis = 0)

Result_Lable_Connect = np.concatenate((np.atleast_1d(Label), Label_result), axis = 0)

# Save the Result to CSV file
submission = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017_oneday.csv')
submission["Result_Rc01_PumpFlow"] = Result_Rc01_PumpFlow
submission["Result_Rc02_PumpFlow"] = Result_Rc02_PumpFlow
submission["Result_Tank_PumpFlow"] = Result_Tank_PumpFlow
submission["Result_HEX_PumpFlow"] = Result_HEX_PumpFlow
submission["Result_Load_PumpFlow"] = Result_Load_PumpFlow
submission["Result_Rc01_LOWTpara"] = Result_Rc01_LOWTpara
submission["Result_Rc01_COPpara"] = Result_Rc01_COPpara
submission["Result_Rc02_LOWTpara"] = Result_Rc02_LOWTpara
submission["Result_Rc02_COPpara"] = Result_Rc02_COPpara
submission["Result_HEX_Para"] = Result_HEX_Para
submission["Result_Tank_Para"] = Result_Tank_Para
submission["Result_Rc01_IWT"] = Result_Rc01_ChillIWT
submission["Result_Rc02_IWT"] = Result_Rc02_ChillIWT
submission["Result_HEX_SIWT"] = Result_HEX_SIWT
submission["Result_HEX_LIWT"] = Result_HEX_LIWT
submission["Result_Tank_SIWT"] = Result_Tank_SIWT
submission["Result_Tank_LIWT"] = Result_Tank_LIWT
submission["Result_Supply_Header"] = Result_Supply_Header
submission["Result_Rc01_OWT"] = Result_Rc01_ChillOWT
submission["Result_Rc02_OWT"] = Result_Rc02_ChillOWT
submission["Result_HEX_SOWT"] = Result_HEX_SOWT
submission["Result_HEX_LOWT"] = Result_HEX_LOWT
submission["Result_Tank_SOWT"] = Result_Tank_SOWT
submission["Result_Tank_LOWT"] = Result_Tank_LOWT
submission["Result_Node1"] = Result_Node1_2nd
submission["Result_Node2"] = Result_Node2_2nd
submission["Result_Node3"] = Result_Node3_2nd
submission["Result_Node4"] = Result_Node4_2nd
submission["Result_Node5"] = Result_Node5_2nd
submission["Result_Node6"] = Result_Node6_2nd
submission["Result_Node7"] = Result_Node7_2nd
submission["Result_Node8"] = Result_Node8_2nd
submission["Result_Node9"] = Result_Node9_2nd
submission["Result_Node10"] = Result_Node10_2nd
submission["Result_Return_Header"] = Result_Return_Header
submission["Result_Lable_Connect"] = Result_Lable_Connect
submission.to_csv('C:\\System_Air\\Result\\System_Result_oneday.csv', index = False)
            
print('Temp Finish')

'--------------------------------------------------'

Predict1 = pd.read_csv('C:\\System_Air\\Result\\System_Result_oneday.csv')

#### Performance of Equipment ####

P01_para_result = []
P02_para_result = []
P03_para_result = []
P04_para_result = []
P05_para_result = []

P01_Power_result = []
P02_Power_result = []
P03_Power_result = []
P04_Power_result = []
P05_Power_result = []

Rc01_InterCOP_result = []
Rc01_COP_result = []
Rc01_Capacity_result = []
Rc02_InterCOP_result = []
Rc02_COP_result = []
Rc02_Capacity_result = []

Rc01_Power_result = []
Rc02_Power_result = []


# Start to Calculate 2nd Time --> Power
for i in range(Outdoor_Temperature) :

    OutTemp_2nd = Predict1['Outdoor_Temperature'].iloc[i]    

    Rc01_PumpFlow_2nd = Predict1['Result_Rc01_PumpFlow'].iloc[i]
    Rc02_PumpFlow_2nd = Predict1['Result_Rc02_PumpFlow'].iloc[i]
    HEX_PumpFlow_2nd = Predict1['Result_HEX_PumpFlow'].iloc[i]
    Tank_PumpFlow_2nd = Predict1['Result_Tank_PumpFlow'].iloc[i]
    Load_PumpFlow_2nd = Predict1['Result_Load_PumpFlow'].iloc[i]

    Rc01_LITemp_2nd = Predict1['Result_Rc01_IWT'].iloc[i]
    Rc02_LITemp_2nd = Predict1['Result_Rc02_IWT'].iloc[i]
    Rc01_LOTemp_2nd = Predict1['Result_Rc01_OWT'].iloc[i]
    Rc02_LOTemp_2nd = Predict1['Result_Rc02_OWT'].iloc[i]
    Rc01_SetTemp_2nd = Predict1['R01_SetTemp'].iloc[i]
    Rc02_SetTemp_2nd = Predict1['R02_SetTemp'].iloc[i]
    Rc01_COPpara_2nd = Predict1['Result_Rc01_COPpara'].iloc[i]
    Rc02_COPpara_2nd = Predict1['Result_Rc02_COPpara'].iloc[i]
    Tank_LOTemp_2nd = Predict1['Result_Tank_LOWT'].iloc[i]
    SHeader_Temp_2nd = Predict1['Result_Supply_Header'].iloc[i]
    RHeader_Temp_2nd = Predict1['Result_Return_Header'].iloc[i]

    Tank_LoadFlow_2nd = Tank_PumpFlow_2nd    

    # Rc Performance
    # (1) Cooling Capacity
    Rc01_Capacity_2nd = CoolingCapa(Rc01_LITemp_2nd, Rc01_LOTemp_2nd, Rc01_PumpFlow_2nd)
    Rc02_Capacity_2nd = CoolingCapa(Rc02_LITemp_2nd, Rc02_LOTemp_2nd, Rc02_PumpFlow_2nd)

    # (2) Interpolate COP
    # Choose setting temperature
    x_Rc01_2nd = OutTemp_2nd
    y_Rc01_2nd = Rc01_Capacity_2nd
    Rc01_InterCOP_2nd = Interpol_Rc(Rc01_SetTemp_2nd, x_Rc01_2nd, y_Rc01_2nd)

    x_Rc02_2nd = OutTemp_2nd
    y_Rc02_2nd = Rc02_Capacity_2nd
    Rc02_InterCOP_2nd = Interpol_Rc(Rc02_SetTemp_2nd, x_Rc02_2nd, y_Rc02_2nd)

    # (3) COP
    Rc01_COP_2nd = Rc01_InterCOP_2nd * (Rc01_COPpara_2nd / 10)
    Rc02_COP_2nd = Rc02_InterCOP_2nd * (Rc02_COPpara_2nd / 10)


    # pump
    # P01 <-- Return Header
    P01_input_2nd = (RHeader_Temp_2nd, Rc01_PumpFlow_2nd)
    P01_input_2nd = np.asarray(P01_input_2nd).astype(np.float16)
    P01_input_2nd = P01_input_2nd.reshape(1, 2)
    P01_para_2nd = P01_para_model.predict(P01_input_2nd)
    P01_para_2nd = P01_para_2nd[0, 0]
    # P02 <-- Return Header
    P02_input_2nd = (RHeader_Temp_2nd, Rc02_PumpFlow_2nd)
    P02_input_2nd = np.asarray(P02_input_2nd).astype(np.float16)
    P02_input_2nd = P02_input_2nd.reshape(1, 2)
    P02_para_2nd = P02_para_model.predict(P02_input_2nd)
    P02_para_2nd = P02_para_2nd[0, 0]
    # P03 <-- Return Header
    P03_input_2nd = (Tank_LOTemp_2nd, Tank_PumpFlow_2nd)
    P03_input_2nd = np.asarray(P03_input_2nd).astype(np.float16)
    P03_input_2nd = P03_input_2nd.reshape(1, 2)
    P03_para_2nd = P03_para_model.predict(P03_input_2nd)
    P03_para_2nd = P03_para_2nd[0, 0]
    # P04 <-- Storage Tank outlet to HEX
    P04_input_2nd = (RHeader_Temp_2nd, HEX_PumpFlow_2nd)
    P04_input_2nd = np.asarray(P04_input_2nd).astype(np.float16)
    P04_input_2nd = P04_input_2nd.reshape(1, 2)
    P04_para_2nd = P04_para_model.predict(P04_input_2nd)
    P04_para_2nd = P04_para_2nd[0, 0]
    # P05 <-- Send Header
    P05_input_2nd = (SHeader_Temp_2nd, Load_PumpFlow_2nd)
    P05_input_2nd = np.asarray(P05_input_2nd).astype(np.float16)
    P05_input_2nd = P05_input_2nd.reshape(1, 2)
    P05_para_2nd = P05_para_model.predict(P05_input_2nd)
    P05_para_2nd = P05_para_2nd[0, 0]


    # Rc01 & Rc02 Performance
    Rc01_Power_2nd = Power(Rc01_Capacity_2nd, Rc01_COP_2nd)
    Rc02_Power_2nd = Power(Rc02_Capacity_2nd, Rc02_COP_2nd)


    # Pump Performance --> Power
        # P01~P05 Performance
    P01_Power_2nd = pump_power(P01_para_2nd, RHeader_Temp_2nd, Rc01_PumpFlow_2nd)
    P02_Power_2nd = pump_power(P02_para_2nd, RHeader_Temp_2nd, Rc02_PumpFlow_2nd)
    P03_Power_2nd = pump_power(P03_para_2nd, Tank_LOTemp_2nd, Tank_PumpFlow_2nd)
    P04_Power_2nd = pump_power(P04_para_2nd, RHeader_Temp_2nd, HEX_PumpFlow_2nd)
    P05_Power_2nd = pump_power(P05_para_2nd, SHeader_Temp_2nd, Load_PumpFlow_2nd)


    # Save to list
    Rc01_InterCOP_result.append(Rc01_InterCOP_2nd)
    Rc01_COP_result.append(Rc01_COP_2nd)
    Rc01_Capacity_result.append(Rc01_Capacity_2nd)
    Rc02_InterCOP_result.append(Rc02_InterCOP_2nd)
    Rc02_COP_result.append(Rc02_COP_2nd)
    Rc02_Capacity_result.append(Rc02_Capacity_2nd)

    P01_para_result.append(P01_para_2nd)
    P02_para_result.append(P02_para_2nd)
    P03_para_result.append(P03_para_2nd)
    P04_para_result.append(P04_para_2nd)
    P05_para_result.append(P05_para_2nd)

    Rc01_Power_result.append(Rc01_Power_2nd)
    Rc02_Power_result.append(Rc02_Power_2nd)
    P01_Power_result.append(P01_Power_2nd)
    P02_Power_result.append(P02_Power_2nd)
    P03_Power_result.append(P03_Power_2nd)
    P04_Power_result.append(P04_Power_2nd)
    P05_Power_result.append(P05_Power_2nd)

    i += 1


    #print(i,'열교환기', P05_Power_result)


# Pump Data Setting
Result_Pump01_Para = np.concatenate((np.atleast_1d(Pump01_para), P01_para_result), axis = 0)
Result_Pump02_Para = np.concatenate((np.atleast_1d(Pump02_para), P02_para_result), axis = 0)
Result_Pump03_Para = np.concatenate((np.atleast_1d(Pump03_para), P03_para_result), axis = 0)
Result_Pump04_Para = np.concatenate((np.atleast_1d(Pump04_para), P04_para_result), axis = 0)
Result_Pump05_Para = np.concatenate((np.atleast_1d(Pump05_para), P05_para_result), axis = 0)

Result_Pump01_Power = np.concatenate((np.atleast_1d(P01_Power), P01_Power_result), axis = 0)
Result_Pump02_Power = np.concatenate((np.atleast_1d(P02_Power), P02_Power_result), axis = 0)
Result_Pump03_Power = np.concatenate((np.atleast_1d(P03_Power), P03_Power_result), axis = 0)
Result_Pump04_Power = np.concatenate((np.atleast_1d(P04_Power), P04_Power_result), axis = 0)
Result_Pump05_Power = np.concatenate((np.atleast_1d(P05_Power), P05_Power_result), axis = 0)

# Rc Performance
Result_Rc01_Capacity = np.concatenate((np.atleast_1d(Rc01_Capacity), Rc01_Capacity_result), axis = 0)
Result_Rc01_InterCOP = np.concatenate((np.atleast_1d(Rc01_Inter_COP), Rc01_InterCOP_result), axis = 0)
Result_Rc01_COP = np.concatenate((np.atleast_1d(Rc01_COP), Rc01_COP_result), axis = 0)

Result_Rc02_Capacity = np.concatenate((np.atleast_1d(Rc02_Capacity), Rc02_Capacity_result), axis = 0)
Result_Rc02_InterCOP = np.concatenate((np.atleast_1d(Rc02_Inter_COP), Rc02_InterCOP_result), axis = 0)
Result_Rc02_COP = np.concatenate((np.atleast_1d(Rc02_COP), Rc02_COP_result), axis = 0)

Result_Rc01_Power = np.concatenate((np.atleast_1d(Rc01_Power), Rc01_Power_result), axis = 0)
Result_Rc02_Power = np.concatenate((np.atleast_1d(Rc02_Power), Rc02_Power_result), axis = 0)

# Save the Result to CSV file
submission = pd.read_csv('C:\\System_Air\\Result\\System_Result_oneday.csv')
submission["Result_Pump01_Para"] = Result_Pump01_Para
submission["Result_Pump02_Para"] = Result_Pump02_Para
submission["Result_Pump03_Para"] = Result_Pump03_Para
submission["Result_Pump04_Para"] = Result_Pump04_Para
submission["Result_Pump05_Para"] = Result_Pump05_Para
submission["Result_Pump01_Power"] = Result_Pump01_Power
submission["Result_Pump02_Power"] = Result_Pump02_Power
submission["Result_Pump03_Power"] = Result_Pump03_Power
submission["Result_Pump04_Power"] = Result_Pump04_Power
submission["Result_Pump05_Power"] = Result_Pump05_Power
submission["Result_Rc01_Capacity"] = Result_Rc01_Capacity
submission["Result_Rc01_InterCOP"] = Result_Rc01_InterCOP
submission["Result_Rc01_COP"] = Result_Rc01_COP
submission["Result_Rc02_Capacity"] = Result_Rc02_Capacity
submission["Result_Rc02_InterCOP"] = Result_Rc02_InterCOP
submission["Result_Rc02_COP"] = Result_Rc02_COP
submission["Result_Rc01_Power"] = Result_Rc01_Power
submission["Result_Rc02_Power"] = Result_Rc02_Power
submission["Result_Tank_Storage"] = Result_Tank_Storage
submission.to_csv('C:\\System_Air\\Result\\System_Result_oneday_power.csv', index = False)


'==============================================================================================================================='

