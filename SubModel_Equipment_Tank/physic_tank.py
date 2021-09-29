
# physic_tank.py

'''
-----------------------------------------------------
function for calculating physics formula

1) Calculate Parameter : Charging Time, Discharging Time
2) Reverse Compute Tank Outlet Temperature
-----------------------------------------------------
'''

'==============================================================================================================================='

import numpy as np


def __init__(self, SWIT, SWOT, LWIT, LWOT, SWF, LWF, SWOT_1min, LOWT_1min, Node10_Heat_Discharg, Node1_Heat_Charg, Tank_ChargStorage, Tank_DischargStorage):
    self.SWIT = SWIT
    self.SWOT = SWOT
    self.LWIT = LWIT
    self.LWOT = LWOT
    self.SWF = SWF
    self.LWF = LWF
    self.Node10_Heat_Discharg = Node10_Heat_Discharg
    self.Tank_ChargStorage = Tank_ChargStorage
    self.Node1_Heat_Charg = Node1_Heat_Charg
    self.Tank_DischargStorage = Tank_DischargStorage
    self.SWOT_1min = SWOT_1min
    self.LOWT_1min = LOWT_1min


'------------------------'
# Parameter Calculate
'------------------------'

# 10 layer's Parmeter (kW - m^3/h)
# charging time 
def Charging_para(SWIT, SWOT, SWF):
    charge_para = ((SWOT - SWIT) * 2 * SWF * 1.162) / 10
    return charge_para

# discharging time
def layer10_para(LWIT, SWOT, LFW, Node10_Heat_Discharg):

    layer10parameter = []

    for i, (a, b, c, d) in enumerate(zip(LWIT, SWOT, LFW, Node10_Heat_Discharg)):
        if i == 0:
            layer10para =  d - (c * 1.162 * (a - b))
            if layer10para < 0:
                layer10para = 0
            else:
                layer10para = layer10para
        else:
            layer10para =  d - (c * 1.162 * (a - b))
            if layer10para < 0:
                layer10para = 0
            else:
                layer10para = layer10para

        layer10parameter.append(layer10para)


# 1 layer's Parmeter
# discharging time
def Discharging_para(LWIT, LWOT, LWF):
    discharge_para = ((LWOT - LWIT) * 2 * LWF * 1.162) / -10
    return discharge_para

# charging time
def layer01_para(SWIT, LOWT, SWF, Node1_Heat_Charg):

    layer01parameter = []

    for i, (a, b, c, d) in enumerate(zip(SWIT, LOWT, SWF, Node1_Heat_Charg)):
        if i == 0:
            layer01para =  d - (c * 1.162 * (a - b))
            if layer01para < 0:
                layer01para = 0
            else:
                layer01para = layer01para
        else:
            layer01para =  d - (c * 1.162 * (a - b))
            if layer01para < 0:
                layer01para = 0
            else:
                layer01para = layer01para

        layer01parameter.append(layer01para)


'------------------------'
# Reverse Calculate
'------------------------'

# 10 layer
def Tank_SWOT(SWIT, LWIT, SWOT_1min, Node10_Heat_Discharg, SWF, LWF, Charging_para, layer10para):
    # charging time
    if SWF != 0 and LWF == 0:
        TankSWOT = SWIT + ((1 / (2 * SWF * 1.162)) * Charging_para * 10)
        # discharging time
    elif LWF != 0 and SWF == 0: 
        if layer10para == 0:
            TankSWOT = SWOT_1min
        else:
            TankSWOT = LWIT - (Node10_Heat_Discharg / (LWF * 1.162)) + (layer10para / (LWF * 1.162))
    # 100% off
    else:
        TankSWOT = SWOT_1min
    return TankSWOT


# Tank Total Charging Heat Sotrage
def Tank_Charging(SWIT, TankSWOT, SWF, LWF):
    if SWF != 0 and LWF == 0:
        TankCharging = ((SWIT-TankSWOT)*SWF*1000*4.184)
    else:
        TankCharging = 0
    return TankCharging


# 10layer Heat Sotrage
def Tank_10layer_charging(LWIT, TankSWOT, SWF, LWF, layer10para):
    if LWF != 0 and SWF == 0: 
        Tank_10layer_charging = ((LWIT - TankSWOT) * LWF * 1.162) + layer10para
    else:
        Tank_10layer_charging = 0
    return Tank_10layer_charging


# 01 layer outlet  temperature
def Tank_LWOT(LWIT, SWIT, LOWT_1min, Node1_Heat_Charg, LWF, SWF, Discharging_para, layer01para):
    # discharging time
    if LWF != 0 and SWF == 0:
        TankLWOT = LWIT + ((1 / (2 * LWF * 1.162)) * Discharging_para * -10)
        # charging time
    elif SWF != 0 and LWF == 0:
        if layer01para == 0:
            TankLWOT = LOWT_1min
        else:
            TankLWOT = SWIT - (Node1_Heat_Charg / (SWF * 1.162)) + (layer01para / (SWF * 1.162))
    # 100% off
    else:
        TankLWOT = LOWT_1min
    return TankLWOT


# Tank Total Discharging Heat Sotrage
def Tank_Discharging(LWIT, TankLWOT, SWF, LWF):
    if LWF != 0 and SWF == 0:
        TankDischarging = ((LWIT-TankLWOT)*LWF*1000*4.184)
    else:
        TankDischarging = 0
    return TankDischarging


# 01layer Heat Sotrage
def Tank_01layer_charging(SWIT, TankLWOT, SWF, LWF, layer01para):
    if SWF != 0 and LWF == 0:
        Tank_01layer_charging = ((SWIT - TankLWOT) * SWF * 1.162) + layer01para
    else:
        Tank_01layer_charging = 0
    return Tank_01layer_charging


'==============================================================================================================================='