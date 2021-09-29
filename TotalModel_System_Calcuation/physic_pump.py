

#__funcphysic__.py

'''
-----------------------------------------------------
function for calculating physics formula
-----------------------------------------------------
1) Density of Water : Rho
    └ the density of air-free water at a presuure of 101.325kPa (1 atomosphere) vaild from 0 to 150 ℃

2) Calcluate for pump parameters : pump_para

'''

import numpy as np

# Density of Water

def __init__ (self, Power, Flow, t, pump_para):
    self.Power = Power
    self.Flow = Flow
    self.t = t
    self.pump_para = pump_para


def Rho(t):
    # Density of Water
    Rho = (999.83952+16.945176*t-7.9870401*(pow(10,-3))*(pow(t,2))-46.170461*(pow(10,-6))*(pow(t,3))
            +105.56302*(pow(10,-9))*(pow(t,4))-280.54253*(pow(10,-12))*(pow(t,4)))/(1+16.897850*(pow(10,-3))*t)
    return Rho


def para(Power, Flow, t):
    # Pump parameter
    Quan = Flow*0.00000039
    if Quan == 0:
        para = 0
    else:
        para = (Power*1.02)/(Rho(t)*Quan)
        
    return para


# Reverse Calculate

def pump_power(pump_para, t, Flow):
    # Reverse Calculate Power
    pump_power = (pump_para*(Rho(t)*Flow*0.00000039))/1.02
    return pump_power


'==============================================================================================================================='