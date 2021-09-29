
# physic_refrigerator.py

'''
-----------------------------------------------------
refrigerator's based model
function for calculating physics formula

1) Calculate Parameter : Chilled water temperautre, COP
2) Reverse Compute Chiller Performance
-----------------------------------------------------
'''

'==============================================================================================================================='


import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import cmath as math
import math
import pickle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import scipy as sp
import scipy.interpolate
import scipy.linalg


def __init__(self, x, y, z, BEMS_COP, InterCOP, ILWT, Prd_Flow, OLWT, LWF, ISWT, SWF, Parameter):
    self.x = x
    self.y = y
    self.z = z
    self.BEMS_COP = BEMS_COP
    self.InterCOP = InterCOP
    self.ILWT = ILWT
    self.Prd_Flow = Prd_Flow
    self.OLWT = OLWT
    self.LWF = LWF
    self.ISWT = ISWT
    self.SWF = SWF
    self.Parameter = Parameter


'------------------------'
# Parameter Calculate
'------------------------'

# First Parmeter 
def ChilledWOT_Parameter(OLWT, ILWT, ISWT, LWF):
    # Calculate ChilledWOT and ChilledWOT_Parameter
    
    if LWF == 0:
        Paramter = 0
    else:
        Paramter = (((OLWT - ILWT) - (1 / (ISWT * LWF * 1.162))) * ILWT * ISWT * LWF * 1.162) * -0.0001
    return Paramter

    for i, (a, b, c, d) in enumerate(zip(OLWT, ILWT, ISWT, LWF)):
        if i == 0:
            LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
            LWOT_Parameter = np.array(LWOT_Para)
        else:
            LWOT_Para = ChilledWOT_Parameter(a, b, c, d)
            LWOT_Parameter = np.append(LWOT_Parameter, LWOT_Para)


# Second Parmeter
def Coef_C(x, y, z):
    # Calculate COP Parameter
    # x: Source side inlet temperature 
    # y: Cooling Capacity(kW) 
    # z: COP
    data = np.c_[x,y,z]
    xi = np.linspace(min(x), max(x), 10*len(x))
    yi = np.linspace(min(y), max(y), 10*len(x)) 
    X, Y = np.meshgrid(xi, yi, indexing='xy')

    XX = X.flatten()
    YY = Y.flatten()

    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
    return C


def InterCOP(Coef_C, x, y, z, XX, YY):
    # Interpolate COP
    C = Coef_C(x, y, z)
    InterCOP = C[4]*XX**2. + C[5]*YY**2. + C[3]*XX*YY + C[1]*XX + C[2]*YY + C[0]
    return InterCOP


def COP_para(BEMS_COP, InterCOP):
    # Calculate COP Parameter

    if InterCOP == 0:
        COP_Para = 0
    else:
        COP_Para = (BEMS_COP / InterCOP) * 10

    return COP_Para


'------------------------'
# Reverse Calculate
'------------------------'

def COP(COPPara, InterCOP):
    # Calculate COP
    
    COP_Pred = []

    for i, (a, b) in enumerate(zip(COPPara, InterCOP)):
        if i == 0:
            COP = b * ( a / 10)
        else:
            COP = b * ( a / 10)

        COP_Pred.append(COP)

    return COP_Pred


def OLWT(ILWT, ISWT, LWF, OLWT_Parameter):
    # Chilled Outlet Temperature

    if LWF == 0:
        OLWT = ILWT
    else:
        OLWT = ILWT + (1 / (ISWT * LWF * 1.162)) + ((1 / (ILWT * ISWT * LWF * 1.162)) * (-10000 * OLWT_Parameter))
    return OLWT


def CoolingCapa(ILWT, OLWT, LWF):
    # Cooling Capacity
    
    if LWF == 0 :
        CoolingCapa = 0
    else:
        CoolingCapa = (ILWT - OLWT) * LWF * 1.162
    return CoolingCapa

    for i, (a, b, c) in enumerate(zip(ILWT, OLWT, LWF)):
        if i == 0:
            Capacity_cool = CoolingCapa(a, b, c)
            Capacitycool = np.array(Capacity_cool)
        else:
            Capacity_cool = CoolingCapa(a, b, c)
            Capacitycool = np.append(Capacitycool, Capacity_cool)


def Chiller_Power(CoolingCapa, COP):
    # Power
    
    Power_pred = []

    for i, (a, b) in enumerate(zip(CoolingCapa, COP)):
        if b == 0:
            Power = 0
        else:
            Power = a / b

        Power_pred.append(Power)

    return Power_pred

def Power(CoolingCapa, COP):
    # Power
    if COP == 0:
        Power = 0
    else:
        Power = CoolingCapa / COP 
    return Power


def OSWT(ISWT, COP, CoolingCapa, SWF):
    # leaving Source side water temperature = Cooling Water (Outlet)
    
    OSWT_Pred = []

    for i, (a, b, c, d) in enumerate(zip(ISWT, COP, CoolingCapa, SWF)):
        if i == 0:
            if d == 0:
                OSWTPred =  a
            else:
                OSWTPred =  a - (((b / c) + c) / d * 1.162)
        else:
            if d == 0:
                OSWTPred =  a
            else:
                OSWTPred =  a - (((b / c) + c) / d * 1.162)
            
        OSWT_Pred.append(OSWTPred)

    return OSWT_Pred


'==============================================================================================================================='
