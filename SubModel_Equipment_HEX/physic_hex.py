

#__hex__.py

'''
-----------------------------------------------------
hex's grey-box model: based on Counter Flow type
-----------------------------------------------------
'''

'==============================================================================================================================='

import numpy as np
import math
from math import sqrt


def __init__ (self, HEX_S_ITemp, HEX_L_ITemp, HEX_S_OTemp, HEX_L_OTemp, HEX_S_flow, HEX_L_flow):
    self.HEX_S_ITemp = HEX_S_ITemp
    self.HEX_L_ITemp = HEX_L_ITemp
    self.HEX_S_OTemp = HEX_S_OTemp
    self.HEX_L_OTemp = HEX_L_OTemp
    self.HEX_S_flow = HEX_S_flow
    self.HEX_L_flow = HEX_L_flow


def HEX_KA(HEX_S_ITemp, HEX_L_ITemp, HEX_S_OTemp, HEX_L_OTemp, HEX_S_flow, HEX_L_flow):

    global wswl
    
    ws = HEX_S_flow * 4.186
    wl = HEX_L_flow * 4.186

    if ws == 0 or wl == 0:
        wswl = 0
    elif HEX_S_ITemp - HEX_L_ITemp > 0:
        wswl = wl / ws
    elif HEX_S_ITemp - HEX_L_ITemp < 0:
        wswl = ws / wl

    if wswl == 0:
        KA = 0
    elif ws < wl:
        KA = -1
    else:
        if HEX_S_ITemp - HEX_L_ITemp < 0:
            if HEX_L_OTemp - HEX_S_ITemp > 0:
                if wswl != 1:
                    y = HEX_L_ITemp - HEX_S_ITemp
                    x = (HEX_S_ITemp - HEX_L_OTemp) * wswl / (HEX_L_ITemp - HEX_L_OTemp - y * wswl)
                    if x >= 0:
                        z = math.log(x)
                        KA = z * ws / (1.0 - wswl)
                elif wswl == 1:
                    KA = (HEX_L_ITemp - HEX_L_OTemp) * ws / (HEX_L_OTemp - HEX_S_ITemp)
            else:
                KA = -1
        else:
            if HEX_L_OTemp < HEX_S_ITemp:
                if wswl != 1:
                    y = HEX_S_ITemp - HEX_L_ITemp
                    x = (HEX_L_ITemp - HEX_S_OTemp) * wswl / (HEX_S_ITemp - HEX_S_OTemp - y * wswl)
                    if x >= 0:
                        z = math.log(x)
                        KA = z * wl / (1.0 - wswl)
                elif wswl == 1:
                    KA = (HEX_S_ITemp - HEX_S_OTemp) * wl / (HEX_S_OTemp - HEX_L_ITemp)
            else:
                KA = -1
    return KA

    for i, (a, b, c, d, e, f) in enumerate(zip(HEX_S_ITemp, HEX_L_ITemp, HEX_S_OTemp, HEX_L_OTemp, HEX_S_flow, HEX_L_flow)):
        if i == 0:
            HEX01_KA = HEX_KA(a, b, c, d, e, f)
            KA01 = np.array(HEX01_KA)
        else:
            HEX01_KA = HEX_KA(a, b, c, d, e, f)
            KA01 = np.append(KA01, HEX01_KA)


def HEX_SOT(HEX_S_ITemp, HEX_L_ITemp, HEX_S_flow, HEX_L_flow, KA):

    global wswl

    ws = HEX_S_flow * 4.186
    wl = HEX_L_flow * 4.186

    if ws == 0 or wl == 0:
        wswl = 0
    elif HEX_S_ITemp > HEX_L_ITemp:
        wswl = wl / ws
    elif HEX_S_ITemp < HEX_L_ITemp:
        wswl = ws / wl

    if wswl == 0:
        OTemp = HEX_S_ITemp
    elif ws < wl:
        OTemp = HEX_S_ITemp
    else:
        if HEX_S_ITemp < HEX_L_ITemp:
            z = KA * (1.0 - wswl) / ws
            if wswl != 1.0:
                x = math.exp(z)
                y = HEX_L_ITemp - HEX_S_ITemp
                OTemp = HEX_S_ITemp + y * (1.0 - x) / (wswl - x)
            elif wswl == 1.0:
                y = HEX_L_ITemp - HEX_S_ITemp
                OTemp = HEX_S_ITemp + y * KA / (ws + KA)
        else:
            z = KA * (1.0 - wswl) / wl
            if wswl != 1.0:
                x = math.exp(z)
                y = HEX_S_ITemp - HEX_L_ITemp
                OTemp = HEX_S_ITemp - y * (1.0 - x) / (1.0 - x / wswl)
            elif wswl == 1.0:
                y = HEX_S_ITemp - HEX_L_ITemp
                OTemp = HEX_S_ITemp - y * KA / (wl + KA)
    return OTemp


    for i, (a, b, c, d, e) in enumerate(zip(HEX_S_ITemp, HEX_L_ITemp, HEX_S_flow, HEX_L_flow, KA)):
        if i == 0:
            HEX_S_OT = HEX_S_OT(a, b, c, d, e)
            HEX_SOWT = np.array(HEX_S_OT)
        else:
            HEX_S_OT = HEX_S_OT(a, b, c, d, e)
            HEX_SOWT = np.append(HEX_SOWT, HEX_S_OT)


def HEX_LOT(HEX_S_ITemp, HEX_L_ITemp, HEX_S_flow, HEX_L_flow, KA):

    global wswl

    ws = HEX_S_flow * 4.186
    wl = HEX_L_flow * 4.186

    if ws == 0 or wl == 0:
        wswl = 0
    elif HEX_S_ITemp > HEX_L_ITemp:
        wswl = wl / ws
    elif HEX_S_ITemp < HEX_L_ITemp:
        wswl = ws / wl

    if wswl == 0:
        OTemp = HEX_L_ITemp
    elif ws < wl:
        OTemp = HEX_L_ITemp
    else:
        if HEX_S_ITemp < HEX_L_ITemp:
            z = KA * (1.0 - wswl) / ws
            if wswl != 1.0:
                x = math.exp(z)
                y = HEX_L_ITemp - HEX_S_ITemp
                OTemp = HEX_L_ITemp - y * (1.0 - x) / (1.0 - x / wswl)
            elif wswl == 1.0:
                y = HEX_L_ITemp - HEX_S_ITemp
                OTemp = HEX_L_ITemp - y * KA / (ws + KA)
        else:
            z = KA * (1.0 - wswl) / wl
            if wswl != 1.0:
                x = math.exp(z)
                y = HEX_S_ITemp - HEX_L_ITemp
                OTemp = HEX_L_ITemp + y * (1.0 - x) / (wswl - x)
            elif wswl == 1.0:
                y = HEX_S_ITemp - HEX_L_ITemp
                OTemp = HEX_L_ITemp + y * KA / (wl + KA)
    return OTemp

    for i, (a, b, c, d, e) in enumerate(zip(HEX_S_ITemp, HEX_L_ITemp, HEX_S_flow, HEX_L_flow, KA)):
        if i == 0:
            HEX_L_OT = HEX_L_OT(a, b, c, d, e)
            HEX_LWOT = np.array(HEX_L_OT)
        else:
            HEX_L_OT = HEX_L_OT(a, b, c, d, e)
            HEX_LWOT = np.append(HEX_LWOT, HEX_L_OT)

'==============================================================================================================================='
