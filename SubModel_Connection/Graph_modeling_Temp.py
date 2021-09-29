

import pickle
import pandas as pd
import numpy as np
import networkx as nx


## Inverse Modeling of HVAC system 
# Finding the connection equipment to equipment


'--------------------------------------------------'

# Read Data
Train = pd.read_csv('C:\\System_Air\\Input\\Train\\TRNSYS_2016.csv')
Predict = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')

'--------------------------------------------------'


### Algorithm for judgement
# Make the Adjacency matrix
Admatrix = []
Lableing = []
Balancing = []

for i in range(len(Train)):

    ### Read the Predict Data
    # flow data
    Rc_flow_1 = Predict['R01_Flow'].iloc[i]
    Rc_flow_2 = Predict['R02_Flow'].iloc[i]
    HEXs_flow = Predict['HEX01_SFlow'].iloc[i]
    HEXl_flow = Predict['HEX01_LFlow'].iloc[i]
    Ts_flow = Predict['ST01_Sflow'].iloc[i]
    Tl_flow = Predict['ST01_Lflow'].iloc[i]
    HS_flow = Predict['SH01_Flow'].iloc[i]
    HR_flow = Predict['RH01_Flow'].iloc[i]

    # Outlet Temperature data
    Rc_outtemp_1 = Predict['R01_OTemp'].iloc[i]
    Rc_outtemp_2 = Predict['R02_OTemp'].iloc[i]
    HEXs_outtemp = Predict['HEX01_SOTemp'].iloc[i]
    HEXl_outtemp = Predict['HEX01_LOTemp'].iloc[i]
    Ts_outtemp = Predict['ST01_SOTemp'].iloc[i]
    Tl_outtemp = Predict['ST01_LOTemp'].iloc[i]
    HR_temp = Predict['RH01_Temp'].iloc[i]

    # Inlet Temperature data --> Final objective data
    Rc_intemp_1 = Predict['R01_ITemp_1aft'].iloc[i]
    Rc_intemp_2 = Predict['R02_ITemp_1aft'].iloc[i]
    HEXs_intemp = Predict['HEX01_SITemp_1aft'].iloc[i]
    HEXl_intemp = Predict['HEX01_LITemp_1aft'].iloc[i]
    Ts_intemp = Predict['ST01_SITemp_1aft'].iloc[i]
    Tl_intemp = Predict['ST01_LITemp_1aft'].iloc[i]
    HS_temp = Predict['SH01_Temp_1aft'].iloc[i]
    
    
    ## 1st Check the Header Flowrate
    if HS_flow == 0:  
        ### Charging Time    
        if Ts_flow != 0:            
            ## Rc1 connect to Tank Source
            if Rc_flow_1 != 0:
                if abs((Ts_flow-Rc_flow_1)/Ts_flow*100) <= 10:
                    A_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                    B_delt = np.zeros((1, 7))
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                    F_delt = np.zeros((1, 7))
                    G_delt = np.zeros((1, 7))
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 0
                    Balance = ["Balance"]
                else:
                    # Judgment Postponement
                    Label = 0.5
                    Balance = ["Unbalance"]
            
            ## Rc2 connect to Tank Source
            elif Rc_flow_2 != 0:
                if abs(((Ts_flow-Rc_flow_2)/Ts_flow)*100) <= 10:
                    A_delt = np.zeros((1, 7))
                    B_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                    F_delt = np.zeros((1, 7))
                    G_delt = np.zeros((1, 7))
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 1
                    Balance = ["Balance"]
                else:
                    # Judgment Postponement
                    Label = 1.5
                    Balance = ["Unbalance"]
            else:                
                # Judgment Postponement
                Label = 1.5
                Balance = ["Unbalance"]

        ## Tank Storage off and system off
        else: 
            if Rc_flow_1 == 0 and Rc_flow_2 == 0:         
                Delta_Temp = np.zeros((7,7))
                Label = 2
                Balance = ["Off"]
            else:
                # Judgment Postponement
                Label = 2.5
                Balance = ["Unbalance"]


    ## Discharging Time : Header Inlet Flowrate ON
    else:
        ## HEX load to Header                   
        if HEXl_flow != 0:
            D2G = 1                 
            G2D = 1                
            leftflow_head = HS_flow-HEXl_flow

                
            ## Rc1, Rc2, Tank load ON - having flowrate 
            if Rc_flow_1 !=0 and Rc_flow_2 !=0 and Tl_flow != 0:

                ## Rc1, Rc2, Tank load to Header 
                if abs((((Rc_flow_1+Rc_flow_2+Tl_flow)-leftflow_head)/leftflow_head)*100) <= 10:
                    # Judgment Postponement : Becuase HEX Source need to Flowrate, But Flowrate already saticified balance, no left Flowrate
                    Label = 3.5
                    Balance = ["Unbalance"]

                ## Find the only two equipment to connect Header, the other one connect HEX Source
                else:
                    # mean_Rc1_Rc2 = ((Rc_flow_1*Rc_outtemp_1)+(Rc_flow_2*Rc_outtemp_2))/(Rc_flow_1+Rc_flow_2)
                    # mean_Rc1_Tl = ((Rc_flow_1*Rc_outtemp_1)+(Tl_flow*Tl_outtemp))/(Rc_flow_1+Tl_flow)
                    # mean_Rc2_Tl = ((Rc_flow_2*Rc_outtemp_2)+(Tl_flow*Tl_outtemp))/(Rc_flow_2+Tl_flow)

                    ## Rc1, Rc2 to Header / Tank Load to HEX Source
                    if abs((((Rc_flow_1+Rc_flow_2)-leftflow_head)/leftflow_head)*100) <= 10:    # abs(((mean_Rc1_Rc2-HS_temp)/HS_temp)*100) <= 10 and 
                        A2G = 1
                        B2G = 1
                        G2A = 1
                        G2B = 1
                        ## Tank load to HEX source 
                        if abs(((Tl_flow-HEXs_flow)/HEXs_flow)*100) <= 10:
                            F2C = 1
                            C2F = 1
                            ## Charing and Discharing: Impossible
                            if Ts_flow != 0:
                                # Judgment Postponement : Flowrate already saticified balance, no left Flowrate
                                Label = 3.5
                                Balance = ["Unbalance"]
                            ## No Charing, Only Discharing
                            else:               
                                A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.zeros((1, 7))
                                F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                G_delt = np.array([1, 1, 0, 1, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 3
                                Balance = ["Balance"]
                        else:
                            # Judgment Postponement
                            Label = 3.5
                            Balance = ["Unbalance"]


                    ## Rc1, Rc2 to Header and Extra Flowrate go to Tank Source (Tank Load to HEX Source)
                    elif abs((((Rc_flow_1+Rc_flow_2)-leftflow_head)/leftflow_head)*100) > 10:    # abs(((mean_Rc1_Rc2-HS_temp)/HS_temp)*100) <= 10 and 
                        A2G = 1
                        B2G = 1
                        G2A = 1
                        G2B = 1
                        ## Tank load to HEX source 
                        if abs(((Tl_flow-HEXs_flow)/HEXs_flow)*100) <= 10:
                            F2C = 1
                            C2F = 1
                            ## Charing and Discharing: Possible
                            if Ts_flow != 0:
                                ## Extra Rc1 to Tank source   
                                if abs(((Rc_outtemp_1-Ts_intemp)/Ts_intemp)*100) <= 10:
                                    ## Left Flowrate of Rc1
                                    leftflow_equip_1 = Rc_flow_1-(leftflow_head-Rc_flow_2)

                                    if abs(((leftflow_equip_1-Ts_flow)/Ts_flow)*100) <= 10:
                                        A2E = 1
                                        E2A = 1

                                        A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                        B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                        C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                                        D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                        E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                        F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                        G_delt = np.array([1, 1, 0, 1, 0, 0, 0])
                                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                        Label = 4
                                        Balance = ["Balance"]
                                    else:
                                        # Judgment Postponement
                                        Label = Label
                                        Balance = ["Unbalance"]

                                ## Extra Rc2 to Tank source                     
                                elif abs(((Rc_outtemp_2-Ts_intemp)/Ts_intemp)*100) <= 10:
                                    ## Left Flowrate of Rc2
                                    leftflow_equip_2 = Rc_flow_2-(leftflow_head-Rc_flow_1)

                                    if abs(((leftflow_equip_2-Ts_flow)/Ts_flow)*100) <= 10:
                                        B2E = 1
                                        E2B = 1
                                    
                                        A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                        B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                        C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                                        D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                        E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                        F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                        G_delt = np.array([1, 1, 0, 1, 0, 0, 0])
                                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                        Label = 5
                                        Balance = ["Balance"]
                                    else:
                                        # Judgment Postponement
                                        Label = 5.5
                                        Balance = ["Unbalance"]
                                else:
                                    # Judgment Postponement
                                    Label = 5.5
                                    Balance = ["Unbalance"]
                            else:
                                # Judgment Postponement
                                Label = 5.5
                                Balance = ["Unbalance"]
                        else:
                            # Judgment Postponement
                            Label = 5.5
                            Balance = ["Unbalance"]
                                

                    ## Rc1, Tank load to Header / Rc2 to HEX Source
                    elif abs((((Rc_flow_1+Tl_flow)-leftflow_head)/leftflow_head)*100) <= 10:    # abs(((mean_Rc1_Tl-HS_temp)/HS_temp)*100) <= 10 and 
                        A2G = 1
                        F2G = 1
                        G2A = 1
                        G2F = 1
                        ## Rc2 to HEX source 
                        if abs(((Rc_flow_2-HEXs_flow)/HEXs_flow)*100) <= 10:
                            B2C = 1
                            C2B = 1
                            ## Charing and Discharing: Impossible
                            if Ts_flow != 0:
                                # Judgment Postponement : Flowrate already saticified balance, no left Flowrate
                                Label = Label
                                Balance = ["Unbalance"]
                            ## No Charing, Only Discharing
                            else:               
                                A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                B_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.zeros((1, 7))
                                F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                G_delt = np.array([1, 0, 0, 1, 0, 1, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 6
                                Balance = ["Balance"]

                        ## Rc2 to HEX source and Rc2 to Tank Source at the same time
                        else:
                            B2C = 1
                            C2B = 1
                            ## Charing and Discharing: Possible
                            if Ts_flow != 0:
                                ## Left Flowrate of Rc2
                                leftflow_equip_3 = Rc_flow_2-HEXs_flow
                                    
                                ## left Rc2 flowrate to Tank Source
                                if abs(((leftflow_equip_3-Ts_flow)/Ts_flow)*100) <= 10:
                                    B2E = 1
                                    E2B = 1

                                    A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    B_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                    C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    G_delt = np.array([1, 0, 0, 1, 0, 1, 0])
                                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                    Label = 7
                                    Balance = ["Balance"]
                                else:
                                    # Judgment Postponement
                                    Label = 7.5
                                    Balance = ["Unbalance"]
                            else:
                                # Judgment Postponement
                                Label = 7.5
                                Balance = ["Unbalance"]
                                

                    ## Rc1, Tank load to Header and Extra Flowrate go to Tank Source (Rc2 to HEX Source)
                    elif abs((((Rc_flow_1+Tl_flow)-leftflow_head)/leftflow_head)*100) > 10:    # abs(((mean_Rc1_Tl-HS_temp)/HS_temp)*100) <= 10 and 
                        A2G = 1
                        F2G = 1
                        G2A = 1
                        G2F = 1
                        ## Rc2 to HEX source 
                        if abs(((Rc_flow_2-HEXs_flow)/HEXs_flow)*100) <= 10:
                            B2C = 1
                            C2B = 1
                            ## Charing and Discharing: Possible
                            if Ts_flow != 0:
                                ## Left Flowrate of Rc1 
                                leftflow_equip_4 = Rc_flow_1-(leftflow_head-Tl_flow)

                                ## Tank Load Flowrate 0 / Rc1 Flowrate left and send to Tank Source
                                if abs(((leftflow_equip_4-Ts_flow)/Ts_flow)*100) <= 10:
                                    A2E = 1
                                    E2A = 1

                                    A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                    B_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                    C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    G_delt = np.array([1, 0, 0, 1, 0, 1, 0])
                                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                    Label = 8
                                    Balance = ["Balance"]
                                else:
                                    # Judgment Postponement
                                    Label = 8.5
                                    Balance = ["Unbalance"]
                            else:
                                # Judgment Postponement
                                Label = 8.5
                                Balance = ["Unbalance"]

                        ## Rc2 to HEX source befer Rc2 flowrate left  
                        else:
                            B2C = 1
                            C2B = 1
                            ## Charing and Discharing: Possible
                            if Ts_flow != 0:
                                ## Left Flowrate of Rc2
                                leftflow_equip_3 = Rc_flow_2-HEXs_flow
                                ## Left Flowrate of Rc1 
                                leftflow_equip_4 = Rc_flow_1-(leftflow_head-Tl_flow)
                                    
                                ## Rc1 and Rc2 Flowrate left and send to Tank Source at the same time
                                if abs((((leftflow_equip_3+ leftflow_equip_4)-Ts_flow)/Ts_flow)*100) <= 10:
                                    A2E = 1
                                    E2A = 1
                                    B2E = 1
                                    E2B = 1

                                    A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                    B_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                    C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    E_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    G_delt = np.array([1, 0, 0, 1, 0, 1, 0])
                                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                    Label = 9
                                    Balance = ["Balance"]
                                else:
                                    # Judgment Postponement
                                    Label = 9.5
                                    Balance = ["Unbalance"]
                            else:
                                # Judgment Postponement
                                Label = 9.5
                                Balance = ["Unbalance"]
                                        

                    ## Rc2, Tank load to Header / Rc1 to HEX Source
                    elif abs((((Rc_flow_2+Tl_flow)-leftflow_head)/leftflow_head)*100) <= 10:    # abs(((mean_Rc2_Tl-HS_temp)/HS_temp)*100) <= 10 and 
                        B2G = 1
                        F2G = 1
                        G2B = 1
                        G2F = 1

                        ## Rc1 to HEX source 
                        if abs(((Rc_flow_2-HEXs_flow)/HEXs_flow)*100) <= 10:
                            A2C = 1
                            C2A = 1
                            ## Charing and Discharing: Impossible
                            if Ts_flow != 0:
                                # Judgment Postponement : Flowrate already saticified balance, no left Flowrate
                                Label = 10.5
                                Balance = ["Unbalance"]
                            ## No Charing, Only Discharing
                            else:               
                                A_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.zeros((1, 7))
                                F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                G_delt = np.array([0, 1, 0, 1, 0, 1, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 10
                                Balance = ["Balance"]

                        ## Rc1 to HEX source and Rc1 to Tank Source at the same time
                        else:
                            A2C = 1
                            C2A = 1
                            ## Charing and Discharing: Possible
                            if Ts_flow != 0:
                                ## Left Flowrate of Rc1 
                                leftflow_equip_5 = Rc_flow_1-HEXs_flow 

                                ## left Rc1 flowrate to Tank Source
                                if abs(((leftflow_equip_5-Ts_flow)/Ts_flow)*100) <= 10:
                                    A2E = 1
                                    E2A = 1

                                    A_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                    B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    G_delt = np.array([0, 1, 0, 1, 0, 1, 0])
                                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                    Label = 11
                                    Balance = ["Balance"]
                                else:
                                    # Judgment Postponement
                                    Label = 11.5
                                    Balance = ["Unbalance"]
                            else:
                                # Judgment Postponement
                                Label = 11.5
                                Balance = ["Unbalance"]
                                

                    ## Rc2, Tank load to Header and Extra Flowrate go to Tank Source (Rc1 to HEX Source)
                    elif abs((((Rc_flow_2+Tl_flow)-leftflow_head)/leftflow_head)*100) > 10:    # abs(((mean_Rc2_Tl-HS_temp)/HS_temp)*100) <= 10 and 
                        B2G = 1
                        F2G = 1
                        G2B = 1
                        G2F = 1
                        ## Rc1 to HEX source 
                        if abs(((Rc_flow_1-HEXs_flow)/HEXs_flow)*100) <= 10:
                            A2C = 1
                            C2A = 1
                            ## Charing and Discharing: Possible
                            if Ts_flow != 0:
                                ## Left Flowrate of Rc2
                                leftflow_equip_6 = Rc_flow_2-(leftflow_head-Tl_flow)

                                ## Tank Load Flowrate 0 / Rc2 Flowrate left and send to Tank Source
                                if abs(((leftflow_equip_6-Ts_flow)/Ts_flow)*100) <= 10:
                                    B2E = 1
                                    E2B = 1

                                    A_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                    B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                    C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    G_delt = np.array([0, 1, 0, 1, 0, 1, 0])
                                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                    Label = 12
                                    Balance = ["Balance"]
                                else:
                                    # Judgment Postponement
                                    Label = 12.5
                                    Balance = ["Unbalance"]
                            else:
                                # Judgment Postponement
                                Label = 12.5
                                Balance = ["Unbalance"]

                        ## Rc1 to HEX source befer Rc2 flowrate left  
                        else:
                            A2C = 1
                            C2A = 1
                            ## Charing and Discharing: Possible
                            if Ts_flow != 0:
                                ## Left Flowrate of Rc1
                                leftflow_equip_5 = Rc_flow_1-HEXs_flow
                                ## Left Flowrate of Rc2
                                leftflow_equip_6 = Rc_flow_2-(leftflow_head-Tl_flow)
                                    
                                ## Rc1 and Rc2 Flowrate left and send to Tank Source at the same time
                                if abs((((leftflow_equip_5+leftflow_equip_6)-Ts_flow)/Ts_flow)*100) <= 10:
                                    A2E = 1
                                    E2A = 1
                                    B2E = 1
                                    E2B = 1

                                    A_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                    B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                    C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    E_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                    G_delt = np.array([0, 1, 0, 1, 0, 1, 0])
                                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                    Label = 13
                                    Balance = ["Balance"]
                                else:
                                    # Judgment Postponement
                                    Label = 13.5
                                    Balance = ["Unbalance"]
                            else:
                                # Judgment Postponement
                                Label = 13.5
                                Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 13.5
                        Balance = ["Unbalance"]
                                    
            
            ## Rc1, Rc2 ON - having flowrate 
            elif Rc_flow_1 !=0 and Rc_flow_2 !=0 and Tl_flow == 0:

                ## Rc1, Rc2 to Header 
                if abs((((Rc_flow_1+Rc_flow_2)-leftflow_head)/leftflow_head)*100) <= 10:
                    # Judgment Postponement : Becuase HEX Source need to Flowrate, But Flowrate already saticified balance, no left Flowrate
                    Label = Label
                    Balance = ["Unbalance"]

                ## Find the only one equipment connect to Header, the other one connect to HEX Source
                else:
                    ### Case1 : Rc1->HEX, Rc2->Header befer left Flowrate each Refrigerator
                    ### Case2 : Rc1->Header, Rc1->HEX befer left Flowrate each Refrigerator


                    ## Rc1 Flowrate to HEX source (Case1)
                    if abs(((Rc_flow_1-HEXs_flow)/HEXs_flow)*100) <= 10:
                        A2C = 1
                        C2A = 1
                        B2G = 1
                        G2B = 1
                        ## Charing and Discharing: Possible --> Rc2 Flowrate left
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc2
                            leftflow_equip_7 = Rc_flow_2-leftflow_head
                                
                            ## Rc1 left Flowrate 0 / Rc2 left Flowrate to Tank Source
                            if abs(((leftflow_equip_7-Ts_flow)/Ts_flow)*100) <= 10:
                                B2E = 1
                                E2B = 1

                                A_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                F_delt = np.zeros((1, 7))
                                G_delt = np.array([0, 1, 0, 1, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 14
                                Balance = ["Balance"]
                            ## Rc1 left Flowrate 0 / Rc2 Flowrate still left
                            else:
                                # Judgment Postponement
                                Label = 14.5
                                Balance = ["Unbalance"]
                        ## Tank Source Flowrate 0 --> Rc2 Flowrate left so Flowrate Unbalance
                        else:
                            # Judgment Postponement
                            Label = 14.5
                            Balance = ["Unbalance"]

                    ## Rc1 to HEX source and Rc1 left flowrate --> Rc1 and Rc2 Flowrate also left
                    else:
                        A2C = 1
                        C2A = 1
                        B2G = 1
                        G2B = 1
                        ## Charing and Discharing: Possible
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc2
                            leftflow_equip_7 = Rc_flow_2-leftflow_head
                            ## Left Flowrate of Rc1
                            leftflow_equip_8 = Rc_flow_1-HEXs_flow

                            ## Rc1 and Rc2 left Flowrate send to Tank Source at the same time
                            if abs((((leftflow_equip_7+leftflow_equip_8)-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                B2E = 1
                                E2A = 1
                                E2B = 1
                                    
                                A_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                                F_delt = np.zeros((1, 7))
                                G_delt = np.array([0, 1, 0, 1, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 15
                                Balance = ["Balance"]
                            ## Rc1 and Rc2 left Flowrate, but Over the Tank Source Flowrate
                            else:
                                # Judgment Postponement
                                Label = 15.5
                                Balance = ["Unbalance"]
                        ## Rc1 and Rc2 left Flowrate, but Tank Source Flowrate 0
                        else:
                            # Judgment Postponement
                            Label = 15.5
                            Balance = ["Unbalance"]


                    ## Rc2 Flowrate to HEX source (Case2)
                    if abs(((Rc_flow_2-HEXs_flow)/HEXs_flow)*100) <= 10:
                        B2C = 1
                        C2B = 1
                        A2G = 1
                        G2A = 1
                        ## Charing and Discharing: Possible --> Rc1 Flowrate left
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc1
                            leftflow_equip_9 = Rc_flow_1-leftflow_head
                                
                            ## Rc2 left Flowrate 0 / Rc1 left Flowrate to Tank Source
                            if abs(((leftflow_equip_9-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                E2A = 1

                                A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                B_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                F_delt = np.zeros((1, 7))
                                G_delt = np.array([1, 0, 0, 1, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 16
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 16.5
                                Balance = ["Unbalance"]
                        ## Tank Source Flowrate 0 / Rc1 Flowrate still left so system flowrate unbalance
                        else:
                            # Judgment Postponement
                            Label = 16.5
                            Balance = ["Unbalance"]
                    ## Rc2 to HEX source and Rc2 left flowrate --> Rc1 and Rc2 Flowrate also left
                    else:
                        B2C = 1
                        C2B = 1
                        A2G = 1
                        G2A = 1
                        ## Charing and Discharing: Possible
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc1
                            leftflow_equip_9 = Rc_flow_1-leftflow_head
                            ## Left Flowrate of Rc2
                            leftflow_equip_10 = Rc_flow_2-HEXs_flow

                            ## Rc1 and Rc2 left Flowrate send to Tank Source at the same time
                            if abs((((leftflow_equip_9+leftflow_equip_10)-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                B2E = 1
                                E2A = 1
                                E2B = 1
                                    
                                A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                B_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                                F_delt = np.zeros((1, 7))
                                G_delt = np.array([1, 0, 0, 1, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 17
                                Balance = ["Balance"]
                            ## Rc1 and Rc2 left Flowrate, but Over the Tank Source Flowrate
                            else:
                                # Judgment Postponement
                                Label = 17.5
                                Balance = ["Unbalance"]
                        ## Rc1 and Rc2 left Flowrate, but Tank Source Flowrate 0
                        else:
                            # Judgment Postponement
                            Label = 17.5
                            Balance = ["Unbalance"]


            ## Rc1, Tank load ON - having flowrate 
            elif Rc_flow_1 !=0 and Rc_flow_2 ==0 and Tl_flow != 0:

                ## Rc1, Tank load to Header 
                if abs((((Rc_flow_1+Tl_flow)-leftflow_head)/leftflow_head)*100) <= 10:
                    # Judgment Postponement : Becuase HEX Source need to Flowrate, But Flowrate already saticified balance, no left Flowrate
                    Label = Label
                    Balance = ["Unbalance"]

                ## Find the only one equipment connect to Header, the other one connect to HEX Source
                else:                     
                    ### Case1 : Rc1->HEX Source, Tank Load->Header befer left Flowrate each Equipment
                    if abs(((Rc_flow_1-HEXs_flow)/HEXs_flow)*100) <= 10 and abs(((Tl_flow-leftflow_head)/leftflow_head)*100) <= 10:
                        A2C = 1
                        C2A = 1
                        F2G = 1
                        G2F = 1
                            
                        A_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                        B_delt = np.zeros((1, 7))
                        C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                        D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        E_delt = np.zeros((1, 7))
                        F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        G_delt = np.array([0, 0, 0, 1, 0, 1, 0])
                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                        Label = 18
                        Balance = ["Balance"]

                    elif abs(((Rc_flow_1-HEXs_flow)/HEXs_flow)*100) > 10 and abs(((Tl_flow-leftflow_head)/leftflow_head)*100) <= 10:
                        A2C = 1
                        C2A = 1
                        F2G = 1
                        G2F = 1

                        ## Charing and Discharing: Possible
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc1
                            leftflow_equip_11 = Rc_flow_1-HEXs_flow
                            ## Rc1 left Flowrate send to Tank Source at the same time
                            if abs(((leftflow_equip_11-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                E2A = 1
                                    
                                A_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                B_delt = np.zeros((1, 7))
                                C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                G_delt = np.array([0, 0, 0, 1, 0, 1, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 19
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 19.5
                                Balance = ["Unbalance"]
                        else:
                            # Judgment Postponement
                            Label = 19.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 19.5
                        Balance = ["Unbalance"]

                    ### Case2 : Rc1->Header, Tank Load->HEX Source befer left Flowrate each Equipment
                    if abs(((Tl_flow-HEXs_flow)/HEXs_flow)*100) <= 10 and abs(((Rc_flow_1-leftflow_head)/leftflow_head)*100) <= 10:
                        F2C = 1
                        C2F = 1
                        A2G = 1
                        G2A = 1

                        A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        B_delt = np.zeros((1, 7))
                        C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                        D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        E_delt = np.zeros((1, 7))
                        F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                        G_delt = np.array([1, 0, 0, 1, 0, 0, 0])
                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                        Label = 20
                        Balance = ["Balance"]

                    elif abs(((Tl_flow-HEXs_flow)/HEXs_flow)*100) <= 10 and abs(((Rc_flow_1-leftflow_head)/leftflow_head)*100) > 10:
                        F2C = 1
                        C2F = 1
                        A2G = 1
                        G2A = 1

                        # Charing and Discharing: Possible
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc1
                            leftflow_equip_12 = Rc_flow_1-leftflow_head

                            ## Rc1 left Flowrate send to Tank Source at the same time
                            if abs(((leftflow_equip_12-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                E2A = 1
                                    
                                A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                B_delt = np.zeros((1, 7))
                                C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                G_delt = np.array([1, 0, 0, 1, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 21
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 21.5
                                Balance = ["Unbalance"]
                        else:
                            # Judgment Postponement
                            Label = 21.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 21.5
                        Balance = ["Unbalance"]
            
                
            ## Rc2, Tank load ON - having flowrate 
            elif Rc_flow_1 ==0 and Rc_flow_2 !=0 and Tl_flow != 0:

                ## Rc2, Tank load to Header 
                if abs((((Rc_flow_2+Tl_flow)-leftflow_head)/leftflow_head)*100) <= 10:
                    # Judgment Postponement : Becuase HEX Source need to Flowrate, But Flowrate already saticified balance, no left Flowrate
                    Label = Label
                    Balance = ["Unbalance"]

                ## Find the only one equipment connect to Header, the other one connect to HEX Source
                else:                     
                    ### Case1 : Rc2->HEX Source, Tank Load->Header befer left Flowrate each Equipment
                    if abs(((Rc_flow_2-HEXs_flow)/HEXs_flow)*100) <= 10 and abs(((Tl_flow-leftflow_head)/leftflow_head)*100) <= 10:
                        B2C = 1
                        C2B = 1
                        F2G = 1
                        G2F = 1
                            
                        A_delt = np.zeros((1, 7))
                        B_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                        C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                        D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        E_delt = np.zeros((1, 7))
                        F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        G_delt = np.array([0, 0, 0, 1, 0, 1, 0])
                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                        Label = 22
                        Balance = ["Balance"]

                    elif abs(((Rc_flow_2-HEXs_flow)/HEXs_flow)*100) > 10 and abs(((Tl_flow-leftflow_head)/leftflow_head)*100) <= 10:
                        B2C = 1
                        C2B = 1
                        F2G = 1
                        G2F = 1

                        ## Charing and Discharing: Possible
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc2
                            leftflow_equip_13 = Rc_flow_2-HEXs_flow

                            ## Rc2 left Flowrate send to Tank Source at the same time
                            if abs(((leftflow_equip_13-Ts_flow)/Ts_flow)*100) <= 10:
                                B2E = 1
                                E2B = 1
                                    
                                A_delt = np.zeros((1, 7))
                                B_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                                C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                G_delt = np.array([0, 0, 0, 1, 0, 1, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 23
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 23.5
                                Balance = ["Unbalance"]
                        else:
                            # Judgment Postponement
                            Label = 23.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 23.5
                        Balance = ["Unbalance"]

                    ### Case2 : Rc2->Header, Tank Load->HEX Source befer left Flowrate each Equipment
                    if abs(((Tl_flow-HEXs_flow)/HEXs_flow)*100) <= 10 and abs(((Rc_flow_2-leftflow_head)/leftflow_head)*100) <= 10:
                        F2C = 1
                        C2F = 1
                        B2G = 1
                        G2B = 1

                        A_delt = np.zeros((1, 7))
                        B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                        D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        E_delt = np.zeros((1, 7))
                        F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                        G_delt = np.array([0, 1, 0, 1, 0, 0, 0])
                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                        Label = 24
                        Balance = ["Balance"]

                    elif abs(((Tl_flow-HEXs_flow)/HEXs_flow)*100) <= 10 and abs(((Rc_flow_2-leftflow_head)/leftflow_head)*100) > 10:
                        F2C = 1
                        C2F = 1
                        B2G = 1
                        G2B = 1

                        ## Charing and Discharing: Possible
                        if Ts_flow != 0:
                            ## Left Flowrate of Rc2
                            leftflow_equip_14 = Rc_flow_2-leftflow_head

                            ## Rc2 left Flowrate send to Tank Source at the same time
                            if abs(((leftflow_equip_14-Ts_flow)/Ts_flow)*100) <= 10:
                                B2E = 1
                                E2B = 1
                                    
                                A_delt = np.zeros((1, 7))
                                B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                                D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                                G_delt = np.array([0, 1, 0, 1, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 25
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 25.5
                                Balance = ["Unbalance"]
                        else:
                            # Judgment Postponement
                            Label = 25.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 25.5
                        Balance = ["Unbalance"]


            ## Rc1 ON - having flowrate 
            elif Rc_flow_1 !=0 and Rc_flow_2 ==0 and Tl_flow == 0:
                    
                ## Rc1 have to send to Flowrate HEX Source
                if abs(((Rc_flow_1-HEXs_flow)/HEXs_flow)*100) <= 10:
                    A2C = 1
                    C2A = 1

                    A_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                    B_delt = np.zeros((1, 7))
                    C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    E_delt = np.zeros((1, 7))
                    F_delt = np.zeros((1, 7))
                    G_delt = np.array([0, 0, 0, 1, 0, 0, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 26
                    Balance = ["Balance"]
                ## Rc1 Flowrate left
                else:
                    A2C = 1
                    C2A = 1
                    ## Left Flowrate of Rc1
                    leftflow_equip_15 = Rc_flow_1-HEXs_flow
                    
                    ## Rc1 Flowrate left send to Tank Source
                    if Ts_flow != 0:
                        if abs(((leftflow_equip_15-Ts_flow)/Ts_flow)*100) <= 10:
                            A2E = 1
                            E2A = 1

                            A_delt = np.array([0, 0, 1, 0, 1, 0, 0])
                            B_delt = np.zeros((1, 7))
                            C_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                            D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                            E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                            F_delt = np.zeros((1, 7))
                            G_delt = np.array([0, 0, 0, 1, 0, 0, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 27
                            Balance = ["Balance"]
                        else:
                            # Judgment Postponement
                            Label = 27.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 27.5
                        Balance = ["Unbalance"]


            ## Rc2 ON - having flowrate 
            elif Rc_flow_1 ==0 and Rc_flow_2 !=0 and Tl_flow == 0:

                ## Rc2 have to send to Flowrate HEX Source
                if abs(((Rc_flow_2-HEXs_flow)/HEXs_flow)*100) <= 10:
                    B2C = 1
                    C2B = 1

                    A_delt = np.zeros((1, 7))
                    B_delt = np.array([0, 0, 1, 0, 0, 0, 0]) 
                    C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    E_delt = np.zeros((1, 7))
                    F_delt = np.zeros((1, 7))
                    G_delt = np.array([0, 0, 0, 1, 0, 0, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 28
                    Balance = ["Balance"]
                ## Rc2 Flowrate left
                else:
                    B2C = 1
                    C2B = 1
                    ## Left Flowrate of Rc2
                    leftflow_equip_16 = Rc_flow_2-HEXs_flow
                    
                    ## Rc2 Flowrate left send to Tank Source
                    if Ts_flow != 0:
                        if abs(((leftflow_equip_16-Ts_flow)/Ts_flow)*100) <= 10:
                            B2E = 1
                            E2B = 1

                            A_delt = np.zeros((1, 7))
                            B_delt = np.array([0, 0, 1, 0, 1, 0, 0]) 
                            C_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                            D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                            E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                            F_delt = np.zeros((1, 7))
                            G_delt = np.array([0, 0, 0, 1, 0, 0, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 29
                            Balance = ["Balance"]
                        else:
                            # Judgment Postponement
                            Label = 29.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 29.5
                        Balance = ["Unbalance"]


            ## Tank load ON - having flowrate 
            elif Rc_flow_1 ==0 and Rc_flow_2 ==0 and Tl_flow != 0:
                    
                ## Tank load have to send to Flowrate HEX Source
                if abs(((Tl_flow-HEXs_flow)/HEXs_flow)*100) <= 10:
                    F2C = 1
                    C2F = 1

                    A_delt = np.zeros((1, 7))
                    B_delt = np.zeros((1, 7))
                    C_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                    D_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    E_delt = np.zeros((1, 7))
                    F_delt = np.array([0, 0, 1, 0, 0, 0, 0])
                    G_delt = np.array([0, 0, 0, 1, 0, 0, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 30
                    Balance = ["Balance"]
                else:
                    # Judgment Postponement
                    Label = 30.5
                    Balance = ["Unbalance"]
            else:
                # Judgment Postponement
                Label = 30.5
                Balance = ["Unbalance"]


        ## No HEX Load Flowrate so, find the other equipment
        else:
            ## Rc1, Rc2, Tank load ON - having flowrate 
            if Rc_flow_1 != 0 and Rc_flow_2 != 0 and Tl_flow != 0:

                ## Tank Load have to connect the Header
                leftflow_head_2 = HS_flow-Tl_flow
                ## In this case, Rc_flow_1+Rc_flow_2 > leftflow_equip_17

                ## Rc1, Rc2, Tank load to Header
                if abs((((Rc_flow_1+Rc_flow_2+Tl_flow)-HS_flow)/HS_flow)*100) <= 10:
                    A2G = 1
                    B2G = 1
                    F2G = 1
                    G2A = 1
                    G2B = 1
                    G2F = 1 

                    A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.zeros((1, 7))
                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    G_delt = np.array([1, 1, 0, 0, 0, 1, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 31
                    Balance = ["Balance"]


                ## Rc1, Rc2, Tank load to Header and befer flowrate left
                elif abs((((Rc_flow_1+Rc_flow_2+Tl_flow)-HS_flow)/HS_flow)*100) > 10:
                    A2G = 1
                    B2G = 1
                    F2G = 1
                    G2A = 1
                    G2B = 1
                    G2F = 1 
                               
                    ## Left Flowrate of Rc1
                    leftflow_equip_17 = Rc_flow_1-(leftflow_head_2-Rc_flow_2)

                    ## Left Flowrate of Rc2
                    leftflow_equip_18 = Rc_flow_2-(leftflow_head_2-Rc_flow_1)

                    ## Rc1, Rc2 left Flowrate send to Tank Source at the same time
                    mean_Rc1_Rc2_2 = ((leftflow_equip_17*Rc_outtemp_1)+(leftflow_equip_18*Rc_outtemp_2))/(leftflow_equip_17+leftflow_equip_18)

                    if Ts_flow != 0:
                        ## Extra Rc1 to Tank source   
                        if abs(((Rc_outtemp_1-Ts_intemp)/Ts_intemp)*100) <= 10:
                            if abs(((leftflow_equip_17-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                E2A = 1

                                A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                C_delt = np.zeros((1, 7))
                                D_delt = np.zeros((1, 7))
                                E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                G_delt = np.array([1, 1, 0, 0, 0, 1, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 32
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 32.5
                                Balance = ["Unbalance"]

                        ## Extra Rc2 to Tank source  
                        elif abs(((Rc_outtemp_2-Ts_intemp)/Ts_intemp)*100) <= 10:
                            if abs(((leftflow_equip_18-Ts_flow)/Ts_flow)*100) <= 10:
                                B2E = 1
                                E2B = 1

                                A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                C_delt = np.zeros((1, 7))
                                D_delt = np.zeros((1, 7))
                                E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                G_delt = np.array([1, 1, 0, 0, 0, 1, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 33
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 33.5
                                Balance = ["Unbalance"]
                        
                        ## Rc1, Rc2 both have extra flowrate, send to the Tank source 
                        elif abs(((mean_Rc1_Rc2_2-Ts_intemp)/Ts_intemp)*100) <= 10:
                            if abs((((leftflow_equip_17+leftflow_equip_18)-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                E2A = 1
                                B2E = 1
                                E2B = 1

                                A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                C_delt = np.zeros((1, 7))
                                D_delt = np.zeros((1, 7))
                                E_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                                F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                G_delt = np.array([1, 1, 0, 0, 0, 1, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 34
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 34.5
                                Balance = ["Unbalance"]
                        else:
                            # Judgment Postponement
                            Label = 34.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 34.5
                        Balance = ["Unbalance"]
                    
                
                ## Rc1, Tank Load send to Header, Rc2 100% send to Tank Source
                elif abs((((Rc_flow_1+Tl_flow)-HS_flow)/HS_flow)*100) <= 10 and abs(((Rc_flow_2-Ts_flow)/Ts_flow)*100) <= 10:

                    A2G = 1
                    G2A = 1
                    F2G = 1
                    G2F = 1
                    B2E = 1
                    E2B = 1

                    A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    B_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    G_delt = np.array([1, 0, 0, 0, 0, 1, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 35
                    Balance = ["Balance"]


                ## Rc1, Tank Load send to Header and extra flowrate send to Tank Source, Rc2 100% send to Tank Source
                elif abs((((Rc_flow_1+Tl_flow)-HS_flow)/HS_flow)*100) > 10 and abs(((Rc_flow_2-Ts_flow)/Ts_flow)*100) > 10:
                    A2G = 1
                    G2A = 1
                    F2G = 1
                    G2F = 1
                    B2E = 1
                    E2B = 1
                    
                    ## Rc1 left Flowrate and send to Tank Source, with Rc2 at the same time
                    ## Left Flowrate of Rc1
                    leftflow_equip_19 = Rc_flow_1-leftflow_head_2

                    if abs((((leftflow_equip_19+Rc_flow_2)-Ts_flow)/Ts_flow)*100) <= 10:
                        A2E = 1
                        E2A = 1

                        A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                        B_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                        C_delt = np.zeros((1, 7))
                        D_delt = np.zeros((1, 7))
                        E_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                        F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        G_delt = np.array([1, 0, 0, 0, 0, 1, 0])
                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                        Label = 36
                        Balance = ["Balance"]
                    
                    else:
                        # Judgment Postponement
                        Label = 36.5
                        Balance = ["Unbalance"]
                    

                ## Rc2, Tank Load send to Header, Rc1 100% send to Tank Source
                elif abs((((Rc_flow_2+Tl_flow)-HS_flow)/HS_flow)*100) <= 10 and abs(((Rc_flow_1-Ts_flow)/Ts_flow)*100) <= 10:
                    A2E = 1
                    E2A = 1
                    F2G = 1
                    G2F = 1
                    B2G = 1
                    G2B = 1

                    A_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                    B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    G_delt = np.array([0, 1, 0, 0, 0, 1, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 37
                    Balance = ["Balance"]


                ## Rc2, Tank Load send to Header and extra flowrate send to Tank Source, Rc1 100% send to Tank Source
                elif abs((((Rc_flow_2+Tl_flow)-HS_flow)/HS_flow)*100) > 10 and abs(((Rc_flow_1-Ts_flow)/Ts_flow)*100) > 10:
                    A2E = 1
                    E2A = 1
                    F2G = 1
                    G2F = 1
                    B2G = 1
                    G2B = 1

                    ## Rc2 left Flowrate and send to Tank Source, with Rc1 at the same time
                    ## Left Flowrate of Rc2
                    leftflow_equip_20 = Rc_flow_2-leftflow_head_2

                    if abs((((leftflow_equip_20+Rc_flow_1)-Ts_flow)/Ts_flow)*100) <= 10:
                        B2E = 1
                        E2B = 1

                        A_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                        B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                        C_delt = np.zeros((1, 7))
                        D_delt = np.zeros((1, 7))
                        E_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                        F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        G_delt = np.array([0, 1, 0, 0, 0, 1, 0])
                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                        Label = 38
                        Balance = ["Balance"]
                    else:
                        # Judgment Postponement
                        Label = 38.5
                        Balance = ["Unbalance"]
                else:
                    # Judgment Postponement
                    Label = 38.5
                    Balance = ["Unbalance"]


            ## Rc1, Rc2 ON - having flowrate 
            elif Rc_flow_1 != 0 and Rc_flow_2 != 0 and Tl_flow == 0:

                ## Tank Charging impossible
                if Ts_flow == 0:

                    ## Rc1, Rc2 to Header
                    if abs((((Rc_flow_1+Rc_flow_2)-HS_flow)/HS_flow)*100) <= 10:
                        A2G = 1
                        B2G = 1
                        G2A = 1
                        G2B = 1

                        A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                        C_delt = np.zeros((1, 7))
                        D_delt = np.zeros((1, 7))
                        E_delt = np.zeros((1, 7))
                        F_delt = np.zeros((1, 7))
                        G_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                        Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                        Label = 39
                        Balance = ["Balance"]
                    else:
                        # Judgment Postponement
                        Label = 39.5
                        Balance = ["Unbalance"]


                ## Tank Charging possible, Rc1, Rc2 to Header and left extra Flowrate for Tank Source
                else:
                    ## Rc1, Rc2 to Header
                    if abs((((Rc_flow_1+Rc_flow_2)-HS_flow)/HS_flow)*100) <= 10:
                        # Judgment Postponement
                        Label = 40.5
                        Balance = ["Unbalance"]
                    else:
                        A2G = 1
                        B2G = 1
                        G2A = 1
                        G2B = 1
                        ## Extra Rc1 to Tank source   
                        if abs(((Rc_outtemp_1-Ts_intemp)/Ts_intemp)*100) <= 10:
                            ##  Left Flowrate of Rc1
                            leftflow_equip_21 = Rc_flow_1-(HS_flow-Rc_flow_2)

                            if abs(((leftflow_equip_21-Ts_flow)/Ts_flow)*100) <= 10:
                                A2E = 1
                                E2A = 1

                                A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                C_delt = np.zeros((1, 7))
                                D_delt = np.zeros((1, 7))
                                E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                                F_delt = np.zeros((1, 7))
                                G_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 40
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 40.5
                                Balance = ["Unbalance"]

                        ## Extra Rc2 to Tank source                     
                        elif abs(((Rc_outtemp_2-Ts_intemp)/Ts_intemp)*100) <= 10:
                            ## Left Flowrate of Rc2
                            leftflow_equip_22 = Rc_flow_2-(HS_flow-Rc_flow_1)

                            if abs(((leftflow_equip_22-Ts_flow)/Ts_flow)*100) <= 10:
                                B2E = 1
                                E2B = 1
                                    
                                A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                                B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                                C_delt = np.zeros((1, 7))
                                D_delt = np.zeros((1, 7))
                                E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                                F_delt = np.zeros((1, 7))
                                G_delt = np.array([1, 1, 0, 0, 0, 0, 0])
                                Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                                Label = 41
                                Balance = ["Balance"]
                            else:
                                # Judgment Postponement
                                Label = 41.5
                                Balance = ["Unbalance"]
                        else:
                            # Judgment Postponement
                            Label = 41.5
                            Balance = ["Unbalance"]
                

            ## Rc1, Tank Load ON - having flowrate 
            elif Rc_flow_1 != 0 and Rc_flow_2 == 0 and Tl_flow != 0:

                ## Rc1, Tank Load to Header at the same time
                if abs((((Rc_flow_1+Tl_flow)-HS_flow)/HS_flow)*100) <= 10:
                    A2G = 1
                    F2G = 1
                    G2A = 1
                    G2F = 1
                
                    A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    B_delt = np.zeros((1, 7))
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.zeros((1, 7))
                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    G_delt = np.array([1, 0, 0, 0, 0, 1, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 42
                    Balance = ["Balance"]

                ## Tank Load have to send Header all the time, Rc1 to Tank Sourece
                elif abs((((Rc_flow_1+Tl_flow)-HS_flow)/HS_flow)*100) > 10:
                    A2G = 1
                    F2G = 1
                    G2A = 1
                    G2F = 1

                    if Ts_flow != 0:
                        ## Rc1 left Flowrate to Tank Source
                        leftflow_equip_23 = Rc_flow_1-(HS_flow-Tl_flow)

                        if abs(((leftflow_equip_23-Ts_flow)/Ts_flow)*100) <= 10:
                            A2E = 1
                            E2A = 1

                            A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                            B_delt = np.zeros((1, 7))
                            C_delt = np.zeros((1, 7))
                            D_delt = np.zeros((1, 7))
                            E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                            F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                            G_delt = np.array([1, 0, 0, 0, 0, 1, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 43
                            Balance = ["Balance"]  
                    else:
                        # Judgment Postponement
                        Label = 43.5
                        Balance = ["Unbalance"]

                ## Only connected Tank Load to Header
                elif abs(((Tl_flow-HS_flow)/HS_flow)*100) <= 10:
                    F2G = 1
                    G2F = 1

                    if Ts_flow != 0:
                        if abs(((Rc_flow_1-Ts_flow)/Ts_flow)*100) <= 10:
                            A2E = 1
                            E2A = 1

                            A_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                            B_delt = np.zeros((1, 7))
                            C_delt = np.zeros((1, 7))
                            D_delt = np.zeros((1, 7))
                            E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                            F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                            G_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 44
                            Balance = ["Balance"]  
                        else:
                            # Judgment Postponement
                            Label = 44.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 44.5
                        Balance = ["Unbalance"]
                else:
                    # Judgment Postponement
                    Label = 44.5
                    Balance = ["Unbalance"]

            ## Rc2, Tank Load ON - having flowrate 
            elif Rc_flow_1 == 0 and Rc_flow_2 != 0 and Tl_flow != 0:

                ## Rc2, Tank Load to Header at the same time
                if abs((((Rc_flow_2+Tl_flow)-HS_flow)/HS_flow)*100) <= 10:
                    B2G = 1
                    F2G = 1
                    G2B = 1
                    G2F = 1
                
                    A_delt = np.zeros((1, 7))
                    B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.zeros((1, 7))
                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    G_delt = np.array([0, 1, 0, 0, 0, 1, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 45
                    Balance = ["Balance"]

                ## Tank Load have to send Header all the time, Rc1 to Tank Sourece
                elif abs((((Rc_flow_1+Tl_flow)-HS_flow)/HS_flow)*100) > 10:
                    B2G = 1
                    F2G = 1
                    G2B = 1
                    G2F = 1

                    if Ts_flow != 0:
                        ## Rc1 left Flowrate to Tank Source
                        leftflow_equip_24 = Rc_flow_2-(HS_flow-Tl_flow)

                        if abs(((leftflow_equip_24-Ts_flow)/Ts_flow)*100) <= 10:
                            B2E = 1
                            E2B = 1

                            A_delt = np.zeros((1, 7))
                            B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                            C_delt = np.zeros((1, 7))
                            D_delt = np.zeros((1, 7))
                            E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                            F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                            G_delt = np.array([0, 1, 0, 0, 0, 1, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 46
                            Balance = ["Balance"]  
                    else:
                        # Judgment Postponement
                        Label = 46.5
                        Balance = ["Unbalance"]

                ## Only connected Tank Load to Header
                elif abs(((Tl_flow-HS_flow)/HS_flow)*100) <= 10:
                    F2G = 1
                    G2F = 1

                    if Ts_flow != 0:
                        if abs(((Rc_flow_1-Ts_flow)/Ts_flow)*100) <= 10:
                            A2E = 1
                            E2A = 1

                            A_delt = np.array([0, 0, 0, 0, 1, 0, 0])
                            B_delt = np.zeros((1, 7))
                            C_delt = np.zeros((1, 7))
                            D_delt = np.zeros((1, 7))
                            E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                            F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                            G_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 47
                            Balance = ["Balance"]  
                        else:
                            # Judgment Postponement
                            Label = 47.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 47.5
                        Balance = ["Unbalance"]
                else:
                    # Judgment Postponement
                    Label = 47.5
                    Balance = ["Unbalance"]


            ## Rc1 ON - having flowrate 
            elif Rc_flow_1 != 0 and Rc_flow_2 == 0 and Tl_flow == 0:

                ## Rc1 to Header at the same time
                if abs(((Rc_flow_1-HS_flow)/HS_flow)*100) <= 10:
                    A2G = 1
                    G2A = 1

                    A_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    B_delt = np.zeros((1, 7))
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.zeros((1, 7))
                    F_delt = np.zeros((1, 7))
                    G_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 48
                    Balance = ["Balance"]

                ## Rc1 left Flowrate 
                else:
                    ## left Flowrate to Tank Source
                    if Ts_flow !=0:
                        leftflow_equip_25 = Rc_flow_1-HS_flow

                        if abs(((leftflow_equip_25-Ts_flow)/Ts_flow)*100) <= 10:
                            A2E = 1
                            E2A = 1

                            A_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                            B_delt = np.zeros((1, 7))
                            C_delt = np.zeros((1, 7))
                            D_delt = np.zeros((1, 7))
                            E_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                            F_delt = np.zeros((1, 7))
                            G_delt = np.array([1, 0, 0, 0, 0, 0, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 49
                            Balance = ["Balance"]
                        else:
                            # Judgment Postponement
                            Label = 49.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 49.5
                        Balance = ["Unbalance"]
            

            ## Rc2 ON - having flowrate 
            elif Rc_flow_1 == 0 and Rc_flow_2 != 0 and Tl_flow == 0:

                ## Rc2 to Header at the same time
                if abs(((Rc_flow_2-HS_flow)/HS_flow)*100) <= 10:
                    B2G = 1
                    G2B = 1

                    A_delt = np.zeros((1, 7))
                    B_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.zeros((1, 7))
                    F_delt = np.zeros((1, 7))
                    G_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 50
                    Balance = ["Balance"]
                
                ## Rc2 left Flowrate 
                else:
                    ## left Flowrate to Tank Source
                    if Ts_flow !=0:
                        leftflow_equip_26 = HS_flow-Rc_flow_2

                        if abs(((leftflow_equip_26-Ts_flow)/Ts_flow)*100) <= 10:
                            B2E = 1
                            E2B = 1

                            A_delt = np.zeros((1, 7))
                            B_delt = np.array([0, 0, 0, 0, 1, 0, 1])
                            C_delt = np.zeros((1, 7))
                            D_delt = np.zeros((1, 7))
                            E_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                            F_delt = np.zeros((1, 7))
                            G_delt = np.array([0, 1, 0, 0, 0, 0, 0])
                            Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                            Label = 51
                            Balance = ["Balance"]
                        else:
                            # Judgment Postponement
                            Label = 51.5
                            Balance = ["Unbalance"]
                    else:
                        # Judgment Postponement
                        Label = 51.5
                        Balance = ["Unbalance"]

            
            ## Tank Load ON - having flowrate
            elif Rc_flow_1 == 0 and Rc_flow_2 == 0 and Tl_flow != 0:

                ## Tank Load to Header at the same time
                if abs(((Tl_flow-HS_flow)/HS_flow)*100) <= 10:
                    F2G = 1
                    G2F = 1

                    A_delt = np.zeros((1, 7))
                    B_delt = np.zeros((1, 7))
                    C_delt = np.zeros((1, 7))
                    D_delt = np.zeros((1, 7))
                    E_delt = np.zeros((1, 7))
                    F_delt = np.array([0, 0, 0, 0, 0, 0, 1])
                    G_delt = np.array([0, 0, 0, 0, 0, 1, 0])
                    Delta_Temp = np.vstack([A_delt, B_delt, C_delt, D_delt, E_delt, F_delt, G_delt])
                    Label = 52
                    Balance = ["Balance"]
                else:
                    # Judgment Postponement
                    Label = 52.5
                    Balance = ["Unbalance"]
            else:
                # Judgment Postponement
                Label = 52.5
                Balance = ["Unbalance"]
    

    Admatrix.append(Delta_Temp)
    Lableing.append(Label)
    Balancing.append(Balance)

    print(i, Label, Balance)


Admatrix = np.array(Admatrix).reshape(122400, 49)
ResultAdmatrix = pd.DataFrame(Admatrix)
ResultLableing = pd.DataFrame(Lableing)
ResultBalancing = pd.DataFrame(Balancing)


ResultAdmatrix.to_csv('Admatrix_Predict.csv', index=False)
ResultLableing.to_csv('Lableing_Predict.csv', index=False)
ResultBalancing.to_csv('Balancing_Predict.csv', index=False)


'====================================================================================='