

# Raw data save to pickle.py

'''
-----------------------------------------------------
rawdata processing
-----------------------------------------------------
'''

'==============================================================================================================================='

import pandas as pd
import pickle


# open csv rawdata
Train = pd.read_csv('C:\\System_Air\\Input\\Train\\TRNSYS_2016.csv')
Predict = pd.read_csv('C:\\System_Air\\Input\\Test\\TRNSYS_2017.csv')


with open('Train.bin', 'wb') as file:
    pickle.dump(Train, file)

with open('Predict.bin', 'wb') as file:
    pickle.dump(Predict, file)


'==============================================================================================================================='