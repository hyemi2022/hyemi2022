
# main.py

import subprocess


subprocess.call(["python", 'train_hex_Pearson20.py']) 
subprocess.call(["python", 'train_hex_Pearson40.py']) 
subprocess.call(["python", 'train_hex_Pearson60.py']) 
subprocess.call(["python", 'train_hex_Pearson80.py']) 

subprocess.call(["python", 'train_hex_Spearman20.py']) 
subprocess.call(["python", 'train_hex_Spearman40.py'])
subprocess.call(["python", 'train_hex_Spearman60.py'])
subprocess.call(["python", 'train_hex_Spearman80.py'])

subprocess.call(["python", 'train_hex_Defualt(100).py']) 
