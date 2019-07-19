"""
File Name: summarize_models.py

Authors: Kyle Seidenthal

Date: 19-07-2019

Description: A script to print a nice table summarizing the performance of all the models

"""


import os
import pandas as pd

model_dirs = os.listdir(".")

summary = None

for model_dir in model_dirs:
    if os.path.isdir(model_dir):
        test_files =[x for x in os.listdir(model_dir) if 'scores' in x ]

        for test_file in test_files:
            data = pd.read_csv(os.path.join(model_dir, test_file))
            
            model_name = model_dir
            
            if 'best' in test_file:
                model_name+="_best"

            data.insert(loc=0, column='Model_Name', value=model_name)
        
            if summary is None:
                summary = data
                
            else:
                summary = pd.concat([summary, data], ignore_index=True, sort=False)


sorted_summary = summary.sort_values(by=['mean_squared_error'], ascending=True)

print(sorted_summary)
    
