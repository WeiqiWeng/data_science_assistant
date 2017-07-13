import data_science_assistant as dsa
import pandas as pd
import numpy as np


data = pd.read_csv('bank-additional.csv', sep=';')
ds_assist = dsa.DataScienceAssistant()

data['age_level'] = ds_assist.discretize_numerical_variable(data.age, [20, 40, 60, 80], [1, 2, 3, 4, 5])

print(data['age_level'].unique())