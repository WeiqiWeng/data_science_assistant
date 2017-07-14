import statistical_modeling_assistant as sma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('bank-additional.csv', sep=';')
sma_assist = sma.StatisticalModelingAssistant()

data = sma_assist.quick_fix_feature_name(data)

model = sma_assist.linear_regression(data, ['nr_employed'], 'nr_employed', True)
model_metrics = sma_assist.model_metrics(model)
print(model_metrics)