import statistical_modeling_assistant as sma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('bank-additional.csv', sep=';')

print(list(data))
data = data[['age', 'marital', 'education', 'emp.var.rate', 'euribor3m', 'nr.employed', 'y']].copy()

sma_assist = sma.StatisticalModelingAssistant()

data = sma_assist.quick_fix_feature_name(data)

# feature_space, metric_space, min_metric_feature, min_metric = sma_assist.experiment_linear_regression(data, 'nr_employed', 10)

# print(feature_space)
model = sma_assist.linear_regression(data, ['euribor3m'], 'nr_employed', missing_handle='none', intercept=True, printout=True)
# model_metrics = sma_assist.model_metrics(model)
# print(model_metrics)

data = sma_assist.dummy_encode_catagorical_variable(data, 'y')
model1 = sma_assist.logistic_regression(data, ['euribor3m', 'nr_employed'], 'y_yes', missing_handle='none', intercept=True, printout=True)
pred_y = model1.predict(data)

pred_y[pred_y > 0.5] = 1
pred_y[pred_y <= 0.5] = 0
print(np.linalg.norm(pred_y - data['y'], 2))

print(pred_y)

print(1)