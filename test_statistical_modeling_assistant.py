import statistical_modeling_assistant as sma
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

data = pd.read_csv('bank-additional.csv', sep=';')

a = np.ones((1, 3))
print(np.sum(a == 1))

normalize(a, norm='l1', axis=1, copy=False)

print(a.sum(axis=1))
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

pred_y = sma_assist.logistic_regression_predict(model1, data, label=True, threshold=0.5)

print(sma_assist.get_accuracy(pred_y, data['y_yes']))

print(1)