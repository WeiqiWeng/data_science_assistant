import statistical_modeling_assistant as sma
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

data = pd.read_csv('bank-additional.csv', sep=';')

# a = np.ones((1, 3))
# print(np.sum(a == 1))

# normalize(a, norm='l1', axis=1, copy=False)

# print(a.sum(axis=1))
# print(list(data))
data = data[['age', 'marital', 'education', 'emp.var.rate', 'euribor3m', 'nr.employed', 'y']].copy()

sma_assist = sma.StatisticalModelingAssistant()

data = sma_assist.batch_dummy_encode_catagorical_variable(data, ['marital', 'education', 'y'], [True] * 3)
data = sma_assist.quick_fix_feature_name(data)

# lr = sma_assist.LinearRegression()
# opt_metric_feature, opt_metric, feature_space, metric_space = sma_assist.experiment(lr, data, 'nr_employed', n=10, metric='rsquared')
#
# print(opt_metric_feature)
# print(opt_metric)

# log_r = sma_assist.LogisticRegression()
# opt_metric_feature, opt_metric, feature_space, metric_space = sma_assist.experiment(log_r, data, 'y_yes', n=2, metric='accuracy')
# # log_r.train(data, ['euribor3m', 'nr_employed'], 'y_yes', missing_handle='none', intercept=True, printout=True)
#
# print(opt_metric_feature)
# print(opt_metric)
#
# print(log_r.model_metrics(data, 'y_yes'))
#
# cm = log_r.confusion_matrix(data.y_yes, log_r.predict(data))
#
# print(cm)


data.loc[30:80, 'y_yes'] = 2
data.y_yes = data.y_yes.astype(int)
data = sma_assist.dummy_encode_catagorical_variable(data, 'y_yes')

soft_r = sma_assist.SoftmaxRegression()

soft_r.train(data, ['euribor3m', 'nr_employed'], ['y_yes_0', 'y_yes_2', 'y_yes_1'], missing_handle='none', intercept=True)

print(soft_r.predict_sigmoid_values(data))

print(soft_r.accuracy(soft_r.confusion_matrix(data.y_yes, soft_r.predict(data))))

print(soft_r.confusion_matrix(data.y_yes, soft_r.predict(data)))

print(1)