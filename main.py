import data_science_assistant as dsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('bank-additional.csv', sep=';')
ds_assist = dsa.DataScienceAssistant()

data['age_level'] = ds_assist.discretize_numerical_variable(data.age, [20, 40, 60, 80], [1, 2, 3, 4, 5])

print(data['age_level'].unique())

print(data['job'].unique())

newvar_vargroup_map = {'employed':['blue-collar', 'services', 'admin.', 'entrepreneur', 'self-employed',
 'technician', 'management', 'housemaid'], 'unemployed': ['student', 'retired', 'unknown']}
data['job_level'] = ds_assist.bucket_catagorical_variable(data.job, newvar_vargroup_map)

print(data['job_level'].unique())

data = ds_assist.dummy_encode_catagorical_variable(data, 'job_level', False)
data = ds_assist.dummy_encode_catagorical_variable(data, 'y', False)

print(data.head(10))

data['log_euribor3m'] = ds_assist.epsilon_natural_log(data['euribor3m'], 0.0001)
data['normalized_euribor3m'], mean_euribor3m, std_euribor3m = ds_assist.normalize(data['euribor3m'])

# plt.hist(data['log_euribor3m'], bins=20)
# fig = plt.figure()
# plt.hist(data['normalized_euribor3m'], bins=20)
# fig.savefig('pic1.jpg')

label = data['y_yes']
data.drop(['y_yes', 'y_no'], inplace=True, axis=1)

train_set, train_set_y, validation_set, validation_set_y, test_set, test_set_y = ds_assist.split_data_set(data, label, 0.6)

print('proportion of y = 1 in training set: %.3f' % (train_set_y.sum()/len(train_set_y)))
print('proportion of y = 1 in validation set: %.3f' % (validation_set_y.sum()/len(validation_set_y)))
print('proportion of y = 1 in testing set: %.3f' % (test_set_y.sum()/len(test_set_y)))