## author: Weiqi Weng
## 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataScienceAssistant:
	def __init__(self):

	def discretize_numerical_variable(feature, thresholds, discrete_levels):
		'''
    	discretize the given numerical variable into several levels
    	Args:
    		feature (pandas data series): the numerical variable user wants to discretize
    		thresholds (list): a list of thresholds [a1, a2, a3, ... , ak] so that
    		the given numerical variable X will be discretized according to
    		X < a1, a1 <= X < a2, ... , ak-1 <= X < ak, ak <= X
    	Returns:
    		pandas data series: the discretized numerical variable
    	'''
		threshold_cnt = len(thresholds)
		level_cnt = len(discrete_levels)

		feature.loc[feature < threshold_cnt[0]] = discrete_levels[0]

		for i in range(1, threshold_cnt):
			feature.loc[(feature >= threshold_cnt[i - 1]) &
			             (feature < threshold_cnt[i])] = discrete_levels[i]

		feature.loc[feature >= threshold_cnt[-1]] = discrete_levels[-1]

		return feature


	def bucket_catagorical_variable(feature, newvar_vargroup_map):
		'''
    	divide the value in given catagorical variable into several buckets
    	Args:
    		feature (pandas data series): the catagorical variable user wants to bucket
    		newvar_vargroup_map (dict): a dictionary mapping bucket name to value levels, for example,
    		{'excellent': [9, 10], 'good': [7, 8], 'not bad'" [3, 4, 5, 6]"},
    		then 9 and 10 => 'excellent', 7 and 8 => 'good', 3, 4, 5, 6 => 'not bad'
    	Returns:
    		pandas data series: the catagorical variable after bucket
    	'''
		for new_var, old_var_group in newvar_vargroup_map:
			feature.loc[feature.isin(old_var_group)] = new_var

		return feature


    def dummy_encode_catagorical_variable(data, variable, remove=False):
    	'''
    	dummy encode the given catagorical variable within data frame
    	Args:
    		data (pandas data frame): the data frame user works on
    		variable (str): name of catagorical variable user wants to dummy encode
    		remove (bool): whether to remove the catagorical variable or not
    	Returns:
    		pandas data frame: new data frame with given variable dummy encoded
    	'''
    	value_list = data[variable].unique()
    	row_cnt, _ = data.shape
    	tmp_zeros = np.zeros(row_cnt)
    	
    	for level in value_list:
    		str_level = '_'.join.(str(level).split())
    		dummy_code_name = str(variable) + '_' + str_level
    		data[dummy_code_name] = tmp_zeros
    		data.loc[data.variable == level, dummy_code_name] = 1

    	if remove:
    		data.drop(variable, inplace=True, axis=1)

    	return data


    def epsilon_natural_log(feature, epsilon):
    	'''
    	take the natural log of given feature plus a tiny number to avoid taking log of 0
    	Args:
    		feature (pandas series): the series to be transformed
    		epsilon (real): tiny number to be added to the series
    	Returns:
    		pandas series: the transformed series
    	'''
    	return pd.Series(np.log(feature + epsilon))


    def normalize(feature):
    	'''
    	normalize with (X - mean) / std
    	Args:
    		feature (pandas series): the series to be normalized
    	Returns:
    		pandas series: the normalized series
    		real: mean of given series
    		real: standard deviation of given series
    	'''
    	mean = feature.mean()
    	std = feature.std()
    	feature = (feature - mean) / std

    	return feature, mean, std 


    def split_data_set(init_data, train_set_proportion=0.6):
    	'''
    	split given data set into training, validation and testing set
    	Args:
    		init_data (pandas data frame): the whole data set user works on
    		train_set_proportion (real): the proportion of training set size in the whole data set
    	Returns:
    		pandas data frame: training set
    		pandas data frame: validation set
    		pandas data frame: testing set
    		
    	'''
    	train_set, test = train_test_split(init_data, test_size = 1 - train_set_proportion)
		validation_set, test_set = train_test_split(test, test_size = 0.5)

		return train_set, validation_set, test_set


	def check_null(data, printout=True):
		'''
    	check if each feature has NULL value
    	Args:
    		data (pandas data frame): the data set to be checked
    		printout (bool): whether to print out intermediate result or not
    	Returns:
    		list: empty list if no null in data set
    	'''
		features_with_null = []
		for feature in list(data):
    		lst = pd.isnull(data[feature])
    		any_null = lst.any()
    		if any_null:
    			features_with_null.append(feature)
    		if printout:
    			print('Null in feature %s: %r' % (feature, any_null))

    	return features_with_null


    def proportion_of_level_catagorical_variable(feature, printout=True):
    	'''
    	compute the proportion of each value level in a catagorical variable
    	Args:
    		feature (pandas series): the catagorical variable user wants to compute proportion
    		printout (bool): whether to print out intermediate result or not
    	Returns:
    		dict: a dictionary mapping value level to its proportion
    	'''
    	value_list = feature.unique()
    	proportion = np.array([])
    	value_to_proportion = dict()

    	for value in value_list:
    		proportion.append(len(feature[feature == value]))

    	proportion = proportion / sum(proportion)

    	for i in range(len(value_list)):
    		if printout:
    			print('value = %s covers %.2f of data' % (value_list[i], proportion[i]))
    		value_to_proportion[value_list[i]] = proportion[i]

    	return value_to_proportion

    	


