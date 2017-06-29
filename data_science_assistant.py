import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataScienceAssistant:
	def __init__(self):
        



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