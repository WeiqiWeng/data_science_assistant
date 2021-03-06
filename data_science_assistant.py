# author: Weiqi Weng
# contact: allen.weng9210@gmail.com

import pandas as pd
import numpy as np


class DataScienceAssistant(object):
    def __init__(self):
        pass

    @staticmethod
    def check_feature_name(data):
        """
        check if any feature name contains '.' which will cause problem during modeling
        Args:
            data (pandas data frame): the data frame the user is working on 
        Returns:
            bool: True if there is some problematic feature name
            [str*]: list of features that contain illegal signs
        """
        features = list(data)
        res = ['.' in x for x in features]

        problematic_features = []
        for i in range(len(features)):
            if res[i]:
                problematic_features.append(features[i])

        return any(res), problematic_features

    def quick_fix_feature_name(self, data):
        """
        check if any feature name contains '.' and quickly substitute it with '_'
        Args:
            data (pandas data frame): the data frame the user is working on 
        Returns:
            pandas data frame: the data frame with illegal feature names fixed
        """
        illegal_flag, problematic_features = self.check_feature_name(data)
        res = self.rename_feature_name(data, problematic_features, ['_'.join(
            x.split('.')) for x in problematic_features]) if illegal_flag else data
        return res

    @staticmethod
    def rename_feature_name(data, old_features, new_features):
        """
        rename features according to given old and new feature names
        Args:
            data (pandas data frame): the data frame user works on
            old_features ([str*]): old feature names
            new_features ([str*]): new feature names user wants to use
        Returns:
            pandas data frame: the pandas data frame with features renamed
        """
        rename_dict = dict(zip(old_features, new_features))
        data.rename(columns=rename_dict, inplace=True)

        return data

    @staticmethod
    def discretize_numerical_variable(ori_feature, thresholds, discrete_levels):
        """
        discretize the given numerical variable into several levels
        Args:
            feature (pandas data series): the numerical variable user wants to discretize
            thresholds (list): a list of thresholds [a1, a2, a3, ... , ak] so that 
            the given numerical variable X will be discretized according to
            X < a1, a1 <= X < a2, ... , ak-1 <= X < ak, ak <= X
        Returns:
            pandas data series: the discretized numerical variable
        """

        feature = ori_feature.copy()

        threshold_cnt = len(thresholds)
        level_cnt = len(discrete_levels)

        if threshold_cnt * level_cnt < 1:
            raise ValueError("Empty threshold list ({threshold_cnt}) or discrete level list ({level_cnt}).".format(
                threshold_cnt=threshold_cnt, level_cnt=level_cnt))

        if threshold_cnt != level_cnt - 1:
            raise ValueError("length of thresholds ({threshold_cnt}) should be length of discrete levels ({level_cnt}) minus one.".format(
                threshold_cnt=threshold_cnt, level_cnt=level_cnt))

        type_thresholds = [type(x) not in {int, float} for x in thresholds]
        type_levels = [type(x) not in {int, float} for x in discrete_levels]
        if any(type_thresholds) or any(type_levels):
            raise TypeError(
                "Type of threshold or type of discrete level not numerical.")

        feature.loc[feature < thresholds[0]] = discrete_levels[0]

        for i in range(1, threshold_cnt):
            feature.loc[(feature >= thresholds[i - 1]) &
                        (feature < thresholds[i])] = discrete_levels[i]

        feature.loc[feature >= thresholds[-1]] = discrete_levels[-1]

        return feature

    @staticmethod
    def bucket_catagorical_variable(ori_feature, newvar_vargroup_map):
        """
        divide the value in given catagorical variable into several buckets
        Args:
            feature (pandas data series): the catagorical variable user wants to bucket
            newvar_vargroup_map (dict): a dictionary mapping bucket name to value levels, for example,
            {'excellent': [9, 10], 'good': [7, 8], 'not bad'" [3, 4, 5, 6]"},
            then 9 and 10 => 'excellent', 7 and 8 => 'good', 3, 4, 5, 6 => 'not bad'
        Returns:
            pandas data series: the catagorical variable after bucket
        """
        feature = ori_feature.copy()

        for new_var in newvar_vargroup_map:
            feature.loc[feature.isin(newvar_vargroup_map[new_var])] = new_var

        return feature

    @staticmethod
    def dummy_encode_catagorical_variable(data, variable, remove=False):
        """
        dummy encode the given catagorical variable within data frame
        Args:
            data (pandas data frame): the data frame user works on
            variable (str): name of catagorical variable user wants to dummy encode
            remove (bool): whether to remove the catagorical variable or not
        Returns:
            pandas data frame: new data frame with given variable dummy encoded
        """
        value_list = data[variable].unique()
        row_cnt, _ = data.shape
        tmp_zeros = np.zeros(row_cnt)

        for level in value_list:
            str_level = '_'.join(str(level).split())
            dummy_code_name = variable + '_' + str_level
            data[dummy_code_name] = tmp_zeros
            data.loc[data[variable] == level, dummy_code_name] = 1

        if remove:
            data.drop(variable, inplace=True, axis=1)

        return data

    def batch_dummy_encode_catagorical_variable(self, data, variables, remove=[]):
        """
        dummy encode the given list of catagorical variables within data frame in a batch manner
        Args:
            data (pandas data frame): the data frame user works on
            variables ([str*]): names of catagorical variables user wants to dummy encode
            remove ([bool*]): whether to remove each of the catagorical variable or not
        Returns:
            pandas data frame: new data frame with given variable dummy encoded
        """
        variables_cnt = len(variables)
        if variables_cnt < 1:
            raise ValueError("Empty variable list.")

        if not remove:
            remove = [False] * len(variables)
        for i in range(len(variables)):
            data = self.dummy_encode_catagorical_variable(
                data, variables[i], remove[i])

        return data

    @staticmethod
    def epsilon_natural_log(feature, epsilon=0.0001):
        """
        take the natural log of given feature plus a tiny number to avoid taking log of 0
        Args:
            feature (pandas series): the series to be transformed
            epsilon (real): tiny number to be added to the series
        Returns:
            pandas series: the transformed series
        """
        return pd.Series(np.log(feature + epsilon))

    @staticmethod
    def normalize(feature):
        """
        normalize with (X - mean) / std
        Args:
            feature (pandas series): the series to be normalized
        Returns:
            pandas series: the normalized series
            real: mean of given series
            real: standard deviation of given series
        """
        mean = feature.mean()
        std = feature.std()
        feature = (feature - mean) / std

        return feature, mean, std

    @staticmethod
    def split_data_set(init_data, y, train_set_proportion=0.6):
        """
        split given data set into training, validation and testing set
        Args:
            init_data (pandas data frame): the whole data set user works on
            train_set_proportion (real): the proportion of training set size in the whole data set
        Returns:
            pandas data frame: training set
            pandas data frame: validation set
            pandas data frame: testing set

        """
        from sklearn.model_selection import train_test_split

        train_set, test_set, train_set_y, test_set_y = train_test_split(
            init_data, y, test_size=1 - train_set_proportion)
        validation_set, test_set, validation_set_y, test_set_y = train_test_split(
            test_set, test_set_y, test_size=0.5)

        return train_set, train_set_y, validation_set, validation_set_y, test_set, test_set_y

    @staticmethod
    def check_null(data, printout=True):
        """
        check if each feature has NULL value
        Args:
            data (pandas data frame): the data set to be checked
            printout (bool): whether to print out intermediate result or not
        Returns:
            list: empty list if no null in data set
        """
        features_with_null = []
        for feature in list(data):
            lst = pd.isnull(data[feature])
            any_null = lst.any()
            if any_null:
                features_with_null.append(feature)
            if printout:
                print('Null in feature %s: %r' % (feature, any_null))

        return features_with_null

    @staticmethod
    def proportion_of_level_catagorical_variable(feature, printout=True):
        """
        compute the proportion of each value level in a catagorical variable
        Args:
            feature (pandas series): the catagorical variable user wants to compute proportion
            printout (bool): whether to print out intermediate result or not
        Returns:
            dict: a dictionary mapping value level to its proportion
        """
        value_list = feature.unique()
        value_list.sort()
        proportion = np.array([])
        value_to_proportion = dict()

        for value in value_list:
            proportion = np.append(proportion, np.sum(feature == value))

        proportion = proportion / np.sum(proportion)

        for i in range(len(value_list)):
            if printout:
                print('value = %s covers %.2f%% of data' %
                      (value_list[i], proportion[i] * 100))
            value_to_proportion[value_list[i]] = proportion[i]

        return value_to_proportion

    @staticmethod
    def balance_with_SMOTETomek(data_x, data_y, ratio=0.99, printout=True):
        """
        balance the data set with SMOTE and Tomek link
        Args:
            data_x (pandas data frame): the data frame with all features
            data_y (pandas series): labels
        Returns:
            pandas data frame: the balanced data frame
            pandas series: balanced label
        """
        from imblearn.combine import SMOTETomek

        sm = SMOTETomek(ratio=ratio)
        resample_x, resample_y = sm.fit_sample(data_x, data_y)

        if printout:
            print('%d positive samples out of %d: %.2f%%' % (
                resample_y.sum(), len(resample_y), 1.0 * resample_y.sum() / len(resample_y)))
        return resample_x, resample_y
