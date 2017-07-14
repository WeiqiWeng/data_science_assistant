import pandas as pd
import numpy as np
import data_science_assistant as dsa
import statsmodels.formula.api as smf
import statsmodels as stat


class StatisticalModelingAssistant(dsa.DataScienceAssistant):
    def __init__(self):
        pass

    def linear_regression(self, data, variables, y, intercept=True, printout=False):
        """
        train a linear regression model
        Args:
            data (pandas data frame): the data frame with all features
            variables ([str*]): explanatory variables as strings
            y (string): responsive variable as a string
            printout (bool): whether to print out summary
        Returns:
            statsmodels.formula.api.fit object: fitted model
        """
        reg_formula = y + '~' + '+'.join(variables)
        if not intercept:
            reg_formula += '-1'
        model = smf.ols(formula=reg_formula, data=data, missing=).fit()

        if printout:
            print(model.summary())

        return model


    def model_metrics(self, fitted_model):
        """
        get metrics of a fitted model as a dictionary
        Args:
            fitted_model (statsmodels.formula.api.fit object): the fitted model user wants to get metrics
        Returns:
            dict: metrics 
        """
        metrics = ['aic', 'bic', 'mse_resid', 'rsquared', 'rsquared_adj']
        metric_values = [stat.regression.linear_model.RegressionResults.aic(fitted_model), 
                         stat.regression.linear_model.RegressionResults.bic(fitted_model),
                         stat.regression.linear_model.RegressionResults.mse_resid(fitted_model),
                         stat.regression.linear_model.RegressionResults.rsquared(fitted_model),
                         stat.regression.linear_model.RegressionResults.rsquared_adj(fitted_model)]

        return dict(zip(metrics, metric_values))


    def experiment_regression(self, data, y, n=10):
        """
        randomly select several explanatory variable groups, run linear regression models and get corresponding metrics
        Args:
            data (pandas data frame): the data frame the user is working on
            y (string): a string represention the responsive variable
            n (int): times of randomly selection
        Returns:
            dict: explanatory variable group to model metrics 
        """

        features = list(data)
        feature_cnt = len(features)
        feature_size_range = range(1, feature_cnt + 1, 1)
        feature_space = []
        metric_space = []
        for i in range(n):
            feature_space_size = np.random.choice(feature_size_range, 1)
            feature_sample = np.random.choice(features, feature_space_size)
            feature_space.append(feature_sample)
            model = self.linear_regression(data, feature_sample, y, True, False)
            metrics = self.model_metrics(model)
            metric_space.append(metrics)

        return feature_space, metric_space
