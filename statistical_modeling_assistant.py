import numpy as np
import data_science_assistant as dsa
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels as stat


class StatisticalModelingAssistant(dsa.DataScienceAssistant):
    def __init__(self):
        super().__init__()

    @staticmethod
    def linear_regression(data, variables, y, missing_handle='none', intercept=True, printout=False):
        """
        train a linear regression model
        Args:
            data (pandas data frame): the data frame with all features
            variables ([str*]): explanatory variables list as strings
            y (string): responsive variable as a string
            missing_handle (string): method to handle missing observations, none/drop/raise
            intercept (bool): whether to maintain intercept in the model
            printout (bool): whether to print out summary
        Returns:
            statsmodels.formula.api.fit object: fitted model
        """
        reg_formula = y + '~' + '+'.join(variables)
        if not intercept:
            reg_formula += '-1'
        model = smf.ols(formula=reg_formula, data=data,
                        missing=missing_handle).fit()

        if printout:
            print(model.summary())

        return model

    @staticmethod
    def model_metrics(fitted_model):
        """
        get metrics of a fitted model as a dictionary
        Args:
            fitted_model (statsmodels.formula.api.fit object): the fitted model user wants to get metrics
        Returns:
            dict: metrics 
        """
        metrics = ['aic', 'bic', 'mse_resid', 'rsquared', 'rsquared_adj']
        metric_values = [stat.regression.linear_model.RegressionResults.aic(fitted_model),
                         stat.regression.linear_model.RegressionResults.bic(
                             fitted_model),
                         stat.regression.linear_model.RegressionResults.mse_resid(
                             fitted_model),
                         stat.regression.linear_model.RegressionResults.rsquared(
                             fitted_model),
                         stat.regression.linear_model.RegressionResults.rsquared_adj(fitted_model)]

        return dict(zip(metrics, metric_values))

    def experiment_linear_regression(self, data, y, n=10, metric='rsquared'):
        """
        randomly select several explanatory variable groups, run linear regression models and get corresponding metrics
        Args:
            data (pandas data frame): the data frame the user is working on
            y (string): a string represention the responsive variable
            n (int): times of randomly selection
            metric (string): the metric name user wants to return as optimal
        Returns:
            [str*]: list of explanatory variables that correspond to optimal metric defined
            float: optimal metric
            [[str*]*]: lists of explanatory variables randomly selected during experiment
            [dict*]: corresponding metric dictionaries
        """

        features = [x for x in list(data) if x != y]

        feature_cnt = len(features)
        feature_size_range = range(1, feature_cnt + 1, 1)
        feature_space = []
        metric_space = []
        if metric in {'rsquared', 'rsquared_adj'}:
            opt_metric = - float('inf')

            def compare(x, y): return x > y
        else:
            opt_metric = float('inf')

            def compare(x, y): return x < y
        opt_metric_feature = None
        for i in range(n):
            feature_space_size = np.random.choice(feature_size_range, 1)
            feature_sample = np.random.choice(
                features, feature_space_size, replace=False)
            feature_space.append(feature_sample)
            model = self.linear_regression(
                data, feature_sample, y, 'drop', True, False)
            metrics = self.model_metrics(model)
            if compare(metrics[metric], opt_metric):
                opt_metric = metrics[metric]
                opt_metric_feature = feature_sample
            metric_space.append(metrics)

        return opt_metric_feature, opt_metric, feature_space, metric_space

    @staticmethod
    def logistic_regression(data, variables, y, missing_handle='none', intercept=True, printout=False):
        """
        train a logistic regression model
        Args:
            data (pandas data frame): the data frame with all features
            variables ([str*]): features list of strings
            y (string): label name as a string
            missing_handle (string): method to handle missing observations, none/drop/raise
            intercept (bool): whether to maintain intercept in the model
            printout (bool): whether to print out summary
        Returns:
            statsmodels.formula.api.fit object: fitted model
        """
        reg_formula = y + '~' + '+'.join(variables)
        if not intercept:
            reg_formula += '-1'
        model = smf.glm(formula=reg_formula, data=data,
                        missing=missing_handle, family=sm.families.Binomial()).fit()

        if printout:
            print(model.summary())

        return model

    @staticmethod
    def logistic_regression_predict(model, data, label=False, threshold=0.5):
        """
        predict label through logistic regression model under given threshold
        Args:
            model (statsmodels.formula.api.fit object): fitted logistic model
            pred_y (numpy array): the predicted logistic value or label under given threshold
            label (bool): whether to predict label or logistic value
            threshold (float in (0, 1)): the threshold, 1 if logistic value > threshold, 0 otherwise.
        Returns:
            numpy array: the predicted label under given threshold
        """
        pred_y = model.predict(data)
        if label:
            pred_y[pred_y > threshold] = 1
            pred_y[pred_y <= threshold] = 0

        return pred_y

    @staticmethod
    def get_accuracy(pred_y, y):
        """
        compute the accuracy from given predicted labels and true labels
        Args:
            pred_y (numpy array): predicted labels
            y (numpy array): true labels
        Returns:
            float: accuracy
        """
        return np.sum(pred_y == y) * 1.0 / len(y)

    def softmax_regression(self, data, variables, ys, missing_handle='none', intercept=True):
        """
        train a softmax regression model
        Args:
            data (pandas data frame): the data frame with all features
            variables ([str*]): features list of strings
            ys ([str*]): dummy encoded responsive variables
            missing_handle (string): method to handle missing observations, none/drop/raise
            intercept (bool): whether to maintain intercept in the model
        Returns:
            [statsmodels.formula.api.fit object*]: list of fitted models
        """
        ys.sort()
        models = []
        for y in ys:
            model = self.logistic_regression(
                data, variables, y, missing_handle, intercept, printout=False)
            models.append(model)

        return models

    @staticmethod
    def softmax_regression_predict(models, data, label=False):
        """
        predict logistic values or labels through softmax regression model
        Args:
            models [statsmodels.formula.api.fit object*]: list of fitted models
            data (pandas data frame): the data frame with all features
            label (bool): whether to return logistic values or labels
        Returns:
            statsmodels.formula.api.fit object: fitted model
        """
        from sklearn.preprocessing import normalize
        models_cnt = len(models)
        sample_size = len(data)
        pred_y = np.zeros((sample_size, models_cnt))
        for i in range(models_cnt):
            pred_y[:, i] = models[i].predict(data).reshape(sample_size, 1)

        normalize(pred_y, norm='l1', axis=1, copy=False)

        res = np.argmax(pred_y, axis=1) if label else pred_y
        return res
