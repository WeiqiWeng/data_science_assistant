import numpy as np
import data_science_assistant as dsa
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels as stat


class StatisticalModelingAssistant(dsa.DataScienceAssistant):
    def __init__(self):
        super().__init__()

    def experiment(self, model_obj, data, y, n=10, metric='rsquared'):
        """
        randomly select several explanatory variable groups, run regression models and get corresponding metrics
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
        if metric in {'rsquared', 'rsquared_adj', 'accuracy'}:
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
            model_obj.train(data, feature_sample, y, 'drop', True, False)
            model_metric = model_obj.get_metric(model_obj, metric)
            if compare(model_metric, opt_metric):
                opt_metric = metrics[metric]
                opt_metric_feature = feature_sample
            metric_space.append(metrics)

        return opt_metric_feature, opt_metric, feature_space, metric_space

    class LinearRegression():

        AVAILABLE_METRICS = ('aic', 'bic', 'mse_resid', 'rsquared', 'rsquared_adj')

        def __init__(self, model_obj=None):
            self.model = model_obj
    
        def train(self, data, variables, y, missing_handle='none', intercept=True, printout=False):
            """
            train a linear regression model
            Args:
                data (pandas data frame): the data frame with all features
                variables ([str*]): explanatory variables list as strings
                y (string): responsive variable as a string
                missing_handle (string): method to handle missing observations, none/drop/raise
                intercept (bool): whether to maintain intercept in the model
                printout (bool): whether to print out summary
            """
            reg_formula = y + '~' + '+'.join(variables)
            if not intercept:
                reg_formula += '-1'
            model = smf.ols(formula=reg_formula, data=data,
                            missing=missing_handle).fit()

            if printout:
                print(model.summary())

            self.model = model

        def predict(self, data):
            """
            use the linear regression model to predict given data
            Args:
                data (pandas data frame): the data user wants to predict
            Returns:
                numpy array: predicted values
            """
            return self.model.predict(data)


        @staticmethod
        def get_metric(trained_obj, metric, data):
            """
            get the metric specified
            Args:
                trained_obj (a LinearRegression object): a trained LinearRegression object
                metric (string): name of metric, aic/bic/mse_resid/rsquared/rsquared_adj
            Returns:
                float: the metric specified
            """
            fitted_model = trained_obj.model
            if not fitted_model:
                raise ValueError("The model is not trained yet.")
            if not metric in trained_obj.AVAILABLE_METRICS:
                raise ValueError("Given metirc not available.")

            metric_map = {'aic': stat.regression.linear_model.RegressionResults.aic(),
                          'bic': stat.regression.linear_model.RegressionResults.bic(), 
                          'mse_resid': stat.regression.linear_model.RegressionResults.mse_resid(), 
                          'rsquared': stat.regression.linear_model.RegressionResults.rsquared(), 
                          'rsquared_adj': stat.regression.linear_model.RegressionResults.rsquared_adj()}

            return metric_map[metric](fitted_model)

        @staticmethod
        def model_metrics(trained_obj):
            """
            get metrics of a fitted model as a dictionary
            Args:
                trained_obj (a LinearRegression object): a trained LinearRegression object
            Returns:
                dict: metrics to value
            """
            return dict(zip(metrics, [self.get_metric(trained_obj, x) for x in self.AVAILABLE_METRICS]))

    class LogisticRegression():

        AVAILABLE_METRICS = ('accuracy', 'sensitivity', 'specificity')

        def __init__(self, model_obj=None):
            self.model = model_obj

        def train(self, data, variables, y, missing_handle='none', intercept=True, printout=False):
            """
            train a logistic regression model
            Args:
                data (pandas data frame): the data frame with all features
                variables ([str*]): features list of strings
                y (string): label name as a string
                missing_handle (string): method to handle missing observations, none/drop/raise
                intercept (bool): whether to maintain intercept in the model
                printout (bool): whether to print out summary
            """
            reg_formula = y + '~' + '+'.join(variables)
            if not intercept:
                reg_formula += '-1'
            model = smf.glm(formula=reg_formula, data=data,
                            missing=missing_handle, family=sm.families.Binomial()).fit()

            if printout:
                print(model.summary())

            self.model = model

        def predict(self, data):
            """
            use the logistic regression to predict
            Args:
                data (pandas data frame): the data user wants to predict
            Returns:
                numpy array: the predicted sigmoid values
            """
            return self.model.predict(data)

        @staticmethod
        def confusion_matrix(y, pred_y):
            from sklearn.metrics import confusion_matrix
            return confusion_matrix(y, pred_y)

        @staticmethod
        def get_metric(trained_obj, metric):
            """
            compute the accuracy from given predicted labels and true labels
            Args:
                pred_y (numpy array): predicted labels
                y (numpy array): true labels
            Returns:
                float: accuracy
            """
            fitted_model = trained_obj.model
            if not fitted_model:
                raise ValueError("The model is not trained yet.")
            if not metric in trained_obj.AVAILABLE_METRICS:
                raise ValueError("Given metirc not available.")

            metric_map = {'accuracy': fitted_model.predict(data),
                          'bic': stat.regression.linear_model.RegressionResults.bic(fitted_model), 
                          'mse_resid': stat.regression.linear_model.RegressionResults.mse_resid(fitted_model), 
                          'rsquared': stat.regression.linear_model.RegressionResults.rsquared(fitted_model), 
                          'rsquared_adj': stat.regression.linear_model.RegressionResults.rsquared_adj(fitted_model)}

            return metric_map[metric]

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

    def cross_validation(self, variables, y, data, k=5):
        metrics = []
        size = len(data)
        fold_size = int(size / k)

        metric_list = []

        for i in range(k):
            out_fold_start, out_fold_end = i*fold_size, (i+1)*fold_size
            train_set = pd.concat(data.iloc[0:out_fold_start], data.iloc[out_fold_end:-1]) 
            validation_set = data.iloc[out_fold_start:out_fold_end]
            model = 



