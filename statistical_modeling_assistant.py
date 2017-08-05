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

        features = [x for x in list(data) if x != y] # TODO need regex operation: y_yes, y_no

        feature_cnt = len(features)
        feature_size_range = range(1, feature_cnt + 1, 1)
        feature_space = []
        metric_space = []
        if metric in {'aic', 'bic', 'mse_resid'}:
            opt_metric = float('inf')

            def compare(x, y): return x < y
        else:
            opt_metric = - float('inf')

            def compare(x, y): return x > y
        opt_metric_feature = None
        for i in range(n):
            feature_space_size = np.random.choice(feature_size_range, 1)
            feature_sample = np.random.choice(
                features, feature_space_size, replace=False)
            feature_space.append(feature_sample)
            model_obj.train(data, feature_sample, y, 'drop', True, False)
            model_metric = model_obj.get_metric(data, y, metric)
            if compare(model_metric, opt_metric):
                opt_metric = model_metric
                opt_metric_feature = feature_sample
            metric_space.append(model_metric)

        return opt_metric_feature, opt_metric, feature_space, metric_space

    class LinearRegression(object):

        AVAILABLE_METRICS = ('aic', 'bic', 'mse_resid',
                             'rsquared', 'rsquared_adj')

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


        def get_metric(self, data, y, metric):
            """
            get the metric specified
            Args:
                trained_obj (a LinearRegression object): a trained LinearRegression object
                metric (string): name of metric, aic/bic/mse_resid/rsquared/rsquared_adj
            Returns:
                float: the metric specified
            """
            fitted_model = self.model
            if not fitted_model:
                raise ValueError("The model is not trained yet.")
            if not metric in self.AVAILABLE_METRICS:
                raise ValueError("Given metirc not available.")

            metric_map = {'aic': stat.regression.linear_model.RegressionResults.aic,
                          'bic': stat.regression.linear_model.RegressionResults.bic,
                          'mse_resid': stat.regression.linear_model.RegressionResults.mse_resid,
                          'rsquared': stat.regression.linear_model.RegressionResults.rsquared,
                          'rsquared_adj': stat.regression.linear_model.RegressionResults.rsquared_adj}

            return metric_map[metric](fitted_model)

        def model_metrics(self):
            """
            get metrics of a fitted model as a dictionary
            Args:
                trained_obj (a LinearRegression object): a trained LinearRegression object
            Returns:
                dict: metrics to value
            """
            return dict(zip(self.AVAILABLE_METRICS, [self.get_metric(x) for x in self.AVAILABLE_METRICS]))

    class LogisticRegression(object):

        AVAILABLE_METRICS = ('accuracy', 'sensitivity', 'specificity', 'precision', 'negative_predictive_value',
                             'fall_out', 'false_negative_rate', 'f1_score', 'false_discover_rate')

        def __init__(self, model_obj=None, init_threshold=0.5):
            if init_threshold <= 0 or init_threshold >= 1:
                raise ValueError("Initial threshold should be in (0, 1). ")
            self.model = model_obj
            self.threshold = init_threshold

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
            pred_y = self.model.predict(data)
            pred_y[pred_y >= self.threshold] = 1
            pred_y[pred_y < self.threshold] = 0

            return pred_y

        @staticmethod
        def confusion_matrix(y, pred_y):
            from sklearn.metrics import confusion_matrix
            return confusion_matrix(y, pred_y)

        @staticmethod
        def accuracy(confusion_matrix):
            return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

        @staticmethod
        def sensitivity(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return 1.0 * confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])

        @staticmethod
        def specificity(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return 1.0 * confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

        @staticmethod
        def precision(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return 1.0 * confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])

        @staticmethod
        def negative_predictive_value(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return 1.0 * confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])

        @staticmethod
        def fall_out(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return 1.0 * confusion_matrix[0, 1] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

        @staticmethod
        def false_negative_rate(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return 1.0 * confusion_matrix[1, 0] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])

        @staticmethod
        def f1_score(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return 2.0 * confusion_matrix[1, 1] / (2.0 * confusion_matrix[1, 1] + confusion_matrix[0, 1] + confusion_matrix[1, 0])

        @staticmethod
        def false_discover_rate(confusion_matrix):
            if confusion_matrix.shape[0] > 2:
                raise ValueError("Only 2D confusion matrix available.")
            return confusion_matrix[0, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])

        def get_metric(self, data, y, metric):
            """
            compute the accuracy from given predicted labels and true labels
            Args:
                pred_y (numpy array): predicted labels
                y (numpy array): true labels
            Returns:
                float: accuracy
            """
            if not self.model:
                raise ValueError("The model is not trained yet.")
            if not metric in self.AVAILABLE_METRICS:
                raise ValueError("Given metirc not available.")

            confusion_matrix = self.confusion_matrix(
                data[y], self.predict(data))

            metric_map = {'accuracy': self.accuracy,
                          'sensitivity': self.sensitivity,
                          'specificity': self.specificity,
                          'precision': self.precision,
                          'negative_predictive_value': self.negative_predictive_value,
                          'fall_out': self.fall_out,
                          'false_negative_rate': self.false_negative_rate,
                          'f1_score': self.f1_score,
                          'false_discover_rate': self.false_discover_rate
                          }

            return metric_map[metric](confusion_matrix)

        def model_metrics(self, data, y):
                """
                get metrics of a fitted model as a dictionary
                Args:
                    trained_obj (a LinearRegression object): a trained LinearRegression object
                Returns:
                    dict: metrics to value
                """
                return dict(zip(self.AVAILABLE_METRICS, [self.get_metric(data, y, x) for x in self.AVAILABLE_METRICS]))

    class SoftmaxRegression(LogisticRegression):
        def __init__(self, model_objs=[]):
            super().__init__()
            self.models = model_objs

        def train(self, data, variables, ys, missing_handle='none', intercept=True):
            """
            train a softmax regression model
            Args:
                data (pandas data frame): the data frame with all features
                variables ([str*]): features list of strings
                ys ([str*]): dummy encoded responsive variables
                missing_handle (string): method to handle missing observations, none/drop/raise
                intercept (bool): whether to maintain intercept in the model
            """
            ys.sort()
            models = []
            for y in ys:
                super().train(
                    data, variables, y, missing_handle, intercept, printout=False)
                models.append(self.model)

            self.models = models

        def predict_sigmoid_values(self, data):
            """
            predict logistic values through softmax regression model
            Args:
                data (pandas data frame): the data frame with all features
            Returns:
                numpy array: predicted sigmoid values
            """
            from sklearn.preprocessing import normalize
            models_cnt = len(self.models)
            sample_size = len(data)
            pred_y = np.zeros((sample_size, models_cnt))
            for i in range(models_cnt):
                pred_y[:, i] = self.models[i].predict(data)

            normalize(pred_y, norm='l1', axis=1, copy=False)

            return pred_y

        def predict(self, data):
            """
            predict labels through softmax regression model
            Args:
                data (pandas data frame): the data frame with all features
            Returns:
                statsmodels.formula.api.fit object: fitted model
            """
            pred_y = self.predict_sigmoid_values(data)

            res = np.argmax(pred_y, axis=1)
            return res

    # def cross_validation(self, variables, y, data, k=5):
    #     metrics = []
    #     size = len(data)
    #     fold_size = int(size / k)

    #     metric_list = []

    #     for i in range(k):
    #         out_fold_start, out_fold_end = i * fold_size, (i + 1) * fold_size
    #         train_set = pd.concat(
    #             data.iloc[0:out_fold_start], data.iloc[out_fold_end:-1])
    #         validation_set = data.iloc[out_fold_start:out_fold_end]
    #         model =
