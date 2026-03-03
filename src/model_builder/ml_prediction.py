import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from scipy.stats import sem, t
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from tqdm.notebook import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

class ml_trainer:

    """
    A class to train machine learning models with different feature sets.

    Attributes
    ----------
    models_dict : dict
        Dictionary of different machine learning models.

    hyperparameters : dict
        Dictionary of hyperparameters for the machine learning models.

    roc_curves_list : list
        List of ROC curves for different feature sets and models.

    results : list
        List of results for different feature sets and models.

    Methods
    -------
    trainer(X_train, y_train, X_test, y_test, feature_dict, models_dict, hyperparameter_dict)
        Train machine learning models with different
        feature sets and hyperparameters.
    """

    models_dict = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(verbose=False, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'XGBoost': GradientBoostingClassifier(random_state=42),
        'Ridge': CalibratedClassifierCV(RidgeClassifier(random_state=42))}

    hyperparameters = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Naive Bayes': {},
        'Neural Network': {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.01, 0.1]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, None], 'min_samples_split': [2, 10]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]},
        'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]},
        'Ridge': {}
    }

    def __init__(self):
        pass

    def trainer(self, X_train, y_train, X_test, y_test, feature_dict, models_dict=models_dict, hyperparameter_dict=hyperparameters):

        """
        Train machine learning models with different feature sets and hyperparameters.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training set to train the models.
            
        y_train : pandas.Series
            Target variable for the training set.

        X_test : pandas.DataFrame
            Test set to evaluate the models.

        y_test : pandas.Series
            Target variable for the test set.

        feature_dict : dict
            Dictionary of different feature sets.

        models_dict : dict
            Dictionary of different machine learning models.

        hyperparameter_dict : dict
            Dictionary of hyperparameters for the machine learning models.

        Returns:
        
        results : list
            List of results for different feature sets and models.
        """

        # Suppress the ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        self.feature_dict = feature_dict
        self.hyperparameters = hyperparameter_dict
        
        # Initialize an empty list to store the ROC curves
        
        self.roc_curves_list = []
        self.results = []

        # Record the ROC curves and feature importance
        self.roc_curves = {}

        le = LabelEncoder()
        smote = SMOTE(random_state=42)

        for feature_set_name, feature_set in tqdm(self.feature_dict.items()):
            roc_curves_per_feature_set = []
            X_train_mod = X_train[feature_set]
            X_test_mod = X_test[feature_set]
            y_train_mod = y_train
            y_val_mod = y_train
            y_test_mod = y_test

            results_per_feature_set = []

            for model_name, model in tqdm(models_dict.items()):
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

                # Hyperparameter tuning
                model = GridSearchCV(model, param_grid=self.hyperparameters[model_name], cv=cv, scoring=make_scorer(roc_auc_score, needs_proba=True), n_jobs=-1)

                model.fit(X_train_mod, y_train_mod)

                best_parameters = model.best_estimator_

                cv1 = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
                val_predictions = cross_val_predict(model, X_train_mod, y_train, method="predict_proba", cv=cv1, n_jobs=-1)

                y_pred_train_labels = model.predict(X_train_mod)
                y_pred_val_labels = np.round(val_predictions[:, 1])
                y_pred_test_labels = model.predict(X_test_mod)

                # Get the probability estimates
                y_pred_train_proba = model.predict_proba(X_train_mod)[:, 1]
                y_pred_val_proba = val_predictions[:,1]
                y_pred_test_proba = model.predict_proba(X_test_mod)[:, 1]

                precision_train = precision_score(y_train_mod, y_pred_train_labels)
                precision_val = precision_score(y_val_mod, y_pred_val_labels)
                precision_test = precision_score(y_test_mod, y_pred_test_labels)

                recall_train = recall_score(y_train_mod, y_pred_train_labels)
                recall_val = recall_score(y_val_mod, y_pred_val_labels)
                recall_test = recall_score(y_test_mod, y_pred_test_labels)

                f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)
                f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
                f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

                    # for train data
                fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
                roc_auc_train = auc(fpr_train, tpr_train)

                # for validation data
                fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
                roc_auc_val = auc(fpr_val, tpr_val)

                # for test data
                fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)
                roc_auc_test = auc(fpr_test, tpr_test)
                
                cm_train = confusion_matrix(y_train_mod, y_pred_train_labels)
                cm_val = confusion_matrix(y_val_mod, y_pred_val_labels)
                cm_test = confusion_matrix(y_test_mod, y_pred_test_labels)

                tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
                tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
                tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

                sensitivity_train = tp_train / (tp_train + fn_train)
                sensitivity_val = tp_val / (tp_val + fn_val)
                sensitivity_test = tp_test / (tp_test + fn_test)

                specificity_train = tn_train / (tn_train + fp_train)
                specificity_val = tn_val / (tn_val + fp_val)
                specificity_test = tn_test / (tn_test + fp_test)
                
                # Get ROC curves
                fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
                fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
                fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)
            
                roc_curves_per_feature_set.append({
                    'model': model_name,
                    'best_parameters': best_parameters.get_params(),
                    'train': {'fpr': fpr_train, 'tpr': tpr_train},
                    'val': {'fpr': fpr_val, 'tpr': tpr_val},
                    'test': {'fpr': fpr_test, 'tpr': tpr_test}
                })
                
            
                # Append the results to the lists
                results_per_feature_set.append({
                    'model': model_name,
                    'best_parameters': best_parameters.get_params(),
                    'model_instance': model,
                    'roc_auc_train': roc_auc_train,
                    'roc_auc_val': roc_auc_val,
                    'roc_auc_test': roc_auc_test,
                    'f1_train': f1_train,
                    'f1_val': f1_val,
                    'f1_test': f1_test,
                    'recall_train': recall_train,
                    'recall_val': recall_val,
                    'recall_test': recall_test,
                    'precision_train': precision_train,
                    'precision_val': precision_val,
                    'precision_test': precision_test,
                    'sensitivity_train': sensitivity_train,
                    'sensitivity_val': sensitivity_val,
                    'sensitivity_test': sensitivity_test,
                    'specificity_train': specificity_train,
                    'specificity_val': specificity_val,
                    'specificity_test': specificity_test,
                })

            self.results.append({
                'feature_set': feature_set_name,
                'results': results_per_feature_set
            })
            self.roc_curves_list.append({
                'feature_set': feature_set_name,
                'roc_curves': roc_curves_per_feature_set
            })

        warnings.filterwarnings("default")

        # Define the models and feature selection techniques
        self.models = [roc_curve['model'] for roc_curve in self.results[0]['results']]  # Assuming the same models are used for each feature selection method
        self.feature_select_methods = [result['feature_set'] for result in self.results]

        # Initialize lists to store the results
        self.results_dict = {
            'roc_auc_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'roc_auc_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'roc_auc_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'f1_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'f1_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'f1_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'precision_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'precision_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'precision_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'recall_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'recall_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'recall_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'sensitivity_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'sensitivity_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'sensitivity_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'specificity_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'specificity_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'specificity_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
        }

        # Iterate over the results
        for i, result in enumerate(self.results):
            for j, model_result in enumerate(result['results']):
                # Store the metrics in the respective arrays
                for metric in self.results_dict.keys():
                    self.results_dict[metric][j, i] = model_result[metric]  # Transpose the matrix by swapping i and j

class ml_evaluator:

    """
    Evaluate the performance of machine learning models.

    Parameters
    ----------
    results : list
        List of results for different feature sets and models obtained from the ml_trainer class.

    Returns:
    None
    """
        
    def __init__(self, results):
        self.results = results
        self.models = [roc_curve['model'] for roc_curve in self.results[0]['results']]  # Assuming the same models are used for each feature selection method
        self.feature_select_methods = [result['feature_set'] for result in self.results]

        # Initialize lists to store the results
        self.results_dict = {
            'roc_auc_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'roc_auc_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'roc_auc_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'f1_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'f1_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'f1_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'precision_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'precision_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'precision_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'recall_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'recall_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'recall_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'sensitivity_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'sensitivity_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'sensitivity_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'specificity_train': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'specificity_val': np.zeros((len(self.models), len(self.feature_select_methods ))),
            'specificity_test': np.zeros((len(self.models), len(self.feature_select_methods ))),
        }

        # Iterate over the results
        for i, result in enumerate(self.results):
            for j, model_result in enumerate(result['results']):
                # Store the metrics in the respective arrays
                for metric in self.results_dict.keys():
                    self.results_dict[metric][j, i] = model_result[metric]  # Transpose the matrix by swapping i and j

    # Calculate Youden index cutoff for classfication

    @staticmethod
    def sensivity_specifity_cutoff(y_true, y_score):

        """
        Find data-driven cut-off for classification
        
        Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
        
        Parameters
        ----------
        
        y_true : array, shape = [n_samples]
            True binary labels.
            
        y_score : array, shape = [n_samples]
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            “decision_function” on some classifiers).
            
        References
        ----------
        
        Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
        Journal of clinical epidemiology, 59(8), 798-801.
        
        Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
        prediction models and markers: evaluation of predictions and classifications.
        Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
        
        Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
        of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.argmax(tpr - fpr)
        return thresholds[idx]


    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    @staticmethod
    def compute_midrank(x):

        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    @staticmethod
    def fastDeLong(predictions_sorted_transposed, label_1_count):
    
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    @staticmethod
    def calc_pvalue(aucs, sigma):

        """Computes log(10) of p-values.
        Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances
        Returns:
        log10(pvalue)
        """
        l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

    @staticmethod
    def compute_ground_truth_statistics(ground_truth):

        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    @staticmethod
    def delong_roc_variance(ground_truth, predictions):

        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = compute_ground_truth_statistics(ground_truth)
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
        return aucs[0], delongcov

    @staticmethod
    def delong_roc_test(ground_truth, predictions_one, predictions_two):
        
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = compute_ground_truth_statistics(ground_truth)
        predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
        aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
        return calc_pvalue(aucs, delongcov)
    
    @staticmethod
    def compute_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.05, random_state=42):

        """
        Compute the AUC and its confidence interval using bootstrapping.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.
        y_score : array-like of shape (n_samples,)
            Target scores.
        n_bootstraps : int, default=1000
            Number of bootstraps.
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for a 95% confidence interval).
        
        Returns
        -------
        auc : float
            AUC score.
        ci_lower : float
            Lower bound of the confidence interval.
        ci_upper : float
            Upper bound of the confidence interval.
        """
        assert len(y_true) == len(y_score), "Lengths of y_true and y_score should be equal."
        
        auc = roc_auc_score(y_true, y_score)
        bootstrapped_scores = []
        
        for _ in range(n_bootstraps):
            # Bootstrap by sampling with replacement
            y_true_resampled, y_score_resampled = resample(y_true, y_score, random_state=random_state+_)
            score = roc_auc_score(y_true_resampled, y_score_resampled)
            bootstrapped_scores.append(score)
        
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        
        # Compute the lower and upper bound of the confidence interval
        confidence_lower = sorted_scores[int((alpha / 2.0) * n_bootstraps)]
        confidence_upper = sorted_scores[int((1 - alpha / 2.0) * n_bootstraps)]
        
        return auc, confidence_lower, confidence_upper

    def results_heatmap(self):

        """
        Generate a heatmap of the results.

        Parameters
        ----------
        None

        Returns:
        Heatmap : matplotlib.pyplot
            Heatmap of the results for machine learning and feature selection techniques.

        """

        # Create the subplots grid
        fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(40, 40))

        # Flatten the axes to iterate over them
        axes = axes.flatten()

        # Generate the heatmaps
        for (metric, matrix), ax in zip(self.results_dict.items(), axes):
            sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=False)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel("Model")  # Swap x and y labels
            ax.set_xlabel("Feature Selection Technique")  # Swap x and y labels
            ax.set_yticklabels(self.models, rotation=0)  # Swap x and y tick labels
            ax.set_xticklabels(self.feature_select_methods, rotation=90)  # Swap x and y tick labels
            ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
            ax.set_aspect('equal')  # Ensure equal aspect ratio for each subplot

            # Adjust font size for better readability
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

        # Adjust the layout
        plt.tight_layout()

        # Show the plot
        plt.show()

    def ranked_model_performance(self, train_test_val_set="val", metric="roc_auc"):

        """
        Get the ranked validation model performance based on average

        Parameters
        ----------
        train_test_val_set : str
            Set to get the ranked model performance.
            train : Training set.
            test : Test set.
            val : Validation set.

        metric : str
            Metric to rank the models.
            roc_auc : ROC AUC score.
            f1 : F1 score.
            precision : Precision score.
            recall : Recall score.
            sensitivity : Sensitivity score.
            specificity : Specificity score.

        Returns:
        val_model_performance : list
            List of validation model performance based on the mean AUC score.

        """

        matrix = self.results_dict[f'{metric}_{train_test_val_set}']

        set_model_performance = []

        for i,m in enumerate(matrix):
            set_model_performance.append((np.mean(m),self.models[i]))

        set_model_performance.sort(reverse=True)
        print(set_model_performance)
        

    def ranked_feature_selection_performance(self, train_test_val_set="val", metric="roc_auc"):

        """
        Get the ranked validation model performance based on average

        Parameters
        ----------
        train_test_val_set : str
            Set to get the ranked model performance.
            train : Training set.
            test : Test set.
            val : Validation set.

        metric : str
            Metric to rank the models.
            roc_auc : ROC AUC score.
            f1 : F1 score.
            precision : Precision score.
            recall : Recall score.
            sensitivity : Sensitivity score.
            specificity : Specificity score.

        Returns:
        val_model_performance : list
            List of validation model performance based on the mean AUC score.

        """
        # Validation set - average performance of each feature selection technique across all models

        matrix = self.results_dict[f'{metric}_{train_test_val_set}']

        set_feature_performance = []

        for i in range(0,matrix.shape[1]):
            set_feature_performance.append((np.mean(matrix[:,i]),self.feature_select_methods[i]))

        # Sort the feature selection techniques based on the mean AUC score
        set_feature_performance.sort(reverse=True)
        print(set_feature_performance)


    def top_3_combinations(self, train_test_val_set="val", metric="roc_auc"):

        """
        Get the top 3 models and feature selection techniques combinations for specified metric and set.

        Parameters
        ----------
        train_test_val_set : str
            Set to get the top 3 combinations.
            train : Training set.
            test : Test set.
            val : Validation set.

        metric : str
            Metric to rank the models.
            roc_auc : ROC AUC score.
            f1 : F1 score.
            precision : Precision score.
            recall : Recall score.
            sensitivity : Sensitivity score.
            specificity : Specificity score.

        Returns:
        None
        """

        # Determine top 3 models and feature selection techniques combinations

        matrix = self.results_dict[f'{metric}_{train_test_val_set}']

        best_feature_index = []

        for i in range(0,matrix.shape[0]):
            best_feature_index.append(np.argmax(matrix[i,:]))

        model_1 = (0,0)
        model_2 = (0,0)
        model_3 = (0,0)

        for i in range(0,matrix.shape[0]):
            if matrix[i,best_feature_index[i]] > matrix[model_1[0],model_1[1]]:
                model_3 = model_2
                model_2 = model_1
                model_1 = (i,best_feature_index[i])
            elif matrix[i,best_feature_index[i]] > matrix[model_2[0],model_2[1]]:
                model_3 = model_2
                model_2 = (i,best_feature_index[i])
            elif matrix[i,best_feature_index[i]] > matrix[model_3[0],model_3[1]]:
                model_3 = (i,best_feature_index[i])

        print(f"Best model: {self.models[model_1[0]]} with feature selection technique: {self.feature_select_methods[model_1[1]]}, val_auc: {matrix[model_1[0],model_1[1]]}")
        print(f"Second best model: {self.models[model_2[0]]} with feature selection technique: {self.feature_select_methods[model_2[1]]}, val_auc: {matrix[model_2[0],model_2[1]]}")
        print(f"Third best model: {self.models[model_3[0]]} with feature selection technique: {self.feature_select_methods[model_3[1]]}, val_auc: {matrix[model_3[0],model_3[1]]}")


    def ensemble_top_3(self, X_train, y_train, X_test, y_test, feature_dict, train_test_val_set="val", metric="roc_auc", random_state=42):

        self.feature_dict = feature_dict

        def model_predictor(X_train, y_train, X_test, y_test, model_name, feature_set_name, random_state=random_state):

            models_blank_dict = {
            'Logistic Regression': LogisticRegression,
            'Naive Bayes': GaussianNB,
            'Neural Network': MLPClassifier,
            'Random Forest': RandomForestClassifier,
            'Support Vector Machine': SVC,
            'XGBoost': GradientBoostingClassifier,
            'Ridge': CalibratedClassifierCV()
            }


            # Get the features selected by Best Method

            feature_set_name = feature_set_name  # Replace this with the correct key if it's different
            feature_set = feature_dict[feature_set_name]

            # Get the training, validation, and test sets
            X_train_mod = X_train[feature_set]
            X_test_mod = X_test[feature_set]
            y_train_mod = y_train
            y_val_mod = y_train
            y_test_mod = y_test

            # Initialize Best model
            model_index = list(models_blank_dict.keys()).index(model_name)
            feature_set_index = list(feature_dict.keys()).index(feature_set_name)
            model = self.results[feature_set_index]["results"][model_index]["model_instance"]

            # if model_name == "Ridge":
            #     model = CalibratedClassifierCV(RidgeClassifier(random_state=42))
            # else:   
            #     model = models_blank_dict[model_name](**self.results[feature_set_index]["results"][model_index]["best_parameters"])

            # Fit the model
            model.fit(X_train_mod, y_train_mod)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            val_predictions = cross_val_predict(model, X_train_mod, y_train, method="predict_proba", cv=cv, n_jobs=-1)

            # Predict the probabilities
            y_pred_train_proba = model.predict_proba(X_train_mod)[:, 1]
            y_pred_val_proba = val_predictions[:,1]
            y_pred_test_proba = model.predict_proba(X_test_mod)[:, 1]

            # Calculate the ROC curves
            fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
            fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
            fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)

            # Get ROC curves
            fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
            fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
            fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)

            auc_train = self.compute_auc_ci(y_train_mod, y_pred_train_proba)
            auc_val = self.compute_auc_ci(y_val_mod, y_pred_val_proba)
            auc_test = self.compute_auc_ci(y_test_mod, y_pred_test_proba)

            youden_index = self.sensivity_specifity_cutoff(y_val_mod, y_pred_val_proba)

            y_pred_train_label = np.array(y_pred_train_proba > youden_index).astype(int)
            y_pred_val_label = np.array(y_pred_val_proba > youden_index).astype(int)
            y_pred_test_label = np.array(y_pred_test_proba > youden_index).astype(int)

            result = permutation_importance(model, X_train_mod, y_train_mod, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
            feature_importance = result.importances_mean

            feature_importance_dict = dict(zip(feature_set, feature_importance))

            model_predictions = {}

            model_predictions = {
            "model_name":model_name,
            "feature_set_name":feature_set_name,
            "train_truth":[y_train_mod],
            "val_truth":[y_val_mod],
            "test_truth":[y_test_mod],
            "train_proba":[y_pred_train_proba],
            "val_proba":[y_pred_val_proba],
            "test_proba":[y_pred_test_proba],
            "train_auc":[auc_train],
            "val_auc":[auc_val],
            "test_auc":[auc_test],
            "train_fpr":[fpr_train],
            "train_tpr":[tpr_train],
            "val_fpr":[fpr_val],
            "val_tpr":[tpr_val],
            "test_fpr":[fpr_test],
            "test_tpr":[tpr_test],
            "youden_index":[youden_index],
            "y_pred_train_label":[y_pred_train_label],
            "y_pred_val_label":[y_pred_val_label],
            "y_pred_test_label":[y_pred_test_label],
            "new_model_predictor":model,
            "feature_importance":feature_importance_dict
            }

            return model_predictions
        
        def ensemble_predictor(X_train, y_train, X_test, y_test, model_1_predictions, model_2_predictions, model_3_predictions, random_state=random_state):

            models_blank_dict = {
            'Logistic Regression': LogisticRegression,
            'Naive Bayes': GaussianNB,
            'Neural Network': MLPClassifier,
            'Random Forest': RandomForestClassifier,
            'Support Vector Machine': SVC,
            'XGBoost': GradientBoostingClassifier,
            'Ridge': CalibratedClassifierCV()
            }

            y_train_mod = y_train
            y_val_mod = y_train
            y_test_mod = y_test

            # Get the features selected by Best Method

            # Predict the probabilities
            y_pred_train_proba = (model_1_predictions["train_proba"][0] + model_2_predictions["train_proba"][0] + model_3_predictions["train_proba"][0])/3
            y_pred_val_proba = (model_1_predictions["val_proba"][0] + model_2_predictions["val_proba"][0] + model_3_predictions["val_proba"][0])/3
            y_pred_test_proba = (model_1_predictions["test_proba"][0] + model_2_predictions["test_proba"][0] + model_3_predictions["test_proba"][0])/3

            # Calculate the ROC curves
            fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
            fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
            fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)

            # Get ROC curves
            fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
            fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
            fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)

            auc_train = self.compute_auc_ci(y_train_mod, y_pred_train_proba)
            auc_val = self.compute_auc_ci(y_val_mod, y_pred_val_proba)
            auc_test = self.compute_auc_ci(y_test_mod, y_pred_test_proba)

            youden_index = self.sensivity_specifity_cutoff(y_val_mod, y_pred_val_proba)

            y_pred_train_label = np.array(y_pred_train_proba > youden_index).astype(int)
            y_pred_val_label = np.array(y_pred_val_proba > youden_index).astype(int)
            y_pred_test_label = np.array(y_pred_test_proba > youden_index).astype(int)

            new_model_predictors = {"model_1":model_1_predictions["new_model_predictor"], "model_2":model_2_predictions["new_model_predictor"], "model_3":model_3_predictions["new_model_predictor"]}

            feature_importance_dicts = {"model_1":model_1_predictions["feature_importance"], "model_2":model_2_predictions["feature_importance"], "model_3":model_3_predictions["feature_importance"]}

            model_predictions = {}

            model_predictions = {
            "model_name":"ensemble",
            "feature_set_name":"ensemble",
            "train_truth":[y_train_mod],
            "val_truth":[y_val_mod],
            "test_truth":[y_test_mod],
            "train_proba":[y_pred_train_proba],
            "val_proba":[y_pred_val_proba],
            "test_proba":[y_pred_test_proba],
            "train_auc":[auc_train],
            "val_auc":[auc_val],
            "test_auc":[auc_test],
            "train_fpr":[fpr_train],
            "train_tpr":[tpr_train],
            "val_fpr":[fpr_val],
            "val_tpr":[tpr_val],
            "test_fpr":[fpr_test],
            "test_tpr":[tpr_test],
            "youden_index":[youden_index],
            "y_pred_train_label":[y_pred_train_label],
            "y_pred_val_label":[y_pred_val_label],
            "y_pred_test_label":[y_pred_test_label],
            "new_model_predictors":new_model_predictors,
            "feature_importance":feature_importance_dicts
            }

            return model_predictions

        # Determine top 3 models and feature selection techniques combinations

        matrix = self.results_dict[f'{metric}_{train_test_val_set}']

        best_feature_index = []

        for i in range(0,matrix.shape[0]):
            best_feature_index.append(np.argmax(matrix[i,:]))

        model_1 = (0,0)
        model_2 = (0,0)
        model_3 = (0,0)

        for i in range(0,matrix.shape[0]):
            if matrix[i,best_feature_index[i]] > matrix[model_1[0],model_1[1]]:
                model_3 = model_2
                model_2 = model_1
                model_1 = (i,best_feature_index[i])
            elif matrix[i,best_feature_index[i]] > matrix[model_2[0],model_2[1]]:
                model_3 = model_2
                model_2 = (i,best_feature_index[i])
            elif matrix[i,best_feature_index[i]] > matrix[model_3[0],model_3[1]]:
                model_3 = (i,best_feature_index[i])

        model_1_predictions = model_predictor(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name=self.models[model_1[0]], feature_set_name=self.feature_select_methods[model_1[1]], random_state=random_state)
        model_2_predictions = model_predictor(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name=self.models[model_2[0]], feature_set_name=self.feature_select_methods[model_2[1]], random_state=random_state)
        model_3_predictions = model_predictor(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_name=self.models[model_3[0]], feature_set_name=self.feature_select_methods[model_3[1]], random_state=random_state)

        ensemble_model_predictions = ensemble_predictor(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_1_predictions=model_1_predictions, model_2_predictions=model_2_predictions, model_3_predictions=model_3_predictions, random_state=random_state)

        all_model_predictions = {"model_1":model_1_predictions, "model_2":model_2_predictions, "model_3":model_3_predictions, "ensemble_model":ensemble_model_predictions}
        
        print("\nModel 1")
        print(f"{self.models[model_1[0]], self.feature_select_methods[model_1[1]]}")
        print(all_model_predictions["model_1"]["train_auc"])
        print(all_model_predictions["model_1"]["val_auc"])
        print(all_model_predictions["model_1"]["test_auc"])

        print("\nModel 2")
        print(f"{self.models[model_2[0]], self.feature_select_methods[model_2[1]]}")
        print(all_model_predictions["model_2"]["train_auc"])
        print(all_model_predictions["model_2"]["val_auc"])
        print(all_model_predictions["model_2"]["test_auc"])

        print("\nModel 3")
        print(f"{self.models[model_3[0]], self.feature_select_methods[model_3[1]]}")
        print(all_model_predictions["model_3"]["train_auc"])
        print(all_model_predictions["model_3"]["val_auc"])
        print(all_model_predictions["model_3"]["test_auc"])

        print("\nEnsemble Model")
        print(all_model_predictions["ensemble_model"]["train_auc"])
        print(all_model_predictions["ensemble_model"]["val_auc"])
        print(all_model_predictions["ensemble_model"]["test_auc"])

        self.all_model_predictions = all_model_predictions

        return all_model_predictions
    
    def ensemble_model_new_predictor(self, new_X_data):

        model_1 = self.all_model_predictions["model_1"]["new_model_predictor"]
        model_1_feature_set = self.feature_dict[self.all_model_predictions["model_1"]["feature_set_name"]]
        model_2 = self.all_model_predictions["model_2"]["new_model_predictor"]
        model_2_feature_set = self.feature_dict[self.all_model_predictions["model_2"]["feature_set_name"]]
        model_3 = self.all_model_predictions["model_3"]["new_model_predictor"]
        model_3_feature_set = self.feature_dict[self.all_model_predictions["model_3"]["feature_set_name"]]

        new_y_pred_data = (model_1.predict_proba(new_X_data[model_1_feature_set])+
                           model_2.predict_proba(new_X_data[model_2_feature_set])+
                            model_3.predict_proba(new_X_data[model_3_feature_set]))/3

        return new_y_pred_data[:,1]
    

def custom_model_predictor(X_train, y_train, X_test, y_test, custom_model_name, custom_feature_set, random_state=42):
    
    # Get the features selected by Best Method
    feature_set_name = custom_feature_set  # Replace this with the correct key if it's different
    feature_set = custom_feature_set

    # Calculate Youden index cutoff for classfication

    def sensivity_specifity_cutoff(y_true, y_score):

        """
        Find data-driven cut-off for classification
        
        Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
        
        Parameters
        ----------
        
        y_true : array, shape = [n_samples]
            True binary labels.
            
        y_score : array, shape = [n_samples]
            Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions (as returned by
            “decision_function” on some classifiers).
            
        References
        ----------
        
        Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
        Journal of clinical epidemiology, 59(8), 798-801.
        
        Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
        prediction models and markers: evaluation of predictions and classifications.
        Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
        
        Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
        of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.argmax(tpr - fpr)
        return thresholds[idx]


    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/

    def compute_midrank(x):

        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    def fastDeLong(predictions_sorted_transposed, label_1_count):

        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    def calc_pvalue(aucs, sigma):

        """Computes log(10) of p-values.
        Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances
        Returns:
        log10(pvalue)
        """
        l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

    def compute_ground_truth_statistics(ground_truth):

        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    def delong_roc_variance(ground_truth, predictions):

        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = compute_ground_truth_statistics(ground_truth)
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
        return aucs[0], delongcov

    def delong_roc_test(ground_truth, predictions_one, predictions_two):
        
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = compute_ground_truth_statistics(ground_truth)
        predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
        aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
        return calc_pvalue(aucs, delongcov)
    
    def compute_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.05, random_state=42):

        """
        Compute the AUC and its confidence interval using bootstrapping.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.
        y_score : array-like of shape (n_samples,)
            Target scores.
        n_bootstraps : int, default=1000
            Number of bootstraps.
        alpha : float, default=0.05
            Confidence level (e.g., 0.05 for a 95% confidence interval).
        
        Returns
        -------
        auc : float
            AUC score.
        ci_lower : float
            Lower bound of the confidence interval.
        ci_upper : float
            Upper bound of the confidence interval.
        """
        assert len(y_true) == len(y_score), "Lengths of y_true and y_score should be equal."
        
        auc = roc_auc_score(y_true, y_score)
        bootstrapped_scores = []
        
        for _ in range(n_bootstraps):
            # Bootstrap by sampling with replacement
            y_true_resampled, y_score_resampled = resample(y_true, y_score, random_state=random_state+_)
            score = roc_auc_score(y_true_resampled, y_score_resampled)
            bootstrapped_scores.append(score)
        
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        
        # Compute the lower and upper bound of the confidence interval
        confidence_lower = sorted_scores[int((alpha / 2.0) * n_bootstraps)]
        confidence_upper = sorted_scores[int((1 - alpha / 2.0) * n_bootstraps)]
        
        return auc, confidence_lower, confidence_upper

    models_dict = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(verbose=False, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'XGBoost': GradientBoostingClassifier(random_state=42),
        'Ridge': CalibratedClassifierCV(RidgeClassifier(random_state=42))}

    hyperparameters = {
        'Logistic Regression': {'C': [0.1, 1, 10]},
        'Naive Bayes': {},
        'Neural Network': {'hidden_layer_sizes': [(10,), (20,), (30,)], 'alpha': [0.0001, 0.01, 0.1]},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [10, None], 'min_samples_split': [2, 10]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]},
        'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]},
        'Ridge': {}
    }


    # Get the training, validation, and test sets
    X_train_mod = X_train[custom_feature_set]
    X_test_mod = X_test[custom_feature_set]
    y_train_mod = y_train
    y_val_mod = y_train
    y_test_mod = y_test

    # Initialize model
    model = models_dict[custom_model_name]

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

    # Hyperparameter tuning
    model = GridSearchCV(model, param_grid=hyperparameters[custom_model_name], cv=cv, scoring=make_scorer(roc_auc_score, needs_proba=True), n_jobs=-1)

    model.fit(X_train_mod, y_train_mod)

    cv1 = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    val_predictions = cross_val_predict(model, X_train_mod, y_train, method="predict_proba", cv=cv1, n_jobs=-1)

    y_pred_train_label = model.predict(X_train_mod)
    y_pred_val_label = np.round(val_predictions[:, 1])
    y_pred_test_label = model.predict(X_test_mod)

    # Fit the model
    model.fit(X_train_mod, y_train_mod)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    val_predictions = cross_val_predict(model, X_train_mod, y_train, method="predict_proba", cv=cv, n_jobs=-1)

    # Predict the probabilities
    y_pred_train_proba = model.predict_proba(X_train_mod)[:, 1]
    y_pred_val_proba = val_predictions[:,1]
    y_pred_test_proba = model.predict_proba(X_test_mod)[:, 1]

    # Calculate the ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
    fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)

    # Get ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train_mod, y_pred_train_proba)
    fpr_val, tpr_val, _ = roc_curve(y_val_mod, y_pred_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test_mod, y_pred_test_proba)

    auc_train = compute_auc_ci(y_train_mod, y_pred_train_proba)
    auc_val = compute_auc_ci(y_val_mod, y_pred_val_proba)
    auc_test = compute_auc_ci(y_test_mod, y_pred_test_proba)

    youden_index = sensivity_specifity_cutoff(y_val_mod, y_pred_val_proba)

    model_predictions = {}

    model_predictions = {
    "model_name":custom_model_name,
    "feature_set_name":custom_feature_set,
    "train_truth":[y_train_mod],
    "val_truth":[y_val_mod],
    "test_truth":[y_test_mod],
    "train_proba":[y_pred_train_proba],
    "val_proba":[y_pred_val_proba],
    "test_proba":[y_pred_test_proba],
    "train_auc":[auc_train],
    "val_auc":[auc_val],
    "test_auc":[auc_test],
    "train_fpr":[fpr_train],
    "train_tpr":[tpr_train],
    "val_fpr":[fpr_val],
    "val_tpr":[tpr_val],
    "test_fpr":[fpr_test],
    "test_tpr":[tpr_test],
    "youden_index":[youden_index],
    "y_pred_train_label":[y_pred_train_label],
    "y_pred_val_label":[y_pred_val_label],
    "y_pred_test_label":[y_pred_test_label],
    "new_model_predictor":model
    }

    return model_predictions
