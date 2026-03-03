import numpy as np
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.utils import resample
from scipy.stats import sem, t
from sklearn.metrics import roc_auc_score

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

