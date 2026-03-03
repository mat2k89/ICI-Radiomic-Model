import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import median_survival_times
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter
from sklearn.utils import resample

def km_curve_at_risk(event, duration, cluster=None, title=None, ax=None):

    """
    Kaplan-Meier curve with at risk counts for each cluster and median survival times.

    Parameters
    ----------
    event : array-like
        Event column.
    duration : array-like
        Duration column.
    cluster : array-like
        Cluster column.
    title : str
        Title of the plot.
    ax : matplotlib.pyplot
        Axis of the plot.

    Returns
    -------
    Kaplan-Meier curve with at risk counts for each cluster and median survival
    
    """

    # Survival Curve - OS from external validation set
    if ax:
        ax = ax
    else:
        plt.subplot(111)

    plt.title(title)

    if cluster is not None and not cluster.empty:

        risk_count_label = []

        for g in np.sort(cluster.unique()):
            h = str(g)
            ix = cluster == g
            h = KaplanMeierFitter()
            h.fit(duration[ix], event[ix], label=g)
            h.plot(ax=ax, ci_show=False)
            risk_count_label.append(h)

        # add at risk counts for each cluster on the same graph

        cluster_number = len(cluster.unique())

        if cluster_number==4:
            add_at_risk_counts(risk_count_label[0], risk_count_label[1], risk_count_label[2], risk_count_label[3], ax=ax)
        elif cluster_number==3:
            add_at_risk_counts(risk_count_label[0], risk_count_label[1], risk_count_label[2], ax=ax)
        elif cluster_number==2:
            add_at_risk_counts(risk_count_label[0], risk_count_label[1], ax=ax)
        elif cluster_number==1:
            add_at_risk_counts(risk_count_label[0], ax=ax)

        # Median Survival

        from lifelines.utils import median_survival_times

        a=[]
        b=[]
        c=np.sort(cluster.unique())

        for g in np.sort(cluster.unique()):
            ix=cluster==g
            km=KaplanMeierFitter()
            km.fit(duration[ix], event[ix], label=g)
            a.append(km.median_survival_time_)
            b.append(median_survival_times(km.confidence_interval_))

        a_array=np.array(a).reshape(len(cluster.unique()),)
        b_array=np.array(b).reshape(len(cluster.unique()),2)
        c_array=np.array(c).reshape(len(cluster.unique()),)
        d_array=b_array[:,0].reshape(len(cluster.unique()),)
        e_array=b_array[:,1].reshape(len(cluster.unique()),)

        median_array=np.stack([c_array,a_array,d_array,e_array], axis=1)
        median_array_df = pd.DataFrame(data=np.round(median_array,1), columns=["Cluster","Median","Lower CI", "Upper CI"], index=None)

        print(pd.DataFrame(data=np.round(median_array,1), columns=["Cluster","Median OS","Lower CI", "Upper CI"]))

        ax.text(0, -0.6, str(median_array_df), fontsize=12, va="top", ha='left')

        from lifelines.statistics import logrank_test

        for i in np.sort(cluster.unique()):
            for j in np.sort(cluster.unique()):
                if i<j:
                    result=logrank_test(event[cluster==i], event[cluster==j],
                                        duration[cluster==i], duration[cluster==j])
                    print("\nCluster", i, "vs Cluster", j)
                    print(result.summary)    

        # logrank multiple

        from lifelines.statistics import multivariate_logrank_test

        result=multivariate_logrank_test(duration, cluster, event)
        print("\nMultiple log rank")
        print(result.summary)

        p_value = result.p_value
        if p_value < 0.001:
            ax.text(np.max(duration), -0.6, f"p value<0.001", fontsize=12, va="top", ha='right')
        else:
            ax.text(np.max(duration), -0.6, f"p value= {np.round(result.p_value,3)}", fontsize=12, va="top", ha='right')

    else:

        a=[]
        b=[]
        c=1

        h = KaplanMeierFitter()
        h.fit(duration, event)
        h.plot(ax=ax, ci_show=False)
        a.append(h.median_survival_time_)
        b.append(median_survival_times(h.confidence_interval_))

        a_array=np.array(a).reshape(1,)
        b_array=np.array(b).reshape(1,2)
        c_array=np.array(c).reshape(1,)
        d_array=b_array[:,0].reshape(1,)
        e_array=b_array[:,1].reshape(1,)

        add_at_risk_counts(h, ax=ax)

        median_array=np.stack([c_array,a_array,d_array,e_array], axis=1)
        median_array_df = pd.DataFrame(data=np.round(median_array,1), columns=["Cluster","Median","Lower CI", "Upper CI"], index=None)

        print(pd.DataFrame(data=np.round(median_array,1), columns=["Cluster","Median OS","Lower CI", "Upper CI"]))

        ax.text(0, -0.4, str(median_array_df), fontsize=12, va="top", ha='left')
  

def cox_univariate(df, features, event, duration):

    """
    Univariate Cox regression analysis with p-values and hazard ratios and concordance index

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the features and target variable.
    features : list
        List of features to be used in the analysis.
    event : str
        Event column.
    duration : str
        Duration column.

    Returns
    -------
    Dataframe with p-values, hazard ratios and concordance index for each feature.

    """

    cox_values = []
    variables = []
    cox_summary_columns = []
    ci = []

    for i in features:
        try:
            cph = CoxPHFitter()
            cph.fit(df[[duration,event,i]], duration_col=duration, event_col=event)
            cox_summary_columns = cph.summary.columns
            cox_values.append(np.array(cph.summary)[0])
            variables.append(i)
            ci.append(cph.concordance_index_)
        except:
            variables.append(i)
            cox_values.append(["NA","NA","NA","NA","NA"])
            ci.append("NA")
            print(i + " failed")
            continue

    cox_values = pd.DataFrame(cox_values, index=variables, columns=cox_summary_columns)

    cox_values["concordance index"] = ci

    cox_df = pd.DataFrame(np.round(cox_values[["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","concordance index"]],2), index=cox_values.index, columns=["exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","concordance index"])

    cox_df["p"] = np.round(cox_values[["p"]],4)

    return(cox_df)



def compute_concord_ci(ml_probabilities, duration, event, n_bootstraps=1000, alpha=0.05, random_state=42):
    
    c_index = float(cox_univariate(pd.DataFrame({"Model_probabilities": np.array(ml_probabilities), "duration": np.array(duration), "event": np.array(event)}), ["Model_probabilities"], "event", "duration")["concordance index"])


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
    
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        resampled_indices = resample(range(len(duration)), random_state=random_state+_)
        duration_resampled = np.array(duration[resampled_indices])
        event_resampled = np.array(event[resampled_indices])
        ml_probabilities_resampled = np.array(ml_probabilities[resampled_indices])
        c_index_score = float(cox_univariate(pd.DataFrame({"Model_probabilities": ml_probabilities_resampled, "duration": duration_resampled, "event": event_resampled}), ["Model_probabilities"], "event", "duration")["concordance index"])
        
        bootstrapped_scores.append(c_index_score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Compute the lower and upper bound of the confidence interval
    confidence_lower = sorted_scores[int((alpha / 2.0) * n_bootstraps)]
    confidence_upper = sorted_scores[int((1 - alpha / 2.0) * n_bootstraps)]
    
    return c_index, confidence_lower, confidence_upper
