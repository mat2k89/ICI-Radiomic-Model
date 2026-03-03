import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SequentialFeatureSelector 
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder



def remove_correlated_features(df, feature_set, threshold):

    """
    Remove highly correlated features from the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to remove correlated features from.

    feature_set : list
        List of features to remove correlated features from.
    
    threshold : float
        Threshold to remove correlated features.

    Returns:
        df : pandas.DataFrame
            Dataframe with correlated features removed.
    """

    corr_matrix = pd.DataFrame(data = np.triu(df[feature_set].corr().abs(), k=1), columns = feature_set, index = feature_set)

    filtered_train_features = []

    for i in corr_matrix:
        if corr_matrix.loc[i].max() < threshold:
            filtered_train_features.append(i)

    df_filtered = df[filtered_train_features].join(df.drop(df[feature_set], axis=1))

    print("Remaining radiomic features: " + str(len(filtered_train_features)))

    return df_filtered


def standardise_features(training_set, test_set, feature_set):

    """
    Standardise features in the dataframe.

    Parameters
    ----------
    training_set : pandas.DataFrame
        Training set to fit and standardise features.

    test_set : pandas.DataFrame
        Test set to standardise features based on training set.

    feature_set : list
        List of features to standardise.

    Returns:
        tuple : tuple
        (X_train_standard, X_test_standard)
            Tupble of X_train, X_test dataframes with standardised features.
    """

    scaler = StandardScaler()

    scaler.fit(training_set[feature_set])

    X_train_standard=pd.DataFrame(scaler.transform(training_set[feature_set]), index=training_set.index, columns=feature_set)

    X_test_standard=pd.DataFrame(scaler.transform(test_set[feature_set]), index=test_set.index, columns=feature_set)

    X_train_standard = X_train_standard.join(training_set.drop(training_set[feature_set], axis=1))
    X_test_standard = X_test_standard.join(test_set.drop(test_set[feature_set], axis=1))

    return (X_train_standard, X_test_standard)


def feature_selector(X_train, y_train, random_state=42):

    """
    Select features based on different dimensionality reduction techniques.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training set to select features from.

    y_train : pandas.Series
        Target variable to select features from.

    random_state : int
        Random state for reproducibility.

    Returns:
        selected_features : dict
            Dictionary of selected features based on different dimensionality reduction techniques.
    """
    
    # turn off convergence warning

    warnings.filterwarnings('ignore')

    # Dimensionality Reduction Techniques

    # No feature reduction
    # PCA
    # LASSO
    # Elastic Net
    # Recursive Feature Elimination
    # Boruta
    # Mutual Information
    # Pearson
    # Spearman
    # Kendall

    # Define dimensionality reduction techniques

    pca = PCA(random_state=random_state)
    lasso = Lasso(random_state=random_state)
    enet = ElasticNet(random_state=random_state)
    boruta = BorutaPy(estimator=RandomForestClassifier(random_state=random_state), n_estimators='auto', verbose=2)
    mi = mutual_info_classif(X_train, y_train, random_state=random_state)

    # Feature Selection : LASSO Regression

    # LASSO regression - is L1 regularization
    # LASSO regression can be used for feature selection - need to optimise the alpha parameter (penalty term)
    # Use grid search to optimise alpha parameter

    np.random.seed(123)

    # Define the model

    model = LogisticRegression(penalty='l1', solver='saga', random_state=random_state)


    # Define the grid of values for alpha

    param_grid = {'C': np.logspace(-4, 4, 50)}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Fit best paramter to model

    grid_search.best_params_

    model = grid_search.best_estimator_

    # Fit model to training data
    model.predict(X_train)

    # Highlight the features with non-zero coefficients

    lasso_features = X_train.columns[model.coef_[0] !=0]

    print("\nLASSO Features")
    print(f"Best param: {grid_search.best_params_}")
    print(len(lasso_features))
    print(lasso_features)

    # Feature Selection : Elastic Net Regression

    # Elastic Net regression - is between L1/L2 regularization
    # Elastic Net regression can be used for feature selection - need to optimise the alpha parameter (penalty term) and the L1 ratio
    # Use grid search to optimise alpha parameter and L1 ratio

    np.random.seed(123)

    # Define the model

    model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=random_state)


    # Define the grid of values for alpha

    param_grid = {'C': np.logspace(-4, 4, 50), 'l1_ratio': np.linspace(0, 1, 10)}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Fit best paramter to model

    grid_search.best_params_

    model = grid_search.best_estimator_

    # Fit model to training data
    model.predict(X_train)

    # Highlight the features with non-zero coefficients

    elasticnet_features = X_train.columns[model.coef_[0] !=0]

    print("\nElastic Net Features")
    print(f"Best param: {grid_search.best_params_}")
    print(len(elasticnet_features))
    print(elasticnet_features)

    # Feature Selection: Recursive Feature Elimination

    # Recursive Feature Elimination - is a wrapper method

    np.random.seed(123)

    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    # Define the model
    model = RandomForestClassifier()

    # Perform Recursive Feature Elimination
    selector = RFECV(model, step=1, cv=10)
    selector = selector.fit(X_train, y_encoded)

    # Get the names of the selected features
    RFE_selected_features = np.array(X_train.columns)[selector.support_]

    print("\nRFE Features")
    print(selector.support_)
    print(selector.ranking_)
    print(len(RFE_selected_features))
    print(RFE_selected_features)


    # Feature Selection: Principle Component Analysis

    # This determines the principle components that explain the most variance in the data

    # Perform PCA
    pca = PCA(random_state=42)
    pca.fit(X_train)

    # Print explained variance ratio
    var_explained = pca.explained_variance_ratio_
    cum_var_explained = np.cumsum(var_explained)

    print("\nPCA Features")
    print("Variance explained by each component:")
    print(var_explained)

    print("Cumulative variance explained:")
    print(cum_var_explained)

    # Scree plot
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(var_explained) + 1), var_explained, 'o-')
    plt.title('Scree plot')
    plt.xlabel('Number of components')
    plt.ylabel('Proportional variance explained')
    plt.show()

    # Get the names of the selected features
    PCA_selected_features = np.array(X_train.columns)[pca.components_[0] != 0]

    print("\nPCA Features")
    print(len(PCA_selected_features))
    print(PCA_selected_features)

    # Boruta

    # Define the model
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state)

    # Define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=random_state)

    # Find all relevant features
    feat_selector.fit(X_train.values, y_train.values)

    # Check selected features
    Boruta_selected_features = X_train.columns[feat_selector.support_].tolist()

    print("\nBoruta Features")
    print(len(Boruta_selected_features))
    print(Boruta_selected_features)


    # Feature Selection: Random Forest Feature Importance

    # Random Forest Feature Importance is an embedded method

    np.random.seed(123)

    # Define the model
    model = RandomForestClassifier(n_estimators=1000, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    print("\nRandom Forest Features")

    # Plot feature importances
    plt.figure(figsize=(10, 7))
    plt.barh(X_train.columns, importances)
    plt.xlabel('Importance')
    plt.title('Feature importances from Random Forest')
    plt.show()

    # Get the names of the selected features
    RF_selected_features = X_train.columns[importances > 0]

    print("\nRandom Forest Features")
    print(len(RF_selected_features))
    print(RF_selected_features)


    # Feature Selection: Mutual Information

    # Mutual information is a filter method


    # Select the 10 best features based on Mutual Information
    if len(X_train.columns) < 10:
        k = len(X_train.columns)
    else:
        k = 10
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_train, y_train)

    # Get the names of the selected features
    MIM_selected_features = X_train.columns[selector.get_support()]

    print("\nMutual Information Features")
    print(len(MIM_selected_features))
    print(MIM_selected_features)


    # Feature Selection: Pearson Correlation

    # Pearson Correlation is a filter method

    # Calculate absolute Pearson correlation with the outcome
    pearson_corr = []

    for i in range(0,len(X_train.columns)):
        corr_coef, p_value = pearsonr(np.array(X_train.iloc[:,i]),np.array(y_train))
        pearson_corr.append(abs(corr_coef))

    pearson_df = pd.DataFrame(pearson_corr,index=X_train.columns,columns=["Pearson_coef"])

    # Select the top 40% of features
    top_40_percent = int(0.4 * len(pearson_df))
    pearson_selected_features = pearson_df.nlargest(top_40_percent,"Pearson_coef").index

    print("\nPearson Features")
    print(len(pearson_selected_features))
    print(pearson_selected_features)

    # Feature Selection: Spearman Correlation

    # Spearman Correlation is a filter method

    # Calculate absolute Spearman correlation with the outcome
    spearman_corr = []

    for i in range(0,len(X_train.columns)):
        corr_coef, p_value = spearmanr(np.array(X_train.iloc[:,i]),np.array(y_train))
        spearman_corr.append(abs(corr_coef))

    spearman_df = pd.DataFrame(spearman_corr,index=X_train.columns,columns=["Spearman_coef"])

    # Select the top 40% of features
    top_40_percent = int(0.4 * len(spearman_df))
    spearman_selected_features = spearman_df.nlargest(top_40_percent,"Spearman_coef").index

    print("\nSpearman Features")
    print(len(spearman_selected_features))
    print(spearman_selected_features)

    # Feature Selection: Kendall Correlation

    # Kendall Correlation is a filter method

    # Calculate absolute Kendall correlation with the outcome
    kendall_corr = []

    for i in range(0,len(X_train.columns)):
        corr_coef, p_value = kendalltau(np.array(X_train.iloc[:,i]),np.array(y_train))
        kendall_corr.append(abs(corr_coef))

    kendall_df = pd.DataFrame(kendall_corr,index=X_train.columns,columns=["Kendall_coef"])

    # Select the top 40% of features
    top_40_percent = int(0.4 * len(kendall_df))
    kendall_selected_features = kendall_df.nlargest(top_40_percent,"Kendall_coef").index

    print("\nKendall Features")
    print(len(kendall_selected_features))
    print(kendall_selected_features)

    # Feature Selection: f_classif

    # f_classif is a filter method

    # Select the 10 best features based on ANOVA F-value
    if len(X_train.columns) < 10:
        k = len(X_train.columns)
    else:
        k = 10
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X_train, y_train)

    # Get the names of the selected features
    fvalue_selected_features = X_train.columns[selector.get_support()]

    print("\nANOVA F-test Features")
    print(len(fvalue_selected_features))
    print(fvalue_selected_features)

    # Feature Selection: Variance Threshold

    # Variance Threshold is a filter method


    # Remove all features with zero variance
    selector = VarianceThreshold()
    selector.fit(X_train)

    # Get the names of the selected features
    variance_selected_features = X_train.columns[selector.get_support()]

    print("\nVariance Threshold Features")
    print(len(variance_selected_features))
    print(variance_selected_features)

    # Feature Selection: Sequential Forward Selection

    # Sequential Forward Selection is a wrapper method
    
    if len(X_train.columns) < 11:
        n_features_to_select = len(X_train.columns)-1
    else:
        n_features_to_select = 10

    sfs = SequentialFeatureSelector(LogisticRegression(random_state=random_state), n_features_to_select=n_features_to_select, direction='forward', scoring='accuracy')

    sfs = sfs.fit(X_train, y_train)

    sfs_selected_features = sfs.get_feature_names_out()

    print("\nSequential Forward Selection Features")
    print(len(sfs_selected_features))
    print(sfs_selected_features)

    # Create a dictonary of the selected features

    feature_dictionary = {"All":X_train.columns, "LASSO": lasso_features, "Elastic Net": elasticnet_features, "RFE": RFE_selected_features, "PCA": PCA_selected_features, "Boruta": Boruta_selected_features, "Mutual Information": MIM_selected_features, "Pearson": pearson_selected_features, "Spearman": spearman_selected_features, "Kendall": kendall_selected_features, "ANOVA F-test": fvalue_selected_features, "Variance Threshold": variance_selected_features, "Sequential Forward Selection": sfs_selected_features}

    selected_features = {}

    for i in feature_dictionary:
        if len(feature_dictionary[i]) > 0:
            selected_features[i] = feature_dictionary[i]

    return selected_features


def feature_table(features_dictionary, output_file_name):
    """
    Create a table of selected features from the feature selection techniques.

    Args:
        features_dictionary (dict): Dictionary of selected features from the feature selection techniques.
        output_file_name (file_path): File path to save the csv file.

    Returns:
        csv : csv file of the selected features.

    """

    feature_table = {}

    for i in features_dictionary.keys():
        feature_list = []
        for j in features_dictionary["All"]:
                present = j in features_dictionary[i]
                feature_list.append(present)
                feature_table[i] = feature_list

    feature_table_df = pd.DataFrame(feature_table, index=features_dictionary["All"])
            
    feature_table_df.to_csv(output_file_name)