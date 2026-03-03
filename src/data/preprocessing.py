import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from neuroCombat import neuroCombat

def merger(radiomic_file, clinical_file, radiomic_key, clinical_key):

    """
    Merge radiomic and clinical dataframes based on the keys provided.

    Parameters
    ----------
    radiomic_file : str
        Path to the radiomic file.
    clinical_file : str
        Path to the clinical file.
    radiomic_key : str
        Key to merge the radiomic file.
    clinical_key : str
        Key to merge the clinical file.

    Returns
    -------
    df : pandas.DataFrame
        Merged dataframe.
        Index is set to the radiomic_key.
    """

    radiomic = pd.read_csv(radiomic_file)
    radiomic.set_index(radiomic_key, inplace=True)
    clinical = pd.read_csv(clinical_file)
    clinical.set_index(clinical_key, inplace=True)
    
    df = radiomic.join(clinical, how='inner')
    return df


def mice_imputer(df, feature_set):

    """
    Impute missing values in the dataframe using MICE imputation.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to impute missing values.

    feature_set : list
        List of features to impute missing values.

    Returns:
        df_imputed : pandas.DataFrame
            Dataframe with imputed missing values.
    """

    imputer = IterativeImputer(random_state=42)
    imputer.fit(df[feature_set])
    features_imputed = imputer.transform(df[feature_set])

    df_features_imputed = pd.DataFrame(features_imputed, columns=feature_set, index=df.index)

    df_imputed = df_features_imputed.join(df.drop(df[feature_set], axis=1))

    return df_imputed


def harmoniser(df, feature_set, batch_column, reference_batch=False):

    """
    Harmonise radiomic features based on batch column

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to harmonise.

    feature_set : list
        List of radiomic features to harmonise.
    
    batch_column : str
        Column name of the batch column.

    Returns:
        _type_: _description_
    """

    scan_batch_id = []

    for e,i in enumerate(df[batch_column].unique()):
        scan_batch_id.append(e)

    scan_batch_dict = dict(zip(df[batch_column].unique(),scan_batch_id))

    df["scan_batch_id"] = df[batch_column].map(scan_batch_dict)

    df["scan_batch_id"].value_counts()

    # Combat harmonisation - based on scanner manufacturer

    radiomic_features = df[feature_set]

    covars = {'batch':list(df["scan_batch_id"])}

    covars = pd.DataFrame(covars)

    covars

    batch_col = "batch"

    radiomic_features = np.array(radiomic_features).transpose()

    radiomic_features

    if reference_batch==False:
        harm_radiomic_features = neuroCombat(dat=radiomic_features, covars=covars, batch_col=batch_col)["data"]
    
    else:
        harm_radiomic_features = neuroCombat(dat=radiomic_features, covars=covars, batch_col=batch_col, ref_batch=scan_batch_dict[reference_batch])["data"]

    harm_radiomic_features = pd.DataFrame(harm_radiomic_features.transpose(), columns=feature_set, index=df.index)

    harmonisation_features_df = harm_radiomic_features.join(df.drop(df[feature_set], axis=1))

    return harmonisation_features_df
