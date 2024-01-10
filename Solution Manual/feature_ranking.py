"""
Description: Ranking features based on multiple methods based on Unupervised
             or Supervised modeling of an ordinal or nominal classification.
Author:      Razi Gaskari
Created:     Dec, 2022
"""
from typing import Dict
from xmlrpc.client import boolean

import numpy as np
import pandas as pd
from feature_ranking_methods import (
    method_anova,
    method_chi2,
    method_extratree,
    method_logisreg,
    method_lsf,
    method_mutual,
    method_pca,
    method_rf,
    method_univariable,
    method_xgboost,
)
from sklearn.preprocessing import (  
    LabelEncoder,
    MinMaxScaler,
)

# list of the methods for unsupervised analysis
unsupervised_method = [method_pca, method_lsf]

# list of the methods for supervised analysis
supervised_method = {
    "nominal": [
        method_chi2,
        method_anova,
        method_logisreg,
        method_mutual,
        method_xgboost,
        method_rf,
    ],
    "ordinal": [method_univariable, method_extratree],
}


def feature_selection_unsupervised(df_input: pd.DataFrame) -> pd.DataFrame:
    """Feature ranking analysis uning unsupervide methods.

    Args:
        df_input (pd.DataFrame): Input data for feature ranking analsyis.

    Returns:
        pd.DataFrame: dataframe, Including the method's name and feature raking (descending).
    """
    selection_method = unsupervised_method
    feature_name = list(df_input.columns)
    # normalized data
    input_scale = normalize_df(df_input)

    print(type(input_scale))
    print(type(feature_name))

    ranking: list[Dict[str, float]] = []
    methods: list[str] = []
    for mehod_fc in selection_method:  
        try:
            ranking.append(mehod_fc(input_scale, feature_name))  
            methods.append(mehod_fc.__name__)  
        except:
            pass

    df = pd.DataFrame(ranking)
    df = df.set_axis(methods)
    df = df.round(2)
    return df


def feature_selection_supervised(
    df_input: pd.DataFrame,
    output: np.ndarray,
    method: str = "nominal",
    recursive: boolean = False,
    apply_encoder: boolean = False,
) -> pd.DataFrame:
    """Feature ranking analysis using supervised methods for ordinal or nominal classification.
        Also, give the user option to use the recursive method (if applicable) and encode the output
        category.

    Args:
        df_input (pd.DataFrame): Input data for feature ranking analsyis.
        output (np.ndarray): Ouput data for feature ranking analsyis.
        method (str, optional): Nature of the output data. "nominal" or "ordinal". Defaults to "nominal".
        recursive (boolean, optional): Apply recursive techniques for selected methods. Defaults to False.
        apply_encoder (boolean, optional): Apply encoder to output data. Defaults to False.

    Returns:
        pd.DataFrame: Including the method's name and feature raking (descending).
    """
    selection_method = supervised_method[method]
    feature_name = list(df_input.columns)
    # normalized data
    input_scale = normalize_df(df_input)

    if apply_encoder:
        le = LabelEncoder()
        le.fit(output)
        output = le.transform(output)

    ranking: list[Dict[str, float]] = []
    methods: list[str] = []
    for mehod_fc in selection_method:  
        ranking.append(mehod_fc(input_scale, output, feature_name, recursive))  
        methods.append(mehod_fc.__name__)  

    df = pd.DataFrame(ranking)
    df = df.set_axis(methods)
    df = df.round(2)
    return df


# ---------------------------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """normalize dataframe

    Args:
        df (pd.DataFrame): input data

    Returns:
        pd.DataFrame: normalized input
    """
    scale = MinMaxScaler()
    df_scale = scale.fit_transform(df)
    return df_scale
