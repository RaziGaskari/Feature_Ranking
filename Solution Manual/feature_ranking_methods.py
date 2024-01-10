"""
Description: Method for feature ranking analsysis
Author:      Razi Gaskari
Created:     Dec, 2022
"""

from typing import Dict

import numpy as np
from linselect import FwdSelect  
from sklearn.decomposition import PCA  
from sklearn.ensemble import (  
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import (  
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression  
from xgboost.sklearn import XGBClassifier


# ---------------------------------------------
def method_rf(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Random Forest Method"""
    model = RandomForestClassifier(n_estimators=20)
    fit = model.fit(input_scale, output)
    score = fit.feature_importances_
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_xgboost(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Gradient Boost Method"""
    model = XGBClassifier()
    fit = model.fit(input_scale, output)
    score = fit.feature_importances_
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_mutual(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Mutual Information Statistic Method"""
    model = SelectKBest(score_func=mutual_info_classif, k=4)
    fit = model.fit(input_scale, output)
    score = fit.scores_
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_extratree(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Extra Tree Mehtod"""
    model = ExtraTreesClassifier()
    # apply recursive method
    if recursive:
        model = RFE(estimator=model, n_features_to_select=3)
        fit = model.fit(input_scale, output)
        score = fit.ranking_
    else:
        fit = model.fit(input_scale, output)
        score = fit.feature_importances_

    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_logisreg(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Logistic Regression Mehtod"""
    model = LogisticRegression()
    # apply recursive method
    if recursive:
        model = RFE(estimator=model, n_features_to_select=1)
        fit = model.fit(input_scale, output)
        score = -1 * fit.ranking_
    else:
        fit = model.fit(input_scale, output)
        score = fit.coef_[0]

    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_chi2(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Chi Square Mehtod"""
    model = SelectKBest(score_func=chi2, k=4)
    fit = model.fit(input_scale, output)
    score = fit.scores_
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_anova(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Anova Mehtod"""
    model = SelectKBest(score_func=f_classif, k=4)
    fit = model.fit(input_scale, output)
    score = fit.scores_
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_univariable(
    input_scale: np.ndarray, output: np.ndarray, feature_names: list[str], recursive: str
) -> Dict[str, float]:

    """Univariate linear regression Mehtod"""
    model = SelectKBest(score_func=f_regression, k=4)
    fit = model.fit(input_scale, output)
    score = fit.scores_
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_pca(
    input_scale: np.ndarray,
    feature_names: list[str],
) -> Dict[str, float]:

    """Principal Component Analysis Mehtod"""
    no_components = min(3, input_scale.shape[1])
    model = PCA(svd_solver="full", n_components=no_components)
    fit = model.fit(input_scale)
    # use the first component
    score = fit.components_[0]
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------
def method_lsf(
    input_scale: np.ndarray,
    feature_names: list[str],
) -> Dict[str, float]:

    """least-squares feature selection Mehtod"""
    model = FwdSelect()
    fit = model.fit(input_scale)
    # use the first component
    score = fit.ordered_features
    # ranking on descending order
    score_normal = normalize_score(score)
    rank_dic = dict(zip(feature_names, score_normal))
    return rank_dic


# ---------------------------------------------


def normalize_score(score: np.ndarray) -> np.ndarray:
    """Normalize score value betweeen 0-1"""
    score_normal = (score - np.min(score)) / (np.max(score) - np.min(score))
    return score_normal
