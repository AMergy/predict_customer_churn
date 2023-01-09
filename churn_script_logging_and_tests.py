#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   churn_script_logging_and_tests.py
@Time    :   2023/01/09 13:35:21
@Author  :   AMergy
@Version :   1.0
@Contact :   anne.mergy@gmail.com
@Desc    :   Testing & logging file for churn_library.py
'''
import logging
import joblib
import matplotlib.pyplot as plt

import churn_library as cls
# Import constants from file constants.py
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(path):
    """
    test data import
    Args:
        path (str): path to bank data
    """
    try:
        churn_prediction = cls.CustomerChurn(path)
        logging.info("Testing CustomerChurn class with import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing CustomerChurn class with import_data: The file wasn't found")
        raise err

    try:
        assert churn_prediction.df_data.shape[0] > 0
        assert churn_prediction.df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing CustomerChurn class with import_data: The file doesn't appear" +
            " to have rows and columns")
        raise err


def test_eda(path):
    """
    test perform eda function
    Args:
        path (str): path to bank data
    """
    try:
        churn_prediction = cls.CustomerChurn(path)
        churn_prediction.perform_eda()
        plt.imread("./images/eda/bivariate_plot.png")
        logging.info("Testing CustomerChurn class with perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing CustomerChurn class with perform_eda: one eda image wasn't found")
        raise err


def test_encoder_helper(path, cat_columns):
    """
    test encoder helper
    Args:
        path (str): path to bank data
        cat_columns: (list) list of str of categorical columns
    """
    try:
        churn_prediction = cls.CustomerChurn(path)
        churn_prediction.df_data[cat_columns]
        logging.info(
            "Testing CustomerChurn class with encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing CustomerChurn class with encoder_helper: cat_columns must " +
            "be columns of the data")
        raise err
    try:
        churn_prediction = cls.CustomerChurn(path)
        df_encoded = churn_prediction.encoder_helper(churn_prediction.df_data,
                                                     cat_columns)
        assert sum([
            cat +
            "_Churn" in df_encoded.columns for cat in cat_columns]) == len(cat_columns)
        assert df_encoded.shape[1] == (churn_prediction.df_data.shape[1] +
                                       len(cat_columns))
        logging.info(
            "Testing CustomerChurn class with encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing CustomerChurn class with encoder_helper: inconsistent number" +
            " of encoded columns")
        raise err


def test_perform_feature_engineering(path, keep_columns):
    """
    test perform_feature_engineering
    Args:
        path (str): path to bank data
        keep_columns: (list) list of str of columns to keep
    """
    try:
        churn_prediction = cls.CustomerChurn(path)
        x_train, x_test, y_train, y_test = churn_prediction.perform_feature_engineering(
            keep_columns)
        assert x_train.shape[1] == len(keep_columns)
        assert y_train.shape[0] == x_train.shape[0]
        assert len(y_train.shape) == 1
        assert x_test.shape[1] == len(keep_columns)
        assert y_test.shape[0] == x_test.shape[0]
        assert len(y_test.shape) == 1
        logging.info(
            "Testing CustomerChurn class with test_perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing CustomerChurn class with test_perform_feature_engineering: " +
            "inconsistent shapes")
        raise err


def test_train_models(path, keep_columns):
    """
    test train_models
    Args:
        path (str): path to bank data
        keep_columns: (list) list of str of columns to keep
    """
    try:
        churn_prediction = cls.CustomerChurn(path)
        x_train, x_test, y_train, y_test = churn_prediction.perform_feature_engineering(
            keep_columns)
        churn_prediction.train_models(x_train, x_test, y_train, y_test)
        joblib.load('./models/rfc_model.pkl')
        joblib.load('./models/logistic_model.pkl')
        logging.info(
            "Testing CustomerChurn class with test_train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing CustomerChurn class with test_train_models: models' weight " +
            "files weren't found")
        raise err


if __name__ == "__main__":
    test_import(constants.PATH)
    test_eda(constants.PATH)
    test_encoder_helper(constants.PATH, constants.CAT_COLUMNS)
    test_perform_feature_engineering(constants.PATH, constants.KEEP_COLUMNS)
    test_train_models(constants.PATH, constants.KEEP_COLUMNS)
