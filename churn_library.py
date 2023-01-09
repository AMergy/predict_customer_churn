#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   churn_library.py
@Time    :   2023/01/09 10:53:01
@Author  :   AMergy
@Version :   1.0
@Contact :   anne.mergy@gmail.com
@Desc    :   Library of functions to find customers who are likely to churn
'''
# import libraries
import os
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import dataframe_image as dfi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
# Import constants from file constants.py
import constants

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# pylint: disable=E1101


class CustomerChurn():
    """
    Class to find customers who are likely to churn
    """

    def __init__(self,
                 pth: str):
        """
        Initialization method
        Input:
            pth (str): a path to the csv with input data
        """
        self.df_data = self.import_data(pth)

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
                pth (str): a path to the csv
        output:
                df: pandas dataframe
        '''
        df_ = pd.read_csv(pth)
        df_['Churn'] = df_['Attrition_Flag'].apply(lambda val: 0 if val ==
                                                   "Existing Customer" else 1)
        return df_

    def perform_eda(self, save=True):
        '''
        perform eda on df and save figures to images folder
        input:
                save (boolean) : if True (default), saves figures of EDA

        output:
                None
        '''
        df_eda = self.df_data
        print("Data shape is ", df_eda.shape)
        print(
            "The sums of null numbers in each column are ",
            df_eda.isnull().sum())
        print("\nData statistics: ", df_eda.describe())

        for cat in constants.CAT_COLUMNS:
            plt.figure(figsize=(20, 10))
            df_eda[cat].value_counts('normalize').plot(kind='bar')
            if save:
                plt.savefig(f"images/eda/univ_cat_plot-{cat}.png")
            plt.close()

        for quant in constants.QUANT_COLUMNS:
            plt.figure(figsize=(20, 10))
            df_eda[quant].hist()
            if save:
                plt.savefig(f"images/eda/univ_quant_plot-{quant}.png")
            plt.close()

        if save:
            plt.figure(figsize=(20, 10))
            plt.xlabel("Total_Trans_Amt")
            plt.ylabel("Churn")
            plt.scatter(df_eda["Total_Trans_Amt"], df_eda.Churn)
            plt.savefig("images/eda/bivariate_plot.png")
            plt.close()

    def encoder_helper(self, df_, category_lst, response=None):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the
        notebook

        input:
                df_: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be
                        used for naming variables or index y column]

        output:
                df_: pandas dataframe with new columns for propotion of churn in
                    each category
        '''
        df_ = df_.copy()
        if response:
            try:
                assert len(category_lst) == len(response)
            except ValueError:
                return "response and category_lst should have the same length"
        else:
            response = [cat + "_Churn" for cat in category_lst]
        for i, cat in enumerate(category_lst):
            cat_groups = df_.groupby(cat).mean()['Churn']
            df_[response[i]] = [cat_groups.loc[val] for val in df_[cat]]
        return df_

    def perform_feature_engineering(self, keep_cols, response=None):
        '''
        input:
                keep_cols: list of strings of columns to keep when performing
                            feature engineering
                response: string of response name [optional argument that could be
                            used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''
        df_ = self.encoder_helper(
            self.df_data, constants.CAT_COLUMNS, response)
        xtot = pd.DataFrame()
        xtot[keep_cols] = df_[keep_cols]
        ytot = df_['Churn']
        xtrain, xtest, ytrain, ytest = train_test_split(
            xtot, ytot, test_size=0.3, random_state=42)
        return xtrain, xtest, ytrain, ytest

    def classification_report_image(self, y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder

        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''
        def save_classification_report_image(algo, y_train, y_test,
                                             y_train_preds, y_test_preds):
            """
            transforms a classification report into an image for an algorithm

            input:
                algo (str): name of the algorithm
                y_train: training response values
                y_test:  test response values
                y_train_preds: training predictions from algo
                y_test_preds: test predictions from algo
            """
            plt.rc('figure', figsize=(5, 5))
            # approach improved by OP -> monospace!
            plt.text(0.01, 1.25, str(algo + ' Train'),
                     {'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.05, str(
                    classification_report(
                        y_test, y_test_preds)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.text(0.01, 0.6, str(algo + ' Test'),
                     {'fontsize': 10}, fontproperties='monospace')
            plt.text(
                0.01, 0.7, str(
                    classification_report(
                        y_train, y_train_preds)), {
                    'fontsize': 10}, fontproperties='monospace')
            plt.savefig(
                "images/results/classification_report-" +
                algo +
                ".png")
            plt.axis('off')
        save_classification_report_image("Random Forest", y_train, y_test,
                                         y_train_preds_rf, y_test_preds_rf)
        save_classification_report_image(
            "Logistic Regression",
            y_train,
            y_test,
            y_train_preds_lr,
            y_test_preds_lr)

    def feature_importance_plot(self, importances, x_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                importances: np.array object containing feature_importances_
                x_data: pandas dataframe of x values
                output_pth: path to store the figure

        output:
                None
        '''
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [x_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)

        plt.savefig(output_pth)
        plt.close()

    def train_models(self, x_train, x_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                x_train: x training data
                x_test: x testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        xtot = pd.concat([x_train, x_test])

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        # store model results: images + scores
        lrc_plot = plot_roc_curve(lrc, x_test, y_test)
        plt.figure(figsize=(15, 8))
        ax_ = plt.gca()
        _ = plot_roc_curve(
            cv_rfc.best_estimator_,
            x_test,
            y_test,
            ax=ax_,
            alpha=0.8)
        lrc_plot.plot(ax=ax_, alpha=0.8)
        plt.savefig("images/results/roc_curve.png")
        plt.close()
        self.classification_report_image(y_train,
                                         y_test,
                                         y_train_preds_lr,
                                         y_train_preds_rf,
                                         y_test_preds_lr,
                                         y_test_preds_rf)
        self.feature_importance_plot(
            cv_rfc.best_estimator_.feature_importances_,
            xtot,
            "images/results/feature_importance-rf.png")
        # influence of a parameter in a logistic regression is given by
        # the magnitude of its coefficient times the standard deviation of the
        # corresponding parameter in the data
        self.feature_importance_plot(
            np.abs(
                np.std(
                    xtot.values,
                    0) *
                lrc.coef_[0]),
            xtot,
            "images/results/feature_importance-lr.png")

        # Store models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

    def classification_report_image_as_df(self, y_train,
                                          y_test,
                                          y_train_preds_lr,
                                          y_train_preds_rf,
                                          y_test_preds_lr,
                                          y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder

        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''
        def get_classification_report(y_test, y_pred):
            """
            transforms a classification report into a dataframe

            input:
                y_test: test response values
                y_pred: predictions from algorithm
            output:
                pd.Dataframe: classification report
            """
            report = classification_report(y_test, y_pred, output_dict=True)
            df_classification_report = pd.DataFrame(report).transpose()
            return df_classification_report

        # classification reports from random forest
        dfi.export(get_classification_report(y_test, y_test_preds_rf),
                   "images/results/classification_report_rf_test.png")
        dfi.export(get_classification_report(y_train, y_train_preds_rf),
                   "images/results/classification_report_rf_train.png")

        # classification reports from logistic regression
        dfi.export(get_classification_report(y_test, y_test_preds_lr),
                   "images/results/classification_report_lr_test.png")
        dfi.export(get_classification_report(y_train, y_train_preds_lr),
                   "images/results/classification_report_lr_train.png")


if __name__ == "__main__":
    churn_prediction = CustomerChurn(constants.PATH)
    print(churn_prediction.df_data.head())

    churn_prediction.perform_eda()

    train_x, test_x, train_y, test_y = churn_prediction.perform_feature_engineering(
        constants.KEEP_COLUMNS)

    churn_prediction.train_models(train_x, test_x, train_y, test_y)
