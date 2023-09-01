#    Copyright © 2023 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.
#
#    This file is part of the PUQSAR package, an open source software for computing the Prediction Uncertainty for QSAR.
#
#    Prediction Uncertainty for QSAR (PUQSAR) is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sys
import os
import time
import math
import numpy as np
import pandas as pd

"""
Prepare data for DNN_dropout or any single-task DNN model:
    Create random splits of Proper Training and Calibration sets to prepare for conformal prediction
    Normalize output labels
    Transform input descriptors
"""
def preprocessing_DNN_default_train(dat_train, p_cal = 0.2, seed = 1234):
    df_label = dat_train[['MOLECULE', 'Act']]
    X = dat_train.drop(labels=['MOLECULE', 'Act'], axis = 1, inplace = False).to_numpy(dtype = "uint8")
    np.random.seed(seed)
    # Random split the training set into (1-p_cal) proper training and p_cal calibration set
    cal_index = np.random.choice(dat_train.shape[0],size = round(p_cal*dat_train.shape[0]), replace = False)
    train_index = np.setdiff1d(range(dat_train.shape[0]), cal_index)
    #
    X_train = X[train_index,]
    X_cal = X[cal_index,]
    df_label_train = df_label.iloc[train_index]
    df_label_cal = df_label.iloc[cal_index]
    # transform input features
    X_train = np.log(X_train + 1.0).astype(np.float32)
    X_cal = np.log(X_cal + 1.0).astype(np.float32)
    #
    y_train = df_label_train['Act'].to_numpy(dtype = np.float64)
    y_cal = df_label_cal['Act'].to_numpy(dtype = np.float64)
    # normalize true label
    mu_tr = y_train.mean()
    sd_tr = y_train.std()
    y_train_norm = (y_train-mu_tr)/sd_tr
    y_cal_norm = (y_cal-mu_tr)/sd_tr
    return (X_train, X_cal, y_train_norm, y_cal_norm, mu_tr, sd_tr, df_label_train, df_label_cal)

def preprocessing_DNN_default_test(dat_test, mu_tr, sd_tr):
    if 'Act' in list(dat_test):
        df_label_test = dat_test[['MOLECULE', 'Act']]
        X_test = dat_test.drop(labels=['MOLECULE', 'Act'], axis = 1, inplace = False).to_numpy(dtype = "uint8")
    else:
        df_label_test = dat_test[['MOLECULE']]
        X_test = dat_test.drop(labels=['MOLECULE'], axis = 1, inplace = False).to_numpy(dtype = "uint8")
    # transform input features
    X_test = np.log(X_test + 1.0).astype(np.float32)
    return X_test, df_label_test

"""
Prepare data for DNN_multitask model:
    Create random splits of Proper Training and Calibration sets to prepare for conformal prediction
    Normalize output labels
    Transform input descriptors
    Create multioutputs training set label with random missing for training
"""
def preprocessing_DNN_multitask_train(dat_train, p_cal = 0.2, n_out=50, p_missing=0.6,seed=1234):
    df_label = dat_train[['MOLECULE', 'Act']]
    X = dat_train.drop(labels=['MOLECULE', 'Act'], axis = 1, inplace = False).to_numpy(dtype = "uint8")
    np.random.seed(seed)
    # Random split the training set into (1-p_cal) proper training and p_cal calibration set
    cal_index = np.random.choice(dat_train.shape[0],size = round(p_cal*dat_train.shape[0]), replace = False)
    train_index = np.setdiff1d(range(dat_train.shape[0]), cal_index)
    #
    X_train = X[train_index,]
    X_cal = X[cal_index,]
    df_label_train = df_label.iloc[train_index]
    df_label_cal = df_label.iloc[cal_index]
    # transform input features
    X_train = np.log(X_train + 1.0).astype(np.float32)
    X_cal = np.log(X_cal + 1.0).astype(np.float32)
    #
    y_train = df_label_train['Act'].to_numpy(dtype = np.float64)
    y_cal = df_label_cal['Act'].to_numpy(dtype = np.float64)
    # normalize true label
    mu_tr = y_train.mean()
    sd_tr = y_train.std()
    y_train_norm = (y_train-mu_tr)/sd_tr
    y_cal_norm = (y_cal-mu_tr)/sd_tr
    # multitask output with missing
    min_value = np.min(y_train_norm)
    mask_value = min_value-1000.0
    y_train_mat = np.tile(y_train_norm,(n_out,1)).transpose()
    y_cal_mat = np.tile(y_cal_norm,(n_out,1)).transpose()
    y_train_mask = np.random.choice([0,1], size = y_train_mat.shape, p=[p_missing,(1-p_missing)])
    y_train_missing = y_train_mat*y_train_mask + mask_value*(1-y_train_mask)
    return (X_train, X_cal, y_train_missing, y_cal_mat, mu_tr, sd_tr, min_value, df_label_train, df_label_cal)

"""
Prepare data for RF_OOB model:
    Since OOB data will be used for calibration, there is no need of splitting or preprocessing for both training and test sets.
"""
def preprocessing_RF(dat, returnAct = False):
    if 'Act' in list(dat):
        df_label = dat[['MOLECULE', 'Act']]
        X = dat.drop(labels=['MOLECULE', 'Act'], axis = 1, inplace = False).to_numpy(dtype = np.float32)
        if returnAct:
            y = df_label['Act'].to_numpy(dtype = np.float64)
            return X, y, df_label
    else:
        df_label = dat[['MOLECULE']]
        X = dat.drop(labels=['MOLECULE'], axis = 1, inplace = False).to_numpy(dtype = np.float32)
    return X, df_label

"""
Prepare data for LGB_tail or other general ML models:
    Create random splits of Proper Training and Calibration sets to prepare for conformal prediction
"""
def preprocessing_default_train(dat_train, p_cal = 0.2, seed = 1234):
    df_label = dat_train[['MOLECULE', 'Act']]
    X = dat_train.drop(labels=['MOLECULE', 'Act'], axis = 1, inplace = False).to_numpy(dtype = "uint8")
    np.random.seed(seed)
    # Random split the training set into (1-p_cal) proper training and p_cal calibration set
    cal_index = np.random.choice(dat_train.shape[0],size = round(p_cal*dat_train.shape[0]), replace = False)
    train_index = np.setdiff1d(range(dat_train.shape[0]), cal_index)
    #
    X_train = X[train_index,].astype(np.float32)
    X_cal = X[cal_index,].astype(np.float32)
    df_label_train = df_label.iloc[train_index]
    df_label_cal = df_label.iloc[cal_index]
    y_train = df_label_train['Act'].to_numpy(dtype = np.float64)
    y_cal = df_label_cal['Act'].to_numpy(dtype = np.float64)
    return (X_train, X_cal, y_train, y_cal, df_label_train, df_label_cal)

def preprocessing_default_test(dat_test):
    if 'Act' in list(dat_test):
        df_label_test = dat_test[['MOLECULE', 'Act']]
        X_test = dat_test.drop(labels=['MOLECULE', 'Act'], axis = 1, inplace = False).to_numpy(dtype = np.float32)
    else:
        df_label_test = dat_test[['MOLECULE']]
        X_test = dat_test.drop(labels=['MOLECULE'], axis = 1, inplace = False).to_numpy(dtype = np.float32)
    return X_test, df_label_test
