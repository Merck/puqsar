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


#!/usr/bin/env python

# CUDA_VISIBLE_DEVICES=7 python run_DNN_multitask.py

# Prerequisites
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import dill
sys.path.append("../puqsar")

# Create folder to save results
saveFolder = '../results/DNN-multitask'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

# Load Example Data
dat_train = pd.read_csv("../data/3A4_processed/dat_train.csv")
dat_test= pd.read_csv("../data/3A4_processed/dat_test_rs.csv")

# Hyperparameters for DNN-multitask outputs
n_out = 50
p_missing = 0.6

# Preparing Data for DNN-multitask model training and testing
from utils.preprocessing import *
X_train, X_cal, y_train_norm, y_cal_norm, mu_tr, sd_tr,min_value,df_label_train, df_label_cal = preprocessing_DNN_multitask_train(dat_train, p_cal = 0.2, n_out=50,p_missing=0.6,seed=99)
X_test, df_label_test = preprocessing_DNN_default_test(dat_test, mu_tr, sd_tr)

# Load functions for DNN-multitask model, and specify Hyperparameters
from models.DNN_multitask import *

# Hyperparameters for DNN structure and training
p_batchSize = 0.05
learn_rate = 0.001

nn_pars = {'nodes' : [4000, 2000, 1000, 1000],
           'dropout': [0.25, 0.25, 0.25, 0.1],
           'batch_size' : min(128,round(X_train.shape[0]*p_batchSize)),
           'learn_rate' : learn_rate,
           'epochs' : 500,
           'n_out': n_out,
           'min_value': min_value,
           'wt_decay' : 0.00005}

# Train a DNN-multitask model
model = train_DNN_multitask(X_train, y_train_norm, X_cal, y_cal_norm, nn_pars)

# Prediction on Calibration set
pred_cal_mat = model.predict(X_cal)
pred_cal_avg = np.mean(pred_cal_mat, 1) * sd_tr + mu_tr
pred_cal_sd = np.std(pred_cal_mat, 1) * sd_tr
df_pred_cal = pd.concat([df_label_cal.reset_index(drop=True, inplace=False),
                         pd.DataFrame({"Pred": pred_cal_avg, "Pred_UNC": pred_cal_sd})], axis=1, ignore_index = False, sort = False)

# Calibration step
#  Conformal algorithm options: CP_ACE, CP_expSD, CP_homo
from calibrators.ICP import *
nominal_level=0.8
fun_PI = CP_ACE(df_pred_cal, nominal_level)

# Save the model to .h5, prediction for calibration set (including raw unertainty score) to .csv, and the calibration results to .pkl file
model_path = os.path.join(saveFolder, 'model.h5')
model.save(model_path)
df_pred_cal.to_csv(os.path.join(saveFolder,"df_pred_cal.csv"), header=True, index=False)
with open(os.path.join(saveFolder, 'calibration.pkl'), 'wb') as file:
    dill.dump([fun_PI,nominal_level,mu_tr,sd_tr], file)

# Application on Test set and save results as CSV
"""
from tensorflow.keras.models import load_model
model = load_model(os.path.join(saveFolder, 'model.h5'))

with open(os.path.join(saveFolder, 'calibration.pkl'), 'rb') as file:
    fun_PI,nominal_level,mu_tr,sd_tr = dill.load(file)
"""

pred_test_mat = model.predict(X_test)
pred_test_avg = np.mean(pred_test_mat, 1) * sd_tr + mu_tr
pred_test_sd = np.std(pred_test_mat, 1) * sd_tr
df_pred_test = pd.concat([df_label_test.reset_index(drop=True, inplace=False),
                         pd.DataFrame({"Pred": pred_test_avg,"Pred_UNC": pred_test_sd})], axis=1, ignore_index = False, sort = False)
df_pred_test = fun_PI(df_pred_test)
df_pred_test.to_csv(os.path.join(saveFolder,"df_pred_test.csv"), header=True, index=False)
