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

# python run_LGB-tail.py

# Prerequisites
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
sys.path.append("../puqsar")

# Create folder to save results
saveFolder = '../results/LGB-tail'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

# Load Example Data
dat_train = pd.read_csv("../data/3A4_processed/dat_train.csv")
dat_test= pd.read_csv("../data/3A4_processed/dat_test_rs.csv")

# Prepare Data for LGB-tail model training and testing
from utils.preprocessing import *
X_train, X_cal, y_train, y_cal, df_label_train, df_label_cal = preprocessing_default_train(dat_train, p_cal = 0.2, seed = 666)
X_test, df_label_test = preprocessing_default_test(dat_test)

from scipy import sparse
import lightgbm as lgb
train_xy = lgb.Dataset(sparse.csr_matrix(X_train), label=y_train)

# Train a LGB model
#import lightgbm as lgb

param = {"num_leaves": 64,
         "objective": "regression",
         "metric": "mse",
         "bagging_freq": 1,
         "bagging_fraction": 0.7,
         "feature_fraction": 0.7,
         "learning_rate": 0.01,
         "num_iterations": 1500,
         "random_state": 1357,
         "boosting_type": 'gbdt',
         }
model = lgb.train(param, train_xy)

# Prediction on Calibration set
from models.LGB_tail import *
pred_cal, pred_cal_sd = lgb_tail_preds(model, X_cal, w=0.2)
df_pred_cal = pd.concat([df_label_cal.reset_index(drop=True, inplace=False),
                         pd.DataFrame({"Pred": pred_cal, "Pred_UNC": pred_cal_sd})], axis=1, ignore_index = False, sort = False)

# Calibration step
#  Conformal algorithm options: CP_ACE, CP_expSD, CP_homo
from calibrators.ICP import *
nominal_level=0.8
fun_PI = CP_ACE(df_pred_cal, nominal_level)

# Save the model to .txt file, the calibration results to .pkl file, and the prediction for calibration set (including raw unertainty score) to .csv
import dill
model.save_model(os.path.join(saveFolder, 'model.txt'))
df_pred_cal.to_csv(os.path.join(saveFolder,"df_pred_cal.csv"), header=True, index=False)
with open(os.path.join(saveFolder, 'calibration.pkl'), 'wb') as file:
    dill.dump([fun_PI,nominal_level], file)

# Application on Test set
"""
model = lgb.Booster(model_file=os.path.join(saveFolder, 'model.txt'))

with open(os.path.join(saveFolder, 'calibration.pkl'), 'rb') as file:
    fun_PI,nominal_level = dill.load(file)
"""

pred_test, pred_test_sd = lgb_tail_preds(model, X_test, w=0.2)
df_pred_test = pd.concat([df_label_test.reset_index(drop=True, inplace=False),
                         pd.DataFrame({"Pred": pred_test,"Pred_UNC": pred_test_sd})], axis=1, ignore_index = False, sort = False)
df_pred_test = fun_PI(df_pred_test)
df_pred_test.to_csv(os.path.join(saveFolder,"df_pred_test.csv"), header=True, index=False)
