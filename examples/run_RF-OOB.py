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

# python run_RF-OOB.py

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
saveFolder = '../results/RF-OOB'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

# Load Example Data
dat_train = pd.read_csv("../data/3A4_processed/dat_train.csv")
dat_test= pd.read_csv("../data/3A4_processed/dat_test_rs.csv")

# Prepare Data for RF_OOB model training and testing
from utils.preprocessing import *
X_train, y_train, df_label_train = preprocessing_RF(dat_train, returnAct = True)
X_test, df_label_test = preprocessing_RF(dat_test, returnAct = False)

# Find number of CPUs
import multiprocessing
nCores = multiprocessing.cpu_count()

# Train a RF model
from sklearn.ensemble import RandomForestRegressor
ntree = 500
model = RandomForestRegressor(n_estimators=ntree, random_state=9876, max_features=0.33, min_samples_leaf=5, oob_score=True, n_jobs=nCores)
model.fit(X_train, y_train)

# Prediction on Calibration set
from models.RF_OOB import *
pred_oob_avg, pred_oob_sd = rf_oob_preds(model, X_train)
df_pred_cal = pd.concat([df_label_train.reset_index(drop=True, inplace=False),
                         pd.DataFrame({"Pred": pred_oob_avg, "Pred_UNC": pred_oob_sd})], axis=1, ignore_index = False, sort = False)

# Calibration step
#  Conformal algorithm options: CP_ACE, CP_expSD, CP_homo
from calibrators.ICP import *
nominal_level=0.8
fun_PI = CP_ACE(df_pred_cal, nominal_level)

# Save the model and the calibration results to .pkl files, and save the prediction for calibration set (including raw unertainty score) to .csv
with open(os.path.join(saveFolder, 'model.pkl'), 'wb') as file:
    dill.dump([model], file)

df_pred_cal.to_csv(os.path.join(saveFolder,"df_pred_cal.csv"), header=True, index=False)

with open(os.path.join(saveFolder, 'calibration.pkl'), 'wb') as file:
    dill.dump([fun_PI,nominal_level], file)

# Application on Test set and save results as CSV
"""
with open(os.path.join(saveFolder, 'model.pkl'), 'rb') as file:
    model = dill.load(file)

with open(os.path.join(saveFolder, 'calibration.pkl'), 'rb') as file:
    fun_PI,nominal_level = dill.load(file)
"""

pred_test_avg, pred_test_sd = rf_test_preds(model, X_test)
df_pred_test = pd.concat([df_label_test.reset_index(drop=True, inplace=False),
                         pd.DataFrame({"Pred": pred_test_avg,"Pred_UNC": pred_test_sd})], axis=1, ignore_index = False, sort = False)
df_pred_test = fun_PI(df_pred_test)
df_pred_test.to_csv(os.path.join(saveFolder,"df_pred_test.csv"), header=True, index=False)
