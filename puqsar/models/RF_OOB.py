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


# Companion functions for using out-of-bag predictions in Random Forest model

import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _get_n_samples_bootstrap, _generate_unsampled_indices

def rf_oob_preds(rf_model, X, returnTrees=False):
    n_samples = X.shape[0]
    oob_predictions = np.zeros((n_samples, rf_model.n_estimators))
    oob_mask = np.zeros((n_samples,rf_model.n_estimators), dtype = np.int64)
    n_predictions = np.zeros((n_samples,), dtype = np.int64)
    n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, rf_model.max_samples)
    for k, estimator in enumerate(rf_model.estimators_):
        unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap)
        p_estimator = estimator.predict(X[unsampled_indices, :], check_input=False)
        oob_predictions[unsampled_indices, k] = p_estimator
        oob_mask[unsampled_indices, k] = 1
        n_predictions[unsampled_indices] += 1
    assert((np.sum(oob_mask, axis = 1)==n_predictions).all())
    if (n_predictions == 0).any():
        warn("Some inputs do not have OOB scores. ")
        temp = np.copy(n_predictions)
        temp[temp == 0] = 1
        pred_oob_avg = np.sum(oob_predictions, axis = 1)/temp
    else:
        pred_oob_avg = np.sum(oob_predictions, axis = 1)/n_predictions
    assert(np.max(np.abs(pred_oob_avg - rf_model.oob_prediction_)) < 10**(-6))
    pred_oob_sd = np.zeros(pred_oob_avg.shape)
    for i in range(oob_predictions.shape[0]):
        if (n_predictions[i] > 1):
            pred_oob_sd[i] = np.std(oob_predictions[i,oob_mask[i,:]>0])
        else:
            pred_oob_sd[i] = 0.0
    if returnTrees:
        return (pred_oob_avg, pred_oob_sd, oob_predictions, oob_mask, n_predictions)
    else:
        return (pred_oob_avg, pred_oob_sd)

def rf_test_preds(rf_model, X_test, returnTrees=False):
    pred_test_trees = np.zeros((X_test.shape[0],rf_model.n_estimators),dtype = np.float32)
    for k, tree in enumerate(rf_model.estimators_):
        pred_test_trees[:,k] = tree.predict(X_test)
    pred_test_avg = np.mean(pred_test_trees, axis = 1)
    pred_test_sd = np.std(pred_test_trees, axis = 1)
    if returnTrees:
        return (pred_test_avg, pred_test_sd, pred_test_trees)
    else:
        return (pred_test_avg, pred_test_sd)
