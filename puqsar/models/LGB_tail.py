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


# Companion functions for making predictions using LGB-tail method
# Reference: Xu, Y., Liaw, A., Sheridan, R.P. and Svetnik, V., 2023. Development and Evaluation of Conformal Prediction Methods for QSAR. arXiv preprint arXiv:2304.00970.

import numpy as np

def lgb_tail_preds(model, X, w=0.2, returnAll = False):
    pred = model.predict(X)
    nrounds = model.num_trees()
    n_tail = int(nrounds * w / 100)
    if returnAll:
        pred_mat = np.zeros((X.shape[0],nrounds))
        for k in range(nrounds):
            pred_mat[:,k] = model.predict(X,start_iteration=k,num_iteration=1)
        pred_sd = np.mean(abs(pred_mat[:,(nrounds-n_tail):nrounds]), axis = 1)
        return (pred, pred_sd, pred_mat)
    else:
        pred_mat = np.zeros((X.shape[0],n_tail))
        for k in range(n_tail):
            pred_mat[:,k] = model.predict(X,start_iteration=(nrounds-n_tail+k),num_iteration=1)
        pred_sd = np.mean(abs(pred_mat), axis = 1)
        return (pred, pred_sd)
