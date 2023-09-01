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


# DNN (Fully Connected Neural Network) model
# Use multitask method to compute the raw prediction uncertainty scores
# Reference: Xu, Y., Liaw, A., Sheridan, R.P. and Svetnik, V., 2023. Development and Evaluation of Conformal Prediction Methods for QSAR. arXiv preprint arXiv:2304.00970.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError
from keras import backend as K

def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.greater_equal(y_true, min_value), K.floatx()) # mask for non-missing elements
    return tf.keras.losses.mean_squared_error(y_true * mask, y_pred * mask)

def train_DNN_multitask(x, y, x_val, y_val, pars):
    global min_value
    min_value = pars['min_value']
    inputs = keras.Input(shape=(x.shape[1], ))
    dense = layers.Dense(pars['nodes'][0],
                         activation="relu",
                         kernel_regularizer=keras.regularizers.l2(pars['wt_decay']))
    prv_layer = dense(inputs)
    prv_layer = layers.Dropout(pars['dropout'][0])(prv_layer)
    for i in range(1, len(pars['nodes'])):
        prv_layer = layers.Dense(pars['nodes'][i],
                                 activation="relu",
                                 kernel_regularizer=keras.regularizers.l2(pars['wt_decay']))(prv_layer)
        prv_layer = layers.Dropout(pars['dropout'][i])(prv_layer)
    outputs = layers.Dense(pars['n_out'])(prv_layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name="DNN_multitask")
    callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    model.compile(
        loss=masked_loss_function,
        optimizer=tf.keras.optimizers.SGD(learning_rate=pars['learn_rate'], momentum=0.9, nesterov=False),
        metrics=[RootMeanSquaredError()]
        )
    model.fit(x, y,
              batch_size=pars['batch_size'],
              epochs=pars['epochs'],
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=[callback_ES]
             )
    return model
