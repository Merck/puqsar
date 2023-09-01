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
# Use random dropout during test time to compute the raw prediction uncertainty scores
# Reference: Cortes-Ciriano, I.; Bender, A. Reliable prediction errors for deep neural networks using test-time dropout. Journal of chemical information and modeling 2019, 59, 3330–3339.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError

def train_DNN_dropout(x, y, x_val, y_val, pars):
    inputs = keras.Input(shape=(x.shape[1], ))
    dense = layers.Dense(pars['nodes'][0],
                         activation="relu",
                         kernel_regularizer=keras.regularizers.l2(pars['wt_decay']))
    prv_layer = dense(inputs)
    prv_layer = layers.Dropout(pars['dropout'][0])(prv_layer,training=True)
    for i in range(1, len(pars['nodes'])):
        prv_layer = layers.Dense(pars['nodes'][i],
                                 activation="relu",
                                 kernel_regularizer=keras.regularizers.l2(pars['wt_decay']))(prv_layer)
        prv_layer = layers.Dropout(pars['dropout'][i])(prv_layer,training=True)
    outputs = layers.Dense(1)(prv_layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name="DNN_dropout")
    callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    model.compile(
        loss=keras.losses.MeanSquaredError(),
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
