import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd
import numpy as np

#from official.nlp import optimization #addons da un warning

import matplotlib.pyplot as plt
import bert_model as bm
import cargadatos as cd

import conjuntosEVT
"""
  preprocessing_layer = hub.KerasLayer(bm.tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(bm.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  net = outputs['pooled_output']
"""



def build_classifier_model():
  input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')
  net = tf.keras.layers.Dropout(0.1)(input)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(input, net)


classifier_model = build_classifier_model()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

epochs = 50 #mirarlo bien

folds = 5

classifier_model.compile(
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {bm.tfhub_handle_encoder}')

for fold in range(folds):
    path = f"Data/CV/Fold-{fold+1}/test.csv"
    test = pd.read_csv(path)
    test_f = test.drop(columns=['class']).to_numpy()
    test_c = test['class'].to_numpy()
    for fold_ in range(folds):
        train_path = f"Data/CV/Fold-{fold+1}/Train-Val/Fold-{fold_+1}/train.csv"
        val_path = f"Data/CV/Fold-{fold+1}/Train-Val/Fold-{fold_+1}/val.csv"
        train = pd.read_csv(train_path)
        train_f = train.drop(columns=['class']).to_numpy()
        train_c = train['class'].to_numpy()
        val = pd.read_csv(val_path)
        val_f = val.drop(columns=['class']).to_numpy()
        val_c = val['class'].to_numpy()
        history = classifier_model.fit(x = train_f, y = train_c , validation_data=(val_f,val_c),
                                epochs=epochs)
    loss, accuracy = classifier_model.evaluate(x= test_f , y = test_c )
    print(f"Fold {fold + 1}:")
    print(f'\tLoss: {loss}')
    print(f'\tAccuracy: {accuracy}')
    
"""
print(f'Training model with {bm.tfhub_handle_encoder}')#entrenamiento simple
history = classifier_model.fit(x=conjuntosEVT.train_f, y= conjuntosEVT.train_c,
                               epochs=epochs)

loss, accuracy = classifier_model.evaluate(x = conjuntosEVT.test_f , y = conjuntosEVT.test_c)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
"""