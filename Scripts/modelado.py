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

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
"""
  preprocessing_layer = hub.KerasLayer(bm.tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(bm.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  net = outputs['pooled_output']
"""

data = pd.read_csv("Data/finaldpck.csv")

def build_classifier_model():
  input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')
  net = tf.keras.layers.Dropout(0.1)(input)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(input, net)


classifier_model = build_classifier_model()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()


"""
steps_per_epoch = tf.data.experimental.cardinality(conjuntosEVT.train_f).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5

optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
"""
classifier_model.compile(
                         loss=loss,
                         metrics=metrics)


"""
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
"""
print(f'Training model with {bm.tfhub_handle_encoder}')#entrenamiento simple
history = classifier_model.fit(x=conjuntosEVT.train_f, y= conjuntosEVT.train_c,
                               epochs=epochs)

loss, accuracy = classifier_model.evaluate(x = conjuntosEVT.test_f , y = conjuntosEVT.test_c)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
"""

epochs = 50 

#Carga de los datos
data_f = data.drop(columns=['class'])
data_c = data['class']

#Entrenamiento:

#CV(externa)
cv_ext = StratifiedKFold(n_splits=5, shuffle=True, random_state= 42)

#CV(interna)
cv_int = StratifiedKFold(n_splits=5, shuffle=True, random_state= 42)

splits = cv_ext.split(data_f,data_c)

#Bucle_externo
for index, (train_val_index, test_index) in enumerate(splits):
   
  #Entrenamiento_validación y test
  x_train_val_fold, x_test_fold = data_f.iloc[train_val_index], data_f.iloc[test_index]

  y_train_val_fold, y_test_fold = data_c.iloc[train_val_index], data_c.iloc[test_index]

  splits_ = cv_int.split(x_train_val_fold,y_train_val_fold)

  modelos = []

  max_accuracy = -1

  #Bucle_interno
  for index_, (train_index, val_index) in enumerate(splits_):#parametros: epochs

    x_train_fold, x_val_fold = data_f.iloc[train_index], data_f.iloc[val_index]

    y_train_fold, y_val_fold = data_c.iloc[train_index], data_c.iloc[val_index]

      
    history = classifier_model.fit(x = x_train_fold, y = y_train_fold , validation_data=(x_val_fold,y_val_fold),
                                  epochs=epochs) 
    
    #if( history > )
    
    #best_model = 
    
    modelos.append(history)

  #best_model = max(modelos, key=lambda x: (x["accuracy"]))

  loss, accuracy = classifier_model.evaluate(x= x_test_fold , y = y_test_fold )
  print(f'\tLoss: {loss}')
  print(f'\tAccuracy: {accuracy}')

