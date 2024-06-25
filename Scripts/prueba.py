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



print("TensorFlow version:", tf.__version__)
print("TensorFlow_hub version:", hub.__version__)
print("TensorFlow Text version:", text.__version__)

tf.get_logger().setLevel('ERROR')

bert_preprocess_model = hub.KerasLayer(bm.tfhub_handle_preprocess)

text_test = ['still faster than ur dads cock']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

bert_model = hub.KerasLayer(bm.tfhub_handle_encoder)

bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {bm.tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')



def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(bm.tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(bm.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

#falta optimizador

epochs = 1

classifier_model.compile(
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {bm.tfhub_handle_encoder}')
history = classifier_model.fit(x=cd.train_dataset,
                               validation_data=cd.val_dataset,
                               epochs=epochs)

loss, accuracy = classifier_model.evaluate(cd.test_dataset)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

