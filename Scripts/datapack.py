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
"""
tf.get_logger().setLevel('ERROR')

bert_preprocess_model = hub.load(bm.tfhub_handle_preprocess)


text_test = ['still faster than ur dads cock']
tok = bert_preprocess_model.tokenize(tf.constant(text_test))

print(tok)

text_preprocessed = bert_preprocess_model.bert_pack_inputs([tok, tok], tf.constant(35))

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}') #128 tokens
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :19]}')#id de las palabras
print(f'Input Mask : {text_preprocessed["input_mask"][0, :19]}')#palabras enmascaradas
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :19]}')#todo a cero por que se usan frases separadas


bert_model = hub.KerasLayer(bm.tfhub_handle_encoder)

bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {bm.tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')
"""
tf.get_logger().setLevel('ERROR')

dataframe = pd.read_csv('Data/Labeled Dota 2 Player Messages Dataset.csv')

mensajes = dataframe['text'].to_numpy().tolist()


bert_preprocess_model = hub.load(bm.tfhub_handle_preprocess)

bert_model = hub.KerasLayer(bm.tfhub_handle_encoder)

tokens = bert_preprocess_model.tokenize(tf.constant(mensajes))
preprocessed = bert_preprocess_model.bert_pack_inputs([tokens, tokens], tf.constant(34)) 

print(f'Shape      : {preprocessed["input_word_ids"].shape}')

bert_results = bert_model(preprocessed)

print(f'Loaded BERT: {bm.tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

"""
def preprocesado(mensaje):
    tokens = bert_preprocess_model.tokenize(tf.constant(mensaje))
    preprocessed = bert_preprocess_model.bert_pack_inputs([tokens, tokens], tf.constant(34)) 
    return preprocessed

mensajes_preprocesados = preprocesado(dataframe['text'].to_list())
"""