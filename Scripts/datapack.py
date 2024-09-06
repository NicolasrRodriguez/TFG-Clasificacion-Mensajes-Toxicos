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

print("TensorFlow version:", tf.__version__)
print("TensorFlow_hub version:", hub.__version__)
print("TensorFlow Text version:", text.__version__)


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

pooled_outputs = bert_results['pooled_output'].numpy()
sequence_outputs = bert_results['sequence_output'].numpy()

#np.save('pooled_outputs.npy', pooled_outputs)
#np.save('sequence_outputs.npy', sequence_outputs)

pooled_outputs_df = pd.DataFrame(pooled_outputs)
pooled_outputs_df['class'] = dataframe['cls'].values
pooled_outputs_df.to_csv('pooled_outputs.csv', index=False)



