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


dataframe = pd.read_csv('Data/pooled_outputs.csv')

dataframe_ = dataframe.drop(columns=['class'])


print(dataframe_.describe())


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
standardized_data = scaler.fit_transform(dataframe_)

print(standardized_data)

standardized_df = pd.DataFrame(standardized_data, columns=dataframe_.columns)

standardized_df['class'] = dataframe['class'].values

print(standardized_df.describe())


standardized_df.to_csv('Data/standar_datapack.csv', index=False)





