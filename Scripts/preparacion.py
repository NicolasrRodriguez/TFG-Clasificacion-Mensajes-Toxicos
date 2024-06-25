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

import nltk  

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

dataframe = pd.read_csv('Data/Labeled Dota 2 Player Messages Dataset.csv')

for mensaje in dataframe['text']:
    word_tokens = word_tokenize(mensaje)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    print(word_tokens)
    print(filtered_sentence)
