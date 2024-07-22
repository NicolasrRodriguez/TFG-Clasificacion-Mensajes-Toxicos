from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


dataframe = pd.read_csv('Data/finaldpck.csv')

clase = dataframe['class']

caracteristicas = dataframe.drop(columns=['class'])


train_f, test_f, train_c, test_c   = train_test_split(caracteristicas, clase  , test_size=0.2, random_state=42)

#train_f, val_f, train_c, val_c   = train_test_split(train_f, test_f  , test_size=0.2, random_state=42)

"""
train_data.to_csv('train.csv', index=False)
val_data.to_csv('validation.csv', index=False)
test_data.to_csv('test.csv', index=False)
"""