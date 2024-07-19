import pandas as pd 
from sklearn.model_selection import train_test_split, StratifiedKFold

import numpy as np


datapack = pd.read_csv('Data/finaldpck.csv')

caracteristicas = datapack.drop(columns=['class'])

clases = datapack['class']

folds = 5

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= 42)

splits = cv.split(caracteristicas,clases)

for index, (train_val_index, test_index) in enumerate(splits):
    print(f"Fold {index} : train_val | test")

    x_train_val_fold, x_test_fold = caracteristicas.iloc[train_val_index], caracteristicas.iloc[test_index]

    y_train_val_fold, y_test_fold = clases.iloc[train_val_index], clases.iloc[test_index]

    #Guardo test en el directorio

    path = f"Data/CV/Fold-{index+1}/test.csv"

    x_test_fold['class'] = y_test_fold

    x_test_fold.to_csv( path , index = False )

    splits = cv.split(x_train_val_fold,y_train_val_fold)

    for index, (train_index, val_index) in enumerate(splits):
        print(f"Fold {index} : train | val")

        path = f"Data/CV/Fold-{index+1}/Train-Val"

        x_train_fold, x_val_fold = caracteristicas.iloc[train_index], caracteristicas.iloc[val_index]

        y_train_fold, y_val_fold = clases.iloc[train_index], clases.iloc[val_index]

"""
for train_val_index, test_index in splits:

    x_train_val_fold, x_test_fold = caracteristicas[train_val_index], caracteristicas[test_index]

    y_train_val_fold, y_test_fold = clases[train_val_index], clases[test_index]

    cv = StratifiedKFold(1)
    splits = cv.split(x_train_val_fold,y_train_val_fold)

    for train_index, val_index in splits:

        x_train_fold, x_val_fold = caracteristicas[train_index], caracteristicas[val_index]

        y_train_fold, y_val_fold = clases[train_index], clases[val_index]
"""
