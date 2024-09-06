
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np

from multicolinearidad import atributos_redundantes

import joblib
#----------------------------------------------------------------

scaler = StandardScaler()

datapack = pd.read_csv('Data/pooled_outputs.csv')

caracteristicas = datapack.drop(columns=['class'])

caracteristicas = caracteristicas.drop(labels = atributos_redundantes ,axis= 1)

X_train = scaler.fit_transform(caracteristicas)

joblib.dump(scaler, 'scaler.pkl')#para reutilar el scaler

clf = LogisticRegression(max_iter=1000, random_state=42)

cv = StratifiedKFold(5)

rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring='accuracy', min_features_to_select=1, n_jobs=-1)

rfecv.fit(X_train, datapack['class'].values)

print(f"Número óptimo de características : {rfecv.n_features_}")

x = pd.DataFrame(X_train, columns=[f"{i}" for i in range(X_train.shape[1])])

resp = x.columns[rfecv.get_support()] 

print(np.array(resp))


df_selected_columns = x[np.array(resp)] 

df_all = pd.concat([df_selected_columns,datapack['class']] , axis= 1)

df_all.to_csv('df_all.csv', index=False)

with open('selected_features.txt', 'w') as f:
    f.write("Número óptimo de características : {rfecv.n_features_}")
    f.write("Características seleccionadas:\n")
    for feature in resp:
        f.write("{}\n".format(feature))


cv_results = pd.DataFrame(rfecv.cv_results_)


plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.plot(range(1, len(cv_results["mean_test_score"]) + 1), cv_results["mean_test_score"])
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()