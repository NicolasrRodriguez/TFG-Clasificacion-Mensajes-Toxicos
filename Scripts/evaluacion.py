import tensorflow as tf
import tensorflow_hub as hub
import bert_model as bm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
import joblib


#----Datos---
data = pd.read_csv("Data/finaldpck.csv")
data_f = data.drop(columns=['class'])#caracteristicas
data_c = data['class']#etiquetas

#----Separación de un conjunto de prueba-------

cv_externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#---Definicion del modelo---

def build_classifier_model_1(learning_rate=1e-3):
  input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')#Entrada: Pooled_outputs procesadas(de 768 caracteristicas a 169)
  net = tf.keras.layers.Dropout(0.1)(input)
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)#Salida: Utiliza una función sigmoidal
  model = tf.keras.Model(input, net)
  opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)#Optimizador Adam
  loss = tf.keras.losses.BinaryCrossentropy()
  model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
  return model

def build_classifier_model_2(learning_rate=1e-3):
    input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')
    net = tf.keras.layers.Dense(64, activation='relu')(input)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(16, activation='relu')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    model = tf.keras.Model(input, net)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model




#---Entrenamiento final con los mejores hiperparametros---- 
# Modelo 1 {'batch_size': 32, 'epochs': 60, 'learning_rate': 0.001}
# Modelo 2 {'batch_size': 64, 'epochs': 60, 'learning_rate': 0.001}
# Modelo 3 {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
accuracys = [] #Exactitudes durante el proceso de hiperparametrización
precisions = []
recalls = []
confusion_matrices = [] #Matrices de confusion 

for index ,(train_idx, test_idx) in enumerate(cv_externo.split(data_f, data_c)):
   X_train, X_test = data_f.iloc[train_idx], data_f.iloc[test_idx]
   y_train, y_test = data_c.iloc[train_idx], data_c.iloc[test_idx]

   #final_model = build_classifier_model_1(learning_rate=0.001)
   #final_model = build_classifier_model_2(learning_rate=0.001)
   final_model = SVC(C= 10, gamma= 'auto' , kernel= 'rbf')
   
   #final_model.fit(X_train, y_train, epochs=60, batch_size=32)
   #final_model.fit(X_train, y_train, epochs=60, batch_size=64)
   final_model.fit(X_train, y_train)

   out_predicts = final_model.predict(X_test)

   binary_predictions = (out_predicts >= 0.5).astype(int)

   acc = accuracy_score(y_pred= binary_predictions, y_true= y_test)

   prec = precision_score(y_pred= binary_predictions, y_true= y_test)

   rec = recall_score(y_pred= binary_predictions, y_true= y_test)

   conf_mat = confusion_matrix(y_pred= binary_predictions, y_true = y_test)

   confusion_matrices.append(conf_mat)

   accuracys.append(acc)

   precisions.append(prec)
   
   recalls.append(rec)

for index ,mat in enumerate(confusion_matrices):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False,annot_kws={"size": 18})
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(f'Matríz de confusión fold {index}')
 
  
with open('resultado_eval.txt', 'w') as file:
   file.write("----------------\n")
   for index, acc in enumerate(accuracys):
    file.write(f"Fold {index}\n")
    file.write(f"Exactitud: {accuracys[index]} Precision: {precisions[index]} Sensibilidad: {recalls[index]}\n")
    file.write("----------------\n")

   file.write(f"Exactitud media en test': {np.mean(accuracys)} Desviacion tipica: {np.std(accuracys)} \n")
   file.write("----------------\n")

plt.show()  