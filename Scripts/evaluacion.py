import tensorflow as tf
import tensorflow_hub as hub
import bert_model as bm
from sklearn.model_selection import GridSearchCV , cross_val_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold , train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(data_f, data_c , test_size=0.2, random_state=42)

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
# Modelo 1 {'batch_size': 256, 'epochs': 60, 'learning_rate': 0.01}
# Modelo 2 {'batch_size': 128, 'epochs': 60, 'learning_rate': 0.001}
# Modelo 3 {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}

#final_model = build_classifier_model_1(learning_rate=0.01)
final_model = build_classifier_model_2(learning_rate=0.001)
#final_model = SVC(C= 10, gamma= 'scale' , kernel= 'rbf')

final_model.fit(X_train, y_train, epochs=60, batch_size=128)

#final_model.fit(X_train, y_train)

#joblib.dump(final_model, "model3.pkl")

final_model.save_weights('model2_weights.h5')



#--Evaluación--

predictions = final_model.predict(X_test)

predictions = (predictions >= 0.5).astype(int)


acuracy = accuracy_score( y_pred = predictions, y_true= y_test)

conf_mat = confusion_matrix(y_pred= predictions, y_true = y_test)

with open('resultado_eval.txt', 'w') as file:
    file.write(f"Exactitud en el conjunto de test: {acuracy}")

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False,annot_kws={"size": 18} ,xticklabels=['Negativos', 'Positivos'], 
            yticklabels=['Negativos', 'Positivos'],)
plt.xlabel('Predicción', fontsize=19)
plt.ylabel('Real', fontsize=19)
plt.title(f'Matríz de confusión de Test: Modelo 2', fontsize=20)
plt.show()  
