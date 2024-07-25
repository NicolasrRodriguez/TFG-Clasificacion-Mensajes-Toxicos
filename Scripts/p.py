import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score



def build_classifier_model():
  input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')
  net = tf.keras.layers.Dropout(0.1)(input)
  net = tf.keras.layers.Dense(2, activation=None, name='classifier')(net)
  return tf.keras.Model(input, net)

def build_classifier_model_():
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
    return tf.keras.Model(input, net)


#def build_model( capas , ):



def cross_val_nested(X, y, outer_folds=5, inner_folds=5):

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
    
    outer_scores = []
    
    for train_index, test_index in outer_cv.split(X,y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Inner loop for hyperparameter tuning
        best_score = -np.inf
        best_model = None
        for inner_train_index, inner_val_index in inner_cv.split(X_train, y_train):
            X_inner_train, X_inner_val = X_train.iloc[inner_train_index], X_train.iloc[inner_val_index]
            y_inner_train, y_inner_val = y_train.iloc[inner_train_index], y_train.iloc[inner_val_index]
            
            # Build and compile the model
            model = build_classifier_model_()
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train the model
            model.fit(X_inner_train, y_inner_train, epochs=50)
            
            # Evaluate on validation set
            y_val_pred = model.predict(X_inner_val)
            y_val_pred = (y_val_pred > 0.5).astype(int)  # Assuming binary classification
            score = accuracy_score(y_inner_val, y_val_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # Test the best model on the outer test set
        y_test_pred = best_model.predict(X_test)
        y_test_pred = (y_test_pred > 0.5).astype(int)
        test_score = accuracy_score(y_test, y_test_pred)
        outer_scores.append(test_score)
    
    return np.mean(outer_scores), np.std(outer_scores)


data = pd.read_csv("Data/finaldpck.csv")
data_f = data.drop(columns=['class'])
data_c = data['class']

mean_score, std_score = cross_val_nested(data_f, data_c)
print(f'Accuracy media: {mean_score:.4f}')
print(f'Desviación estándar de la accuracy: {std_score:.4f}')