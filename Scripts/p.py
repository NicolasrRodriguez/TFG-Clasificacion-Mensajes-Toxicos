import tensorflow as tf
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score


#tensorflow
def build_classifier_model():
  input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')
  net = tf.keras.layers.Dropout(0.1)(input)
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
  return tf.keras.Model(input, net)

#gpt
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

#kagle
def build_model(max_len = 169):
    input = tf.keras.layers.Input(shape=(max_len,), dtype=tf.float64, name='Entrada')
    lay = tf.keras.layers.Dense(64, activation='relu')(input)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(lay)
    model = tf.keras.Model(inputs = input, outputs = out)
    return model

def fine_tune(model, X_train, x_val, y_train, y_val):
    max_epochs = 60
    batch_size = 32
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy()
    best_weights_file = "weights.h5"
    m_ckpt = tf.keras.callbacks.ModelCheckpoint(best_weights_file, monitor='val_accuracy', mode='max', verbose=2,
                             save_weights_only=True, save_best_only=True)
    model.compile(loss=loss, optimizer=opt, metrics=[
                                                     'accuracy'])
    model.fit(
        X_train, y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[m_ckpt],
        verbose=2
    )


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
            model = build_classifier_model()
            #model.compile(loss='binary_crossentropy', metrics=['accuracy'])


            fine_tune(model,X_inner_train,X_inner_val,y_inner_train,y_inner_val)
            
            # Train the model
            #model.fit(X_inner_train, y_inner_train, epochs=10)
            
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