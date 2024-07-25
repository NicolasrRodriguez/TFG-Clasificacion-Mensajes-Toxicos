import tensorflow as tf

def build_classifier_model():
  input = tf.keras.layers.Input(shape=(169,), dtype=tf.float64, name='Entrada')
  net = tf.keras.layers.Dropout(0.1)(input)
  net = tf.keras.layers.Dense(2, activation=None, name='classifier')(net)
  return tf.keras.Model(input, net)


classifier_model = build_classifier_model()

def Estimator(BaseEstimator):
  def __init__ (self):
    self.clasifier = classifier_model

  def fit(self, x, y, w=None):
    self.clasifier.fit()