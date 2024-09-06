import tensorflow as tf
import pandas as pd
import numpy
print("TensorFlow version:", tf.__version__)

print("TensorFlow version:", numpy.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


