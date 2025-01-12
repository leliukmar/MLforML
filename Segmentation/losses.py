import tensorflow as tf
import numpy as np

def dice_loss(y_true, y_pred, smoothing=1e-5):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)

  intersection = tf.reduce_sum(y_true*y_pred)
  sum = tf.reduce_sum(y_true + y_pred)

  dice_score = (2*intersection + smoothing) / (sum + smoothing)
  return 1 - dice_score

def cross_entropy(y_true, y_pred):
  y_pred = tf.cast(y_pred, tf.float32)
  y_true = tf.cast(y_true, tf.float32)

  log_sum = tf.math.log(y_pred + tf.keras.backend.epsilon())*y_true
  ce_matrix = -tf.reduce_sum(log_sum, axis=-1)
  return tf.reduce_mean(ce_matrix)