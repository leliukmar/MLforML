import tensorflow as tf
import numpy as np

# y_true shape: (N, D, H, W, C)
# y_pred shape: (N, D, H, W, C)
def dice_loss(y_true, y_pred, smoothing=1e-5):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)

  intersection = tf.reduce_sum(y_true*y_pred)
  sum = tf.reduce_sum(y_true + y_pred)

  dice_score = (2*intersection + smoothing) / (sum + smoothing)
  return 1 - dice_score

# y_true shape: (N, D, H, W, C)
# y_pred shape: (N, D, H, W, C)
def cross_entropy(y_true, y_pred):
  y_pred = tf.cast(y_pred, tf.float32)
  y_true = tf.cast(y_true, tf.float32)

  log_sum = tf.math.log(y_pred + tf.keras.backend.epsilon())*y_true
  ce_matrix = -tf.reduce_sum(log_sum, axis=-1)
  return tf.reduce_mean(ce_matrix)


# y_true shape: (N, D, H, W, C)
# y_pred shape: (N, D, H, W, C)
def tversky_loss(y_true, y_pred, alpha=0.5, smoothing=1e-5):
  y_true_pos = tf.cast(y_true, tf.float32)
  y_pred_pos = tf.cast(y_pred, tf.float32)

  true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
  false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
  false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)

  return 1 - (true_pos + smoothing) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smoothing)