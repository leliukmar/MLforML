import tensorflow as tf
import numpy as np

# y_true shape: (N, D, H, W)
# y_pred shape: (N, D, H, W)
def intersection_over_union(y_true, y_pred, class_idx):
  class_iou = []
  for class_id in class_idx:
    true_mask = y_true == class_id
    pred_mask = y_pred == class_id

    intersection = tf.reduce_sum(tf.cast(true_mask & pred_mask, tf.float32))
    union = tf.reduce_sum(tf.cast(true_mask | pred_mask, tf.float32))
    iou = (intersection / union).numpy()
    class_iou.append(iou)

  return tf.reduce_mean(class_iou)