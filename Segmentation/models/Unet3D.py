import tensorflow as tf
import numpy as np

def _conv_block(input,
               filter_size,
               kernel_size=3,
               strides=1,
               transpose=False,
               norm=True,
               activation=tf.keras.layers.LeakyReLU()):
  if transpose:
    conv = tf.keras.layers.Conv3DTranspose
  else:
    conv = tf.keras.layers.Conv3D

  x = conv(filters=filter_size,
           kernel_size=kernel_size,
           strides=strides,
           padding='same')(input)

  if norm:
      x = tf.keras.layers.BatchNormalization()(x)
  if activation:
      x = activation(x)
  return x


def _downsample_block(input, filter_size):
  x = _conv_block(input, filter_size, strides=2)
  x = _conv_block(x, filter_size)
  return x


def _upsample_block(inp, filter_size, skip_connection):
  init_filter_size = inp.shape[-1]
  x = _conv_block(inp, init_filter_size, transpose=True, kernel_size=2, strides=2, norm=False, activation=None)
  x = tf.keras.layers.Concatenate(axis=-1)([x, skip_connection])

  x = _conv_block(x, filter_size)
  return _conv_block(x, filter_size)


def unet_3d(class_num, size = 96, filter_size = 32):
  assert size % 16 == 0, "Size should be a multiple of 16"

  inputs = tf.keras.layers.Input(shape=(size, size, size, 3))
  inp = _conv_block(inputs, filter_size)
  inp = _conv_block(inp, filter_size)

  ds_1 = _downsample_block(inp, 2*filter_size)   # 48x48x48x64
  ds_2 = _downsample_block(ds_1, 4*filter_size)  # 24x24x24x128
  ds_3 = _downsample_block(ds_2, 8*filter_size)  # 12x12x12x256
  ds_4 = _downsample_block(ds_3, 16*filter_size) # 6x6x6x512

  ups_1 = _upsample_block(ds_4, 8*filter_size, ds_3)  # 12x12x12x256
  ups_2 = _upsample_block(ups_1, 4*filter_size, ds_2) # 24x24x24x128
  ups_3 = _upsample_block(ups_2, 2*filter_size, ds_1) # 48x48x48x64
  ups_4 = _upsample_block(ups_3, filter_size, inp)    # 96x96x96x32

  output = _conv_block(ups_4, class_num, activation=tf.keras.layers.Softmax())
  return tf.keras.Model(inputs=inputs, outputs=output)