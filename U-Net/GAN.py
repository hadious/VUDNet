import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, losses, optimizers


OUTPUT_CHANNELS = 1
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 256
INPUT_NUM = 1

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[512, 256, INPUT_NUM])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    # downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    # upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    # print (x.shape)
    # print("LAYER", down.input_shape, down.output_shape)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    # print ("X shape", x.shape)
    # print("Skip shape", skip.shape)
    x = up(x)
    # print (x.shape, skip.shape)
    # print ("after X shape", x.shape)
    # print("after Skip shape", skip.shape)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[512, 256, INPUT_NUM], name='input_image')
  tar = tf.keras.layers.Input(shape=[512, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)



#####################################################################################################
# Generator with U-Net architecture 
# def build_generator():
#     inputs = layers.Input(shape=[256, 512, 9])

#     # Downsample
#     down_stack = [
#         layers.Conv2D(64, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         layers.LeakyReLU(),

#         layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.LeakyReLU(),

#         layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.LeakyReLU(),

#         layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.LeakyReLU(),

#         layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.LeakyReLU(),

#         # layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         # layers.LeakyReLU(),

#         # layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         # layers.LeakyReLU(),

#         # layers.Conv2D(512, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         # layers.LeakyReLU(),
#     ]

#     # Upsample
#     up_stack = [
#         # layers.Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         # layers.ReLU(),

#         # layers.Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         # layers.ReLU(),

#         # layers.Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         # layers.ReLU(),

#         # layers.Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False),
#         # layers.BatchNormalization(),
#         # layers.ReLU(),

#         layers.Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),

#         layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),

#         layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),

#         layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
#         layers.BatchNormalization(),
#         layers.ReLU(),
#     ]

#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = layers.Conv2DTranspose(1, 4, strides=2, padding='same',
#                                   kernel_initializer=initializer, activation='tanh')

#     x = inputs
#     skips = []
#     for down in down_stack:
#         x = down(x)
#         # print(x.shape)
#         skips.append(x)
#     skips = reversed(skips[:-1])

#     for up, skip in zip(up_stack, skips):
#         # print(x.shape, skip.shape)
#         x = up(x)
#         # print(x.shape, skip.shape)
#         x = layers.Concatenate()([x, skip])

#     x = last(x)
#     return Model(inputs=inputs, outputs=x)


# # Discriminator with PatchGAN
# def build_discriminator():
#     init = tf.random_normal_initializer(0., 0.02)
#     inp = layers.Input(shape=[512, 256, 9], name='input_image')
#     tar = layers.Input(shape=[512, 256, 1], name='target_image')

#     x = layers.concatenate([inp, tar])

#     down1 = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=init)(x)
#     down1 = layers.LeakyReLU()(down1)

#     down2 = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=init)(down1)
#     down2 = layers.BatchNormalization()(down2)
#     down2 = layers.LeakyReLU()(down2)

#     down3 = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=init)(down2)
#     down3 = layers.BatchNormalization()(down3)
#     down3 = layers.LeakyReLU()(down3)

#     last = layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=init)(down3)

#     return Model(inputs=[inp, tar], outputs=last)


with tf.device('/cpu:0'):  # Move loss to CPU for memory efficiency
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(disc_real_output, disc_generated_output):
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (100 * l1_loss)
        return total_gen_loss



 
