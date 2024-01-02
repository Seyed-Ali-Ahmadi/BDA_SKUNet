from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Concatenate, Multiply, Resizing, DepthwiseConv2D, Lambda, Softmax, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Activation, Add, GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, maximum, Layer # , DropBlock2D
import tensorflow as tf
# from keras.initializers import he_normal



def SKConv(M=2, r=16, L=32, G=32, convolutions='same', dropout_rate=0.001, name='skconv'):
    def wrapper(inputs):
        b, h, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        filters = inputs.get_shape().as_list()[-1]
        d = max(filters//r, L)      # Middle channels

        x = inputs

        xs = []
        for m in range(M):
            if G == 1:
                if convolutions == 'same':
                    _x = Conv2D(filters, kernel_size=3, dilation_rate=m+1, padding='same', use_bias=False, name=name+'_conv%d'%m)(x)
                elif convolutions == 'different':
                    _x = Conv2D(filters, kernel_size=3+m*2, dilation_rate=m+1, padding='same', use_bias=False, name=name+'_conv%d'%m)(x)
            else:
                c = filters // G
                if convolutions == 'same':
                    _x = DepthwiseConv2D(kernel_size=3, dilation_rate=m+1, padding='same', use_bias=False, depth_multiplier=c, name=name+'_conv%d'%m)(x)
                elif convolutions == 'different':
                    _x = DepthwiseConv2D(kernel_size=3+m*2, dilation_rate=m+1, padding='same', use_bias=False, depth_multiplier=c, name=name+'_conv%d'%m)(x)

                _x = Reshape([h, w, G, c, c], name=name+'_conv%d_reshape1'%m)(_x)
                _x = Lambda(lambda x: tf.reduce_sum(x, axis=-1), output_shape=[b, h, w, G, c], name=name+'_conv%d_sum'%m)(_x)
                _x = Reshape([h, w, filters], name=name+'_conv%d_reshape2'%m)(_x)

            _x = BatchNormalization(name=name+'_conv%d_bn'%m)(_x)
            _x = Activation('relu', name=name+'_conv%d_relu'%m)(_x)
            _x = Dropout(rate=dropout_rate)(_x)

            xs.append(_x)

        U = Add(name=name+'_add')(xs)
        s = Lambda(lambda x: tf.reduce_mean(x, axis=[1,2], keepdims=True), output_shape=[b, 1, 1, filters], name=name+'_gap')(U)

        z = Conv2D(d, 1, name=name+'_fc_z')(s)
        z = BatchNormalization(name=name+'_fc_z_bn')(z)
        z = Activation('relu', name=name+'_fc_z_relu')(z)
        z = Dropout(rate=dropout_rate)(z)

        x = Conv2D(filters*M, 1, name=name+'_fc_x')(z)
        x = Reshape([1, 1, filters, M],name=name+'_reshape')(x)
        scale = Softmax(name=name+'_softmax')(x)

        x = Lambda(lambda x: tf.stack(x, axis=-1), output_shape=[b, h, w, filters, M], name=name+'_stack')(xs) # b, h, w, c, M
        x = Axpby(name=name+'_axpby')([scale, x])

        return x
    return wrapper


class Axpby(Layer):
  def __init__(self, **kwargs):
        super(Axpby, self).__init__(**kwargs)

  def build(self, input_shape):
        super(Axpby, self).build(input_shape)  # Be sure to call this at the end

  def call(self, inputs):
    scale, x = inputs
    f = tf.multiply(scale, x, name='product')
    f = tf.reduce_sum(f, axis=-1, name='sum')
    return f

  def compute_output_shape(self, input_shape):
    return input_shape[0:4]



def squeeze_excite_block(inputs, ratio=8):
    init = inputs       ## (b, 128, 128, 32)
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)     ## (b, 32)   -> (b, 1, 1, 32)
    se = Reshape(se_shape)(se)
    se = Dense(filters//ratio, activation="relu", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", use_bias=False)(se)

    x = Multiply()([inputs, se])
    return x


def ASPP(x, filter):
    shape = x.shape

    y1 = layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = layers.Conv2D(filter, 1, padding="same")(y1)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.Activation("relu")(y1)
    y1 = layers.UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    y2 = layers.Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.Activation("relu")(y2)

    y3 = layers.Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.Activation("relu")(y3)

    y4 = layers.Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.Activation("relu")(y4)

    y5 = layers.Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = layers.BatchNormalization()(y5)
    y5 = layers.Activation("relu")(y5)

    y = layers.Concatenate()([y1, y2, y3, y4, y5])

    y = layers.Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    return y
    

