# SEGAR is resnet
# SEGARN is resNext

from math import ceil
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, initializers, regularizers, constraints

alpha = 1e-4


def SE_Block(inputs, num_filters, ratio=4):
    squeeze = tf.keras.layers.GlobalAveragePooling3D()(inputs)
    excitation = tf.keras.layers.Dense(units=num_filters / ratio)(squeeze)
    excitation = tf.keras.layers.Activation("relu")(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, 1, num_filters])(excitation)
    scale = inputs * excitation
    return scale


class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.ReLU()

        self.conv_list_layer_1 = []
        self.conv_list_layer_2 = []
        for i in range(self.groups):
            self.conv_list_layer_1.append(tf.keras.layers.Conv2D(filters=self.group_out_num * 2,
                                                                 kernel_size=(1, 1),
                                                                 strides=strides,
                                                                 padding=padding,
                                                                 data_format=data_format,
                                                                 dilation_rate=dilation_rate,
                                                                 activation=activations.get(activation),
                                                                 use_bias=use_bias,
                                                                 kernel_initializer=initializers.get(
                                                                     kernel_initializer),
                                                                 bias_initializer=initializers.get(bias_initializer),
                                                                 kernel_regularizer=regularizers.get(
                                                                     kernel_regularizer),
                                                                 bias_regularizer=regularizers.get(bias_regularizer),
                                                                 activity_regularizer=regularizers.get(
                                                                     activity_regularizer),
                                                                 kernel_constraint=constraints.get(kernel_constraint),
                                                                 bias_constraint=constraints.get(bias_constraint),
                                                                 **kwargs))
            self.conv_list_layer_2.append(tf.keras.layers.Conv2D(filters=self.group_out_num,
                                                                 kernel_size=kernel_size,
                                                                 strides=strides,
                                                                 padding=padding,
                                                                 data_format=data_format,
                                                                 dilation_rate=dilation_rate,
                                                                 activation=activations.get(activation),
                                                                 use_bias=use_bias,
                                                                 kernel_initializer=initializers.get(
                                                                     kernel_initializer),
                                                                 bias_initializer=initializers.get(bias_initializer),
                                                                 kernel_regularizer=regularizers.get(
                                                                     kernel_regularizer),
                                                                 bias_regularizer=regularizers.get(bias_regularizer),
                                                                 activity_regularizer=regularizers.get(
                                                                     activity_regularizer),
                                                                 kernel_constraint=constraints.get(kernel_constraint),
                                                                 bias_constraint=constraints.get(bias_constraint),
                                                                 **kwargs))
    @tf.function
    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i_layer1 = self.conv_list_layer_1[i](
                inputs[:, :, :, :, i * self.group_in_num: (i + 1) * self.group_in_num])
            x_i_layer1 = self.batch_norm(x_i_layer1)
            x_i_layer1 = self.ReLU(x_i_layer1)
            x_i_out = self.conv_list_layer_2[i](x_i_layer1)
            feature_map_list.append(x_i_out)
        out = tf.concat(feature_map_list, axis=-1)
        return out


def SE_ResNeXt_Identity(inputs, strides, groups):
    shortcut = inputs
    filters = inputs.shape[-1]

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.15)(x)

    x = tf.keras.layers.Conv2D(filters=2 * filters,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = GroupConv2D(input_channels=filters,
                    output_channels=filters,
                    kernel_size=(3, 3),
                    strides=strides,
                    padding="same",
                    groups=groups,
                    kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    x = SE_Block(inputs=x, num_filters=filters, ratio=4)

    output = tf.keras.layers.Add()([x, shortcut])
    return output

def SE_ResNeXt_BottleNeck(inputs, filters, strides, groups):
    shortcut = tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=(1, 1),
                                      strides=strides,
                                      padding="same")(inputs)

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.15)(x)

    x = tf.keras.layers.Conv2D(filters=2 * filters,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = GroupConv2D(input_channels=filters,
                    output_channels=filters,
                    kernel_size=(3, 3),
                    strides=strides,
                    padding="same",
                    groups=groups,
                    kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    x = SE_Block(inputs=x, num_filters=filters, ratio=4)

    output = tf.keras.layers.Add()([x, shortcut])
    return output


def SE_Global_Attention_ResNeXt_BottleNeck(inputs, strides, groups):
    filters = inputs.shape[-1]
    shortcut = inputs
    inputs = tf.keras.layers.BatchNormalization()(inputs)
    inputs = tf.keras.layers.LeakyReLU(alpha=0.15)(inputs)

    min_channels = ceil(((1 / 3) * filters) / groups) * groups

    GPS = tf.keras.layers.Conv2D(filters=min_channels,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding="same",
                                 kernel_regularizer=tf.keras.regularizers.L2(alpha))(inputs)

    GPS = tf.keras.layers.BatchNormalization()(GPS)
    GPS = tf.keras.layers.ReLU()(GPS)

    GPS_MEAN = tf.keras.layers.GlobalAvgPool3D()(GPS)
    GPS_MAX = tf.keras.layers.GlobalMaxPooling3D()(GPS)

    GPS = tf.keras.layers.concatenate([GPS_MEAN, GPS_MAX])
    for _ in range(2):
        GPS = tf.keras.layers.Dense(GPS.shape[-1])(GPS)  # FC
    GPS = tf.keras.layers.Reshape([1, 1, 1, GPS.shape[-1]])(GPS)

    x = GroupConv2D(input_channels=filters,
                    output_channels=2 * min_channels,
                    kernel_size=(3, 3),
                    strides=strides,
                    padding="same",
                    groups=groups,
                    kernel_regularizer=tf.keras.regularizers.L2(alpha))(inputs)
    x = SE_Block(inputs=x, num_filters=x.shape[-1], ratio=4)

    x = tf.keras.layers.Add()([x, GPS])

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding="same",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)

    output = tf.keras.layers.Add()([x, shortcut])
    output = tf.keras.layers.BatchNormalization()(output)

    return output


def build_ResNeXt_block(inputs, filters, strides, groups, repeat_num):
    x = inputs
    for i in range(1, repeat_num + 1):
        x = SE_ResNeXt_BottleNeck(x,
                                  filters=filters,
                                  strides=strides,
                                  groups=groups)
        test_shape = x.shape
        x = SE_Global_Attention_ResNeXt_BottleNeck(inputs=x, strides=strides, groups=2)
        assert np.array_equal(test_shape, x.shape)
        if i % 2 != 0:
            # x = tf.keras.layers.GaussianDropout(rate=0.1)(x)
            x = tf.keras.layers.Dropout(0.1)(x)

    return x


def Get_2dSEGARNB(shape, policy_shape, filters_per_group, groups, blocks):
    inputs = tf.keras.Input(shape, dtype=tf.float32)

    # specific for Gomoku because
    # eye_4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), padding="same")(inputs)
    # eye_4 = SE_Block(eye_4, num_filters=16, ratio=1)
    eye_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(inputs)
    eye_3 = SE_Block(eye_3, num_filters=256, ratio=4)
    # eye_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), padding="same")(inputs)
    # eye_2 = SE_Block(eye_2, num_filters=64, ratio=2)

    # eyes = tf.keras.layers.concatenate([eye_3, eye_2])
    eyes = tf.keras.layers.BatchNormalization()(eye_3)
    eyes = tf.keras.layers.PReLU()(eyes)

    # x = build_ResNeXt_block(inputs=eyes, filters=64, strides=1, groups=2, repeat_num=1)
    x = SE_ResNeXt_BottleNeck(inputs=eyes, filters=196, strides=1, groups=2)
    x = build_ResNeXt_block(inputs=x, filters=filters_per_group, strides=1, groups=groups, repeat_num=blocks)

    # raise ValueError
    policy_head = tf.keras.layers.Conv2D(6, (3, 3), strides=(1, 1), padding="same",
                                         kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    policy_head = tf.keras.layers.BatchNormalization()(policy_head)
    policy_head = tf.keras.layers.PReLU()(policy_head)

    policy_head = tf.keras.layers.Flatten()(policy_head)

    policy_head = tf.keras.layers.Dense(448,
                                        activation="relu",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha / 1.5)
                                        )(policy_head)
    policy_head = tf.keras.layers.Dense(policy_shape, activation='softmax', dtype='float32', name="policy")(policy_head)

    # value head
    value_head = tf.keras.layers.Conv2D(2, (1, 1), strides=(1, 1),
                                        padding='same',
                                        # kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    value_head = tf.keras.layers.BatchNormalization()(value_head)
    value_head = tf.keras.layers.LeakyReLU(0.1)(value_head)

    value_head = tf.keras.layers.Flatten()(value_head)
    value_head = tf.keras.layers.Dropout(0.1)(value_head)
    # value_head = tf.keras.layers.Dense(512, activation="relu",
    #                                    kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(256, activation="relu",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha)
                                       )(value_head)
    value_head = tf.keras.layers.Dense(128, activation="relu",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(96, activation="relu")(value_head)

    value_head = tf.keras.layers.Dense(1, activation="tanh", dtype='float32', name="value")(value_head)

    outputs = {"policy": policy_head, "value": value_head}
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),
                  loss={"policy": "binary_crossentropy", "value": "mse"},
                  loss_weights={"policy": 1, "value": 1},
                  metrics=["accuracy", "binary_crossentropy", "mse"])

    model.summary()
    tf.keras.utils.plot_model(model,
                              to_file="SE_GARNnest_2d.png",
                              show_dtype=True,
                              show_shapes=True,
                              expand_nested=True,
                              # show_trainable=True,
                              show_layer_activations=True,
                              show_layer_names=True,
                              )
    return model


if __name__ == "__main__":
    # import tensorflow_model_optimization as tfmot
    import womoku as gf

    # inputs = tf.keras.Input((1, 128, 128, 4))
    # x = GroupConv2D(inputs,4, 4, (3, 3), groups=4)
    # print(x.get_prunable_weights())

    model = Get_2dSEGARNB((4, 15, 15, 1), gf.WIDTH * gf.HEIGHT, filters_per_group=128, groups=2, blocks=4)
    # model.save_weights(f"../alphazero/models/0.h5", save_format="h5", overwrite=True)
    model.save(f"../alphazero/models/0", overwrite=True, save_traces=True)
