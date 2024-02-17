import tensorflow as tf
#

# Creating Densenet121
def densenet(input_tensor, repetitions=[6, 24, 12], filters=32, alpha=5e-5):
    # input_tensor should receive output from the eyes of the tower
    # batch norm + relu + conv
    def bn_rl_conv(x, filters, kernel=1, strides=1):

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, kernel, strides=strides, padding='same',
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=tf.keras.regularizers.L2(alpha)
                                   )(x)
        return x

    def dense_block(x, repetition):

        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 2)
            x = tf.keras.layers.concatenate([y, x])
        return x

    def transition_layer(x):

        x = bn_rl_conv(x, tf.keras.backend.int_shape(x)[-1] // 2)
        x = tf.keras.layers.AvgPool3D(2, strides=2, padding='same')(x)
        return x

    # input = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(128, 5, strides=1, padding='same',
                               kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(input_tensor)

    # x = tf.keras.layers.MaxPool3D(1, strides=1, padding='same')(x)

    for repetition in repetitions:
        d = dense_block(x, repetition)
        x = transition_layer(d)

    return x


def Get_2ddensenet(input_shape, policy_shape=9, repetitions=[6, 24, 12], alpha=5e-4):
    # for wic wac woe shape should be 19, 3, 3
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float16)
    x = densenet(inputs, repetitions=repetitions, alpha=alpha)
    # print(x.shape)

    policy_head = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding="same",
                                         kernel_initializer="he_normal",
                                         kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    policy_head = tf.keras.layers.BatchNormalization()(policy_head)
    policy_head = tf.keras.layers.Activation("relu")(policy_head)

    policy_head = tf.keras.layers.Flatten()(policy_head)
    policy_head = tf.keras.layers.Dense(policy_shape * 30, activation='relu',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    policy_head = tf.keras.layers.Dense(policy_shape * 20, activation='relu',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    policy_head = tf.keras.layers.Dense(policy_shape * 10, activation='relu',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    policy_head = tf.keras.layers.Dense(policy_shape, activation='softmax', dtype='float32', name="policy")(policy_head)

    # value head
    value_head = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    value_head = tf.keras.layers.BatchNormalization()(value_head)
    value_head = tf.keras.layers.Activation("relu")(value_head)

    value_head = tf.keras.layers.Flatten()(value_head)
    value_head = tf.keras.layers.Dense(policy_shape * 20, activation='relu',
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(policy_shape * 20, activation='relu',
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(policy_shape * 10, activation='relu',
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(policy_shape * 5, activation='relu',
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(1, activation="tanh", dtype='float32', name="value")(value_head)

    outputs = {"policy": policy_head, "value": value_head}
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),
                  loss={"policy": "bce", "value": "mse"},
                  metrics=["accuracy", "bce", "mse"])
    # model.summary()
    tf.keras.utils.plot_model(model,
                              to_file="densenet.png",
                              show_shapes=True,
                              show_dtype=True,
                              show_layer_activations=True,
                              dpi=1000)
    model.summary()
    return model


# if __name__ == "__main__":
#     Get_2ddensenet(input_shape=(19, 3, 3, 1))
    # model = tf.keras.models.Model(inputs=inputs, outputs=)
    #
    #
    # model.summary()
    # tf.keras.utils.plot_model(model,
    #                           show_shapes=True,
    #                           show_dtype=True,
    #                           show_layer_activations=True)
