import tensorflow as tf
import womoku as gf

# tf.keras.mixed_precision.set_global_policy('float16')

def Get_2dresnet(input_shape, policy_shape, blocks):
    # policy shape must be flat, so for board shape 3 x 3 for wic wac woe, the policy shape should be 9
    if not isinstance(policy_shape, int):
        raise TypeError("Policy shape has to be an integer, it will be reshaped/reformatted when alphazero predict is "
                        f"called, shape {policy_shape} is invalid")
    alpha = 8.5e-5

    def dense_block(x, units, reshape=True):
        x = tf.keras.layers.Reshape((gf.SHAPE[1] * gf.SHAPE[2] * gf.SHAPE[3], -1))(x)
        # x = tf.keras.layers.Dense(units=units//1, kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
        x = tf.keras.layers.Dense(units=units, kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
        if reshape:
            x = tf.keras.layers.Reshape((gf.SHAPE[1], gf.SHAPE[2], gf.SHAPE[3], -1))(x)
        return x


    def residual_block(input_tensor, filters, kernel_size=(1, 1), strides=1, activation=tf.keras.layers.PReLU()):
        # Apply a 1x1 convolutional layer to reduce the number of filters in the input
        residue = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                         # kernel_initializer="he_normal",
                                         kernel_regularizer=tf.keras.regularizers.L2(alpha))(input_tensor)
        # Apply the main convolutional layers
        resnet = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
                                        # kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(input_tensor)
        resnet = tf.keras.layers.BatchNormalization()(resnet)
        resnet = tf.keras.layers.Activation(activation)(resnet)

        resnet = tf.keras.layers.Conv2D(filters, (3, 3), padding='same',
                                        # kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(resnet)
        resnet = tf.keras.layers.BatchNormalization()(resnet)
        # Apply a 1x1 convolutional layer to reduce the number of filters in the output

        resnet = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', strides=strides,
                                        # kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(resnet)
        # Add the inputs to the output of the block
        # resnet = tf.keras.layers.Reshape((gf.SHAPE[1] * gf.SHAPE[1] * gf.SHAPE[3], -1))(resnet)
        # resnet = SE(resnet)
        # resnet = tf.keras.layers.Reshape((gf.SHAPE[1], gf.SHAPE[1], gf.SHAPE[3], -1))(resnet)
        resnet = tf.keras.layers.Add()([resnet, residue])
        resnet = tf.keras.layers.Activation(activation)(resnet)
        return resnet

    # def SE(x):
    #     # w = tf.keras.layers.GlobalAvgPoolD()(x)
    #     filters = x.shape[-1]
    #     w = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation="relu")(x)
    #     w = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation="sigmoid")(w)
    #     x = x * w
    #     return x


    inputs = tf.keras.layers.Input(shape=input_shape, dtype="float16")

    # Apply a convolutional layer with 32 filters and a kernel size of 3x3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                               # kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                               # kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                               # kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    for _ in range(blocks):
        x = residual_block(x, 32)
        x = dense_block(x, 64)


    #
    # print(x.shape)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation(tf.keras.layers.PReLU())(x)
    #
    # for i in range(blocks):
    #     x = residual_block(x, 32, (1, 1), strides=1)
    #     if i != blocks - 1 and i % 1 == 0:
    #         x = tf.keras.layers.Dropout(0.15)(x)

    # policy head
    # policy_head = tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), padding="same",
    #                                          kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    # policy_head = tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), padding="same",
    #                                          kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    # policy_head = tf.keras.layers.BatchNormalization()(policy_head)
    # policy_head = tf.keras.layers.Activation(tf.keras.layers.PReLU())(policy_head)
    # policy_head = tf.keras.layers.Reshape((gf.SHAPE[1] * gf.SHAPE[1] * gf.SHAPE[3], -1))(x)
    policy_head = tf.keras.layers.Dense(units=32, kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    policy_head = tf.keras.layers.Dense(units=4, kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    policy_head = tf.keras.layers.Flatten()(policy_head)
    # policy_head = tf.keras.layers.Dense(640,
    #                                     activation="relu",
    #                                     kernel_regularizer=tf.keras.regularizers.L2(alpha)
    #                                     )(policy_head)
    # policy_head = tf.keras.layers.Dense(512,
    #                                     activation="relu",
    #                                     # kernel_regularizer=tf.keras.regularizers.L2(alpha)
    #                                     )(policy_head)
    # policy_head = tf.keras.layers.Dense(512,
    #                                     activation="relu",
    #                                     # kernel_regularizer=tf.keras.regularizers.L2(alpha)
    #                                     )(policy_head)
    # policy_head = tf.keras.layers.Dense(256,
    #                                     activation=tf.keras.layers.PReLU(),
    #                                     # kernel_regularizer=tf.keras.regularizers.L2(alpha)
    #                                     )(policy_head)
    policy_head = tf.keras.layers.Dense(policy_shape, activation='softmax', dtype='float32', name="policy")(policy_head)

    # value head
    value_head = tf.keras.layers.Conv2D(6, (1, 1), strides=(1, 1),
                                        padding='same',
                                        # kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    value_head = tf.keras.layers.BatchNormalization()(value_head)
    value_head = tf.keras.layers.Activation(tf.keras.layers.PReLU())(value_head)

    value_head = tf.keras.layers.Flatten()(value_head)
    # value_head = tf.keras.layers.GlobalAvgPool3D()(value_head)
    value_head = tf.keras.layers.Dense(640, activation="relu",
                                       # kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(512, activation="relu",
                                       # kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(256, activation="relu",
                                       # kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(128, activation="relu",
                                       # kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(1, activation="tanh", dtype='float32', name="value")(value_head)

    outputs = {"policy": policy_head, "value": value_head}
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
                  loss={"policy": "bce", "value": "mse"},
                  loss_weights={"policy": 1, "value": 2},
                  metrics=["accuracy", "bce", "mse"])
    model.summary()
    tf.keras.utils.plot_model(model,
                              to_file="resnet_2d.png",
                              show_dtype=True,
                              show_shapes=True,
                              # show_trainable=True,
                              show_layer_activations=True,
                              show_layer_names=True)
    return model
if __name__ == "__main__":
    model = Get_2dresnet(input_shape=gf.SHAPE[1:], policy_shape=gf.WIDTH*gf.HEIGHT, blocks=5)
    model.save("../alphazero/models/1")