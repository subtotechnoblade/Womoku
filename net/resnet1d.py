# import connect2
import tensorflow as tf


def Get_1dresnet(input_shape, policy_shape=4, blocks=4, alpha=5e-5):
    def residual_block(input_tensor, filters, kernel_size=1, strides=1, activation='relu'):
        # Apply a 1x1 convolutional layer to reduce the number of filters in the input
        residue = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides,
                                         kernel_initializer="he_normal",
                                         kernel_regularizer=tf.keras.regularizers.L2(alpha))(input_tensor)
        # Apply the main convolutional layers
        resnet = tf.keras.layers.Conv1D(filters, 1, padding='same',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(input_tensor)
        resnet = tf.keras.layers.BatchNormalization()(resnet)
        resnet = tf.keras.layers.Activation(activation)(resnet)

        resnet = tf.keras.layers.Conv1D(filters, 3, padding='same',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(resnet)
        resnet = tf.keras.layers.BatchNormalization()(resnet)

        resnet = tf.keras.layers.Conv1D(filters, 1, padding='same', strides=strides,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(resnet)
        # Add the inputs to the output of the block
        resnet = tf.keras.layers.Add()([resnet, residue])
        resnet = tf.keras.layers.Activation(activation)(resnet)
        return resnet

    inputs = tf.keras.layers.Input(shape=input_shape, dtype="float16")

    # Apply a convolutional layer with 32 filters and a kernel size of 3x3
    x = tf.keras.layers.Conv1D(512, 5, padding='same',
                               kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.L2(alpha))(inputs)

    for i in range(blocks):
        x = residual_block(x, 256, kernel_size=1, strides=1)

    policy_head = tf.keras.layers.Conv1D(8, 2, activation='relu', padding="same", kernel_initializer="he_normal",
                                         kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    policy_head = tf.keras.layers.BatchNormalization()(policy_head)
    policy_head = tf.keras.layers.Activation("relu")(policy_head)

    policy_head = tf.keras.layers.Flatten()(policy_head)
    policy_head = tf.keras.layers.Dense(128, activation='relu',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    policy_head = tf.keras.layers.Dense(128, activation='relu',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    policy_head = tf.keras.layers.Dense(64, activation='relu',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(policy_head)
    policy_head = tf.keras.layers.Dense(policy_shape, activation='softmax', dtype='float32', name="policy")(policy_head)

    value_head = tf.keras.layers.Conv1D(16, 2, activation='relu', padding='same',
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.L2(alpha))(x)
    value_head = tf.keras.layers.BatchNormalization()(value_head)
    value_head = tf.keras.layers.Activation("relu")(value_head)


    value_head = tf.keras.layers.Flatten()(value_head)
    value_head = tf.keras.layers.Dense(128, activation='relu',
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(128, activation='relu',
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.L2(alpha))(value_head)
    value_head = tf.keras.layers.Dense(64, activation='relu',
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
    return model
