import os
import numpy as np
from glob import glob
from tqdm import tqdm
import gc

import womoku as gf

import time
import random

from net.resnext2d import Get_2dresnext
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split


# model 1 -> generation 1 self play games -> trains model 1 -> gen 1 games -> ect.
def Train_net(max_generations, lr=1e-4):
    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(12)
    tf.config.threading.set_inter_op_parallelism_threads(12)

    if gf.USE_GPU:
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        assert tf.config.experimental.get_memory_growth(physical_devices[0])

        # tf.config.experimental.set_virtual_device_configuration(physical_devices[1], [
        #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3548)])
    generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
    print(f"loading generation :{generation + 1}")
    prev_2 = np.random.choice(glob(f"alphazero/games/{generation - 1}/*.npz"), 250).tolist()
    prev_1 = np.random.choice(glob(f"alphazero/games/{generation}/*.npz"), 500).tolist()
    alpha_files = glob(f"alphazero/games/{generation + 1}/*.npz") + prev_1 + prev_2

    x_train, policy_labels, value_labels = [0 for _ in range(len(alpha_files))], [0 for _ in range(len(alpha_files))], [
        0 for _ in range(len(alpha_files))]

    for id, file in enumerate(tqdm(alpha_files)):
        data = np.load(file, allow_pickle=True)
        x_train[id] = data["inputs"]
        policy_labels[id] = data["policy"]
        value_labels[id] = data["value"]

    x_train = np.concatenate(x_train).reshape((-1, 4, 15, 15, 1))
    policy_labels = np.concatenate(policy_labels).reshape((-1, 225))
    value_labels = np.concatenate(value_labels).reshape((-1, 1))

    print(x_train.shape, policy_labels.shape, value_labels.shape)

    x_train, x_test, policy_labels, policy_test, value_labels, value_test = train_test_split(x_train, policy_labels,
                                                                                             value_labels,
                                                                                             test_size=0.05,
                                                                                             shuffle=True)
    print(x_train.shape, policy_labels.shape, value_labels.shape)

    player1_wins = 0
    player2_wins = 0
    game_length = []
    for file in alpha_files:
        try:
            data = np.load(file, allow_pickle=True)
        except:
            data = np.load(file, allow_pickle=False)

        if data["value"][0][0] == 1:
            player1_wins += 1
        else:
            player2_wins += 1
        game_length.append(len(data["value"]))
    print(player1_wins, player2_wins)
    print(sum(game_length) / len(game_length))

    model = tf.keras.models.load_model(f"alphazero/models/{generation}", compile=False)

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=lr),
        loss={"policy": "bce", "value": "mse"},
        loss_weights={"policy": 1, "value": 1},
        metrics=["accuracy", "binary_crossentropy", "mse", ],
        # jit_compile=True,
    )
    # epochs = 3 if generation == 0 else 1
    # print(f"Training with {epochs} epochs")
    model.fit(x_train,
              {
                  "policy": policy_labels,
                  "value": value_labels
              },
              epochs=2,
              batch_size=2048,
              shuffle=True,
              validation_data=(x_test, {"policy": policy_test, "value": value_test})
              )

    model.save(f"alphazero/models/{generation + 1}")


if __name__ == "__main__":
    import tensorflow as tf
    import time
    import time
    import multiprocessing as mp

    p = mp.Process(target=Train_net, args=[20, 5e-4])
    p.start()
    p.join()

    # generation = 0
    # model = tf.keras.models.load_model(f"alphazero/models/{generation}")
    #
    # initial_sparsity = 0.0
    # final_sparsity = 0.8
    # begin_step = 100
    # end_step = 500
    # pruning_params = {
    #     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
    #         initial_sparsity=initial_sparsity,
    #         final_sparsity=final_sparsity,
    #         begin_step=begin_step,
    #         end_step=end_step)
    # }
    # model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    # pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
    #
    # model.compile(
    #     optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),
    #     loss={"policy": tf.keras.losses.kl_divergence, "value": "mse"},
    #     loss_weights={"policy": 1, "value": 1},
    #     metrics=["accuracy", "binary_crossentropy", "mse", tf.keras.metrics.kl_divergence],
    #     # jit_compile=True,
    # )
