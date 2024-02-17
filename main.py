import os
import gc
import time
from glob import glob

import womoku as gf
from womoku import Womoku

import tensorflow as tf
# from net.densenet2d import Get_2ddensenet
from net.resnet2d import Get_2dresnet
from onnx_convert import Convert2onnx

# from joblib import Parallel, delayed
# import onnxruntime as rt
# from MCTS_ONNX import MCTS
from build_trt_cache import Build_trt_cache
from self_play import Gen_games
from self_play_mcts import Gen_mcts_games
from reanalysis import Gen_reanalysis
from augment_data import Augment_data
from train import Train_net

import multiprocessing as mp

def create_convert(generation):
    model = Get_2dresnet(input_shape=gf.SHAPE[1:], policy_shape=81, blocks=4)
    model.save(f"alphazero/models/{generation}", overwrite=True)
    Convert2onnx(model, generation)

def load_convert(generation):
    model = tf.keras.models.load_model(f"alphazero/models/{generation}")
    Convert2onnx(model, generation)

if __name__ == "__main__":
    if gf.USE_GPU:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[1], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    workers = 5
    timeout = 999999
    max_generations = 15
    games_per_gen = 1000
    games_per_worker_re = 10

    generation = 0

    restart = False
    if not restart:
        # create the initial model
        print(f"input_shape={gf.SHAPE[1:]}, policy_shape={gf.WIDTH * gf.HEIGHT}")

        # ignore the 1st value in input_shape because it is
        # the batch size

        p = mp.Process(target=create_convert, args=[generation])
        p.start()
        p.join()

        gc.collect()


        if gf.USE_GPU:
            print("Building TRT cache")
            p = mp.Process(target=Build_trt_cache)
            p.start()
            p.join()

        gc.collect()

        Gen_mcts_games(max_generations=max_generations, total_games=10000, num_workers=12)

        gc.collect()
        Augment_data()
        gc.collect()

        p = mp.Process(target=Train_net, args=[max_generations, 3e-4])
        p.start()
        p.join()
        gc.collect()
        #


    for i in range(max_generations):
        generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
        print(f"Generating model: {generation}")
        # convert to onnx

        p = mp.Process(target=load_convert, args=[generation])
        p.start()
        p.join()

        gc.collect()


        if gf.USE_GPU:
            print("Building TRT cache")
            p = mp.Process(target=Build_trt_cache)
            p.start()
            p.join()
            print("\n")

        gc.collect()
        os.makedirs(f"alphazero/game_moves/{generation + 1}", exist_ok=True)
        os.makedirs(f"alphazero/games_unrot/{generation + 1}", exist_ok=True)

        Gen_games(max_generations=max_generations, total_games=games_per_gen, num_workers=workers)
        gc.collect()
        if generation not in [0, 1, 2, 3, 4, 5]:
            Gen_reanalysis(max_games_per_worker=games_per_worker_re, max_generations=max_generations, num_workers=workers)
        gc.collect()

        Augment_data()
        gc.collect()

        learning_rate = 1.5e-4 - (((generation + 1)/max_generations) * (1.5e-4 - 5e-5))
        print(f"Current learning rate at: {learning_rate}")
        p = mp.Process(target=Train_net, args=[learning_rate])
        p.start()
        p.join()

        gc.collect()
