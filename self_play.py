import os
import time
import random
import pickle
import zipfile

import zipfile
import numpy as np
from glob import glob
import onnxruntime as rt
from copy import deepcopy

import womoku as gf
from womoku import Womoku
from MCTS_virtualloss import MCTS

from joblib import Parallel, delayed

try:
    generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
except:
    generation = 0
os.makedirs(f"alphazero/game_moves/{generation + 1}", exist_ok=True)
os.makedirs(f"alphazero/games_unrot/{generation + 1}", exist_ok=True)


class Self_play:
    def __init__(self, session, worker_id, max_generations):
        self.game = Womoku()

        self.random_num = random.randrange(0, 10)
        # if random_num in [1, 1]:
        #     self.game.put((7, 7))
        #     self.game.put(random.choice([(6, 6), (7, 6), (7, 5), (8, 6), (5, 6), (6, 5), (8, 5), (9, 6)]))
        #     self.game.put(random.choice([(6, 1), (7, 1), (8, 1)]))
        # elif random_num == 3:
        #     self.game.put((7, 7))
        #     self.game.put((7, 5))
        #     self.game.put((6, 1))
        # elif random_num == 4:
        #     self.game.put((7, 7))
        #     self.game.put(random.choice([(6, 6), (7, 6), (8, 6), (5, 6), (6, 5), (8, 5), (9, 6)]))
        # elif random_num in [5, 6]:
        #     self.game.put((7, 7))
        #     self.game.put(random.choice([(6, 6), (7, 6), (7, 5), (8, 6), (5, 6), (6, 5), (8, 5), (9, 6)]))

        self.max_generations = max_generations
        self.generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
        self.worker_id = worker_id

        # max_iterations = 1000
        # min_iterations = 500
        # self.iterations = int(min_iterations + (((max_iterations - min_iterations) * (self.generation / self.max_generations))))
        self.time_limit = 8
        self.iterations = 1000

        max_c_puct = 4.5
        min_c_puct = 4
        c_puct = max_c_puct - ((max_c_puct - min_c_puct) * (self.generation / self.max_generations))

        self.mcts1 = MCTS(self.game, session, c_puct=c_puct, explore=False, tau=1)
        self.mcts2 = MCTS(self.game, session, c_puct=c_puct, explore=False, tau=1)

        self.board_state_history = []
        self.board_prob_history = []

    def play(self, i):
        output_policy = []
        output_value = []
        won_player = -2

        while won_player == -2:
            if len(self.game.moves) in []:
                self.iterations = 500
                self.time_limit = None
            else:
                self.iterations = 800
                self.time_limit = None

            self.board_state_history.append(
                np.array(gf.get_inference_board(np.array([(-1, -1)] + self.game.moves, dtype=np.int8)).reshape(gf.SHAPE[1:])))
            # inputs for the NN
            if gf.get_next_player(np.array(self.game.board, dtype=np.int8)) == -1:
                move, line = self.mcts1.run(iteration_limit=self.iterations, time_limit=self.time_limit)
            else:
                move, line = self.mcts2.run(iteration_limit=self.iterations, time_limit=self.time_limit)

            if self.random_num <= 4:
                if len(self.game.moves) == 0:
                    move = (7, 7)
#                 elif len(self.game.moves) == 1:
#                     move = random.choice([(6, 6), (7, 6), (7, 5), (8, 6), (5, 6), (6, 5), (8, 5), (9, 6)])

            print(f"Worker:{self.worker_id} | Generation: {self.generation + 1} | Move: {len(self.game.moves)} | Current Player: {gf.get_next_player(np.array(self.game.board, dtype=np.int8))}")
            print(move, line)

            self.board_prob_history.append(gf.prob_to_board(line, self.game.board))  # output π for the NN
            self.game.put(move)

            self.game.print_board()
            self.mcts1.update_tree(self.game, move)
            self.mcts2.update_tree(self.game, move)

            if 5 >= len(self.game.moves) >= 1:
                self.mcts1.update_hyperparameters(explore=True, c_puct=-0.025, tau=-0.1)
                self.mcts2.update_hyperparameters(explore=True, c_puct=-0.025, tau=-0.1)
            if len(self.game.moves) > (5 + random.randrange(0, 6)):
                self.mcts1.tau = 1e-2
                self.mcts2.tau = 1e-2

            won_player = gf.check_won(np.array(self.game.board), move)

        if won_player == -1:
            print("Player 1 WON")
        elif won_player == 1:
            print("Player 2 WON")
        else:
            print("Game was a draw")

        for player, prob_dis in enumerate(self.board_prob_history):
            if player % 2 == 0:  # if black is the current player
                output_policy.append(prob_dis)
                if won_player == -1:  # and if black won
                    output_value.append(1)
                elif won_player == 1:  # and if white won
                    output_value.append(-1)
                else:
                    output_value.append(0)

            else:  # if white is the current player
                output_policy.append(prob_dis)
                if won_player == 1:  # and white won
                    output_value.append(1)
                elif won_player == -1:  # and black won
                    output_value.append(-1)
                else:
                    output_value.append(0)

        board_inputs = np.array(self.board_state_history, dtype=np.int8).reshape((-1, *gf.SHAPE[1:]))
        policy = np.array(output_policy, dtype=np.float32).reshape(-1, gf.HEIGHT * gf.WIDTH)
        value = np.array(output_value, dtype=np.float32).reshape(-1, 1)

        game_num = len(glob(f"alphazero/games_unrot/{self.generation + 1}/cpu2_sp_{self.worker_id}_*.npz"))
        with open(f"alphazero/game_moves/{self.generation + 1}/cpu2_{self.worker_id}_{game_num}.pickle", "wb") as f:
            pickle.dump(self.game.moves, f)

        for tries in range(10):
            try:
                np.savez_compressed(
                    f'alphazero/games_unrot/{self.generation + 1}/cpu2_sp_{self.worker_id}_{game_num}.npz',
                    allow_pickle=True,
                    inputs=board_inputs,
                    policy=policy,
                    value=value)

                # Try to load the file to verify it was written correctly
                data = np.load(f'alphazero/games_unrot/{self.generation + 1}/cpu2_sp_{self.worker_id}_{game_num}.npz',
                               allow_pickle=True)
                dummy_inputs = data["inputs"]
                dummy_policy = data["policy"]
                dummy_value = data["value"]

                del dummy_inputs, dummy_policy, dummy_value
                break

            except zipfile.BadZipfile:
                print(f"Failed try {tries}")
                print(f'alphazero/games_unrot/{self.generation + 1}/sp_{self.worker_id}_{game_num}.npz')

        del self.game, self.mcts1, self.mcts2, self.board_state_history, self.board_prob_history


def self_play_func(amount_games, max_generations, worker_id):
    time.sleep(worker_id * 2)
    generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = f"alphazero/onnx_optimized/{generation}.onnx"
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    session = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                                  providers=gf.PROVIDERS, sess_options=sess_options)
    for game_num in range(amount_games):
        # try:
        self_player = Self_play(session, worker_id, max_generations)
        self_player.play(game_num)
        # except:
        #     print(f"MCTS_onnx worker {worker_id} failed")
        del self_player
    print(f"Worker: {worker_id} finished its games")


def Gen_games(max_generations, total_games, num_workers, timeout=None):
    # there should always be a model in f"alphazero/models/* when this is called
    try:
        generation = max([int(path.split("\\")[-1][0:]) for path in glob(f"alphazero/models/*")])
    except:
        generation = 0

    played_games = int(len(glob(f"alphazero/games_unrot/{generation + 1}/*.npz")))
    remaining_games = total_games - played_games
    print(f"Played {played_games} games")
    print(f"Total remaining games: {remaining_games} games")
    Parallel(n_jobs=num_workers, backend="multiprocessing", timeout=timeout)(
        delayed(self_play_func)(remaining_games // num_workers, max_generations, i) for i in range(num_workers))


if __name__ == "__main__":
    from joblib import Parallel, delayed
    import onnxruntime as rt

    workers = 1
    total_games = 1500

    Gen_games(max_generations=25, total_games=total_games, num_workers=workers)
