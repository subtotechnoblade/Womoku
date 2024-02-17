import os
import random
import pickle
import numpy as np
from glob import glob
import zipfile

import onnxruntime as rt
from MCTS_test2 import MCTS

import womoku as gf
from womoku import Womoku

from joblib import Parallel, delayed


class Re_analysis:
    def __init__(self, worker_id, max_games_per_worker, max_generations):
        self.game = Womoku()

        self.max_generations = max_generations
        self.generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
        self.worker_id = worker_id

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # sess_options.optimized_model_filepath = f"alphazero/onnx_optimized/{self.generation}_{worker_id}.onnx"

        self.session = rt.InferenceSession(f"alphazero/onnx_models/{self.generation}.onnx",
                                               providers=gf.PROVIDERS, sess_options=sess_options)

        os.makedirs(f"alphazero/games_unrot/{self.generation + 1}", exist_ok=True)

        game_generation = [int(path.split("\\")[-1]) for path in glob("alphazero/game_moves/*")]
        game_generation.sort()
        game_generation = game_generation[-6:-2]
        print(f"Generation that can be reanalyzed are: {game_generation}")
        # game_generation = random.choice(game_generation)
        game_generation = game_generation[worker_id]
        print(
            f"Worker: {worker_id} is searching generation: {game_generation} saving in generation {self.generation + 1}")
        self.pickle_files = glob(f"alphazero/game_moves/{game_generation}/*.pickle")
        random.shuffle(self.pickle_files)
        self.pickle_files = np.random.choice(self.pickle_files, size=max_games_per_worker)


    def analyze(self):
        for game_num, path in enumerate(self.pickle_files):
            with open(path, "rb") as f:
                game_moves = pickle.load(f)
            print(f"Reanalyzing {path}")
            self.game = Womoku()

            max_c_puct = 4.5
            min_c_puct = 4
            c_puct = max_c_puct - ((max_c_puct - min_c_puct) * ((self.generation + 1) / self.max_generations))

            self.mcts1 = MCTS(self.game, self.session, self.worker_id, c_puct=c_puct, explore=False)
            self.mcts2 = MCTS(self.game, self.session, self.worker_id, c_puct=c_puct, explore=False)
            self.board_state_history = []
            self.board_prob_history = []

            output_policy = []
            output_value = []

            won_player = -2
            for move in game_moves:
                self.board_state_history.append(gf.get_inference_board(np.array([(-1, -1)] + self.game.moves)).reshape(gf.SHAPE[1:-1]))

                if gf.get_next_player(np.array(self.game.board)) == -1:
                    _, line = self.mcts1.run(time_limit=None, iteration_limit=1500)
                else:
                    _, line = self.mcts2.run(time_limit=None, iteration_limit=1500)
                print(move, line)
                self.board_prob_history.append(gf.prob_to_board(line, self.game.board))
                self.game.put(move)

                self.game.print_board()

                self.mcts1.update_tree(self.game, move)
                self.mcts2.update_tree(self.game, move)

                if 15 > len(self.game.moves) > 10:
                    self.mcts1.update_hyperparameters(c_puct=-0.025)
                    self.mcts2.update_hyperparameters(c_puct=-0.025)

                won_player = gf.check_won(np.array(self.game.board), move)
                if won_player != -2:
                    break
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
            policy = np.array(output_policy, dtype=np.float32).reshape(-1, gf.WIDTH*gf.HEIGHT)
            value = np.array(output_value, dtype=np.float32).reshape(-1, 1)
            latest_game_num = len(glob(f"alphazero/games_unrot/{self.generation + 1}/re_{self.worker_id}_*.npz"))
            for tries in range(10):
                try:
                    np.savez_compressed(
                        f'alphazero/games_unrot/{self.generation + 1}/re_{self.worker_id}_{latest_game_num}.npz',
                        inputs=board_inputs,
                        policy=policy,
                        value=value)

                    # Try to load the file to verify it was written correctly
                    data = np.load(f'alphazero/games_unrot/{self.generation + 1}/re_{self.worker_id}_{latest_game_num}.npz',
                                   allow_pickle=True)
                    dummy_inputs = data["inputs"]
                    dummy_policy = data["policy"]
                    dummy_value = data["value"]

                    del dummy_inputs, dummy_policy, dummy_value
                    break

                except zipfile.BadZipfile:
                    print(f"Failed try {tries}")
                    print(f'alphazero/games_unrot/{self.generation + 1}/re_{self.worker_id}_{latest_game_num}.npz')
            del self.game, self.mcts1, self.mcts2, self.board_prob_history, self.board_state_history


def run_analyze(max_games_per_worker, max_generations, worker_id):
    analysis = Re_analysis(worker_id=worker_id, max_games_per_worker=max_games_per_worker, max_generations=max_generations)
    # try:
    analysis.analyze()
    # except:
    #     print(f"Worker:{worker_id} failed reanalyzing")
    print(f"Worker:{worker_id} has finished reanalyzing")


def Gen_reanalysis(max_games_per_worker, max_generations, num_workers, timeout=None):
    generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
    amount_reanalyzed_games = len(glob(f"alphazero/games_unrot/{generation + 1}/re_*"))
    print(f"Starting reanalysis!, Reanalyzed {amount_reanalyzed_games} games already")
    print(f"Remaining games are: {num_workers * max_games_per_worker - amount_reanalyzed_games}")
    if max_games_per_worker * num_workers > amount_reanalyzed_games:
        games_per_worker = (num_workers * max_games_per_worker - amount_reanalyzed_games) // num_workers
        Parallel(n_jobs=num_workers, backend="multiprocessing", timeout=timeout)(
            delayed(run_analyze)(games_per_worker, max_generations, i) for i in range(num_workers))


if __name__ == "__main__":
    Gen_reanalysis(1500, 20, 1)
