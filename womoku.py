import numpy as np
from numba import njit, types
import numba as nb

USE_GPU = False
if USE_GPU:
    PROVIDERS = [
        ('TensorrtExecutionProvider',
         {"device_id": 0,
          "trt_fp16_enable": True,
          "trt_max_workspace_size": 0.5 * 1024 * 1024 * 1024,
          "trt_engine_cache_enable": True,
          "trt_engine_cache_path": "alphazero/trt_cache",
          # "trt_timing_cache_enable": True,
          # "trt_force_timing_cache": True,
          # "trt_build_heuristics_enable": True,
          }
         ),
        # 'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
    NP_DTYPE = np.int16
    TF_DTYPE = "INT16"
else:
    PROVIDERS = [
        'CPUExecutionProvider',
        # "OpenVINOExecutionProvider"
        # ("OpenVINOExecutionProvider",
        # {
        #     "device_type": "CPU_FP32",
        #     "num_of_threads": 1,
        # }
        #  )
    ]
    # PROVIDER_OPTIONS = [{"device_type": "CPU_FP32", "num_of_threads": 1}]
    # PROVIDER_OPTIONS = None
    NP_DTYPE = np.int8
    TF_DTYPE = "INT8"

# WIDTH and HEIGHT must be odd for consistency
WIDTH = 15
HEIGHT = 15
CONNECT = 5
# CONNECT can be any number for connect5, 6 or 7
# I'll train on 6 and 7 in the future

NUM_PREVIOUS_MOVES = 3
SHAPE = (1, NUM_PREVIOUS_MOVES + 1, HEIGHT, WIDTH, 1)

MAX_NODES_INFER = 5


class Womoku:
    def __init__(self, board: list = None):
        if board is None:
            self.board = [[0 for _ in range(WIDTH)] for __ in range(HEIGHT)]
        else:
            self.board = board

        self.moves = []

    def __repr__(self):
        return str(np.array(self.board, dtype=np.int8))

    def print_board(self):
        print(np.array(self.board))

    def put(self, move: tuple):
        x, y = move
        if self.board[y][x] != 0:
            raise ValueError(f"Move: {x, y} is not a legal move")

        self.board[y][x] = get_next_player(np.array(self.board))
        self.moves.append(move)


def put(input_board: np.array, move: tuple) -> tuple[list, tuple]:
    x, y = move
    board = [[*row] for row in input_board]
    if input_board[y][x] != 0:
        raise ValueError(f"Stupid, illegal move was attempted to be played")
    board[y][x] = get_next_player(np.array(input_board))
    return board


@njit(cache=True, fastmath=True, nogil=True)
def _puct_select(nodes_info, c_puct):
    best_score = -np.inf
    best_child_index = -1
    for node_id, (value, prob_prior, visits, parent_visits) in enumerate(nodes_info):
        if visits == 0:
            return node_id
        PUCT_score = value / visits + c_puct * prob_prior * ((parent_visits ** 0.5) / (visits + 1))
        if PUCT_score > best_score:
            best_score = PUCT_score
            best_child_index = node_id
    return best_child_index


@njit(cache=True, nogil=True, fastmath=True)
def get_next_player(input_board: np.array) -> int:
    """
    :param input_board: the input board, array of shape 15 by 15 representing the current position of the game
    :return: the next player to play
    """
    return -1 if sum([abs(place) for place in input_board.flat]) % 2 == 0 else 1


@njit(cache=True, fastmath=True)
def get_next_player_histo(move_histo: np.array) -> int:
    """
    :param move_histo: the previous moves
    :return: the next player to play
    """
    return -1 if move_histo.shape[0] % 2 == 0 else 1


@njit(cache=True, nogil=True, fastmath=True)
def get_current_player(input_board: np.array) -> int:
    """
    :param input_board:
    :return: the current player that has just played a move
    """
    return -get_next_player(input_board)


@njit([types.List(types.Tuple((nb.int64, nb.int64)))(nb.int8[:, :])], cache=True, nogil=True)
def get_legal_moves(input_board: np.array):
    """
    :param input_board:
    :return: list of legal moves [(x, y), ...]
    """
    return [(x, y) for y, x in np.argwhere(input_board == 0)]


@njit([types.Set(types.Tuple((nb.int64, nb.int64)))(nb.int8[:, :], nb.int64)], cache=True, nogil=True)
def get_edge_moves(input_board, player=0):
    edge_moves = set()
    if player == -1:
        moves = np.argwhere(input_board == -1)
    elif player == 1:
        moves = np.argwhere(input_board == 1)
    else:
        moves = np.concatenate((np.argwhere(input_board == -1), np.argwhere(input_board == 1)))
    for (y, x) in moves:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if WIDTH > new_x >= 0 == input_board[new_y, new_x] and 0 <= new_y < HEIGHT:
                    edge_moves.add((new_x, new_y))
    return edge_moves


@njit(cache=True, nogil=True, fastmath=True)
def check_won(input_board: np.array, move: tuple) -> int:
    """
    :param input_board:
    :param move: The original move/coordinate point to perform the octagonal pattern check
    :return: The winning player (-1, 1) a draw 1, or no winner -1
    """
    board = input_board
    current_x, current_y = move
    player = get_current_player(board)

    fives = 0
    for i in range(-CONNECT + 1, CONNECT):
        new_x = current_x + i
        if 0 <= new_x <= WIDTH - 1:
            if board[current_y][new_x] == player:
                fives += 1
                if fives == CONNECT:
                    return player
            else:
                fives = 0

    # vertical
    fives = 0
    for i in range(-CONNECT + 1, CONNECT):
        new_y = current_y + i
        if 0 <= new_y <= HEIGHT - 1:
            if board[new_y][current_x] == player:
                fives += 1
                if fives == CONNECT:
                    return player
            else:
                fives = 0

    #  left to right diagonal
    fives = 0
    for i in range(-CONNECT + 1, CONNECT):
        new_x = current_x + i
        new_y = current_y + i
        if 0 <= new_x <= WIDTH - 1 and 0 <= new_y <= HEIGHT - 1:
            if board[new_y][new_x] == player:
                fives += 1
                if fives == CONNECT:
                    return player
            else:
                fives = 0

    # right to left diagonal
    fives = 0
    for i in range(-CONNECT + 1, CONNECT):
        new_x = current_x - i
        new_y = current_y + i
        if 0 <= new_x <= WIDTH - 1 and 0 <= new_y <= HEIGHT - 1:
            if board[new_y][new_x] == player:
                fives += 1
                if fives == CONNECT:
                    return player
            else:
                fives = 0
    # if sum([abs(x) for x in input_board.flat]) == WIDTH * HEIGHT:
    #     return 0
    # remember that draw is very unlikely

    # if there is no winner, and it is not a draw
    return -2


@njit(cache=True, nogil=True)
def find_term_expand(input_board: np.array, next_player=None) -> list:
    """
    :param next_player:
    :param input_board:
    :return: The winning moves that the expand method uses to bottleneck the tree
    """
    if next_player is None:
        next_player = get_next_player(input_board)
    terminal_moves = []
    for x, y in get_edge_moves(input_board, player=next_player):
        input_board[y][x] = next_player
        won_player = check_won(input_board, (x, y))
        input_board[y][x] = 0
        if won_player == -1 or won_player == 1:  # if the winner is -1 or 1
            terminal_moves.append(((x, y), True))
        elif won_player == 0:
            terminal_moves.append(((x, y), False))
    return terminal_moves


@njit(cache=True, nogil=True, fastmath=True)
def find_win_simulate(input_board: np.array) -> tuple:
    """
    :param input_board:
    :return: (-1, -1) means no winning move found. (x, y)
    """
    next_player = get_next_player(input_board)
    for x, y in get_edge_moves(input_board, width=1, player=next_player):
        input_board[y][x] = next_player
        won_player = check_won(input_board, (x, y))
        input_board[y][x] = 0
        if won_player != -2:
            return x, y
    return -1, -1


@njit(cache=True, nogil=True, parallel=False, fastmath=True)
def simulate_jit(state, max_moves, simulations_per_node) -> int:
    # wins = [1 for _ in range(simulations_per_node)]
    wins = 0
    node_player = get_current_player(state)
    for i in range(simulations_per_node):
        simulation_state = np.copy(state)
        legal_moves = get_edge_moves(simulation_state, width=1)
        for _ in range(max_moves):
            if legal_moves:
                move = find_win_simulate(simulation_state)
                if move == (-1, -1):
                    move = legal_moves[int(np.random.choice(len(legal_moves)))]
                simulation_state[move[1]][move[0]] = get_next_player(simulation_state)
                legal_moves = get_edge_moves(simulation_state, width=1)
                won_player = check_won(simulation_state, move)
                if won_player != -2:
                    if won_player == node_player:
                        wins += 1
                    elif won_player == -node_player:
                        wins -= 1
                    break
            else:
                break
    return wins


@njit(nogil=True)
def get_inference_board_old(input_move_history: np.array, num_previous_moves: int = NUM_PREVIOUS_MOVES):
    """
    :param input_move_history: The move history. [(x, y), ...]
    :param num_previous_moves: Number of previous moves to add to the game state
    :return: A specific array of shape (1, num_previous_moves*1 + 1, HEIGHT, WIDTH, 1). Used to train and inference.
    # np.array([current_player board, black_board, white_board], dtype=np.int8).reshape(((1, num_previous_moves*1 + 1, HEIGHT, WIDTH, 1))
    #  A specific part for forward and backpropagation
    """
    black_board_history = []
    white_board_history = []
    black_board = [[0 for _ in range(WIDTH)] for __ in range(HEIGHT)]
    white_board = [[0 for _ in range(WIDTH)] for __ in range(HEIGHT)]

    for i, [x, y] in enumerate(input_move_history):
        if i != 0:
            if (i - 1) % 2 == 0:  # if it is white's turn
                black_board[y][x] = -1
                black_board_history.append([[*row] for row in black_board])
            else:
                white_board[y][x] = 1
                white_board_history.append([[*row] for row in white_board])

    move_history = input_move_history[1:]

    current_player = -1 if move_history.shape[0] % 2 == 0 else 1

    current_player_board = [[[current_player for _ in range(WIDTH)] for __ in range(HEIGHT)]]

    if len(white_board_history) >= num_previous_moves:
        white_board = white_board_history[-num_previous_moves:]
    elif not white_board_history:
        white_board = [[[0 for _ in range(WIDTH)] for __ in range(HEIGHT)] for ___ in range(num_previous_moves)]
    else:
        white_board = [[[0 for _ in range(WIDTH)] for __ in range(HEIGHT)] for ___ in
                       range(num_previous_moves - len(white_board_history))] + white_board_history

    if len(black_board_history) >= num_previous_moves:
        black_board = black_board_history[-num_previous_moves:]
    elif not black_board_history:
        black_board = [[[0 for _ in range(WIDTH)] for __ in range(HEIGHT)] for ___ in range(num_previous_moves)]
    else:
        # print(np.array(black_board_history).shape)
        black_board = [[[0 for _ in range(WIDTH)] for __ in range(HEIGHT)] for ___ in
                       range(num_previous_moves - len(black_board_history))] + black_board_history
    game_state = np.array([current_player_board + black_board + white_board],
                          dtype=NP_DTYPE).reshape((1, num_previous_moves * 2 + 1, HEIGHT, WIDTH, 1))
    return game_state


@njit([nb.int8[:, :, :, :, :](nb.int8[:, :], nb.int64), nb.int16[:, :, :, :, :](nb.int8[:, :], nb.int64)], cache=True,
      nogil=True, fastmath=True)
def get_inference_board(input_move_history: np.array, num_previous_moves: int = NUM_PREVIOUS_MOVES):
    inference_board = []
    board = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]

    for i, (x, y) in enumerate(input_move_history[1:]):
        if i % 2 == 0:
            board[y][x] = -1
        else:
            board[y][x] = 1
        inference_board.append(
            [[*row] for row in board])  # the fastest way to deepcopy knowing the structure of the list

    current_player = 1 if input_move_history.shape[0] % 2 == 0 else -1
    num_placed_moves = len(inference_board)
    if num_placed_moves >= num_previous_moves:
        inference_board = np.array(inference_board, dtype=NP_DTYPE)[-num_previous_moves:]
        return np.concatenate(
            (np.full(shape=(1, HEIGHT, WIDTH), fill_value=current_player, dtype=NP_DTYPE), inference_board)).reshape(
            SHAPE)
    elif num_placed_moves < num_previous_moves and num_placed_moves != 0:
        inference_board = np.array(inference_board, dtype=np.int8)
        return np.concatenate((np.full(shape=(1, HEIGHT, WIDTH), fill_value=current_player, dtype=NP_DTYPE),
                               np.zeros(shape=(num_previous_moves - num_placed_moves, HEIGHT, WIDTH), dtype=NP_DTYPE),
                               inference_board)).reshape(SHAPE)
    else:
        return np.concatenate((np.full(shape=(1, HEIGHT, WIDTH), fill_value=current_player, dtype=NP_DTYPE),
                               np.zeros(shape=(num_previous_moves, HEIGHT, WIDTH), dtype=NP_DTYPE))).reshape(SHAPE)


@njit(cache=True, nogil=True, fastmath=True)
def parse_policy(raw_policy: np.array, game_state: np.array, cutoff: int or None = WIDTH * HEIGHT) -> list:
    """
    :param raw_policy: The raw board output from the NN
    :param game_state: The current position of the board from inferencing, used to remove illegal moves
    :param cutoff: The top moves to use, any below the top cutoff moves are thrown out
    #  top moves are then re-normalized to sum to 1
    :return:
    """
    raw_policy = raw_policy[0]
    policy = []
    # given an array of shape 15, 15 rank the moves x,y by their numbers, including the values [x, y, value]
    sorted_indices = np.argsort(raw_policy, kind="mergesort")[:: -1]
    sum_values = 0
    for idx in sorted_indices:
        x, y = idx % WIDTH, idx // HEIGHT
        if game_state[y][x] == 0:
            prob_prior = raw_policy[y * HEIGHT + x]
            policy.append(((x, y), prob_prior, False))
            sum_values += prob_prior

            # if cutoff is not None and len(policy) >= cutoff:
            #     break
    # normalize the policy distribution
    policy = [(move, prob_prior / sum_values, False) for move, prob_prior, _ in policy]

    return policy


def prob_to_board(prob_dis: list[tuple, float, int, int, int, int], board: list):
    """
    :param prob_dis: line produced by the MCTS + NN. [(x, y), probability, node visits, root visits, accumulated
    value, probability prior]
    # node value is the raw accumulated value, to get win rate so node.value/node.visits
    :param board: :return: A list where the moves, probability are translated to a board + probability for x, y on the board
    """
    prob_board = [[0 for _ in range(WIDTH)] for __ in range(HEIGHT)]
    for (x, y), prob, _, _, _, _ in prob_dis:
        if board[y][x] == 0:
            prob_board[y][x] = prob
        else:
            print("Illegal move found while converting to board distribution")
            print("Stupid")
            print(x)
    return prob_board


if __name__ == "__main__":
    import time
    import timeit
    from numba.typed import List

    # import llvmlite.binding as llvm

    # llvm.set_option('', '--debug-only=loop-vectorize')

    game = Womoku()
    game.put((7, 7))
    game.put((5, 6))
    game.put((7, 8))
    game.put((9, 7))
    game.put((7, 6))
    game.put((6, 6))
    game.put((7, 5))
    game.put((6, 5))
    game.put((7, 4))
    game.put((6, 4))
    # game.put((7, 3))
    #
    # game.put((5, 5))
    # game.put((3, 3))
    game.print_board()
    # game.put(())
    import random

    # dummy_data = [list(set([(random.randint(0, 14), random.randint(0, 14)) for _ in range(250)]))
    #               for _ in range(1)]
    dummy_data = np.array(game.board, dtype=np.int8)
    s = time.time()
    for _ in range(10):
        x = get_legal_moves(dummy_data)
        get_legal_moves1(dummy_data)
    print(time.time() - s)
    s = time.time()
    for _ in range(1000):
        get_legal_moves(dummy_data)
    print(time.time() - s)
    s = time.time()
    for _ in range(1000):
        get_legal_moves1(dummy_data)
    print(time.time() - s)
    print(x)

    # for y in range(5):
    #     for x in range(5):
    #         # x += 5
    #         # y += 3
    #         game.put((x, y))
    #         for _ in range(10):
    #             x = get_inference_board2(np.array([(-1, -1)] + game.moves, dtype=np.int8))
    #         for _ in range(10):
    #             x = get_inference_board(np.array([(-1, -1)] + game.moves, dtype=np.int8))
    # game = Womoku()
    # equal = []
    # f1 = []
    # f2 = []
    # for y in range(5):
    #     for x in range(5):
    #         # x += 5
    #         # y += 3
    #         game.put((x, y))
    #         equal.append(np.array_equal(get_inference_board2(np.array([(-1, -1)] + game.moves, dtype=np.int8)),
    #                                     get_inference_board(np.array([(-1, -1)] + game.moves, dtype=np.int8))))
    #
    #         s = time.time()
    #         for _ in range(10):
    #             x = get_inference_board2(np.array([(-1, -1)] + game.moves, dtype=np.int8))
    #         f2.append(time.time() - s)
    #         s = time.time()
    #         for _ in range(10):
    #             x = get_inference_board(np.array([(-1, -1)] + game.moves, dtype=np.int8))
    #         f1.append(time.time() - s)
    #
    # print(sum(f1), sum(f2))
    # delta = np.array(f2) - np.array(f1)
    # print(sum(delta))
    #
    # print(equal)
    # print(all(equal))

    # print(x)
    # print()
    # print(y)
    # print(get_next_player_hist(np.array(game.moves)))
    # get_next_player.compile((np.array(game.board)))
    # print(check_won.signatures)
    # print('svml' in check_won.inspect_asm(check_won.signatures))
    # print('intel_svmlcc' in check_won.inspect_llvm(check_won.signatures[0]))
    # game.put((4, 4))
    # game.put((1, 1))
    # game.put((3, 3))
    # game.put((1, 1))
    # game.put((5, 5))
    # game.put((3, 1))
    # game.put((6, 6))
    # game.put((4, 1))
    # game.print_board()

    # for y in range(9):
    #     for x in range(9):
    #         game.put((x, y))
    # data = np.array([(-1, -1)] + game.moves)
    # start = time.time()
    #
    # for _ in range(100):
    #     x = get_inference_board(data)
    #     x = get_inference_board_old(data)
    #
    # for _ in range(10000):
    #     x = get_inference_board(data)
    # print(time.time() -start)
    #
    # start = time.time()
    # for _ in range(10000):
    #     x = get_inference_board_old(data)
    # print(time.time() - start)
    # print(get_inference_board(np.array([(-1, -1)] + game.moves)).shape)
    # print(check_won(np.array(game.board), game.moves[-1]))
    # print(find_win_simulate(np.array(game.board)))
    # print(get_edge_moves(np.array(game.board), width=1))

    # import tensorflow as tf

    # model = tf.keras.models.load_model("alphazero/models/1")
    # output = model.predict(np.ones(SHAPE))
    # np_board = np.array(game.board)

    # print(inference_board.reshape((11, 9, 9)))
    # # print("-" * 30)
    # print(original.reshape((11, 9, 9)))

    # raise ValueError
    # for _ in range(100):
    #     parse_policy(output["policy"], np.array(game.board))
    #
    # start = time.time()
    # for _ in range(10000):
    #     parse_policy(output["policy"], np.array(game.board))
    # print(time.time() - start)
    # game.put((3, 4))
    # game.put((4, 3))
    # game.put((3, 3))
    # game.put((4, 1))
    # game.put((3, 1))
    # print(find_win_simulate(np.array(game.board)))
    # for _ in range(1):
    #     simulate_jit(np.array(game.board), 100, 100)
    # start = time.time()
    # simulate_jit(np.array(game.board), 100, 100)
    # print(time.time() - start)

    # print(((WIDTH - 1) // 1, (HEIGHT - 1) // 1))
    # print(get_inference_board(game.moves).reshape(11, 9, 9))
