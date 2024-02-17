import time
import math
import random
import numpy as np
from tqdm import tqdm
from collections import deque
import onnxruntime as rt

import womoku as gf
from womoku import Womoku
from copy import deepcopy


# game functions

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node:
    def __init__(self, state, move, move_history=[], prob_prior=0, parent=None, terminal=None):
        self.state = state
        self.move = move
        self.move_history = move_history
        self.parent = parent
        # None means not terminal, -1 mean player -1 won and 1 means player 1 won
        self.is_terminal = terminal

        self.children = []

        self.visits = 0  # N

        self.value = 0
        self.prob_prior = prob_prior  # P


class MCTS:
    def __init__(self,
                 game, session, worker_id=0,
                 move=None,
                 explore=False,
                 c_puct=4.5, tau=1.0):
        self.game = deepcopy(game)
        # self.global_player = gf.get_next_player(np.array(self.game.board))

        self.root = Node(state=np.array(game.board, dtype=np.int8),
                         move_history=self.game.moves,
                         move=move,
                         parent=None)  # starting state doesn't have a move
        self.explore = explore
        self.c_puct = c_puct
        self.tau = tau  # temperature control for exploration, stochastic exploration

        self.worker_id = worker_id
        self.sess = session
        self.input_name = self.sess.get_inputs()[0].name
        self._expand(self.root)

        # warmup for onnnxruntime
        self.sess.run(["policy", "value"],
                      {self.input_name: np.random.randint(low=-1, high=2, size=gf.SHAPE, dtype=gf.NP_DTYPE)})

    def update_hyperparameters(self, explore=None, c_puct=0.0, tau=0.0):
        if explore is not None:
            self.explore = explore
        self.c_puct += c_puct
        self.tau += tau

    def apply_dirchlet(self, node):
        prob_prior = np.array([child.prob_prior for child in node.children])
        prob_prior = 0.75 * prob_prior + 0.25 * np.random.dirichlet(0.3 * np.ones(len(prob_prior)))
        for child_id, child in enumerate(node.children):
            child.prob_prior = prob_prior[child_id]

    def _puct_select(self, node):
        best_score = -math.inf
        best_child = None
        for child in node.children:
            if child.visits == 0:
                return child
            PUCT_score = (child.value / child.visits) + (
                        child.prob_prior * ((child.parent.visits ** 0.5) / (child.visits + 1)) *
                        (self.c_puct + (math.log10((child.parent.visits + 19653) / 19652))))
            if PUCT_score > best_score:
                best_score = PUCT_score
                best_child = child
        return best_child

    def _expand(self, node, cutoff=None) -> None:
        next_player = gf.get_next_player_histo(np.array(node.state, dtype=np.int8))
        winning_moves = gf.find_term_expand(node.state,
                                            next_player)  # not destructive to the original array (node_state)
        if len(winning_moves) == 0:
            raw_policy, raw_value = self.sess.run(["policy", "value"],
                                                  {self.input_name: gf.get_inference_board(
                                                      np.array([(-1, -1)] + node.move_history, dtype=np.int8))})
            policy = gf.parse_policy(raw_policy=raw_policy,
                                     game_state=node.state,
                                     cutoff=cutoff)

            self._backprop(node, -raw_value[0][0])
        else:
            policy = []
            for move, is_win in winning_moves:
                if is_win is True:
                    policy.append([move, 1, True])
                elif not is_win:
                    policy.append([move, 1, None])
            # True for win, False for no win, None for a drawing move

            policy = [[move, probability / len(policy), terminal] for move, probability, terminal in policy]

        # state_copies = np.tile(node.state, [len(policy), 1, 1])
        for id, (move, prior_prob, policy_terminal) in enumerate(policy):
            # new_state = state_copies[id]
            new_state = np.copy(node.state)
            new_state[move[1]][move[0]] = next_player

            new_move_history = node.move_history + [move]
            child = Node(state=new_state,
                         move=move,
                         move_history=new_move_history,
                         prob_prior=prior_prob,
                         parent=node,
                         terminal=(next_player if policy_terminal else (None if policy_terminal is False else 0)))
            node.children.append(child)
            if policy_terminal is True:
                self._backprop(child, 1)
            elif policy_terminal is None:
                self._backprop(child, 0)

    def _backprop(self, node, value):
        back_val = value
        while node is not None:
            node.value += back_val
            node.visits += 1
            back_val *= -1
            node = node.parent

    #  not usable currently
    def iteration_step(self):
        node = self.root
        if not self.root.children:
            self._expand(self.root, cutoff=gf.WIDTH * gf.HEIGHT)
            self.apply_dirchlet(self.root)  # this is not effective btw, might as well apply to all nodes

        while node.children and node.is_terminal is None:
            nodes_info = np.array(
                [[child.value, child.prob_prior, child.visits, child.parent.visits] for child in node.children])
            node_id = gf._puct_select(nodes_info, c_puct=self.c_puct)
            node = node.children[node_id]

        if node.is_terminal is None:
            self._expand(node, cutoff=None)
        else:
            self._backprop(node, abs(node.is_terminal))

    def run(self, time_limit=None, iteration_limit=None):
        if time_limit is None and iteration_limit is None:
            raise RuntimeError(f"Both iteration limit and time limit is None")
        # print(f"searching for player {self.global_player}")
        winning_moves = gf.find_term_expand(np.array(self.game.board, dtype=np.int8))
        if winning_moves:
            if winning_moves[0][1] is False:
                print(f"Found drawing move")
                return winning_moves[0][0], [[winning_move[0], 0 / int(len(winning_moves)), 1, len(winning_moves), 1,
                                              0 / int(len(winning_moves))] for winning_move in winning_moves]
            print("Found win through game check win")
            return winning_moves[0][0], [[winning_move[0], 1.0 / int(len(winning_moves)), 1, len(winning_moves), 1,
                                          1.0 / int(len(winning_moves))] for winning_move in winning_moves]

        # start_time = time.time()
        # iterations = 0

        if iteration_limit is True:
            # iteration_limit = len(gf.get_edge_moves(np.array(self.game.board), width=1)) * 32
            # iteration_limit = iteration_limit if iteration_limit < 1600 else 1600
            # if len(self.game.moves) in range(3):
            iteration_limit = 1500

        #     bar = tqdm(total=iteration_limit)
        # elif iteration_limit:
        #     bar = tqdm(total=iteration_limit)
        # else:
        #     bar = tqdm(total=time_limit)
        self.apply_dirchlet(self.root)

        # while (time_limit is None or time.time() - start_time < time_limit) and \
        #         (iteration_limit is None or iterations < iteration_limit):
        for _ in tqdm(range(iteration_limit)):
            # if time_limit is not None:
            #     loop_start_time = time.time()

            node = self.root
            while len(node.children) != 0 and node.is_terminal is None:
                # node = self._puct_select(node)
                nodes_info = np.array(
                    [[child.value, child.prob_prior, child.visits, child.parent.visits] for child in node.children])
                node_id = gf._puct_select(nodes_info, c_puct=self.c_puct)
                node = node.children[node_id]

            if node.is_terminal is None:
                self._expand(node, cutoff=gf.WIDTH * gf.HEIGHT)
            else:
                self._backprop(node, abs(node.is_terminal))

            # if iteration_limit:
            # bar.update(1)
            # else:
            #     bar.update(time.time() - loop_start_time)
            # iterations += 1

        move_probs = []
        move_visits = []
        for i, node in enumerate(self.root.children):
            move_probs.append(
                [node.move, None, node.visits, self.root.visits, node.value, node.prob_prior])
            move_visits.append(node.visits)

        move_probs_visits = softmax(np.log(np.array(move_visits) + 1e-10))
        selection_visits = softmax((1.0 / self.tau) * np.log(np.array(move_visits) + 1e-10))

        for i, prob in enumerate(move_probs_visits):
            move_probs[i][1] = prob
        unsorted_move_probs = move_probs

        move_probs = [[move_probs[i], selection_visits[i]] for i in range(len(selection_visits))]
        move_probs = sorted(move_probs, key=lambda x: x[1], reverse=True)  # sorted creates a duplicate

        move_probs, tau_prob = zip(*move_probs)
        move_probs = list(move_probs)
        tau_prob = np.array(tau_prob)
        if self.explore:
            # only explore the top 20 moves or fewer
            indexes = list(range(len(tau_prob)))
            if tau_prob[0] < 1:
                move = move_probs[np.random.choice(indexes, p=tau_prob)][0]
            else:
                move = move_probs[0][0]
        else:
            indexes = list(range(len(move_probs)))
            weights = softmax((1.0 / 5e-4) * np.log(np.array(move_visits) + 1e-10))
            move = unsorted_move_probs[np.random.choice(indexes, p=weights)][0]

        top_line = []
        node = self.root
        while len(node.children) != 0:
            # print()
            # print([[child.move, (child.wins / (child.visits + 1)),  child.wins, child.visits, child.value, child.prob, child.is_terminal] for child in node.children])
            best_node = None
            for child in node.children:
                if best_node is None or (child.visits / child.parent.visits) > (
                        best_node.visits / best_node.parent.visits):
                    best_node = child
            top_line.append(
                [best_node.move, best_node.value / (best_node.visits + 1), best_node.visits,
                 best_node.value,
                 best_node.prob_prior, best_node.is_terminal])
            node = best_node
        print("\nTop Engine Line:")
        print(top_line)
        print()

        # print(move_probs[:10])

        return move, move_probs

    def update_tree(self, game, move):
        del self.game
        self.game = deepcopy(game)
        if len(self.root.children) == 0 or game.moves not in [child.move_history for child in self.root.children]:
            self.root = Node(np.array(self.game.board, dtype=np.int8), move=move, move_history=self.game.moves.copy(),
                             parent=None)
            print("Pruned all nodes")
            return

        move_count = 0
        for node in self.root.children:
            if (node.move == move and game.moves == node.move_history):
                move_count += 1
                print(f"Pruned {len(self.root.children) - 1} nodes")
                self.root = node
                self.root.state = node.state
                self.root.children = node.children
                self.root.parent = None
                if game.moves != node.move_history:
                    print(f"Game moves and node moves are not the same")
                    print(game.moves)
                    print(node.move)
                self.root.move_history = node.move_history
            if move_count > 1:
                raise ValueError("Found duplicate when pruning, something went wrong")
            del node
        if move_count == 0:
            raise ValueError(f"Couldn't prune because node.move_history != game.moves")


if __name__ == "__main__":
    from glob import glob
    from womoku import Womoku
    import womoku as gf

    # generation = max([path.split("\\")[-1][0] for path in glob("alphazero/onnx_models/*.onnx")])
    generation = 6
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads=12
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    session = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                                  providers=gf.PROVIDERS,
                                  # provider_options=[{"num_of_threads": 0}],
                                  sess_options=sess_options)

    game = Womoku()
    game.put((7, 7))
    # game.put((6, 7))
    # game.put((7, 6))
    # game.put((6, 6))
    # game.put((7, 5))
    # game.put((6, 5))
    # game.put((4, 4))
    # moves = [(4, 4), (3, 3), (4, 3), (4, 5), (4, 1), (3, 4), (4, 1), (4, 0), (3, 1), (5, 4), (5, 1), (1, 1), (3, 5), (6, 1), (1, 6), (1, 7), (5, 3), (6, 3), (3, 6), (6, 4), (6, 5), (6, 0), (6, 1), (5, 1), (7, 3), (4, 6), (5, 5), (5, 6), (1, 3), (6, 7), (7, 8), (1, 4), (7, 4), (7, 5), (7, 1), (6, 6), (7, 1), (7, 0), (5, 0), (5, 7), (4, 7), (4, 8), (8, 4), (7, 6), (8, 6), (1, 6), (1, 5), (8, 5), (0, 0), (0, 1), (0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 8), (1, 7), (1, 5), (3, 7), (1, 0), (1, 1), (1, 1), (1, 3), (1, 4), (3, 8), (1, 8), (5, 8), (6, 8), (8, 8), (7, 7), (8, 7), (8, 3), (8, 1), (8, 1), (8, 0), (3, 1)]
    # for i, move in enumerate(moves):
    #     game.put(move)
    #     mcts1.run(1000)
    # game.put((4, 4))
    # game.put((3, 3))
    # game.put((4, 3))
    # game.put((3, 4))
    # game.put((4, 5))

    game.print_board()
    mcts1 = MCTS(game=game, session=session,
                 c_puct=4, explore=True, tau=0.1)
    print(mcts1.run(iteration_limit=2000))

    # mcts2 = MCTS(game=game, session=session,
    #             c_puct=4, explore=False)

    # won_player = -1
    #
    # while won_player == -1:
    #     if gf.get_next_player(np.array(game.board)) == -1:
    #         move, line = mcts1.run(time_limit=None, iteration_limit=800)
    #     else:
    #         move, line = mcts2.run(time_limit=None, iteration_limit=800)
    #     print(move, line)
    #     gf.prob_to_board(line, game.board)
    #
    #     game.put(move)
    #     game.print_board()
    #
    #     won_player = gf.check_won(np.array(game.board), move)
    #     if won_player != -1:
    #         print(won_player)
    #
    #
    #     mcts1.update_tree(game, move)
    #     mcts2.update_tree(game, move)

    # game = Womoku()
    # mcts = MCTS(game, session=session,)
    # won_player = -1
    # while won_player == -1:
    #     if gf.get_next_player(np.array(game.board)) == -1:
    #         valid = False
    #         while not valid:
    #             try:
    #                 x, y = input(f"Move? x,y please no spaces").split(",")
    #                 iterations = int(input("Iterations??"))
    #                 x, y = int(x), int(y)
    #                 move = (x, y)
    #                 valid = True
    #             except:
    #                 print("stop being dumb and put an actual move")
    #     else:
    #         # iterations = 5000
    #
    #         move, line = mcts.run(iteration_limit=iterations)
    #         print(move, line)
    #     game.put(move)
    #     game.print_board()
    #
    #     mcts.update_tree(game, move)
    #     won_player = gf.check_won(np.array(game.board), move)
