import time
import math
import random
import numpy as np
from tqdm import tqdm
import onnxruntime as rt

import womoku as gf
from womoku import Womoku
from copy import deepcopy


# game functions

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class Node(object):
    __slots__ = "state", "move", "move_history", "parent", "children", "prob_prior", "visits", "value", "is_terminal"

    def __init__(self, state, move, move_history=[], prob_prior=0, parent=None, terminal=None):
        self.state = state
        self.move = move
        self.move_history = move_history
        self.parent = parent
        # None means not terminal, -1 mean player -1 won and 1 means player 1 won

        self.children = []

        self.prob_prior = np.array([prob_prior], dtype=np.float16)  # P
        self.visits = 0  # N
        self.value = np.array([0], dtype=np.float16)

        self.is_terminal = terminal


class MCTS:
    def __init__(self,
                 game, session, worker_id=0,
                 move=None,
                 explore=False,
                 max_nodes_infer=gf.MAX_NODES_INFER,
                 c_puct=4.5, tau=1.0):
        self.game = deepcopy(game)

        self.root = Node(state=np.array(game.board, dtype=np.int8),
                         move_history=self.game.moves.copy(),
                         move=move,
                         parent=None)  # starting state doesn't have a move
        self.max_nodes_infer = max_nodes_infer
        self.explore = explore
        self.c_puct = c_puct
        self.tau = tau  # temperature control for exploration, stochastic exploration

        self.worker_id = worker_id
        self.sess = session
        self.input_name = self.sess.get_inputs()[0].name

        # self.virtual_loss = np.array([1], dtype=np.int8)
        self.virtual_loss = 1

        self._expand_root(self.root, cutoff=gf.WIDTH * gf.HEIGHT)

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

    def _expand_root(self, node, cutoff=gf.HEIGHT * gf.WIDTH) -> None:
        winning_moves = gf.find_term_expand(node.state)  # be careful of this line, copying can be very important

        if len(winning_moves) == 0:
            raw_policy, raw_value = self.sess.run(["policy", "value"],
                                                  {self.input_name: gf.get_inference_board(
                                                      np.array([(-1, -1)] + node.move_history, dtype=np.int8))})
            policy = gf.parse_policy(raw_policy=raw_policy,
                                     game_state=node.state,
                                     cutoff=cutoff)

            self._backprop_virtual(node, -raw_value[0][0])
        else:
            policy = []
            for move, is_win in winning_moves:
                if is_win is True:
                    policy.append([move, 1, True])
                elif not is_win:
                    policy.append([move, 1, None])
            # True for win, False for no win, None for a drawing move

            sum_prob = sum([info[1] for info in policy])
            policy = [[move, probability / sum_prob, terminal] for move, probability, terminal in policy]

        # next_player = gf.get_next_player(node.state)
        next_player = gf.get_next_player_histo(np.array(node.move_history))
        # print(next_player)
        # raise ValueError
        for child_id, (move, prior_prob, policy_terminal) in enumerate(policy):
            new_state = np.copy(node.state)  # fastest shallow copy
            new_state[move[1]][move[0]] = next_player

            new_move_history = node.move_history + [move]
            child = Node(state=new_state,
                         move=move,
                         move_history=new_move_history,
                         prob_prior=prior_prob,
                         parent=node,
                         terminal=(next_player if policy_terminal else (None if policy_terminal is False else 0)))
            node.children.append(child)

            if policy_terminal in [-1, 1]:
                self._backprop(child, 1)
            elif policy_terminal is None:
                self._backprop(child, 0)
        if self.explore:
            self.apply_dirchlet(self.root)

    def _expand(self, node, policy, cutoff=gf.HEIGHT * gf.WIDTH) -> None:
        next_player = gf.get_next_player_histo(np.array(node.move_history, dtype=np.int8))
        if node.children:
            print("Something went wrong, if this feature was applied, we would be screwed")
        for child_id, (move, prior_prob, policy_terminal) in enumerate(policy):
            new_state = np.copy(node.state)  # fastest shallow copy
            new_state[move[1]][move[0]] = next_player

            new_move_history = node.move_history + [move]
            child = Node(state=new_state,
                         move=move,
                         move_history=new_move_history,
                         prob_prior=prior_prob,
                         parent=node,
                         terminal=(next_player if policy_terminal else (None if policy_terminal is False else 0)))
            node.children.append(child)

            if policy_terminal in [-1, 1]:
                self._backprop(child, 1)
            elif policy_terminal is None:
                self._backprop(child, 0)
        if self.explore:
            self.apply_dirchlet(node)  # technically not needed but still good for deep tree exploration

    def _backprop_virtual(self, node, value):
        back_val = value
        while node is not None:
            # virtual loss
            node.value += (back_val + self.virtual_loss)
            back_val *= -1
            node = node.parent

    def _backprop(self, node, value):
        back_val = value
        while node is not None:
            node.value += back_val
            node.visits += 1
            back_val *= -1
            node = node.parent

    def iteration_step(self):
        node = self.root
        if not self.root.children:
            self._expand(self.root, cutoff=gf.WIDTH * gf.HEIGHT)

        while len(node.children) != 0 and node.is_terminal is None:
            node = self._puct_select(node)

        if node.is_terminal is None:
            self._expand(node, cutoff=gf.WIDTH * gf.HEIGHT)
        else:
            self._backprop_virtual(node, abs(node.is_terminal))

    def run(self, time_limit=None, iteration_limit=None):
        # print(f"searching for player {self.global_player}")
        winning_moves = gf.find_term_expand(np.array(self.game.board))
        if winning_moves:
            if winning_moves[0][1] is False:
                print(f"Found drawing move")
                return winning_moves[0][0], [[winning_move[0], 0 / int(len(winning_moves)), 1, len(winning_moves), 1,
                                              0 / int(len(winning_moves))] for winning_move in winning_moves]
            print("Found win through game check win")
            return winning_moves[0][0], [[winning_move[0], 1.0 / int(len(winning_moves)), 1, len(winning_moves), 1,
                                          1.0 / int(len(winning_moves))] for winning_move in winning_moves]

        start_time = time.time()
        iterations = 0

        if iteration_limit is True:
            # iteration_limit = len(gf.get_edge_moves(np.array(self.game.board), width=1)) * 32
            # iteration_limit = iteration_limit if iteration_limit < 1600 else 1600
            # if len(self.game.moves) in range(3):
            iteration_limit = 600

            bar = tqdm(total=iteration_limit)
        elif iteration_limit:
            bar = tqdm(total=iteration_limit)
        else:
            bar = tqdm(total=time_limit)

        # for current_iteration in tqdm(range(iteration_limit)):
        if time_limit is None and iteration_limit is None:
            raise RuntimeError(f"Both iteration limit and time limit is None")
        while (time_limit is None or time.time() - start_time < time_limit) and \
                (iteration_limit is None or iterations < iteration_limit):
            if time_limit is not None:
                loop_start_time = time.time()
            batch_nodes = set()
            for _ in range(self.max_nodes_infer):
                node = self.root
                while len(node.children) != 0 and node.is_terminal is None:
                    node = self._puct_select(node)
                    node.parent.value -= self.virtual_loss
                    node.parent.visits += 1
                node.visits += 1
                node.value -= self.virtual_loss

                if node.is_terminal is None and node not in batch_nodes:
                    batch_nodes.add(node)
                    # prop_node = node
                    # while prop_node is not None:
                    #     prop_node.visits += 1
                    #     prop_node.value -= self.virtual_loss
                        # prop_node = prop_node.parent
                elif node.is_terminal is not None:
                    self._backprop_virtual(node, abs(node.is_terminal))
                # else:
                #     break

            policies_values = [(None, 0) for _ in range(len(batch_nodes))]
            batch_inference_boards = {}
            # print(batch_nodes)
            for policy_id, infer_node in enumerate(batch_nodes):
                winning_moves = gf.find_term_expand(
                    infer_node.state)  # be careful of this line, copying can be very important
                if not winning_moves:
                    batch_inference_boards[policy_id] = gf.get_inference_board(
                        np.array([(-1, -1)] + infer_node.move_history, dtype=np.int8)).reshape(gf.SHAPE[1:])
                else:
                    policy = []
                    for move, is_win in winning_moves:
                        if is_win is True:
                            policy.append([move, 1, True])
                        elif not is_win:
                            policy.append([move, 1, None])
                    policies_values[policy_id] = (policy, -1)
            # print(len(batch_inference_boards))
            # raise ValueError
            if len(batch_inference_boards) != 0:
                raw_policies, raw_values = self.sess.run(["policy", "value"],
                                                         {self.input_name: np.array(
                                                             list(batch_inference_boards.values()),
                                                             dtype=np.int8)})
                infer_id = 0  # must use this as the raw policies index doesn't match when check win fills some policies in
                for id, (node, (checked_policy, checked_value)) in enumerate(zip(batch_nodes, policies_values)):
                    if checked_value == 1 or checked_policy is not None:
                        continue
                    policy = gf.parse_policy(raw_policy=raw_policies[infer_id],
                                             game_state=node.state)
                    policies_values[id] = (policy, -raw_values[infer_id][0])
                    infer_id += 1

            # have policy value, have batch nodes
            for node, (policy, value) in zip(batch_nodes, policies_values):
                if node.is_terminal is None:
                    self._expand(node, policy, cutoff=gf.WIDTH * gf.HEIGHT)
                    self._backprop_virtual(node, value)
                else:
                    self._backprop_virtual(node, abs(node.is_terminal))

            if iteration_limit:
                bar.update(1)
            else:
                bar.update(time.time() - loop_start_time)
            iterations += 1

        move_probs = []
        move_visits = []
        for i, node in enumerate(self.root.children):
            move_probs.append(
                [node.move, None, node.visits, self.root.visits, node.value[0], node.prob_prior[0]])
            move_visits.append(node.visits)
        move_visits = np.array(move_visits)

        move_probs_visits = softmax((1.0/1.1) * np.log(np.array(move_visits) + 1e-10))
        selection_visits = softmax((1.0 / self.tau) * np.log(move_visits + 1e-10))
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
            weights = softmax((1.0 / 1e-4) * np.log(np.array(move_visits) + 1e-10))
            move = unsorted_move_probs[np.random.choice(indexes, p=weights)][0]

        top_line = []
        node = self.root
        while len(node.children) != 0:
            best_node = None
            if node.visits == 0:
                break
            for child in node.children:
                if child.visits == 0:
                    continue
                if best_node is None or (child.visits / child.parent.visits) > (
                        best_node.visits / best_node.parent.visits):
                    best_node = child
            if best_node is not None:
                top_line.append(
                    [best_node.move, best_node.value[0] / best_node.visits, best_node.visits,
                     best_node.value[0],
                     best_node.prob_prior[0], best_node.is_terminal])
            else:
                break
            node = best_node
        print("\nTop Engine Line:")
        print(top_line)
        print()

        return move, move_probs

    def update_tree(self, game, move, delete=True):
        del self.game
        self.game = deepcopy(game)

        if move not in [node.move for node in self.root.children]:
            print(f"Create node for new position")
            self.root = Node(state=np.array(self.game.board, dtype=np.int8), move=move, move_history=self.game.moves.copy())
            self._expand_root(self.root)
            return

        move_count = 0
        for node in self.root.children:
            if node.move == move and game.moves == node.move_history:
                move_count += 1
                print(f"Pruned {len(self.root.children) - 1} nodes")
                self.root = node
                self.root.state = node.state
                self.root.children = node.children
                self.root.parent = None
                # if game.moves != node.move_history:
                #     print(f"Game moves and node moves are not the same")
                #     print(game.moves)
                #     print(node.move)
                self.root.move_history = node.move_history
            # if move_count > 1:
            #     raise ValueError("Found duplicate when pruning, something went wrong")
            # if delete:
            #     del node
        if move_count == 0:
            raise ValueError(f"Couldn't prune because node.move_history != game.moves")


if __name__ == "__main__":
    from glob import glob
    from womoku import Womoku
    import womoku as gf

    generation = max([path.split("\\")[-1][0] for path in glob("alphazero/onnx_models/*.onnx")])
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1

    session = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                                  providers=gf.PROVIDERS, sess_options=sess_options)

    game = Womoku()
    # game.put((7, 7))
    # game.put((8, 8))
    # game.put((8, 7))
    # game.put((14, 6))
    # game.put((6,7))
    # game.put((8, 6))
    # game.put((6, 6))
    # game.put((6, 8))
    # game.put((8, 8))
    # game.put((5, 5))
    # game.put((9, 9))
    # game.put((10, 10))
    # game.put((4, 4))
    # game.put((3, 3))
    # game.put((4, 3))
    # game.put((3, 4))
    # game.put((4, 5))

    game.print_board()
    mcts1 = MCTS(game=game, session=session,
                 c_puct=4, explore=False)
    print(mcts1.run(iteration_limit=250))

    # import pickle
    #
    # with open("alphazero/domain/domain1.pickle", "wb") as f:
    #     pickle.dump(mcts1.root, f)
    # with open("alphazero/domain/domain1.pickle", "rb") as f:
    #     tree = pickle.load(f)
    # for node in tree.children:
    #     print(node.children)

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
