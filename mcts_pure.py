import time
import random
import math
from tqdm import tqdm
from glob import glob
import pickle
import numpy as np

from copy import deepcopy
from womoku import Womoku
import womoku as gf


class Node:
    def __init__(self, state, move, parent=None, terminal=None):
        self.state = state
        self.move = move
        self.parent = parent
        # None means not terminal, -1 mean player -1 won and 1 means player 1 won
        self.is_terminal = terminal

        self.children = []
        self.visits = 0
        self.wins = 0


class MCTS:
    def __init__(self,
                 game,
                 move=None,
                 c=2**0.5,
                 tau=1.15,
                 explore=False,
                 use_dirichlet=True):
        self.game = game

        self.root = Node(state=[[*row] for row in game.board],
                         move=move,
                         parent=None)  # starting state doesn't have a move
        self.c = c
        self.tau = tau
        self.explore = explore
        self.use_dirichlet = use_dirichlet

        self.simulations_per_node = 2

    def _select(self, node):
        best_score = -math.inf
        best_child = None
        for child in node.children:
            if child.visits == 0:
                return child
            score = (child.wins / child.visits) + self.c * math.sqrt(math.log(child.parent.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node, width=1) -> Node:
        winning_moves = gf.find_term_expand(np.array(node.state))
        if len(winning_moves) != 0:
            for winning_move, is_win in winning_moves:
                new_state = [[*row] for row in node.state]
                new_state[winning_move[1]][winning_move[0]] = gf.get_next_player(np.array(node.state))
                if is_win:
                    node_terminality = gf.get_current_player(np.array(node.state))
                elif is_win is False:
                    node_terminality = None
                else:
                    node_terminality = 0
                child = Node(state=new_state, move=winning_move, parent=node,
                             terminal=node_terminality)  # check current or next player
                node.children.append(child)
            return child

        for move in gf.get_edge_moves(np.array(node.state)):
            new_state = [[*row] for row in node.state]
            new_state[move[1]][move[0]] = gf.get_next_player(np.array(node.state))
            child = Node(state=new_state, move=move, parent=node, terminal=None)
            node.children.append(child)
        if not node.children:
            return node
        return random.choice(node.children)

    def _backprop(self, node, wins):
        node.visits += self.simulations_per_node
        node.wins += wins
        if node.parent is not None:
            self._backprop(node.parent, -wins)

    def run(self, iteration_limit=5000):
        iterations = 0
        bar = tqdm(total=iteration_limit)
        if len(self.root.children) == 0:
            self._expand(self.root, width=2)
        while iterations < iteration_limit:
            node = self.root
            while node.children and node.is_terminal is None:
                node = self._select(node)
            if node.is_terminal is None:
                node = self._expand(node, width=2)
                if node.is_terminal is None:
                    result = gf.simulate_jit(np.array(node.state), 100, self.simulations_per_node)
                    self._backprop(node, result)
            elif node.is_terminal == 0:
                self._backprop(node, 0)
            else:
                self._backprop(node, self.simulations_per_node)
            iterations += 1
            bar.update(1)

        bar.close()

        move_probs = []
        for i, node in enumerate(self.root.children):
            if node.visits == 0:
                continue
            probability = (node.visits / self.root.visits) ** (1 / self.tau)
            move_probs.append(
                [node.move, probability, node.visits, self.root.visits, node.wins, "I'm Gai"])
        move_probs = sorted(move_probs, key=lambda x: x[1], reverse=True)
        sum_prob = sum([prob for _, prob, _, _, _, _ in move_probs])
        norm_prob = [prob / sum_prob for _, prob, _, _, _, _ in move_probs]

        for i, _ in enumerate(move_probs):
            move_probs[i][1] = norm_prob[i]

        if self.use_dirichlet:
            noisy_probs = 0.85 * np.array(norm_prob) + 0.15 * np.random.dirichlet(0.3 * np.ones(len(norm_prob)))
            for i, noisy_prob in enumerate(noisy_probs):
                move_probs[i][1] = noisy_prob

        move_probs = sorted(move_probs, key=lambda x: x[1], reverse=True)
        if self.explore is None:
            move = move_probs[0]
        elif self.explore:
            weights = np.array([info[1] for info in move_probs])
            index = np.random.choice(np.arange(0, weights.shape[0]), p=weights)
            move = move_probs[index]
        else:
            move = move_probs[0]

        move_probs = sorted(move_probs, key=lambda x: x[1], reverse=True)

        return move[0], move_probs

    def update_tree(self, game, move):
        self.game = deepcopy(game)
        if len(self.root.children) == 0:
            self.root = Node([[*row] for row in game.board], move=move, parent=None)
            print("Pruned all nodes")
            return

        move_count = 0
        for node in self.root.children or move not in [node.move for node in self.root.children]:
            if node.move == move:
                move_count += 1
                print(f"Pruned {len(self.root.children) - 1} nodes")
                self.root = node
                self.root.state = [[*row] for row in node.state]
                self.root.children = node.children
                self.root.parent = None
            if move_count > 1:
                raise ValueError("Found duplicate when pruning, something went wrong")
            del node


if __name__ == "__main__":
    game = Womoku()

    # game.put((7, 7))
    # game.put((6, 7))
    #
    # game.put((7, 6))
    # game.put((6, 6))
    # game.put((7, 5))
    # # game.put((6, 5))
    # game.print_board()
    mcts = MCTS(game, use_dirichlet=False)
    won_player = -2
    while won_player == -2:
        line = None
        if gf.get_next_player(np.array(game.board)) == 1:
            valid = False
            while not valid:
                try:
                    x, y = input(f"Move? x,y please no spaces").split(",")
                    x, y = int(x), int(y)
                    move = (x, y)
                    valid = True
                except:
                    print("stop being dumb and put an actual move")
        else:
            move, line = mcts.run()
        print(move, line)
        game.put(move)
        game.print_board()

        mcts.update_tree(game, move)
        won_player = gf.check_won(np.array(game.board), move)
    #
    # #self play code for MCTS samples
    # game = Womoku()
    # mcts1 = MCTS(game=game, explore=True, use_dirichlet=True)
    #
    # mcts2 = MCTS(game=game, explore=True, use_dirichlet=True)
    #
    # won_player = -1
    #
    # while won_player == -1:
    #     if gf.get_next_player(np.array(game.board)) == -1:
    #         move, line = mcts1.run()
    #     else:
    #         move, line = mcts2.run()
    #     print(move, line)
    #
    #     game.put(move)
    #     game.print_board()
    #
    #     won_player = gf.check_won(np.array(game.board), move)
    #     if won_player != -1:
    #         print(won_player)
    #
    #     mcts1.update_tree(game, move)
    #     mcts2.update_tree(game, move)
