import pygame
import numpy as np
import random
from tqdm import tqdm

from MCTS_ONNX import MCTS
from MCTS_virtualloss import MCTS as MCTS_p
import womoku as gf
from womoku import Womoku

import onnxruntime as rt

# pygame.init()
#
# import ctypes
#
# ctypes.windll.user32.SetProcessDPIAware()
#
# screen = pygame.display.set_mode((800, 800), vsync=True, flags=pygame.RESIZABLE)
# LIME_GREEN = (120, 190, 33)
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
#
# tile_size = 800 / (gf.WIDTH + 1)
#
# background = pygame.image.load('background.jpeg').convert()
#
#
# def draw_background():
#     screen.blit(background, (0, 0))
#
#
# def draw_board():
#     for y in range(gf.HEIGHT + 1):
#         pygame.draw.line(screen, BLACK, (tile_size, y * tile_size - 1), (gf.HEIGHT * tile_size, y * tile_size - 1),
#                          width=1)
#     for x in range(gf.WIDTH + 1):
#         pygame.draw.line(screen, BLACK, (x * tile_size - 1, tile_size), (x * tile_size - 1, (gf.WIDTH * tile_size)),
#                          width=1)
#
#     # Centre circle
#     pygame.draw.circle(screen, BLACK, (400, 400), 10)
#
#     # for x in range(1):
#     #     pygame.draw.circle(screen, BLACK, (200 + 400 * x, 200), 7)
#     # for x in range(1):
#     #     pygame.draw.circle(screen, BLACK, (200 + 400 * x, 600), 7)
#
#
# def draw_move(moves):
#     if not moves:
#         return
#     x, y = moves[-1]
#     pygame.draw.circle(screen, LIME_GREEN, (tile_size * (x + 1), tile_size * (y + 1)), 30)
#     font = pygame.font.Font('freesansbold.ttf', 20)
#
#     for move_num, move in enumerate(moves):
#         move_num += 1
#         x, y = move
#         if move_num % 1 == 0:
#             pygame.draw.circle(screen, WHITE, (tile_size * (x + 1), tile_size * (y + 1)), 20)
#             render_move_num = font.render(str(move_num), True, BLACK)
#             screen.blit(render_move_num, (tile_size * (x + 1) - 11, tile_size * (y + 1) - 10))
#         else:
#             pygame.draw.circle(screen, BLACK, (tile_size * (x + 1), tile_size * (y + 1)), 20)
#             render_move_num = font.render(str(move_num), True, WHITE)
#             screen.blit(render_move_num, (tile_size * (x + 1) - 11, tile_size * (y + 1) - 10))
#
#     pygame.display.update()

draws = 0
AI1_wins = 0
generation = 3
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

session_1 = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                                providers=gf.PROVIDERS, sess_options=sess_options)

AI2_wins = 0
generation = 3
session_2 = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                                providers=gf.PROVIDERS, sess_options=sess_options)
# max_iterations = 1500

max_games = 1
for game_num in range(max_games):
    game = Womoku()
    won_player = -2

    # if the game num is even then AI_1 is the first player and vice versa

    if game_num % 2 == 0:
        player_1 = MCTS_p(game=game, session=session_1, c_puct=4, explore=True, tau=1e-1)
        player_2 = MCTS_p(game=game, session=session_2, c_puct=4, explore=True, tau=1e-1)
        print(f"AI1 is black and AI2 is white")
    else:
        player_2 = MCTS_p(game=game, session=session_1, c_puct=4, explore=True, tau=1e-1)
        player_1 = MCTS_p(game=game, session=session_2, c_puct=4, explore=True, tau=1e-1)
        print(f"AI1 is white and AI2 is black")
    time_limit = 30
    iterations = 0
    while won_player == -2:
        move = (None, None)
        if gf.get_next_player(np.array(game.board)) == -1:
            # if iterations == 0:
            #     bar = tqdm(total=1000)
            # if iterations < max_iterations:
            #     player_1.iteration_step()
            #     iterations += 1
            #     bar.update(1)
            # else:
            #     move, line = player_1.run(iteration_limit=0)
            #     iterations = 0
            #     bar.close()
            move, line = player_1.run(iteration_limit=1000)
        else:
            # if iterations == 0:
            #     bar = tqdm(total=max_iterations)
            # if iterations < max_iterations:
            #     player_2.iteration_step()
            #     iterations += 1
            #     bar.update(1)
            # else:
            #     iterations = 0
            #     move, line = player_2.run(iteration_limit=0)
            #     bar.close()
            move, line = player_2.run(iteration_limit=1000)

        if move != (None, None):
            game.put(move)
            player_1.update_tree(game, move)
            player_2.update_tree(game, move)
        # draw_background()
        # draw_board()
        # draw_move(game.moves)
        # pygame.display.update()
        # pygame_event = pygame.event.get()
        # for event in pygame_event:
        #     if event.type == pygame.QUIT:
        #         running = False
        if move != (None, None):
            print(line)
            print(f"Played: {move}")
            won_player = gf.check_won(np.array(game.board), move=move)
        game.print_board()
        if won_player != -2:
            if game_num % 2 == 0:
                # AI1 is black
                # Ai2 is white
                if won_player == -1:
                    AI1_wins += 1
                elif won_player == 1:
                    AI2_wins += 1
                else:
                    draws += 1
            else:
                # AI1 is white
                # Ai2 is black
                if won_player == -1:
                    AI2_wins += 1
                elif won_player == 1:
                    AI1_wins += 1
                else:
                    draws += 1
            print(f"AI_1 won {AI1_wins} and AI_2 won {AI2_wins}")
            print(f"AI_1 winrate is {AI1_wins / (game_num + 1)} and AI_2 winrate is {AI2_wins / (game_num + 1)}")
