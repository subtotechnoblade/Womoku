import pygame
import numpy as np
from glob import glob
from tqdm import tqdm

from womoku import Womoku
import womoku as gf

import onnxruntime as rt
from MCTS_virtualloss import MCTS

# from mcts_pure import MCTS


pygame.init()

import ctypes

ctypes.windll.user32.SetProcessDPIAware()

screen = pygame.display.set_mode((800, 800), vsync=True, flags=pygame.DOUBLEBUF | pygame.HWACCEL)
LIME_GREEN = (120, 190, 33)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

tile_size = 800 / (gf.WIDTH + 1)

background = pygame.image.load('background.jpeg').convert()


def draw_background():
    screen.blit(background, (0, 0))


def draw_board():
    for y in range(gf.HEIGHT + 1):
        pygame.draw.line(screen, BLACK, (tile_size, y * tile_size - 1), (gf.HEIGHT * tile_size, y * tile_size - 1),
                         width=2)
    for x in range(gf.WIDTH + 1):
        pygame.draw.line(screen, BLACK, (x * tile_size - 1, tile_size), (x * tile_size - 1, (gf.WIDTH * tile_size)),
                         width=2)

    # Centre circle
    pygame.draw.circle(screen, BLACK, (400, 400), 10)

    for x in range(2):
        pygame.draw.circle(screen, BLACK, (200 + 400 * x, 200), 7)
    for x in range(2):
        pygame.draw.circle(screen, BLACK, (200 + 400 * x, 600), 7)


def draw_move(moves):
    if not moves:
        return
    x, y = moves[-1]
    pygame.draw.circle(screen, LIME_GREEN, (tile_size * (x + 1), tile_size * (y + 1)), 30)
    font = pygame.font.Font('freesansbold.ttf', 20)

    for move_num, move in enumerate(moves):
        move_num += 1
        x, y = move
        if move_num % 2 == 0:
            pygame.draw.circle(screen, WHITE, (tile_size * (x + 1), tile_size * (y + 1)), 20)
            render_move_num = font.render(str(move_num), True, BLACK)
            screen.blit(render_move_num, (tile_size * (x + 1) - 11, tile_size * (y + 1) - 10))
        else:
            pygame.draw.circle(screen, BLACK, (tile_size * (x + 1), tile_size * (y + 1)), 20)
            render_move_num = font.render(str(move_num), True, WHITE)
            screen.blit(render_move_num, (tile_size * (x + 1) - 11, tile_size * (y + 1) - 10))

    # pygame.display.update()


coord_list = []
for i in range(gf.HEIGHT):
    temp_list = []
    y = i * tile_size + 25
    for o in range(gf.WIDTH):
        x = o * tile_size + 25
        temp_list.append([x, y, x + tile_size, y + tile_size])
    coord_list.append(temp_list)


def human_move(position, input_board):
    board = input_board.copy()
    cursor_x, cursor_y = position
    for y_move, row in enumerate(coord_list):
        for x_move, (start_x, start_y, end_x, end_y) in enumerate(row):
            if start_y <= cursor_y <= end_y and start_x <= cursor_x <= end_x:
                if board[y_move][x_move] == 0:
                    return x_move, y_move
                return None, None
    return None, None


w, h = gf.WIDTH, gf.HEIGHT

game = Womoku()
# game.put((6, 6))
# game.put((7, 7))
# game.put((8, 6))
# game.put((7, 6))
# game.put((7, 5))
# game.put((5, 7))
# game.put((6, 7))
# game.put((6, 5))
# game.put((8, 7))
# game.put((8, 5))
# game.put((8, 9))
# game.put((7, 8))
# game.put((8, 10))
# game.put((8, 8))
# game.put((8, 4))
# game.put((6, 8))
# game.put((9, 3))
# game.put((10, 2))
# game.put((9, 8))
# game.put((7, 10))
# game.put((7, 9))
# game.put((5, 8))
# game.put((4, 8))
# game.put((6, 9))
#
# game.put((7, 7))
# game.put((5, 7))
# game.put((8, 6))
# game.put((6, 8))
# game.put((7,9))
# game.put((6, 6))
# game.put((7, 5))
# game.put((7, 8))
# game.put((9, 7))
# game.put((10, 8))
# game.put((6, 4))
# game.put((5, 3))
# game.put((8, 8))
# game.put((6, 10))
# game.put((8, 7))
# game.put((6, 7))
# game.put((6, 9))
# game.put((5, 6))

generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
sess_options = rt.SessionOptions()
# generation = 8
# sess_options.add_session_config_entry('session.inter_op_thread_affinities', '3,4;5,6;7,8;9,10;11,12;13,14;15,16')
print(f"Using generation: {generation}")
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 1
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = f"alphazero/onnx_optimized/{generation}.onnx"
session = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                              providers=gf.PROVIDERS, sess_options=sess_options)
mcts = MCTS(game, session, explore=False, c_puct=4, max_nodes_infer=2)

total_iterations = 1500
time_limit = None

move_list = []

mode = 1
# mode 3, human_player_move = 1, for human against human

# mode 1 for the bottom
human_player_move = 1
# 1 for human becoming the first move
# 2 for human to become the second move
# 3 for AI to play against itself

# I no longer know hos this piece of code works, and I probably never will
# When this code only God and I knew how it worked
# Now only God knows
if mode == 1:
    if human_player_move == 1:
        move_player_black = 0
        move_player_white = 1
    elif human_player_move == 2:
        move_player_black = 0
        move_player_white = 0
    elif human_player_move == 3:
        move_player_black = 0
        move_player_white = 0
else:
    if human_player_move == 1:
        move_player_black = 0
        move_player_white = 0

fast = True
ponder = False
turn = len(game.moves)
player_that_won = 0
won_player = -2
enter_pressed = 0
iterations = 0

shown_result = False

# clock = pygame.time.Clock()
running = True
while running:
    # draw the stuff, background and board
    draw_background()
    draw_board()

    pygame_event = pygame.event.get()
    if won_player == -2:
        move = (None, None)
        current_player = gf.get_next_player(np.array(game.board))
        if current_player == -1 and won_player == -2:
            if (1 if current_player == -1 else 2) % human_player_move == move_player_black:
                if pygame.mouse.get_pressed()[0]:
                    move = human_move(pygame.mouse.get_pos(), game.board)

            else:
                if fast:
                    move, line = mcts.run(iteration_limit=total_iterations, time_limit=time_limit)
                    print(line)
                else:
                    if iterations == 0:
                        bar = tqdm(total=total_iterations)
                    if iterations <= total_iterations:
                        mcts.iteration_step()
                        iterations += 1
                        bar.update(1)
                    elif iterations >= total_iterations:
                        move, line = mcts.run(iteration_limit=0)
                        iterations = 0
                        bar.close()

        elif current_player == 1 and won_player == -2:
            if (1 if current_player == -1 else 2) % human_player_move == move_player_white:
                if pygame.mouse.get_pressed()[0]:
                    move = human_move(pygame.mouse.get_pos(), game.board)
            else:
                if fast:
                    move, line = mcts.run(iteration_limit=total_iterations, time_limit=time_limit)
                    print(line)
                else:
                    if iterations == 0:
                        bar = tqdm(total=total_iterations)
                    if iterations <= total_iterations:
                        mcts.iteration_step()
                        iterations += 1
                        bar.update(1)
                    elif iterations >= total_iterations:
                        move, line = mcts.run(iteration_limit=0)
                        iterations = 0
                        bar.close()
    if move != (None, None) and won_player == -2:
        print(move)
        game.put(move)
        if mode != 3:
            mcts.update_tree(game, move)
        won_player = gf.check_won(np.array(game.board), move)
        turn += 1

    # if fast and ponder and len(game.moves) != 0:
    #     if human_player_move == 1 and gf.get_next_player(np.array(game.board)) == -1:  # if the human player is 1
    #         mcts.iteration_step()
    #     elif human_player_move == 1 and gf.get_next_player(np.array(game.board)) == 1:
    #         mcts.iteration_step()

    if won_player != -2 and not shown_result:
        if won_player == -1:
            print("BLACK has won")
        elif won_player == 1:
            print("WHITE has won")
        elif won_player == 0:
            print("GAME was a draw")
        shown_result = True

    # if len(game.moves) == 1:
    #     draw_move(game.moves)
    # # if len(game.moves) != 0 and len(game.moves) == 1:
    # #     draw_move(game.moves)
    # elif len(game.moves) > 1:
    draw_move(game.moves[:turn])

    keys = pygame.key.get_pressed()
    for event in pygame_event:
        if event.type == pygame.KEYDOWN:
            if keys[pygame.K_RETURN]:
                enter_pressed += 1
                # if enter_pressed % 1 == 1:
                #     game_number = len(glob("played_games/*.psq"))
                #     with open(f"played_games/{game_number}.psq", 'a') as data:
                #         for x, y in move_list:
                #             data.write(f"\n{str(x)},{str(y)}")
                if enter_pressed % 2 == 1:
                    print(game.moves)
                    # reset the game
                    player_that_won = 0
                    won_player = -2
                    shown_result = False
                    # reset the board and move list
                    game = Womoku()
                    mcts = MCTS(game, session, explore=False, c_puct=4.25)


            elif keys[pygame.K_LEFT]:
                if turn > 0:
                    turn -= 1
            elif keys[pygame.K_RIGHT]:
                if turn < len(game.moves):
                    turn += 1

    # print(pygame.mouse.get_pos())
    pygame.display.update()

    # pygame.display.flip()
    for event in pygame_event:
        if event.type == pygame.QUIT:
            running = False
    # clock.tick(65)
pygame.quit()
