import numpy as np
from glob import glob

generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
files = glob(f"alphazero/games_unrot/{generation + 1}/*npz")

player1_wins = 0
player2_wins = 0
game_length = []
for file in files:
    try:
        data = np.load(file, allow_pickle=True)
    except:
        data = np.load(file, allow_pickle=False)
    # for move in data["inputs"]:
    #     print(move[-1].reshape(-1, 15, 15))
    # raise ValueError
    if data["value"][0][0] == 1:
        player1_wins += 1
    else:
        player2_wins += 1
    game_length.append(len(data["value"]))
print(player1_wins, player2_wins)
print(sum(game_length)/len(game_length))
print(f"Total states: {sum(game_length)/len(game_length) * len(files) * 8}")