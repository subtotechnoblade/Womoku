import numpy as np
from glob import glob
from tqdm import tqdm
import womoku as gf
import gc


def load_data(alpha_files: list):
    """
    :param alpha_files: list of file paths as strings
    :return: x_train, policy_labels, value_labels
    """
    x_train, policy_labels, value_labels = [0 for _ in range(len(alpha_files))], [0 for _ in
                                                                                  range(len(alpha_files))], [
        0 for _ in range(len(alpha_files))]
    # winning_value = math.log2(1024)
    ones, neg_ones = 0, 0
    divi = 8
    for i, file in enumerate(alpha_files):
        data = np.load(file, allow_pickle=True)
        if 0 <= np.sum(data["value"].reshape(-1)):
            ones += np.sum(data["value"].reshape(-1)) / divi
        else:
            neg_ones += abs(np.sum(data["value"].reshape(-1))) / divi
    wins, losses = 0, 0
    for i, file in enumerate(alpha_files):
        data = np.load(file, allow_pickle=True)
        if ones >= neg_ones:
            if 0 <= np.sum(data["value"].reshape(-1)) and wins <= neg_ones:  # if it is a one
                x_train[i] = data["inputs"]
                policy_labels[i] = data["policy"]
                value_labels[i] = data["value"]
                wins += np.sum(data["value"].reshape(-1)) / divi
            elif np.sum(data["value"].reshape(-1)) <= 0:
                x_train[i] = data["inputs"]
                policy_labels[i] = data["policy"]
                value_labels[i] = data["value"]
                losses += abs(np.sum(data["value"].reshape(-1))) / divi
        else:
            if np.sum(data["value"].reshape(-1)) <= 0 and losses * 1.1 <= ones:
                x_train[i] = data["inputs"]
                policy_labels[i] = data["policy"]
                value_labels[i] = data["value"]
                losses += abs(np.sum(data["value"].reshape(-1))) / divi
            elif 0 <= np.sum(data["value"].reshape(-1)):
                x_train[i] = data["inputs"]
                policy_labels[i] = data["policy"]
                value_labels[i] = data["value"]
                wins += np.sum(data["value"].reshape(-1)) / divi
    print(wins, losses)
    x_temp, pi_temp, v_temp = [], [], []
    for X, pi, val in zip(x_train, policy_labels, value_labels):
        if not isinstance(X, int):
            x_temp.append(X)
            pi_temp.append(pi)
            v_temp.append(val)
    x_train, policy_labels, value_labels = x_temp, pi_temp, v_temp

    x_train, policy_labels, value_labels = np.concatenate((x_train)).reshape((-1, *gf.SHAPE[1:])), np.concatenate(
        (policy_labels)).reshape((-1, 4)), np.concatenate((value_labels)).reshape(-1, 1)

    x_train_black, x_train_white, policy_labels_black, policy_labels_white, value_labels_black, value_labels_white = (
        [0 for _ in range(black_wins)], [0 for _ in range(white_wins)], [0 for _ in range(black_wins)],
        [0 for _ in range(white_wins)], [0 for _ in range(black_wins)], [0 for _ in range(white_wins)])

    black_id = 0
    white_id = 0
    for id, file in enumerate(tqdm(alpha_files)):
        data = np.load(file, allow_pickle=True)
        if np.sum(data["value"]) == 1:
            x_train_black[black_id] = data["inputs"]
            policy_labels_black[black_id] = data["policy"]
            value_labels_black[black_id] = data["value"]
            black_id += np.sum(data["value"])
        else:
            x_train_white[white_id] = data["inputs"]
            policy_labels_white[white_id] = data["policy"]
            value_labels_white[white_id] = data["value"]
            white_id += np.sum(data["value"])
    x_train_black, x_train_white, policy_labels_black, policy_labels_white, value_labels_black, value_labels_white = (
        np.concatenate((x_train_black)).reshape((-1, *gf.SHAPE[1:-1])),
        np.concatenate((x_train_white)).reshape((-1, *gf.SHAPE[1:-1])),
        np.concatenate((policy_labels_black)).reshape((-1, gf.HEIGHT * gf.WIDTH)),
        np.concatenate((policy_labels_white)).reshape((-1, gf.HEIGHT * gf.WIDTH)),
        np.concatenate((value_labels_black)).reshape((-1, 1)),
        np.concatenate((value_labels_white)).reshape((-1, 1)))
    print(x_train_black.shape)
    print(x_train_white.shape)

    # print(policy_labels_black.shape)
    # print(policy_labels_white.shape)
    # print(value_labels_black.shape)
    # print(value_labels_white.shape)
    # raise ValueError
    # smaller, larger = (white_wins, black_wins) if black_wins >= white_wins else (black_wins, white_wins)

    def split_arrays(larger, smaller):
        len_larger, len_smaller = len(larger), len(smaller)
        # smaller = np.concatenate((smaller))
        if len_larger % len_smaller == 0:
            window_size = int(len_larger / len_smaller)
            return_arr = [0 for _ in range(window_size)]
            for i in range(window_size):
                return_arr[i] = np.concatenate((smaller, larger[i * len_smaller: (i + 1) * len_smaller]))
        elif 2 * len_smaller > len_larger:
            return_arr = [0 for _ in range(2)]
            return_arr[0] = np.concatenate((smaller, larger[:len_smaller]))
            return_arr[1] = np.concatenate((smaller, larger[(len_larger - len_smaller):]))
        else:
            window_size = len_larger // len_smaller
            return_arr = [0 for _ in range(window_size)]
            offset = len_smaller - len_larger % len_smaller
            return_arr[0] = np.concatenate((smaller, larger[:len_smaller]))
            for i in range(1, window_size):
                return_arr[i] = np.concatenate(
                    (smaller, larger[(i * len_smaller) - offset: (i + 1) * len_smaller - offset]))
            return_arr[-1] = np.concatenate((smaller, larger[(i + 1) * len_smaller - offset:]))
        return return_arr

    if black_wins >= white_wins:  # blackwins is larger than white_wins
        x_train = np.array(split_arrays(x_train_black, x_train_white), np.int8)
        policy_labels = np.array(split_arrays(policy_labels_black, policy_labels_white), np.float32)
        value_labels = np.array(split_arrays(value_labels_black, value_labels_white), np.float32)
    else:
        x_train = np.array(split_arrays(x_train_white, x_train_black), np.int8)
        policy_labels = np.array(split_arrays(policy_labels_white, policy_labels_black), np.float32)
        value_labels = np.array(split_arrays(value_labels_white, value_labels_black), np.float32)
    del x_train_black, x_train_white, policy_labels_black, policy_labels_white, value_labels_black, value_labels_white
    gc.collect()
    print(x_train.shape)
    print(policy_labels.shape)
    print(value_labels.shape)
    return x_train, policy_labels, value_labels


if __name__ == "__main__":
    files = glob(f"alphazero/games/{6}/*.npz")
    load_data(files)
