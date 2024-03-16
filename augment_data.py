import os
import zipfile

from tqdm import tqdm
import numpy as np
from glob import glob
import womoku as gf


def Augment_data(worker_id, files, generation=None):
    game_num = 0
    for file in tqdm(files):
        try:
            data = np.load(file, allow_pickle=True)
        except:
            data = np.load(file, allow_pickle=False)
        augmented_game_inputs = []
        augmented_game_polices = []
        augmented_game_value = []
        for move_num in range(data["inputs"].shape[0]):
            inputs = data["inputs"][move_num]
            policy = data["policy"][move_num].reshape((gf.HEIGHT, gf.WIDTH))
            value = data["value"][move_num]

            augmented_inputs = []
            augmented_policies = []
            values = []

            for k in range(4):
                # rotate k times and add to the list
                augmented_game_state = []
                for state_id in range(gf.NUM_PREVIOUS_MOVES + 1):
                    inputs_rot = np.rot90(inputs[state_id], k)
                    augmented_game_state.append(inputs_rot)

                augmented_inputs.append(augmented_game_state)
                policy_rot = np.rot90(policy, k)
                augmented_policies.append(policy_rot)
                values.append(value)

                # flip and add to the list
                augmented_game_state = []
                for state_id in range(gf.NUM_PREVIOUS_MOVES + 1):
                    inputs_rot = np.rot90(inputs[state_id], k)
                    inputs_flip = np.fliplr(inputs_rot)
                    augmented_game_state.append(inputs_flip)

                augmented_inputs.append(augmented_game_state)
                policy_flip = np.fliplr(policy_rot)
                augmented_policies.append(policy_flip)
                values.append(value)

            # return as numpy arrays with required shape
            augmented_game_inputs.append(np.stack(augmented_inputs).reshape((-1, *gf.SHAPE[1:])))
            augmented_game_polices.append(np.stack(augmented_policies).reshape((-1, gf.WIDTH * gf.HEIGHT)))
            augmented_game_value.append(np.stack(values))
        for tries in range(10):
            try:
                np.savez_compressed(f"alphazero/games/{generation + 1}/{worker_id}_{game_num}.npz",
                                    inputs=np.array(augmented_game_inputs, dtype=np.int8),
                                    policy=np.array(augmented_game_polices, dtype=np.float32),
                                    value=np.array(augmented_game_value, dtype=np.float32))

                # data = np.load(f"alphazero/games/{generation + 1}/{game_num}.npz", allow_pickle=True)
                # dummy_inputs = data["inputs"]
                # dummy_policy = data["policy"]
                # dummy_value = data["value"]
                #
                # del dummy_inputs, dummy_policy, dummy_value
                break
            except:
                print(f"Failed try {tries}")
                print(f'alphazero/games/{generation + 1}/{game_num}.npz')
        game_num += 1


if __name__ == "__main__":
    import math
    import multiprocessing as mp
    workers = 6
    generation = max([int(path.split("\\")[-1][0:]) for path in glob("alphazero/models/*")])
    print(f"Augmenting data generation: {generation + 1} for future model generation: {generation + 1}")

    os.makedirs(f"alphazero/games/{generation + 1}", exist_ok=True)
    files = glob(f"alphazero/games_unrot/{generation + 1}/*.npz")
    files_per_worker = math.ceil(len(files) / workers)
    worker_files = [None for _ in range(workers)]
    for i in range(workers):
        if i == workers - 1:
            worker_files[i] = files[files_per_worker * i:]
        worker_files[i] = files[files_per_worker * i: files_per_worker * (i + 1)]
    jobs = []
    for worker_id in range(workers):
        p = mp.Process(target=Augment_data, args=(worker_id, worker_files[worker_id], generation,))
        p.start()
        jobs.append(p)
    for job in jobs:
        job.join()
