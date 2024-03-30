import os
import h5py
import time
import numpy as np
from glob import glob
import gc
from multiprocessing.shared_memory import SharedMemory

from MCTS_virtualloss import softmax, Node

# HDF5 file would have
# child indexes # uint32
# parent indexes # create a list and add the node's index to that list for parent indexes
# parent_visits
# move # uint8
# visits # uint16
# sum_value # float 16
# prop_prior # float16


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def Create_domain(path: str) -> None:
    with h5py.File(path, "w", libver="latest") as hdf:
        hdf.create_dataset("moves", shape=(1, 2), maxshape=(None, 2), chunks=True, dtype=np.uint8, compression=9)
        hdf.create_dataset("visits", shape=1, maxshape=(None,), chunks=True, dtype=np.uint32,
                           compression=9)  # node visits
        hdf.create_dataset("p_v", shape=(1, 2), maxshape=(None, 2), chunks=True, dtype=np.float32, compression=9)
        hdf.create_dataset("child_indexes", shape=(1, 225), maxshape=(None, 225), chunks=True,
                           dtype=np.uint32, compression=9)  # index 0 can never be a child as it is the true root


def _nodes_to_stats(stats: list, node: Node, starting_index: int = 0, delta: float = 0.15):
    len_stats = len(stats)
    current_index = int(starting_index) + len_stats
    if starting_index > 0:
        assert node.move is not None
    move = np.array(node.move if node.move is not None else [0, 0], dtype=np.uint8)
    visits = np.array(node.visits, dtype=np.uint32)
    p_v = np.array([node.prob_prior, node.value], dtype=np.float32)
    child_indexes = np.zeros(225, dtype=np.uint32)
    stats.append([move, visits, p_v, child_indexes])

    sum_child_visits = sum([child.visits for child in node.children])
    if sum_child_visits == 0 or sum_child_visits < (node.visits * delta):
        del sum_child_visits
        return current_index
    del sum_child_visits

    for i, child in enumerate(node.children):
        index = _nodes_to_stats(stats, child, starting_index, delta)
        stats[len_stats][-1][i] = index

    return current_index


class Dummy_Lock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Domain:
    def __init__(self, path: str = None,
                 generation: int = max([path.split("\\")[-1][0] for path in glob("alphazero/onnx_models/*.onnx")]),
                 worker_id=0,
                 lock=None,
                 c_puct: float = 4,
                 tau: float = 1,
                 virtual_loss: float = 1,
                 delta: float = 0.1):
        # path overwrites generation
        # Sanity checks
        if path is None and generation is None:
            raise ValueError(f"Both path and generation is None")

        self.path = path
        if path is None:
            self.path = f"alphazero/domains/{generation}.h5"
        if not os.path.exists(self.path):
            Create_domain(self.path)

        self.worker_id = worker_id
        self.lock = lock
        if self.lock is None:
            self.lock = Dummy_Lock()

        self.c_puct = c_puct
        self.tau = tau

        # with self.lock, h5py.File(self.path, mode="r+", locking=False, libver="latest") as hdf:
        with self.lock, h5py.File(self.path, mode="r", locking=False, libver="latest") as hdf:
            self.size = hdf["moves"].shape[0]
            print("pass")

        self.delta = delta
        self.virtual_loss = virtual_loss
        self.visit_inc = np.array([1], dtype=np.uint32)
        self.node_paths = dict()

    def get_path(self) -> str:
        return self.path

    def get_move(self, node: int) -> (int, int):
        with self.lock, h5py.File(self.path, mode="r", locking=False, libver="latest") as hdf:
            move = (*hdf["moves"][node],)
        return move

    def get_node_visits(self, node: int) -> int:
        with self.lock, h5py.File(self.path, mode="r", locking=False, libver="latest") as hdf:
            visits = hdf["visits"][node]
        return visits

    def update_size(self):
        """
        Intermediate method should not be used outside of this class
        :return: None
        """
        with h5py.File(self.path, mode="r", libver="latest") as hdf:
            self.size = hdf["moves"].shape[0]

    def domain_expansion(self, inc, hdf_class=None) -> None:
        """
        Intermediate method, shouldn't be used outside of this class
        :param hdf_class: The hdf class initialized by another method
        :param inc: Number of additions to the array, i.e., the amount of 0's to add to the array/domains to extend it
        :return: None
        """
        self.size += inc
        if hdf_class is not None:
            manager = hdf_class
        else:
            manager = h5py.File(self.path, "r+", libver="latest")
        with manager as hdf:
            hdf["moves"].resize((self.size, 2))
            hdf["visits"].resize((self.size,))
            hdf["p_v"].resize((self.size, 2))
            hdf["child_indexes"].resize((self.size, 225))

    def tree_to_domain(self, true_root: Node, overwrite=False) -> None:
        """
        :param true_root: The starting node of the tree, not the normal root but the very first one in the game Womoku
        :param overwrite: Whether to overwrite the current domains or not
        :return: None
        """
        if true_root.move is not None or len(true_root.children) != 225:
            raise ValueError(
                f"Root given isn't the true root, the true root shouldn't have a move as it's children should be the starting moves")
        elif self.size > 1 and not overwrite:
            raise ValueError(f"Domain has already been created, to add more nodes use update_domain")

        if overwrite and os.path.exists(self.path):
            os.remove(self.path)
            Create_domain(self.path)
        stats = []
        _nodes_to_stats(stats, true_root, delta=self.delta)
        gc.collect()
        self.domain_expansion(len(stats) - 1)
        assert len(stats) > 0
        with self.lock, h5py.File(self.path, mode="r+", locking=False, libver="latest") as hdf:
            moves, visits, p_v, child_indexes = list(zip(*stats))
            to_index = len(moves)
            hdf["moves"][:to_index] = moves
            hdf["visits"][:to_index] = visits
            hdf["p_v"][:to_index] = p_v
            hdf["child_indexes"][:to_index] = child_indexes
            del moves, visits, p_v, child_indexes, stats
            gc.collect()

    def traverse(self, node: int, child_path: list, c_puct: [None, float], use_virtual: bool = False) -> int:
        """
        :param node: domains node
        :param child_path: List to append domains children to
        :param c_puct: A value used to control the PUCT algorithm, if None self.c_puct is used
        :param use_virtual: Whether to use virtual loss or not
        :return: The chosen domains node
        """

        with self.lock, h5py.File(self.path, mode="r+" if use_virtual else "r", locking=False, libver="latest") as hdf:
            parent_visits = hdf["visits"][node]
            children = hdf["child_indexes"][node]
            children = children[children > 0]
            while len(children) > 0:
                best_child = None
                best_score = -np.inf
                for child in children:
                    child_visits = hdf["visits"][child]
                    if child_visits == 0:
                        best_child = child
                        if use_virtual:
                            hdf["p_v"][child] -= self.virtual_loss
                            hdf["visits"][child] += self.visit_inc
                        break
                    prob_prior, value = hdf["p_v"][child]
                    PUCT_score = (value / child_visits) + (
                            prob_prior * ((parent_visits ** 0.5) / (child_visits + 1)) *
                            (c_puct + (np.log10((parent_visits + 19653) / 19652))))
                    if PUCT_score >= best_score:
                        best_child = child
                        best_score = PUCT_score
                # apply virtual loss and add visits
                if use_virtual:
                    hdf["p_v"][node] -= self.virtual_loss
                    hdf["visits"][node] += self.visit_inc
                node = best_child
                parent_visits = hdf["visits"][node]
                children = hdf["child_indexes"][node]
                children = children[children > 0]
                child_path.append(node)
        return best_child

    def stochastic_sampling(self, hdf: h5py.File, node: int, child_path: list, tau: [None, float] = None,
                            explore: bool = True):
        """
        :param hdf: H5py class for opening the .h5 file
        :param node: starting node index of the domains
        :param child_path: A list to append the domains child to
        :param tau: Temp for controlling the sharpening/flattening of the distribution
        :param p_sampling bool, whether to use stochastic sampling
        :param explore: If we use stochastic sampling
        :return:
        """
        if tau is None:
            tau = self.tau
        # nodes and childs are all indexes of the domains, they ARE NOT NODE objects
        child_ids = hdf["child_indexes"][node]
        child_ids = child_ids[child_ids > 0]
        visits = np.zeros(225, dtype=np.uint32)
        for i, child in enumerate(child_ids):
            child_visits = hdf["visits"][child]
            if child_visits == 0:
                child_path.append(child)
                return child
            visits[i] = child_visits
        visits = visits[visits > 0]
        if visits.shape[0] > 0:
            weights = softmax((1.0 / tau) * np.log(visits + 1e-10))
            if explore:
                child = np.random.choice(child_ids, p=weights, size=1, replace=False)[0]
            else:
                child = child_ids[np.argmax(weights)]
        else:
            child = node
        child_path.append(child)
        return child

    def sample(self, sampling: dict[str:int], worker_id: int, c_puct: [None, float] = None,
               tau: [None, float] = None) -> list[int, int]:
        """
        :param sampling: dict with keys s, d
        where each key represents the number of times we sample from the root
        for example {"s":5, "d":True} means to sample 5 nodes stochastically from the root
        and sample deterministically onwards until a node with no visits is found
        If second sampling method is not specified and first is None, then that is used throughout
        If puct is specified, use True and False to determine whether to use virtualloss for different paths
        If not second sampling method is specified and first is a number, puct sampling is used until a node with no visits is found
        There can only be two keys for each dict
        :param worker_id: Used to save the node path into self.node_paths
        :param c_puct: If None self.c_puct is used
        :param tau: If None self.tau will be used
        :return: A list of lists containing the node path to the sampled node
        """
        # Sanity Checks
        if self.size == 0:
            raise ValueError(f"Domain size of 0 cannot be sampled from because there aren't any nodes")
        if not isinstance(worker_id, int):
            raise ValueError(f"We must use the worker_id as a number can't be {worker_id}")
        if (worker_id, False) in self.node_paths or (worker_id, True) in self.node_paths:
            raise ValueError(
                f"Worker {worker_id} has already sampled from this domains, call close worker to delete it from the dict")
        if len(sampling) > 2:
            raise ValueError(f"Len of sampling keys must be 1 or 2")
        if "puct" not in sampling.keys() and None not in sampling.values():
            raise ValueError(
                f"Must sample to a node without children, if not we can't update domains, there's going to be overlap")

        batch_paths, batch_nodes = [], set()
        if tau is None:
            tau = self.tau
        if c_puct is None:
            c_puct = self.c_puct
        while True:
            node = 0
            child_path = []
            is_use_virtual_loss = False
            if len(sampling.keys()) == 1:
                key1 = [*sampling.keys()][0]
                sampling_times = sampling[key1]
                if key1 == "puct" and sampling_times is None:
                    raise ValueError(
                        f"Can't have None for puct sampling, use True for False for using virtual loss")
                if sampling_times is None or (key1 == "puct" and sampling_times is not None):
                    if key1 in ["s", "d"]:
                        with self.lock, h5py.File(self.path, mode="r", locking=False, libver="latest") as hdf:
                            while True:
                                node = self.stochastic_sampling(hdf, node, tau=tau, child_path=child_path,
                                                                explore=True if key1 == "s" else False)
                                if hdf["visits"][node] == 0 or np.sum(hdf["child_indexes"][node]) == 0:
                                    break
                    elif key1 == "puct":
                        node = self.traverse(node, child_path=child_path, c_puct=c_puct, use_virtual=sampling_times)

                        is_use_virtual_loss = sampling_times
                elif key1 in ["s", "d"]:
                    puct_sampling = True
                    with self.lock, h5py.File(self.path, mode="r", locking=False, libver="latest") as hdf:
                        for _ in range(sampling_times):
                            node = self.stochastic_sampling(hdf, node, tau=tau, child_path=child_path,
                                                            explore=True if key1 == "s" else False)
                            if hdf["visits"][node] == 0 or np.sum(hdf["child_indexes"][node]) == 0:
                                puct_sampling = False
                                break
                    if puct_sampling:
                        node = self.traverse(node, child_path, c_puct)

                if node not in batch_nodes:
                    batch_paths.append(child_path)
                    batch_nodes.add(node)
                    self.node_paths[(worker_id, is_use_virtual_loss)] = child_path
                    break
                else:
                    continue

            key1, key2 = sampling.keys()
            sampling_times, _ = sampling.values()

            with self.lock, h5py.File(self.path, mode="r", locking=False, libver="latest") as hdf:
                continue_sampling = True
                if key1 in ["s", "d"]:
                    for _ in range(sampling_times):
                        node = self.stochastic_sampling(hdf, node, tau=tau, child_path=child_path,
                                                        explore=True if key1 == "s" else False)
                        if hdf["visits"][node] == 0 or np.sum(hdf["child_indexes"][node]) == 0:
                            continue_sampling = False
                            break
                if continue_sampling and (("s" == key2 and "d" == key1) or ("d" == key2 and "s" == key1)):
                    while True:
                        node = self.stochastic_sampling(hdf, node, tau=tau, child_path=child_path,
                                                        explore=True if key1 == "s" else False)
                        if hdf["visits"][node] == 0 or np.sum(hdf["child_indexes"][node]) == 0:
                            break
                elif key1 == key2:
                    raise ValueError(f"Two keys where the same")
                # self.node_paths[(worker_id, True)] = child_path

                if node not in batch_nodes:
                    batch_paths.append(child_path)
                    batch_nodes.add(node)
                    break
        return batch_paths[0]

    def get_policy(self, domain_node) -> [[(int, int), float, float, float, float], ...]:
        # We assume that we derive the domains from enough iterations
        with self.lock, h5py.File(self.path, mode="r", locking=False, libver="latest") as hdf:
            children = hdf["child_indexes"][domain_node]
            children = children[children > 0]

            line = [0] * len(children)

            visits = [0] * len(children)
            for i, child in enumerate(children):
                visits[i] = hdf["visits"][child]
            policy = softmax(1.0 * np.log(np.array(visits) + 1e-10))

            for i, child in enumerate(children):
                line[i] = [(*hdf["moves"][child],), policy[i], 0, 0, 0, 0]
        return line

    def backprop_virtual(self, key: [None, (int, bool)], sum_value: float, sum_visits: float):
        """
        Intermediate function, this doesn't have a sanity check nor does it use self._event to protect it from multiple writes
        """
        value_inc = np.array([0, sum_value + self.virtual_loss], dtype=np.float32)
        with h5py.File(self.path, mode="r+", libver="latest") as hdf:
            for node in self.node_paths[key][::-1]:
                hdf["p_v"][node] += value_inc
                hdf["visits"][node] += sum_visits
                sum_value *= -1

    def backprop(self, key: [None, (int, bool)], sum_value: float, sum_visits: float):
        """
        Intermediate function, this doesn't have a sanity check nor does it use self._event to protect it from multiple writes
        """
        value_inc = np.array([0, sum_value], dtype=np.float32)
        with h5py.File(self.path, mode="r+", libver="latest") as hdf:
            for node in self.node_paths[key][::-1]:
                hdf["p_v"][node] += value_inc
                hdf["visits"][node] += sum_visits
                value_inc *= -1

    def update_domain(self, tree_root, chosen_node, key: [None, (str, str)], delta: [None, float] = None) -> None:
        """
        :param tree_root: Node(class) from the tree search with MCTS_virtualloss
        We assume that the chosen node has been poped out except if and only if there is 1 child meaning a winning move
        :param key: The key for self.node_paths
        :param delta: The pruning delta for the nodes saved depending on the amount of visits they have compared to its parent
        given by save if child.visits >= child.parent.visits * self.delta
        :return: None
        # """
        delta = self.delta if delta is None else delta
        if chosen_node.is_terminal is not None:
            delta = 0

        child_path = self.node_paths.get(key)

        domain_root = child_path[-1]  # as it is the last node to the given to the MCTs_virtualloss
        if child_path is None:
            raise KeyError(
                "There are no child paths associated with this key, make sure that they info saved when sampling is accurate")

        with self.lock:
            sub_domain = []
            self.update_size()  # this calls self._event.wait()

            previous_size = self.size
            while True:
                # remember than starting_index is an index not the len of the array
                _nodes_to_stats(stats=sub_domain, node=tree_root, starting_index=self.size-1, delta=delta)
                if len(sub_domain) > 1:
                    break
                sub_domain = []
                delta -= 0.1
                delta = delta if delta > 0 else 0
            print(f"Added {len(sub_domain)} new nodes")

            self.domain_expansion(len(sub_domain))
            # when expanding the domains to fit more nodes, it is interesting because the sub_domain[0](domains index) == node_paths[-1] (domain_index) thus we have to -1 from size, but since we need to fit the chosen_node from the tree, we need to +1 thus cancels out to len(sub_domain)
            # remember that self.size is also updated here, thus self.size = self.size + new_length
            overlapped_node = sub_domain[0]
            sub_domain = sub_domain[1:]

            # attempt the backprop, backprop_virtual
            sum_visits, sum_value = tree_root.visits - chosen_node.visits, tree_root.value - chosen_node.value
            self.backprop(key=key, sum_value=sum_value, sum_visits=sum_visits)

            with h5py.File(self.path, mode="r+", libver="latest") as hdf:
                #Sanity checks
                if not np.array_equal(overlapped_node[0], hdf["moves"][domain_root]):
                    print(self.node_paths)
                    raise ValueError(f"{overlapped_node[0]}, {hdf['moves'][domain_root]}")
                # Update the domains root's children for linking
                if chosen_node.is_terminal is None:
                    overlapped_node[-1][len(overlapped_node[-1][overlapped_node[-1] > 0])] = self.size - 1
                hdf["child_indexes"][domain_root] = overlapped_node[-1]
                # Update the domains root with the chosen root's index, so that it is linked

                # Update the domains with the new children
                moves, visits, p_v, child_indexes = list(zip(*sub_domain))
                hdf["moves"][previous_size: self.size - 1] = moves
                hdf["visits"][previous_size: self.size - 1] = visits
                hdf["p_v"][previous_size: self.size - 1] = p_v
                hdf["child_indexes"][previous_size: self.size - 1] = child_indexes
                # Simple checks to make sure that no bugs occur
                assert np.array_equal(hdf["child_indexes"][self.size - 2], hdf["child_indexes"][-2])
                if chosen_node is not None:
                    # Update the domains with the chosen node's statistics but not the chosen node's children
                    hdf["moves"][-1] = np.array(chosen_node.move, dtype=np.uint32)
                    hdf["visits"][-1] = np.array((chosen_node.visits,), dtype=np.uint32)
                    hdf["p_v"][-1] = np.array([chosen_node.prob_prior, chosen_node.value], dtype=np.float32)
                    assert np.array_equal(hdf["moves"][-1], hdf["moves"][self.size - 1])

            self.node_paths[key].append(self.size - 1)

    def close_worker(self, key: [None, (int, bool)] = None, worker_id: [None, int] = None) -> None:
        if key is None and worker_id is None:
            raise ValueError(f"Both key and worker id is None")
        # Sanity checks and we will perform fill-ins
        if worker_id is not None and key is not None:
            assert key[0] == worker_id
        elif worker_id is not None and key not in self.node_paths:
            raise ValueError(f"Key {key} is not in self.node_path's keys: {self.node_paths.keys()}")

        if key is None:
            try:
                del self.node_paths[(worker_id, False)]
            except:
                try:
                    del self.node_paths[(worker_id, False)]
                except:
                    raise ValueError(
                        f"Couldn't close worker {worker_id} because it wasn't in the dict self.node_paths {self.node_paths}")
            return
        del self.node_paths[key]

    def domain_to_tree(self):
        print("Don't do dat, as we attempt to traverse on the domains rather than loading the domains to tree tree")

    def __repr__(self):
        self.update_size()
        return f"There are {self.size} nodes"


def test(worker_id, lock):
    x = Domain(generation=0, lock=lock)
    x.sample({"puct": True}, worker_id=worker_id)


if __name__ == "__main__":
    import womoku as gf
    from womoku import Womoku
    import onnxruntime as rt
    from MCTS_virtualloss import MCTS
    import multiprocessing as mp

    generation = max([path.split("\\")[-1][0] for path in glob("alphazero/onnx_models/*.onnx")])

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1

    session = rt.InferenceSession(f"alphazero/onnx_models/{generation}.onnx",
                                  providers=gf.PROVIDERS, sess_options=sess_options)

    game = Womoku()
    domain = Domain()
    mcts = MCTS(game=game, session=session, worker_id=0, explore=True, is_apply_dirichlet=True, domain=domain)
    mcts.run(iteration_limit=5200)
    domain.tree_to_domain(mcts.root, overwrite=True)

    # domains = Domain(path="alphazero/domains/0.h5")
    # domain_nodes = domains.sample({"puct":True}, 0)
    # for domain_node in domain_nodes:
    #     move = domains.get_move(domain_node)
    #     game.put(move)
    # mcts = MCTS(session=session, game=game, explore=True, domains=domains)
    # move, line = mcts.run(iteration_limit=20)
    # # domains = Domain(path=None, generation=generation, delta=0.55)
    # # domains.tree_to_domain(mcts.root, overwrite=True)
    #
    # game.put(move)
    # mcts.update_tree_domain(game=game,move=game.moves[-1], domain_key=(0, True))


    # game = Womoku()
    # game.put((7, 7))
    # game.put((6, 7))
    # game.put((7, 6))
    # game.put((6, 6))
    # game.put((7, 5))
    # game.put((6, 5))
    # game.put((7, 4))
    # game.put((7, 3))
    # game.print_board()
    # mcts = MCTS(session=session, game=game, explore=True, domains=domains)
    # move, line = mcts.run(iteration_limit=100)
    # print(move, line)
    # game.put(move)
    #
    #
    # mcts.update_tree_domain(game, game.moves[-1], domain_key=(0, True))



    # num_workers = 2
    # lock = mp.Lock()
    # jobs = []
    # for i in range(num_workers):
    #     p = mp.Process(target=test, args=(i, lock))
    #     jobs.append(p)
    #     p.start()
    # for p in jobs:
    #     p.join()
