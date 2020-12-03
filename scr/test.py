from envs.grid_world import GridWorld
import numpy as np


def get_rmin_rmax(rewards):

    rmin = np.inf
    rmax = - np.inf

    for value in rewards.values():
        rmin = np.min([np.min(value), rmin])
        rmax = np.max([np.max(value), rmax])

    return rmin, rmax


def normalize_rewards(rewards, rmin, rmax):

    for key in rewards.keys():
        rewards[key] = (rewards[key] - rmin) / (rmax - rmin)


def initialize_metric(num_states):

    return np.zeros((num_states, num_states), dtype=np.float32)


def iterate_metric(metric, env):

    new_metric = np.zeros_like(metric)

    for s1 in range(env.num_states):
        for s2 in range(env.num_states):

            tmp = []

            for a in ["up", "down", "left", "right"]:
                r1 = env.rewards[a][s1]
                r2 = env.rewards[a][s2]

                sp1 = env.next_states[a][s1]
                sp2 = env.next_states[a][s2]

                d = metric[sp1, sp2]

                val = 0.1 * np.abs(r1 - r2) + 0.9 * d
                tmp.append(val)

            new_metric[s1, s2] = np.max(tmp)

    return new_metric


env = GridWorld("tmp", grid_file="res/mirrored_rooms.grid", add_noise=False)
rmin, rmax = get_rmin_rmax(env.rewards)
normalize_rewards(env.rewards, rmin, rmax)

m = initialize_metric(env.num_states)

for i in range(50):
    m = iterate_metric(m, env)
    print(m)

env.bisim_metric = m
env.pretty_print_metric()

env.compute_exact_metric(verbose=True)
