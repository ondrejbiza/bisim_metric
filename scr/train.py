import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from modules.FCEncoder import FCEncoder
from models.Sim import Sim
from logger import Logger
from constants import Constants
from envs.grid_world import GridWorld


def sample_batch(env, batch_size):

    states = np.random.choice(list(range(env.num_states)), size=batch_size, replace=True)
    actions = np.random.choice(list(range(4)), size=batch_size, replace=True)
    names = ["up", "down", "left", "right"]
    action_names = [names[action] for action in actions]
    next_states = [env.next_states[action][state] for state, action in zip(states, action_names)]
    rewards = [env.rewards[action][state] for state, action in zip(states, action_names)]

    states = [env.inverse_index_states[state] + 0.5 for state in states]
    next_states = [env.inverse_index_states[state] + 0.5 for state in next_states]

    return states, actions, rewards, next_states


def batch_to_tensor(states, actions, rewards, next_states, device):

    return torch.tensor(states, dtype=torch.float32, device=device), \
           torch.tensor(actions, dtype=torch.int32, device=device), \
           torch.tensor(rewards, dtype=torch.float32, device=device), \
           torch.tensor(next_states, dtype=torch.float32, device=device)


def main(args):

    batch_size = 256
    device = args.device

    env = GridWorld("tmp", grid_file="res/mirrored_rooms.grid", add_noise=False)

    logger = Logger()

    encoder_config = {
        Constants.INPUT_SIZE: 4,
        Constants.NEURONS: [729, 1],
        Constants.USE_BATCH_NORM: False,
        Constants.USE_LAYER_NORM: False,
        Constants.ACTIVATION_LAST: False
    }

    encoder = FCEncoder(encoder_config, logger)
    target_encoder = FCEncoder(encoder_config, logger)

    sim = Sim(encoder, target_encoder, {
        Constants.GAMMA: 0.99
    }).to(device)

    opt = optim.Adam(params=sim.encoder.parameters(), lr=1e-2)

    one_minus_beta = 1
    beta_exp = 0.9
    update_freq = 500
    num_steps = 240000

    losses = []

    for i in range(num_steps):

        if i > 0 and i % update_freq == 0:
            one_minus_beta *= beta_exp
            sim.sync()

            losses = np.mean(losses)
            print("{:d}/{:d} ({:.0f}%), beta={:.3f}, avg. loss={:.4f}".format(
                i, num_steps, 100 * (i / num_steps), 1 - one_minus_beta, losses
            ))
            losses = []

        beta = 1 - one_minus_beta
        batch = sample_batch(env, batch_size)
        batch = batch_to_tensor(*batch, device)

        opt.zero_grad()
        loss = sim.loss(*batch, beta)
        loss.backward()
        opt.step()

        losses.append(loss.item())

    states = list(range(env.num_states))
    states = [env.inverse_index_states[state] + 0.5 for state in states]
    states = torch.tensor(states, dtype=torch.float32, device=device)

    all_state_pairs = sim.encoder(sim.all_state_pairs_(states)).detach().cpu().numpy()
    all_state_pairs = all_state_pairs[:, 0]
    all_state_pairs = all_state_pairs.reshape((env.num_states, env.num_states))

    env.bisim_metric = all_state_pairs
    env.pretty_print_metric()


parser = argparse.ArgumentParser()
parser.add_argument("device")
parsed = parser.parse_args()

main(parsed)
