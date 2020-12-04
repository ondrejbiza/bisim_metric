import torch
import torch.nn as nn
from constants import Constants


class Sim(nn.Module):

    def __init__(self, encoder, target_encoder, config):

        super(Sim, self).__init__()

        c = config
        self.gamma = c[Constants.GAMMA]

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.sync()

    def loss(self, states, actions, rewards, next_states, beta):

        # we have N transitions, create all N^2 possible transition pairs
        state_pairs = self.all_state_pairs_(states)
        action_pairs = self.all_state_pairs_(actions[:, None])
        reward_pairs = self.all_state_pairs_(rewards[:, None])
        next_state_pairs = self.all_state_pairs_(next_states)

        # only select pairs where a_1 == a_2
        # TODO: maybe don't do pairs of the same state?
        mask = action_pairs[:, 0] == action_pairs[:, 1]
        state_pairs = state_pairs[mask]
        reward_pairs = reward_pairs[mask]
        next_state_pairs = next_state_pairs[mask]

        # get state pair distances and next state pair distances using online and target networks
        state_dists = self.encoder(state_pairs)[:, 0]
        state_target_dists = self.target_encoder(state_pairs).detach()[:, 0]
        next_state_target_dists = self.target_encoder(next_state_pairs).detach()[:, 0]

        # get pair reward differences
        reward_diffs = torch.abs(reward_pairs[:, 0] - reward_pairs[:, 1])

        # calculate targets and compute squared loss
        target1 = reward_diffs + self.gamma * beta * next_state_target_dists
        target2 = beta * state_target_dists
        target = torch.max(target1, target2)

        loss = (state_dists - target) ** 2
        return loss.mean()

    def all_state_pairs_(self, states):

        batch_size = states.size(0)
        latent_size = states.size(1)

        states1 = states[:, None, :].repeat(1, batch_size, 1)
        states2 = states[None, :, :].repeat(batch_size, 1, 1)

        return torch.cat([states1, states2], dim=2).reshape((batch_size ** 2, 2 * latent_size))

    def sync(self):

        self.target_encoder.load_state_dict(self.encoder.state_dict())
