# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class for computing bisimulation metrics on deterministic grid worlds.

The grid worlds created will have the form:
  *****
  *  g*
  *   *
  *****
where a reward of `self.reward_value` (set to `1.`) is received upon
entering the cell marked with 'g', and a reward of `-self.reward_value` is
received upon taking an action that would drive the agent to a wall.

One can also specify a deterministic policy by using '^', 'v', '<', and '>'
characters instead of spaces. The 'g' cell will default to the down action.

The class supports:
- Reading a grid specification from a file or creating a square room from a
  specified wall length.
- Computing the exact bisimulation metric up to a desired tolerance using the
  standard dynamic programming method.
- Computing the exact bisimulation metric using sampled pairs of trajectories.
- Computing the approximate bisimulation distance using the bisimulation loss
  function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np


class GridWorld(object):
  """Class defining deterministic grid world MDPs."""

  def __init__(self,
               base_dir,
               wall_length=2,
               grid_file=None,
               gamma=0.99,
               representation_dimension=64,
               batch_size=64,
               target_update_period=100,
               num_iterations=10000,
               starting_learning_rate=0.01,
               use_decayed_learning_rate=False,
               learning_rate_decay=0.96,
               epsilon=1e-8,
               staircase=False,
               add_noise=True,
               bisim_horizon_discount=0.9,
               double_period_halfway=True,
               total_final_samples=1000,
               debug=False):
    """Initialize a deterministic GridWorld from file.

    Args:
      base_dir: str, directory where to store exact metric and event files.
      wall_length: int, length of walls for constructing a 1-room MDP. Ignored
        if grid_file is not None.
      grid_file: str, path to file defining GridWorld MDP.
      gamma: float, discount factor.
      representation_dimension: int, dimension of each state representation.
      batch_size: int, size of sample batches for the learned metric.
      target_update_period: int, period at which target network weights are
        synced from the online network.
      num_iterations: int, number of iterations to run learning procedure.
      starting_learning_rate: float, starting learning rate for AdamOptimizer.
      use_decayed_learning_rate: bool, whether to use decayed learning rate.
      learning_rate_decay: float, amount by which to decay learning rate.
      epsilon: float, epsilon for AdamOptimizer.
      staircase: bool, whether to decay learning rate at discrete intervals.
      add_noise: bool, whether to add noise to grid points (thereby making
        them continuous) when learning the metric.
      bisim_horizon_discount: float, amount by which to increase the horizon for
        estimating the distance.
      double_period_halfway: bool, whether to double the update period halfway
        through training.
      total_final_samples: int, number of samples to draw at the end of training
        (if noise is on).
      debug: bool, whether we are in debug mode.
    """
    self.base_dir = base_dir
    self.exact_bisim_filename = 'exact_bisim_metric.pkl'
    self.wall_length = wall_length
    self.goal_reward = 1.
    self.wall_penalty = -1.
    self.gamma = gamma
    self.representation_dimension = representation_dimension
    self.batch_size = batch_size
    self.target_update_period = target_update_period
    self.num_iterations = num_iterations
    self.starting_learning_rate = starting_learning_rate
    self.use_decayed_learning_rate = use_decayed_learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.epsilon = epsilon
    self.staircase = staircase
    self.add_noise = add_noise
    self.double_period_halfway = double_period_halfway
    self.bisim_horizon_discount = bisim_horizon_discount
    self.total_final_samples = total_final_samples
    self.debug = debug
    self.raw_grid = []
    # Assume no policy by default. If there is a policy, we will compute
    # the on-policy bisimulation metric.
    self.has_policy = False

    if grid_file is not None:
      with open(grid_file, "r") as f:
        for l in f.readlines():
          print(l)
          self.raw_grid.append(list(l)[:-1])
          # If we see a policy character we assume there is a policy.
          if (not self.has_policy and
                  ('<' in l or '>' in l or '^' in l or 'v' in l)):
            self.has_policy = True
      self.raw_grid = np.array(self.raw_grid)
    else:
      # Either read in from file or build a room using wall_length.
      self.raw_grid.append(['*'] * (wall_length + 2))
      for row in range(wall_length):
        self.raw_grid.append(['*'] + [' '] * wall_length + ['*'])
      self.raw_grid.append(['*'] * (wall_length + 2))
      self.raw_grid = np.array(self.raw_grid)
      self.raw_grid[1, wall_length] = 'g'
    # First make walls 0s and cells 1s.
    self.indexed_states = [
        [0 if x == '*' else 1 for x in y] for y in self.raw_grid]
    # Now do a cumsum to get unique IDs for each cell.
    self.indexed_states = (
        np.reshape(np.cumsum(self.indexed_states),
                   np.shape(self.indexed_states)) * self.indexed_states)
    # Subtract 1 so we're 0-indexed, and walls are now -1.
    self.indexed_states -= 1
    self.num_rows = np.shape(self.indexed_states)[0]
    self.num_cols = np.shape(self.indexed_states)[1]
    # The maximum value on any dimension, used for normalizing the inputs.
    self.max_val = max(self.num_rows - 2., self.num_cols - 2.)
    self.num_states = np.max(self.indexed_states) + 1
    # State-action rewards.
    self.rewards = {}
    # Set up the next state transitions.
    self.actions = ['up', 'down', 'left', 'right']
    # Policy to action mapping.
    policy_to_action = {'^': 'up',
                        'v': 'down',
                        '<': 'left',
                        '>': 'right',
                        'g': 'down'}  # Assume down from goal state.
    # Deltas for 4 actions.
    self.action_deltas = {'up': (-1, 0),
                          'down': (1, 0),
                          'left': (0, -1),
                          'right': (0, 1)}
    self.next_states_grid = {}
    self.next_states = {}
    self.inverse_index_states = np.zeros((self.num_states, 2))
    self.policy = [''] * self.num_states if self.has_policy else None
    # So we only update inverse_index_states once.
    first_pass = True
    # Set up the transition and reward dynamics.
    for action in self.actions:
      self.next_states_grid[action] = np.copy(self.indexed_states)
      self.next_states[action] = np.arange(self.num_states)
      self.rewards[action] = np.zeros(self.num_states)
      for row in range(self.num_rows):
        for col in range(self.num_cols):
          state = self.indexed_states[row, col]
          if self.has_policy and state >= 0:
            self.policy[state] = policy_to_action[self.raw_grid[row, col]]
          if first_pass and state >= 0:
            self.inverse_index_states[state] = [row, col]
          next_row = row + self.action_deltas[action][0]
          next_col = col + self.action_deltas[action][1]
          if (next_row < 1 or next_row > self.num_rows or
              next_col < 1 or next_col > self.num_cols or
              state < 0 or self.indexed_states[next_row, next_col] < 0):
            if state >= 0:
              self.rewards[action][state] = self.wall_penalty
            continue
          if self.raw_grid[next_row, next_col] == 'g':
            self.rewards[action][state] = self.goal_reward
          else:
            self.rewards[action][state] = 0.
          next_state = self.indexed_states[next_row, next_col]
          self.next_states_grid[action][row, col] = next_state
          self.next_states[action][state] = next_state
      first_pass = False
    # Initial metric is undefined.
    self.bisim_metric = None
    # Dictionary for collecting statistics.
    self.statistics = {}

  def compute_exact_metric(self, tolerance=0.001, verbose=False):
    """Compute the exact bisimulation metric up to the specified tolerance.

    Args:
      tolerance: float, maximum difference in metric estimate between successive
        iterations. Once this threshold is past, computation stops.
      verbose: bool, whether to print verbose messages.
    """
    # Initial metric is all zeros.
    self.bisim_metric = np.zeros((self.num_states, self.num_states))
    metric_difference = tolerance * 2.
    i = 1
    exact_metric_differences = []
    start_time = time.time()
    while metric_difference > tolerance:
      new_metric = np.zeros((self.num_states, self.num_states))
      for s1 in range(self.num_states):
        for s2 in range(self.num_states):
          if self.has_policy:
            action1 = self.policy[s1]
            action2 = self.policy[s2]
            next_state1 = self.next_states[action1][s1]
            next_state2 = self.next_states[action2][s2]
            rew1 = self.rewards[action1][s1]
            rew2 = self.rewards[action2][s2]
            new_metric[s1, s2] = (
                abs(rew1 - rew2) +
                self.gamma * self.bisim_metric[next_state1, next_state2])
          else:
            for action in self.actions:
              next_state1 = self.next_states[action][s1]
              next_state2 = self.next_states[action][s2]
              rew1 = self.rewards[action][s1]
              rew2 = self.rewards[action][s2]
              act_distance = (
                  abs(rew1 - rew2) +
                  self.gamma * self.bisim_metric[next_state1, next_state2])
              if act_distance > new_metric[s1, s2]:
                new_metric[s1, s2] = act_distance
      metric_difference = np.max(abs(new_metric - self.bisim_metric))
      exact_metric_differences.append(metric_difference)
      if i % 1000 == 0 and verbose:
        print('Iteration {}: {}'.format(i, metric_difference))
      self.bisim_metric = np.copy(new_metric)
      i += 1
    total_time = time.time() - start_time
    exact_statistics = {
        'tolerance': tolerance,
        'time': total_time,
        'num_iterations': i,
        'metric_differences': exact_metric_differences,
        'metric': self.bisim_metric,
    }
    self.statistics['exact'] = exact_statistics
    print('**** Exact statistics ***')
    print('Number of states: {}'.format(self.num_states))
    print('Total number of iterations: {}'.format(i))
    print('Total time: {}'.format(total_time))
    print('*************************')
    if verbose:
      self.pretty_print_metric()

  def compute_sampled_metric(self, tolerance=0.001, verbose=False):
    """Use trajectory sampling to compute the exact bisimulation metric.

    Will compute until the difference with the exact metric is within tolerance.
    Will try for a maximum of self.num_iterations.

    Args:
      tolerance: float, required accuracy before stopping.
      verbose: bool, whether to print verbose messages.
    """
    self.sampled_bisim_metric = np.zeros((self.num_states, self.num_states))
    exact_metric_errors = []
    metric_differences = []
    start_time = time.time()
    exact_metric_error = tolerance * 10
    i = 0
    while exact_metric_error > tolerance:
      new_metric = np.zeros((self.num_states, self.num_states))
      # Generate a pair of sampled trajectories.
      s1 = np.random.randint(self.num_states)
      s2 = np.random.randint(self.num_states)
      if self.has_policy:
        action1 = self.policy[s1]
        action2 = self.policy[s2]
      else:
        action1 = self.actions[np.random.randint(4)]
        action2 = action1
      next_s1 = self.next_states[action1][s1]
      next_s2 = self.next_states[action2][s2]
      rew1 = self.rewards[action1][s1]
      rew2 = self.rewards[action2][s2]
      if self.has_policy:
        new_metric[s1, s2] = (
            abs(rew1 - rew2) +
            self.gamma * self.sampled_bisim_metric[next_s1, next_s2])
      else:
        new_metric[s1, s2] = max(
            self.sampled_bisim_metric[s1, s2],
            abs(rew1 - rew2) +
            self.gamma * self.sampled_bisim_metric[next_s1, next_s2])
      metric_difference = np.max(
          abs(new_metric - self.sampled_bisim_metric))
      metric_differences.append(metric_difference)
      exact_metric_error = np.max(
          abs(self.bisim_metric - self.sampled_bisim_metric))
      exact_metric_errors.append(exact_metric_error)
      if i % 10000 == 0 and verbose:
        print('Iteration {}: {}'.format(i, metric_difference))
      self.sampled_bisim_metric = np.copy(new_metric)
      i += 1
      if i > self.num_iterations:
        break
    total_time = time.time() - start_time
    exact_sampling_statistics = {
        'time': total_time,
        'tolerance': tolerance,
        'num_iterations': i,
        'sampled_metric_differences': metric_differences,
        'exact_metric_errors': exact_metric_errors,
    }
    self.statistics['exact_sampling'] = exact_sampling_statistics
    print('**** Exact sampled statistics ***')
    print('Number of states: {}'.format(self.num_states))
    print('Total number of iterations: {}'.format(i))
    print('Total time: {}'.format(total_time))
    print('*************************')
    if verbose:
      self.pretty_print_metric()

  def pretty_print_metric(self, metric_type='exact', print_side_by_side=True):
    """Print out a nice grid version of metric.

    Args:
      metric_type: str, which of the metrics to print, possible values:
        ('exact', 'sampled', 'learned').
      print_side_by_side: bool, whether to print side-by-side with the exact
        metric.
    """
    for s1 in range(self.num_states):
      print('From state {}'.format(s1))
      for row in range(self.num_rows):
        for col in range(self.num_cols):
          if self.indexed_states[row, col] < 0:
            sys.stdout.write('**********')
            continue
          s2 = self.indexed_states[row, col]
          if metric_type == 'exact' or print_side_by_side:
            val = self.bisim_metric[s1, s2]
          elif metric_type == 'sampled':
            val = self.sampled_bisim_metric[s1, s2]
          elif metric_type == 'learned':
            val = self.learned_distance[s1, s2]
          else:
            raise ValueError('Unknown metric type: {}'.format(metric_type))
          sys.stdout.write('{:10.4}'.format(val))
        if metric_type != 'exact' and print_side_by_side:
          sys.stdout.write('    ||    ')
          for col in range(self.num_cols):
            if self.indexed_states[row, col] < 0:
              sys.stdout.write('**********')
              continue
            s2 = self.indexed_states[row, col]
            if metric_type == 'sampled':
              val = self.sampled_bisim_metric[s1, s2]
            elif metric_type == 'learned':
              val = self.learned_distance[s1, s2]
            sys.stdout.write('{:10.4}'.format(val))
        sys.stdout.write('\n')
        sys.stdout.flush()
      print('')
