import numpy as np
import copy
import queue
from tree_search_utils import expansion
from tree_search_utils import node as node_mod
import torch.multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import time

import logging

logging.basicConfig(level=logging.INFO)

class ObservationCurrupter:
    def __init__(self, corruption_rate):
        self.corruption_rate = corruption_rate

    def corrupt(self, observation):
        corrupted_observation = observation.copy()
        random_mask = np.random.rand(*corrupted_observation.shape) < self.corruption_rate
        random_values = np.random.randn(*corrupted_observation.shape) * corrupted_observation
        corrupted_observation[random_mask] = random_values[random_mask]
        return corrupted_observation

class BoxCorrupter:
    def __init__(self, corruption_rate):
        self.corruption_rate = corruption_rate

    def corrupt(self, node):
        node_ts = node.state.episode_timestep
        # move all the demand and wind scenarios after the current timestep by +- 10% #TODO: to be changed given the arma process generating the errors
        ep_forecast = node.state.episode_forecast
        ep_wind_forecast = node.state.episode_wind_forecast

        ep_forecast[node_ts + 1 :] = ep_forecast[node_ts + 1 :] * (
            1
            + np.random.uniform(
                -self.corruption_rate, self.corruption_rate, size=ep_forecast[node_ts + 1 :].shape
            )
        )
        ep_wind_forecast[node_ts + 1 :] = ep_wind_forecast[node_ts + 1 :] * (
            1
            + np.random.uniform(
                -self.corruption_rate,
                self.corruption_rate,
                size=ep_wind_forecast[node_ts + 1 :].shape,
            )
        )

        node.state.episode_forecast = ep_forecast
        node.state.episode_wind_forecast = ep_wind_forecast
        return node

class TreeSearch:
    @classmethod
    def uniform_cost_search_forest(cls, node, terminal_timestep, **policy_kwargs):
        """Uniform cost search for forest"""
        if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
            return node_mod.get_solution(node)
        frontier = queue.PriorityQueue()
        frontier.put(
            (0, id(node), node)
        )  # include the object id in the priority queue. prevents type error when path_costs are identical.
        while True:
            assert frontier, "Failed to find a goal state"
            node = frontier.get()[2]
            if node.state.is_terminal() or node.state.episode_timestep == terminal_timestep:
                return node_mod.get_solution(node)
            actions = expansion.get_actions(node, **policy_kwargs)
            for action in actions:
                child = expansion.get_child_node(
                    node=node,
                    action=action,
                    demand_scenarios=None,
                    wind_scenarios=None,
                    global_outage_scenarios=None,
                )
                
                node.children[action.tobytes()] = child
                frontier.put((child.path_cost, id(child), child))

            # Early stopping if root has one child
            if node.parent is None and len(actions) == 1:
                return [actions[0]], 0


class TreeBuilder:
    def __init__(self, root_node, num_trees, terminal_timestep, corruption_rate, **policy_kwargs):
        self.root_node = root_node
        self.terminal_timestep = terminal_timestep
        # self.demand_scenarios = demand_scenarios
        # self.wind_scenarios = wind_scenarios
        # self.global_outage_scenarios = global_outage_scenarios
        self.policy_kwargs = policy_kwargs
        self.corruption_rate = corruption_rate
        self.obs_corrupter = BoxCorrupter(corruption_rate)

        self.num_trees = num_trees

        self.forest = None

    def build_forest(self):
        corrupted_root_nodes = self._corrupt_root_nodes()

        forest_actions = []

        for tree_ind in range(self.num_trees):
            path, cost = TreeSearch.uniform_cost_search_forest(
                node=corrupted_root_nodes[tree_ind],
                terminal_timestep=self.terminal_timestep,
                **self.policy_kwargs
            )
            forest_actions.append((path, cost))

        optimal_path, optimal_cost = self._get_majority_action_path(forest_actions)
        return optimal_path, optimal_cost

    def _corrupt_root_nodes(self):
        corrupted_root_nodes = []
        for _ in range(self.num_trees):
            new_node = copy.deepcopy(self.root_node)
            corrupted_root_nodes.append(self.obs_corrupter.corrupt(new_node))
        return corrupted_root_nodes

    def _get_optimal_path(self, forest_actions):
        optimal_path = None
        optimal_cost = np.inf
        for path, cost in forest_actions:
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_path = path
        return optimal_path, optimal_cost

    def _get_majority_action_path(self, forest_actions):
        majority_action = self._get_majority_action(forest_actions)
        for path, cost in forest_actions:
            if np.array_equal(path[0], majority_action):
                return path, cost
        return None, None

    def _get_majority_action(self, forest_actions):
        action_counts = {}
        for path, cost in forest_actions:
            action = tuple(path[0])
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
        majority_action = max(action_counts, key=action_counts.get)
        return list(majority_action)

class DistributedTreeBuilder(TreeBuilder):
    def __init__(self, num_workers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers

    def build_forest(self):
        corrupted_root_nodes = self._corrupt_root_nodes()

        with mp.Pool(self.num_workers) as pool:
            forest_actions = pool.map(self.worker_uniform_cost_search, corrupted_root_nodes)

        optimal_path, optimal_cost = self._get_majority_action_path(forest_actions)
        return optimal_path, optimal_cost

    def worker_uniform_cost_search(self, node):
        path, cost = TreeSearch.uniform_cost_search_forest(
            node=node,
            terminal_timestep=self.terminal_timestep,
            #policy=policy,
            **self.policy_kwargs
        )
        return path, cost

      