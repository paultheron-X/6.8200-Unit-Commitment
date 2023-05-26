# Robust Guided Forest Search for Unit Commitment

Vassili Chesterkine, Thomas Wright, Paul Theron

---

This is the course project for 6.8200 Computational Sensorimotor Learning. The course is taught by [Prof. Pulkit Agrawam]().

Report is available [here](./report.pdf).

## Project Description
In that project, we explore the unit commitment problem, which is a combinatorial optimization problem that aims to find the optimal set of generators to be turned on and off in order to meet the demand of the system while minimizing the cost of the system, using reinforcement learning.

## Method
*Our project introduces a unique solution to the unit commitment problem through reinforcement learning, utilizing a guided forest search algorithm that leverages an RL agent as a heuristic guide. Our innovative method aggregates best action predictions by constructing a forest of trees based on perturbations around the tree's initial state.*

To deal with the exponentially growing action space inherent in Unit Commitment (UC) problems, we adopted local tree search methods on trained policies. Nodes in our Markov decision process (MDP) represent observations, with edges representing feasible actions. At each node (state), only the top M most likely actions, according to the trained policy π, are considered as possible actions, ensuring the search tree's complexity remains scalable.

One key observation we made is that an RL policy can be sensitive to initial conditions and state. We found that by building trees rooted at each perturbed initial state, we explored at least 7% more of the action space. This led us to propose a new search algorithm called Robust Guided Forest Search algorithm.

This algorithm constructs a forest, with each tree corresponding to a perturbation of the initial state. Each tree begins with actions suggested by the trained policy applied to the perturbed state. The tree expands to a set depth, and each leaf node accumulates the cost of the actions leading to it. A Uniform Cost Search (UCS) retrieves the minimum-cost path and the first action along that path for each timestep of the episode. The final action decision is made based on one of two path-finding strategies, either minimizing worst-case costs across scenarios or minimizing average costs across scenarios, which can be tailored to specific robustness requirements of the power system.

This approach focuses on improving the robustness of the algorithm, which is crucial when dealing with power systems, particularly under conditions of poor or highly variable supply or demand forecasts.

### Environment

Our work will use the environment provided by the [RL4UC](https://github.com/pwdemars/rl4uc). The environment is using the gym open ai interface.

## Environment

To create the environment, run the following command in the terminal:

```bash

conda create -n sensorimotor_env python=3.9

conda activate sensorimotor_env
```

To install the required packages, run the following command

```bash

pip install -r requirements.txt

```


