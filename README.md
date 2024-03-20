# rl4caribou

Using *deep reinforcement learning* techniques to optimize Caribou conservation policies.
Our apporach is based on the methods developed in the
[approx-model-or-approx-soln](https://github.com/boettiger-lab/approx-model-or-approx-soln) project:

[![DOI](https://zenodo.org/badge/572256056.svg)](https://zenodo.org/badge/latestdoi/572256056) 

## Problem statement

Caribous are facing increased rates of predation by wolves due to the presence of *linear features* in
their habitat---mainly seismic lines and roads which facilitate wolves' access to high elevations during
winter.
Due to this, Caribou numbers have dwindled over the past decades.
Policy makers now face a question of resource allocation: how to best use finite financial resources in
order to restore Caribou populations?

We consider three types of actions available to policy makers: 
1. **Seismic line restoration.** Restoring seismic lines to forest is an expensive policy which takes several years to have its effect felt, however it can greatly reduce the wolf-caribou predation rate on the long run.
2. **Wolf culling.** A second, more direct, option is to cull wolves to bring their population density down.
3. **Moose culling.** A third option is to cull moose---wolves' main source of food. Reducing moose population density effectively reduces the carrying capacity for the wolf population.

We use a [community model from the literature](https://www.sciencedirect.com/science/article/abs/pii/S0304380019303990?casa_token=QHYvnZHBOLAAAAAA:pHqdBeT8mVzV3tCaIci_0Yxo3f8lFVcrGF8GaHMC2Ch_8YYD6NG-BfcD6g1eK1cn0kZRPYVxqg) to describe the population dynamics of the Moose - Caribou - Wolf system.
The equations are as follow:

Moose:
$$dM/dt = M (r_m - \frac{(r_m \alpha_{mm} M)}{K_m} - \frac{(M^{x - 1} W a_M)}{(1 + B^x h_B a_B + M^x h_M a_M)} - \frac{(r_m \alpha_{mb} B)}{K_m} - \mu_t)$$

Caribou:
$$dB/dt = B(r_b- \frac{r_b \alpha_{bb} B}{K_b}- \frac{B^{x - 1} W a_B}{1 + M^x h_M a_M + B^x h_B a_B} - \frac{r_b \alpha_{bm}  M}{K_b})$$

Wolves:
$$dW/dt = W(\frac{B^x a_B}{1 + M^x h_M aM + B^x h_B a_B} + \frac{u a_M M^x}{1 + M^x h_M a_M + B^x h_B a_B} - d - \omega_t)$$.

Here $\mu_t$ and $\omega_t$ are, respectively, the moose and wolf mortalities due to culling.
The other parameters in the equations are estimated from empirical data in the reference.

**Problem in plain language:** choose culling mortalities $\omega_t$ and $\mu_t$ such that 1. the caribou population can grow, 2. all three populations are maintained at a healthy level, 3. the cost of implementing the policy is as low as possible.

We operationalize this problem in the following way: 

First, we break down time into discrete timesteps representing one year.
At each time-step, the manager observes the vector of populations $(M_t, B_t, W_t)$ and chooses an *action* $(\mu_t, \omega_t)$.
This action produces a *reward* based on the following utility funciton:

```
def utility(pop, effort):
    # pop = [moose, caribou, wolf]
    benefits = 1 * pop[1]  # benefit from Caribou
    costs = 0.1 * (effort[0] + effort[1])  # cost to culling
    if np.any(pop <= [0.03,  0.07, 1e-4]):
        benefits -= 1 # penalty for having low densities
    return benefits - costs
```

**Technical problem statement:** maximize reward over a period of 100 timesteps by choosing an optimal *policy function* which chooses actions based on observing the population vector $(M_t, B_t, W_t)$.


