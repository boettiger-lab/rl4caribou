import gymnasium as gym
import numpy as np

# pop = elk, caribou, wolves
# Population dynamics
def dynamics(pop, effort, harvest_fn, p, timestep=1):
    pop = harvest_fn(pop, effort)
    M, B, W = pop[0], pop[1], pop[2] # moose, caribou, wolf
    denominator  = (1 + B**p['x'] * p['h_B'] * p['a_B'] + M**p['x'] * p['h_M'] * p['a_M'])
    
    return np.float32([
        M + M * (
            p['r_m'] * (1 - p['alpha_mm'] * M / p['K_m'])
            - M**(p['x'] - 1) * W * p['a_M'] / denominator
            - p['r_m'] * p['alpha_mb'] * B / p['K_m']
            + p['sigma_M'] * np.random.normal()
        ),
        #
        B + B * (
            p['r_b']  * (1 - p['alpha_bb'] * B / p['K_b'])
            -  B**(p['x']-1) * W * p['a_B']  / denominator
            - p['r_b'] * p['alpha_bm'] * M / p['K_b']
            + p['sigma_B'] * np.random.normal()
        ),
        #
        W + W * (
            B**p['x'] * p['a_B'] /  denominator
            + M**p['x'] * p['a_M'] * p['u'] / denominator
            - p['d']
            + p['sigma_W'] * np.random.normal()
        ),
    ])


##
## Param vals taken from https://doi.org/10.1016/j.ecolmodel.2019.108891
##
am = {"current": 15.32, "full_rest": 11.00}
ab = {"current": 51.45, "full_rest": 26.39}

parameters = {
    "r_m": np.float32(0.39),
    "r_b": np.float32(0.30),
    #
    "alpha_mm": np.float32(1),
    "alpha_bb": np.float32(1),
    "alpha_bm": np.float32(1),
    "alpha_mb": np.float32(1),
    #
    "a_M": am["current"],
    "a_B": ab["current"],
    #
    "K_m": np.float32(1.1),
    "K_b": np.float32(0.40),
    #
    "h_M": np.float32(0.112),
    "h_B": np.float32(0.112),
    #
    "x": np.float32(2),
    "u": np.float32(1),
    "d": np.float32(1),
    #
    "sigma_M": np.float32(0.1),
    "sigma_B": np.float32(0.1),
    "sigma_W": np.float32(0.1),
}


##
## Harvest, utility
##
def harvest(pop, effort):
    q0 = 0.5  # catchability coefficients -- erradication is impossible
    q2 = 0.5
    pop[0] = pop[0] * (1 - effort[0] * q0)  # pop 0, moose
    pop[2] = pop[2] * (1 - effort[1] * q2)  # pop 2, wolves
    return pop


def utility(pop, effort):
    benefits = 0.5 * pop[1]  # benefit from Caribou
    costs = 0.00001 * (effort[0] + effort[1])  # cost to culling
    if np.any(pop <= 0.01):
        benefits -= 1
    return benefits - costs

class Caribou(gym.Env):
    """A 3-species ecosystem model with two control actions"""

    def __init__(self, config=None):
        config = config or {}

        ## these parameters may be specified in config
        self.Tmax = config.get("Tmax", 800)
        self.max_episode_steps = self.Tmax
        self.threshold = config.get("threshold", np.float32(1e-4))
        self.init_sigma = config.get("init_sigma", np.float32(1e-3))
        self.training = config.get("training", True)
        self.initial_pop = config.get("initial_pop", np.ones(3, dtype=np.float32))
        self.parameters = config.get("parameters", parameters)
        self.dynamics = config.get("dynamics", dynamics)
        self.harvest = config.get("harvest", harvest)
        self.utility = config.get("utility", utility)
        self.observe = config.get(
            "observe", lambda state: state
        )  # default to perfectly observed case
        self.bound = 2

        self.action_space = gym.spaces.Box(
            np.array([-1, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.reset(seed=config.get("seed", None))

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.initial_pop += np.multiply(
            self.initial_pop, np.float32(self.init_sigma * np.random.normal(size=3))
        )
        self.state = self.state_units(self.initial_pop)
        info = {}
        return self.observe(self.state), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        pop = self.population_units()  # current state in natural units
        effort = (action + 1.0) / 2

        # harvest and recruitment
        reward = self.utility(pop, effort)
        nextpop = self.dynamics(
            pop, effort, self.harvest, self.parameters, self.timestep
        )

        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)

        # in training mode only: punish for population collapse
        if any(pop <= self.threshold) and self.training:
            terminated = True
            reward -= 50 / self.timestep

        self.state = self.state_units(nextpop)  # transform into [-1, 1] space
        observation = self.observe(self.state)  # same as self.state
        return observation, reward, terminated, False, {}

    def state_units(self, pop):
        self.state = 2 * pop / self.bound - 1
        self.state = np.clip(
            self.state,
            np.repeat(-1, self.state.__len__()),
            np.repeat(1, self.state.__len__()),
        )
        return np.float32(self.state)

    def population_units(self):
        pop = (self.state + 1) * self.bound / 2
        return np.clip(
            pop, np.repeat(0, pop.__len__()), np.repeat(np.Inf, pop.__len__())
        )


# verify that the environment is defined correctly
# from stable_baselines3.common.env_checker import check_env
# env = s3a2()
# check_env(env, warn=True)
