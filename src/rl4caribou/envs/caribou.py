import gymnasium as gym
import numpy as np

# pop = elk, caribou, wolves
# Population dynamics
def dynamics(pop, effort, harvest_fn, p, timestep=1):
    pop = harvest_fn(pop, effort)
    M, B, W = pop[0], pop[1], pop[2] # moose, caribou, wolf
    p['a_B(t)'] = p['a_B'] * (1 + min(3, 6 * timestep / 800))
    # print(p['a_B(t)'])
    
    denominator  = (1 + B**p['x'] * p['h_B'] * p['a_B(t)'] + M**p['x'] * p['h_M'] * p['a_M'])

    B_zero = 0 if B==0 else 1

    zero_B_mask = np.float32([1, B_zero, 1])
    # wolf and moose could randomly move into the habitat from elsewhere
    
    return np.clip(
        zero_B_mask * np.float32([
            M + 0.2 * M * (
                p['r_m'] * (1 - p['alpha_mm'] * M / p['K_m'])
                - M**(p['x'] - 1) * W * p['a_M'] / denominator
                - p['r_m'] * p['alpha_mb'] * B / p['K_m']
                + p['sigma_M'] * np.random.normal()
            ) + p['additive_sigma'] * np.random.normal(),
            #
            B + 0.2 * B * (
                p['r_b']  * (1 - p['alpha_bb'] * B / p['K_b'])
                -  B**(p['x']-1) * W * p['a_B(t)']  / denominator
                - p['r_b'] * p['alpha_bm'] * M / p['K_b']
                + p['sigma_B'] * np.random.normal()
            ) + p['additive_sigma'] * np.random.normal(),
            #
            W + 0.2 *W * (
                B**(p['x']) * p['a_B'] /  denominator
                + M**(p['x']) * p['a_M'] * p['u'] / denominator
                - p['d']
                + p['sigma_W'] * np.random.normal()
            ) + p['additive_sigma'] * np.random.normal(), 
        ]),
        a_min = np.float32([0,0,0]),
        a_max=None,
    )


##
## Param vals from notebooks/discrete_time.ipynmb experiments
##
# am = {"current": 15.32, "full_rest": 11.00}
# ab = {"current": 51.45, "full_rest": 26.39}

parameters = {
    "r_m": np.float32(0.6),
    "r_b": np.float32(0.6),
    #
    "alpha_mm": np.float32(0.1),
    "alpha_bb": np.float32(0.1),
    "alpha_bm": np.float32(0.1),
    "alpha_mb": np.float32(0.1),
    #
    # "a_M": am["current"],
    # "a_B": ab["current"],
    "a_M": 1,
    "a_B": 2,
    #
    "K_m": np.float32(1.1),
    "K_b": np.float32(0.40),
    #
    "h_M": np.float32(0.11),
    "h_B": np.float32(0.11),
    #
    "x": np.float32(2),
    "u": np.float32(1),
    "d": np.float32(1),
    #
    "sigma_M": np.float32(0.1),
    "sigma_B": np.float32(0.2),
    "sigma_W": np.float32(0.1),
    "additive_sigma": np.float32(0.003),
}


##
## Harvest, utility
##
def harvest(pop, effort):
    q0 = 1  # catchability coefficients -- erradication is impossible
    q2 = 1
    pop[0] = pop[0] * (1 - effort[0] * q0)  # pop 0, moose
    pop[2] = pop[2] * (1 - effort[1] * q2)  # pop 2, wolves
    return pop


def utility(pop, effort, env):
    benefit_vec = [0.2, 0.5, 0.2]
    benefits = sum(benefit_vec * pop) # benefit from populations
    costs = 0.1 * effort[0] + 0.2 * effort[1]  # cost to culling

    thresholds = [env.initial_pop[0], 0.1, env.initial_pop[2]]
    for i, pop_i in enumerate(pop):
        if pop_i < thresholds[i]:
            benefits -= 5 * (thresholds[i] - pop_i)
        if (pop_i == 0) and (i==1):
            # caribou crash
            benefits -= 10
        if (pop_i == 0) and not (i==1):
            # other crash
            benefits -= 5
    return 0.001 * (benefits - costs)

def triv_observe(state):
    return state

class Caribou(gym.Env):
    """A 3-species ecosystem model with two control actions"""

    def __init__(self, config=None):
        config = config or {}

        ## these parameters may be specified in config
        self.Tmax = config.get("Tmax", 800)
        self.max_episode_steps = self.Tmax
        self.threshold = config.get("threshold", np.float32(1e-3))
        self.init_sigma = config.get("init_sigma", np.float32(1e-3))
        self.training = config.get("training", True)
        self.initial_pop = config.get("initial_pop", np.ones(3, dtype=np.float32))
        self.parameters = config.get("parameters", parameters)
        self.dynamics = config.get("dynamics", dynamics)
        self.harvest = config.get("harvest", harvest)
        self.utility = config.get("utility", utility)
        self.observe = config.get(
            "observe", triv_observe
        )  # default to perfectly observed case
        self.bound = 10

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
        # self.initial_pop = self.initial_pop * (1 + self.init_sigma * np.random.normal(size=3))
        self.state = self.state_units(self.initial_pop)
        info = {}
        observation = self.observe(self.state)
        return observation, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        pop = self.population_units()  # current state in natural units
        effort = (action + 1.0) / 2

        # harvest and recruitment
        reward = self.utility(pop, effort, self)
        nextpop = self.dynamics(
            pop, effort, self.harvest, self.parameters, self.timestep
        )

        self.timestep += 1
        terminated = bool(self.timestep > self.Tmax)

        # # in training mode only: punish for population collapse
        # if any(pop <= self.threshold) and self.training:
        #     terminated = True
        #     reward -= 100 / self.timestep

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
