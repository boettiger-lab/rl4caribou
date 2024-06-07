import gymnasium as gym
import numpy as np
from scipy.integrate import odeint
# import nbkode

def dynamics_scipy(pop, effort, p, timestep, singularities):
    #
    # parameters of the ODE are s.t. t is in years, so lets make the time-step a tenth of a year
    # (this ad hoc rule gives better convergence than if we set dt = 1 full year)
    dt = 1./12
    t_interval = np.float32([timestep, timestep+dt])
    y0 = pop 
    timestep_randomness = (
        np.float32(
            [p['sigma_M'],  p['sigma_B'],  p['sigma_W']]
        ) *
        np.random.normal(size=3)
    )
    new_pop = odeint(ode_func, y0, t_interval, args=(effort, p), tcrit=singularities)[1]
    
    return new_pop + timestep_randomness * dt

def ode_func(y, t, effort, p):
    M, B, W = y
    denominator  = (1 + B**p['x'] * p['h_B'] * p['a_B'] + M**p['x'] * p['h_M'] * p['a_M'])
    return np.float32([
        M * (
            p['r_m'] * (1 - p['alpha_mm'] * M / p['K_m'])
            - M**(p['x'] - 1) * W * p['a_M'] / denominator
            - p['r_m'] * p['alpha_mb'] * B / p['K_m']
            - effort[0]
        ),
        #
        B * (
            p['r_b']  * (1 - p['alpha_bb'] * B / p['K_b'])
            -  B**(p['x']-1) * W * p['a_B']  / denominator
            - p['r_b'] * p['alpha_bm'] * M / p['K_b']
        ),
        #
        W * (
            B**p['x'] * p['a_B'] /  denominator
            + M**p['x'] * p['a_M'] * p['u'] / denominator
            - p['d']
            -  effort[1]
        ),
    ])


##
## Param vals taken from https://doi.org/10.1016/j.ecolmodel.2019.108891
##
#am = {"current": 15.32, "full_rest": 11.00}
#ab = {"current": 51.45, "full_rest": 26.39}

#parameters = {
#    "r_m": np.float32(0.39),
#    "r_b": np.float32(0.30),
    #
#    "alpha_mm": np.float32(1),
#    "alpha_bb": np.float32(1),
#    "alpha_bm": np.float32(1),
#    "alpha_mb": np.float32(1),
    #
 #   "a_M": am["current"],
#    "a_B": ab["current"],
    #
#    "K_m": np.float32(1.1),
#    "K_b": np.float32(0.40),
    #
#    "h_M": np.float32(0.112),
#    "h_B": np.float32(0.112),
    #
#    "x": np.float32(2),
#    "u": np.float32(1),
#    "d": np.float32(1),
    #
#    "sigma_M": np.float32(0.05),
#    "sigma_B": np.float32(0.05),
#    "sigma_W": np.float32(0.05),
#}
#
# computed using scipy's fsolve (coordinates where d Pops / dt = 0)
singularities = [
    np.array([1.1000000238415355, 0, 0]),
    np.array([0.26788470722361574, 0.022792841445996415, 0.07873609043869849]),
    np.array([0, 0.4000000059610799, 0]),
    np.array([0, 0.14794518267150766, 0.027967723350836204]),
    np.array([0.2711216103310206, 0, 0.0796758786194987]),
    np.array([0, 0, 0]),
]

# def numba_func(y,t,effort):
#     global parameters
#     return ode_func(y, t, effort, parameters)

# def dynamics_numba(t, pop, effort):
#     #
#     y0 = pop 
#     solver = nbkode.ForwardEuler(numba_func, t, y0, params=effort)
#     #
#     t_interval = np.float32([t, t+1])
#     ts, ys = solver.run(t_interval)
#     return ys[1]
    

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
    benefits = 1 * pop[1]  # benefit from Caribou
    costs = 0.1 * (effort[0] + effort[1]) + 0.4 * effort[2]  # cost to culling + cost of restoring
    if np.any(pop <= [0.05,  0.01, 0.001]):
        benefits -= 1
    return benefits - costs

def observe_3pop_restoration(env):
    rest_obs = -1 + 2 * (
        (env.current_ab - env.parameters["a_B"]) 
        / (env.current_ab - env.restored_ab)
    )
    return np.float32([*env.state, rest_obs])

class CaribouScipy(gym.Env):
    """A 3-species ecosystem model with two control actions"""

    def __init__(self, config=None):
        config = config or {}

        ## these parameters may be specified in config
        self.Tmax = config.get("Tmax", 800)
        self.max_episode_steps = self.Tmax
        self.threshold = config.get("threshold", np.float32(1e-4))
        self.init_sigma = config.get("init_sigma", np.float32(1e-3))
        self.training = config.get("training", True)
        self.initial_pop = np.float32(config.get("initial_pop", [0.268, 0.023, 0.079]))
        #
        self.current_am = 15.32
        self.restored_am = 11.00
        self.current_ab = 51.45
        self.restored_ab = 26.39
        self.parameters = {
            "r_m": config.get("r_m", np.float32(0.39)),
            "r_b": config.get("r_b", np.float32(0.30)),
            #
            "alpha_mm": config.get("alpha_mm", np.float32(1)),
            "alpha_bb": config.get("alpha_bb", np.float32(1)),
            "alpha_bm": config.get("alpha_bm", np.float32(1)),
            "alpha_mb": config.get("alpha_mb", np.float32(1)),
            #
            "a_M": self.current_am,
            "a_B": self.current_ab,
            #
            "K_m": config.get("K_m", np.float32(1.1)),
            "K_b": config.get("K_b", np.float32(0.40)),
            #
            "h_M": config.get("h_M", np.float32(0.112)),
            "h_B": config.get("h_B", np.float32(0.112)),
            #
            "x": config.get("x", np.float32(2)),
            "u": config.get("u", np.float32(1)),
            "d": config.get("d", np.float32(1)),
            #
            "sigma_M": config.get("sigma_M", np.float32(0.05)),
            "sigma_B": config.get("sigma_B", np.float32(0.05)),
            "sigma_W": config.get("sigma_W", np.float32(0.05)),
        }
        self.a_restoration_change = config.get("a_restoration_change", 0.1)
        #
        self.singularities = config.get("singularities", None)
        self.dynamics = config.get("dynamics", dynamics_scipy)
        self.harvest = config.get("harvest", harvest)
        self.utility = config.get("utility", utility)
        self.observe = config.get(
            "observe", observe_3pop_restoration
        )  # default to perfectly observed case
        self.bound = 2

        self.action_space = gym.spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1, -1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.reset(seed=config.get("seed", None))

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.true_initial_pop = self.initial_pop + np.multiply(
            self.initial_pop, np.float32(self.init_sigma * np.random.normal(size=3))
        )
        self.state = self.state_units(self.true_initial_pop)
        info = {}
        return self.observe(self), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        pop = self.population_units()  # current state in natural units
        effort = (action + 1.0) / 2 # (moose_cull, wolf_cull, restoration_intensity)

        # harvest and recruitment
        nextpop = self.dynamics(
            pop, effort, self.parameters, self.timestep, singularities=self.singularities
        )
        
        # restoration
        self.parameters["a_M"] = self.parameters["a_M"] - (self.parameters["a_M"] - self.restored_am) * effort[2] * self.a_restoration_change
        self.parameters["a_B"] = self.parameters["a_B"] - (self.parameters["a_B"] - self.restored_ab) * effort[2] * self.a_restoration_change
        
        ## linear approx to rewards
        reward = self.utility((pop+nextpop)/2., effort)

        self.timestep += 1
        truncated = bool(self.timestep > self.Tmax) # or bool(any(nextpop < 1e-7))
        
        self.state = self.state_units(nextpop)  # transform into [-1, 1] space
        observation = self.observe(self)  # same as self.state
        return observation, reward, False, truncated, {}

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



# class CaribouNumba(gym.Env):
#     """A 3-species ecosystem model with two control actions"""

#     def __init__(self, config=None):
#         config = config or {}

#         ## these parameters may be specified in config
#         self.Tmax = config.get("Tmax", 800)
#         self.max_episode_steps = self.Tmax
#         self.threshold = config.get("threshold", np.float32(1e-4))
#         self.init_sigma = config.get("init_sigma", np.float32(1e-3))
#         self.training = config.get("training", True)
#         self.initial_pop = config.get("initial_pop", np.ones(3, dtype=np.float32))
#         self.parameters = config.get("parameters", parameters)
#         self.dynamics = config.get("dynamics", dynamics_numba)
#         self.harvest = config.get("harvest", harvest)
#         self.utility = config.get("utility", utility)
#         self.observe = config.get(
#             "observe", lambda state: state
#         )  # default to perfectly observed case
#         self.bound = 2

#         self.action_space = gym.spaces.Box(
#             np.array([-1, -1], dtype=np.float32),
#             np.array([1, 1], dtype=np.float32),
#             dtype=np.float32,
#         )
#         self.observation_space = gym.spaces.Box(
#             np.array([-1, -1, -1], dtype=np.float32),
#             np.array([1, 1, 1], dtype=np.float32),
#             dtype=np.float32,
#         )
#         self.reset(seed=config.get("seed", None))

#     def reset(self, *, seed=None, options=None):
#         self.timestep = 0
#         self.initial_pop += np.multiply(
#             self.initial_pop, np.float32(self.init_sigma * np.random.normal(size=3))
#         )
#         self.state = self.state_units(self.initial_pop)
#         info = {}
#         return self.observe(self.state), info

#     def step(self, action):
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         pop = self.population_units()  # current state in natural units
#         effort = (action + 1.0) / 2

#         # harvest and recruitment
#         nextpop = self.dynamics(
#             self.timestep, pop, effort
#         )
#         ## linear approx to rewards
#         reward = self.utility((pop+nextpop)/2., effort)

#         self.timestep += 1
#         terminated = bool(self.timestep > self.Tmax)

#         # in training mode only: punish for population collapse
#         if any(pop <= self.threshold) and self.training:
#             terminated = True
#             reward -= 50 / self.timestepq

#         self.state = self.state_units(nextpop)  # transform into [-1, 1] space
#         observation = self.observe(self.state)  # same as self.state
#         return observation, reward, terminated, False, {}

#     def state_units(self, pop):
#         self.state = 2 * pop / self.bound - 1
#         self.state = np.clip(
#             self.state,
#             np.repeat(-1, self.state.__len__()),
#             np.repeat(1, self.state.__len__()),
#         )
#         return np.float32(self.state)

#     def population_units(self):
#         pop = (self.state + 1) * self.bound / 2
#         return np.clip(
#             pop, np.repeat(0, pop.__len__()), np.repeat(np.Inf, pop.__len__())
#         )