from rl4caribou.envs.caribou import Caribou
from rl4caribou.envs.caribou_ode import CaribouScipy
from gymnasium.envs.registration import register

register(id="Caribou-v0", entry_point="rl4caribou.envs.caribou:Caribou")
register(id="CaribouScipy", entry_point="rl4caribou.envs.caribou_ode:CaribouScipy")
