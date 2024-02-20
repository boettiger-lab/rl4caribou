from rl4caribou.envs.caribou import Caribou
from gymnasium.envs.registration import register
register(id="Caribou-v0", entry_point="rl4caribou.envs.caribou:Caribou")
