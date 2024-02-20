# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
from rl4caribou import Caribou

def test_Caribou():
    check_env(Caribou(), warn=True)

