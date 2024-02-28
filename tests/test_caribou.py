# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env
import numpy as np
from rl4caribou import Caribou
from rl4caribou.agents import constAction, constEsc

def test_Caribou():
    check_env(Caribou(), warn=True)

def test_constAction():
    ca1 = constAction(mortality_vec = [0,0])
    ca2 = constAction(mortality_vec = np.zeros(2))
    obs = np.zeros(3)
    pr1, _ = ca1.predict(obs)
    pr2, _ = ca2.predict(obs)

def test_constEsc():
    ce1 = constEsc(escapement_vec = [0,0])
    ce2 = constEsc(escapement_vec = np.zeros(2))
    obs = np.zeros(3)
    pr1, _ = ce1.predict(obs)
    pr2, _ = ce2.predict(obs)

