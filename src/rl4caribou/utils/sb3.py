import yaml
import os

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, HER, DDPG
from sb3_contrib import TQC, ARS, RecurrentPPO

def algorithm(algo):
    algos = {
        'PPO': PPO, 
        'ppo': PPO,
        'RecurrentPPO': RecurrentPPO,
        'RPPO': RecurrentPPO,
        'recurrentppo': RecurrentPPO,
        'rppo': RecurrentPPO,
        #
        'ARS': ARS,
        'ars': ARS,
        'A2C': A2C, 
        'a2c':A2C ,
        #
        'DDPG': DDPG, 
        'ddpg': DDPG,
        #
        'HER': HER, 
        'her': HER,
        #
        'SAC': SAC, 
        'sac': SAC,
        #
        'TD3': TD3, 
        'td3': TD3,
        #
        'TQC': TQC, 
        'tqc': TQC,
    }
    return algos[algo]

def sb3_train(config_file, **kwargs):
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)
        options = {**options, **kwargs}
        # updates / expands on yaml options with optional user-provided input

    if "n_envs" in options:
        env = make_vec_env(
            options["env_id"], options["n_envs"], env_kwargs={"config": options["config"]}
        )
    else:
        env = gym.make(options["env_id"])
    ALGO = algorithm(options["algo"])
    model_id = options["algo"] + "-" + options["env_id"]  + "-" + options["id"]
    save_id = os.path.join(options["save_path"], model_id)

    model = ALGO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=options["tensorboard"],
        use_sde=options["use_sde"],
    )

    progress_bar = options.get("progress_bar", False)
    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id, progress_bar=progress_bar)

    os.makedirs(options["save_path"], exist_ok=True)
    model.save(save_id)
    print(f"Saved {options['algo']} model at {save_id}")
    
    return model