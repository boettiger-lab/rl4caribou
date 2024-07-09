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

def sb3_train(config_file, *, progress_bar, identifier, **kwargs):
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)
        options = {**options, **kwargs}
        # updates / expands on yaml options with optional user-provided input

    if 'additional_imports' in options:
        import importlib 
        for module in options['additional_imports']:
            print(f"importing {module}")
            module = importlib.import_module(module)
            globals()[module.__name__] = module
    
    if "n_envs" in options:
        env = make_vec_env(
            options["env_id"], options["n_envs"], env_kwargs={"config": options["config"]}
        )
    else:
        env = gym.make(options["env_id"])

    if (
        'policy_kwargs' in options['algo_config'] and 
        isinstance(options['algo_config']['policy_kwargs'], str)
    ):
        options['algo_config']['policy_kwargs'] = eval(options['algo_config']['policy_kwargs'])
    
    ALGO = algorithm(options["algo"])
    # if "id" in options:
    #     options["id"] = "-" + options["id"]
    model_id = options["algo"] + "-" + options["env_id"] + "_id_" + identifier
    save_id = os.path.join(options["save_path"], model_id)

    model = ALGO(
        env=env,
        **options['algo_config']
    )

    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id, progress_bar=progress_bar)

    os.makedirs(options["save_path"], exist_ok=True)
    model.save(save_id)
    print(f"Saved {options['algo']} model at {save_id}")
    
    return save_id, options
    
def sb3_train_old(config_file, **kwargs):
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
    if "id" in options:
        options["id"] = "-" + options["id"]
    model_id = options["algo"] + "-" + options["env_id"]  + options.get("id", "")
    save_id = os.path.join(options["save_path"], model_id)

    model = ALGO(
        options.get("policyType", "MlpPolicy"),
        env,
        verbose=0,
        tensorboard_log=options["tensorboard"],
        **{opt: options[opt] for opt in options if opt in ['use_sde']}, # oof, something nicer soon?
    )

    progress_bar = options.get("progress_bar", False)
    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id, progress_bar=progress_bar)

    os.makedirs(options["save_path"], exist_ok=True)
    model.save(save_id)
    print(f"Saved {options['algo']} model at {save_id}")
    
    return save_id, options