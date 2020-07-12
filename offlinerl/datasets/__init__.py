from .d4rl_dataset import D4RL_ENVS, get_d4rl_env_dataset


def get_dataset_env(id, env_params):
    if id in D4RL_ENVS:
        env, dataset = get_d4rl_env_dataset(id, **env_params)
        return env, dataset
