from .d4rl_dataset import D4RL_ENVS, get_d4rl_env_dataset


def get_dataset_env(id):
    if id in D4RL_ENVS:
        env, dataset = get_d4rl_env_dataset(id)
        return env, dataset
