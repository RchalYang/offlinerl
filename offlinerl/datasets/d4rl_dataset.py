from torch.utils.data import Dataset
import gym
import d4rl
import numpy as np
from gym.wrappers import TimeLimit

MAZE2D_ENVS = [
    'maze2d-open-v0',
    'maze2d-umaze-v0',
    'maze2d-medium-v0',
    'maze2d-large-v0',
    'maze2d-open-dense-v0',
    'maze2d-umaze-dense-v0',
    'maze2d-medium-dense-v0',
    'maze2d-large-dense-v0'
]

MINIGRID_ENVS = [
    'minigrid-fourrooms-v0',
    'minigrid-fourrooms-random-v0'
]

ADROIT_ENVS = [
    'pen-human-v0',
    'pen-cloned-v0',
    'pen-expert-v0',
    'hammer-human-v0',
    'hammer-cloned-v0',
    'hammer-expert-v0',
    'relocate-human-v0',
    'relocate-cloned-v0',
    'relocate-expert-v0',
    'door-human-v0',
    'door-cloned-v0',
    'door-expert-v0'
]

MUJOCO_ENVS = [
    'halfcheetah-random-v0',
    'halfcheetah-medium-v0',
    'halfcheetah-expert-v0',
    'halfcheetah-mixed-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-random-v0',
    'walker2d-medium-v0',
    'walker2d-expert-v0',
    'walker2d-mixed-v0',
    'walker2d-medium-expert-v0',
    'hopper-random-v0',
    'hopper-medium-v0',
    'hopper-expert-v0',
    'hopper-mixed-v0',
    'hopper-medium-expert-v0'
]

ANTMAZE_ENVS = [
    'antmaze-umaze-v0',
    'antmaze-umaze-diverse-v0',
    'antmaze-medium-play-v0',
    'antmaze-medium-diverse-v0',
    'antmaze-large-play-v0',
    'antmaze-large-diverse-v0'
]


FRANKAKITCHEN_ENVS = [
    'mini-kitchen-microwave-kettle-light-slider-v0',
    'kitchen-microwave-kettle-light-slider-v0',
    'kitchen-microwave-kettle-bottomburner-light-v0',
]

D4RL_ENVS = MAZE2D_ENVS + MINIGRID_ENVS + ADROIT_ENVS + MUJOCO_ENVS + \
    ANTMAZE_ENVS + FRANKAKITCHEN_ENVS


def get_d4rl_env_dataset(id):
    assert id in D4RL_ENVS
    env = gym.make(id)
    if isinstance(env, TimeLimit):
        dataset = env.env.get_dataset()
    else:
        dataset = env.get_dataset()
    return env, D4RLDataset(dataset)


class D4RLDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.obs = self.dataset["observations"]
        self.acts = self.dataset["actions"]
        # self.next_obs = self.dataset["observation"]
        self.rews = self.dataset["rewards"]
        self.dones = self.dataset["terminals"]

        self.len = self.obs.shape[0]

    def __getitem__(self, index):
        return {
            "obs": self.obs[index],
            "acts": self.acts[index],
            "next_obs": self.obs[index+1],
            "rews": self.rews[index, np.newaxis],
            "dones": self.dones[index, np.newaxis]
        }

    def __len__(self):
        return self.len-1
