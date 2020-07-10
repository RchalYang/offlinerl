import torch
import os.path as osp
from .ddpg import DDPGAgent


def get_agent(agent_id, env, agent_params):
    if agent_id == "ddpg":
        agent_cls = DDPGAgent

    return agent_cls(env=env, **agent_params)
