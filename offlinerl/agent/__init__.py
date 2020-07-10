import torch
import os.path as osp
from .ddpg import DDPGAgent
from .twin_sac_q import TwinSACQAgent
from .sac import SACAgent


def get_agent(agent_id, env, agent_params):
    if agent_id == "ddpg":
        agent_cls = DDPGAgent

    if agent_id == "twin_sac_q":
        agent_cls = TwinSACQAgent

    if agent_id == "sac":
        agent_cls = SACAgent

    return agent_cls(env=env, **agent_params)
