import torch
import os.path as osp
from .behaviour_clone import BehaviorCloneAgent
from .ddpg import DDPGAgent
from .twin_sac_q import TwinSACQAgent
from .sac import SACAgent
from .bcq import ContBCQAgent


def get_agent(agent_id, env, agent_params):
    if agent_id == "ddpg":
        agent_cls = DDPGAgent

    if agent_id == "twin_sac_q":
        agent_cls = TwinSACQAgent

    if agent_id == "sac":
        agent_cls = SACAgent

    if agent_id == "behavior_clone":
        agent_cls = BehaviorCloneAgent

    if agent_id == "bcq":
        agent_cls = ContBCQAgent

    return agent_cls(env=env, **agent_params)
