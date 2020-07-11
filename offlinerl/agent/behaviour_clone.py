import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from offlinerl.agent.base_agent import Agent
import offlinerl.policies as policies
import offlinerl.networks as networks


class BehaviorCloneAgent(Agent):
    """
    DDPG
    """
    def __init__(
            self,
            env,
            policy_params,
            plr,
            optimizer_class=optim.Adam,
            **kwargs
    ):
        super(BehaviorCloneAgent, self).__init__(env=env, **kwargs)

        self.pf = policies.get_policy(
            input_shape=self.env.observation_space.shape,
            output_shape=self.env.action_space.shape[0],
            policy_cls=policies.DetContPolicy,
            policy_params=policy_params)

        self.to(self.device)

        self.plr = plr

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.behavior_clone_loss = nn.MSELoss()

    def update(self, batch):
        self.training_update_num += 1

        obs = batch['obs']
        actions = batch['acts']

        obs = torch.Tensor(obs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)

        """
        Policy Loss.
        """

        new_actions = self.pf(obs)

        policy_loss = self.behavior_clone_loss(new_actions, actions)

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        # Information For Logger
        info = {}
        info['Training/policy_loss'] = policy_loss.item()

        info['new_actions/mean'] = new_actions.mean().item()
        info['new_actions/std'] = new_actions.std().item()
        info['new_actions/max'] = new_actions.max().item()
        info['new_actions/min'] = new_actions.min().item()

        return info

    @property
    def functions(self):
        return [
            self.pf
        ]

    @property
    def snapshot_functions(self):
        return [
            ["pf", self.pf]
        ]
