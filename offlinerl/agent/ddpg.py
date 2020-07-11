import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from offlinerl.agent.base_agent import Agent
import offlinerl.policies as policies
import offlinerl.networks as networks


class DDPGAgent(Agent):
    """
    DDPG
    """
    def __init__(
            self,
            env,
            policy_params,
            q_params,
            plr, qlr,
            optimizer_class=optim.Adam,
            **kwargs
    ):
        super(DDPGAgent, self).__init__(env=env, **kwargs)

        self.pf = policies.get_policy(
            input_shape=self.env.observation_space.shape,
            output_shape=self.env.action_space.shape[0],
            policy_cls=policies.FixGuassianContPolicy,
            policy_params=policy_params)
        self.target_pf = copy.deepcopy(self.pf)

        self.qf = networks.get_network(
            input_shape=(self.env.observation_space.shape[0] +
                         self.env.action_space.shape[0],),
            output_shape=1,
            network_cls=networks.FlattenNet,
            network_params=q_params)
        self.target_qf = copy.deepcopy(self.qf)

        self.to(self.device)

        self.plr = plr
        self.qlr = qlr

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qlr,
        )

        self.qf_criterion = nn.MSELoss()

    def update(self, batch):
        self.training_update_num += 1

        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rews']
        terminals = batch['dones']

        rewards = torch.Tensor(rewards).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)
        obs = torch.Tensor(obs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)

        """
        Policy Loss.
        """

        new_actions = self.pf(obs)
        new_q_pred = self.qf([obs, new_actions])

        policy_loss = -new_q_pred.mean()

        """
        QF Loss
        """
        target_actions = self.target_pf(next_obs)
        target_q_values = self.target_qf([next_obs, target_actions])

        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        q_pred = self.qf([obs, actions])
        assert q_pred.shape == q_target.shape
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self._update_target_functions()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/qf_loss'] = qf_loss.item()

        info['new_actions/mean'] = new_actions.mean().item()
        info['new_actions/std'] = new_actions.std().item()
        info['new_actions/max'] = new_actions.max().item()
        info['new_actions/min'] = new_actions.min().item()

        return info

    @property
    def functions(self):
        return [
            self.pf,
            self.qf,
            self.target_pf,
            self.target_qf
        ]

    @property
    def snapshot_functions(self):
        return [
            ["pf", self.pf],
            ["qf", self.qf],
        ]

    @property
    def target_functions(self):
        return [
            (self.pf, self.target_pf),
            (self.qf, self.target_qf)
        ]
