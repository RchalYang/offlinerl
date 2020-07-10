import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from offlinerl.agent.base_agent import Agent
import offlinerl.policies as policies
import offlinerl.networks as networks


class SACAgent(Agent):
    """
    SAC
    """
    def __init__(
            self,
            env,
            policy_params,
            q_params,
            v_params,
            plr, qlr, vlr,
            optimizer_class=optim.Adam,
            policy_std_reg_weight=0,
            policy_mean_reg_weight=0,

            reparameterization=True,
            automatic_entropy_tuning=True,
            target_entropy=None,
            **kwargs
    ):
        super(SACAgent, self).__init__(env=env, **kwargs)

        self.pf = policies.get_policy(
            input_shape=self.env.observation_space.shape,
            output_shape=2 * self.env.action_space.shape[0],
            policy_cls=policies.GuassianContPolicy,
            policy_params=policy_params)

        self.qf = networks.get_network(
            input_shape=(self.env.observation_space.shape[0] +
                         self.env.action_space.shape[0],),
            output_shape=1,
            network_cls=networks.FlattenNet,
            network_params=q_params)

        self.vf = networks.get_network(
            input_shape=self.env.observation_space.shape,
            output_shape=1,
            network_cls=networks.Net,
            network_params=v_params)
        self.target_vf = copy.deepcopy(self.vf)

        self.to(self.device)

        self.plr = plr
        self.vlr = vlr
        self.qlr = qlr

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qlr,
        )

        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=self.vlr,
        )

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()  # from rlkit
            self.log_alpha = torch.zeros(1).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=self.plr,
            )

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight

        self.reparameterization = reparameterization

    def update(self, batch):
        self.training_update_num += 1

        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rews = batch['rews']
        terminals = batch['dones']

        rews = torch.Tensor(rews).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)
        obs = torch.Tensor(obs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)

        """
        Policy operations.
        """
        sample_info = self.pf.explore(obs, return_log_probs=True )

        mean = sample_info["mean"]
        log_std = sample_info["log_std"]
        new_actions = sample_info["action"]
        log_probs = sample_info["log_prob"]

        q_pred = self.qf([obs, actions])
        v_pred = self.vf(obs)

        if self.automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (
                log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rews + (1. - terminals) * self.discount * target_v_values
        assert q_pred.shape == q_target.shape
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = self.qf([obs, new_actions])
        v_target = q_new_actions - alpha * log_probs
        assert v_pred.shape == v_target.shape
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        """
        if not self.reparameterization:
            log_policy_target = q_new_actions - v_pred
            policy_loss = (
                log_probs * (alpha * log_probs - log_policy_target).detach()
            ).mean()
        else:
            policy_loss = (alpha * log_probs - q_new_actions).mean()

        std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()

        policy_loss += std_reg_loss + mean_reg_loss

        """
        Update Networks
        """

        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self._update_target_functions()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rews.mean().item()

        if self.automatic_entropy_tuning:
            info["Alpha"] = alpha.item()
            info["Alpha_loss"] = alpha_loss.item()
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/vf_loss'] = vf_loss.item()
        info['Training/qf_loss'] = qf_loss.item()

        info['log_std/mean'] = log_std.mean().item()
        info['log_std/std'] = log_std.std().item()
        info['log_std/max'] = log_std.max().item()
        info['log_std/min'] = log_std.min().item()

        info['log_probs/mean'] = log_std.mean().item()
        info['log_probs/std'] = log_std.std().item()
        info['log_probs/max'] = log_std.max().item()
        info['log_probs/min'] = log_std.min().item()

        info['mean/mean'] = mean.mean().item()
        info['mean/std'] = mean.std().item()
        info['mean/max'] = mean.max().item()
        info['mean/min'] = mean.min().item()

        return info

    @property
    def functions(self):
        return [
            self.pf,
            self.qf,
            self.vf,
            self.target_vf
        ]

    @property
    def snapshot_functions(self):
        return [
            ["pf", self.pf],
            ["qf", self.qf],
            ["vf", self.vf]
        ]

    @property
    def target_functions(self):
        return [
            (self.vf, self.target_vf)
        ]
