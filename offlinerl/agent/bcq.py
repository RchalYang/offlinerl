import copy
import torch
import torch.nn as nn
import torch.optim as optim
from .base_agent import Agent
import offlinerl.networks as networks


class ContBCQAgent(Agent):
    def __init__(
            self,
            env,
            policy_params,
            q_params,
            encoder_params,
            decoder_params,
            plr, qlr, vaelr,
            phi, lmbda,
            optimizer_class=optim.Adam,
            **kwargs):
        super(ContBCQAgent, self).__init__(env=env, **kwargs)

        # Here Policy function serve as a additional noise
        if "tanh_action" in policy_params:
            self.tanh_action = policy_params["tanh_action"]
            del policy_params["tanh_action"]
        else:
            self.tanh_action = False
        self.pf = networks.get_network(
            input_shape=(self.env.observation_space.shape[0] +
                         self.env.action_space.shape[0],),
            output_shape=self.env.action_space.shape[0],
            network_cls=networks.FlattenNet,
            network_params=policy_params)
        self.target_pf = copy.deepcopy(self.pf)

        self.qf1 = networks.get_network(
            input_shape=(self.env.observation_space.shape[0] +
                         self.env.action_space.shape[0],),
            output_shape=1,
            network_cls=networks.FlattenNet,
            network_params=q_params)
        self.target_qf1 = copy.deepcopy(self.qf1)

        self.qf2 = networks.get_network(
            input_shape=(self.env.observation_space.shape[0] +
                         self.env.action_space.shape[0],),
            output_shape=1,
            network_cls=networks.FlattenNet,
            network_params=q_params)
        self.target_qf2 = copy.deepcopy(self.qf2)

        latent_shape = decoder_params["latent_shape"][0]
        self.encoder = networks.get_network(
            input_shape=(self.env.observation_space.shape[0] +
                         self.env.action_space.shape[0],),
            output_shape=2 * latent_shape,
            network_cls=networks.FlattenEncoder,
            network_params=q_params)

        self.phi = phi
        self.lmbda = lmbda

        self.decoder = networks.get_network(
            input_shape=self.env.observation_space.shape,
            output_shape=self.env.action_space.shape[0],
            network_cls=networks.FlattenDecoder,
            network_params=decoder_params)

        self.qlr = qlr
        self.plr = qlr
        self.vaelr = vaelr

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=self.qlr,
        )

        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=self.qlr,
        )

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(),
            lr=self.plr,
        )

        from itertools import chain
        self.vae_optimizer = optimizer_class(
            [{"params": self.decoder.parameters()},
             {"params": self.encoder.parameters()}],
            lr=self.vaelr,
        )

        self.qf_criterion = nn.MSELoss()
        self.vae_criterion = nn.MSELoss()

    def explore(self, obs):
        with torch.no_grad():
            obs = torch.repeat_interleave(obs.unsqueeze(0), 10, 0)
            sampled_action = self.decoder([obs])
            sampled_noise = self.pf([obs, sampled_action])
            sampled_action = sampled_action + self.phi * sampled_noise
            sampled_action = torch.tanh(sampled_action)
            q1 = self.qf1([obs, sampled_action])
            ind = q1.max(0, keepdim=True)[1]
            ind = torch.repeat_interleave(
                ind, sampled_action.shape[-1], -1)
        return {
            "action": sampled_action.gather(0, ind).squeeze(0)
        }

    def eval_act(self, obs):
        return self.explore(obs)["action"].cpu().data.numpy().flatten()

    def update(self, batch):
        self.training_update_num += 1

        obs = batch['obs']
        acts = batch['acts']
        next_obs = batch['next_obs']
        rews = batch['rews']
        ters = batch['dones']

        """
        Reconstruction Loss
        """
        sampled, mean, std, _ = self.encoder([obs, acts])
        recon = self.decoder([obs, sampled])
        recon = torch.tanh(recon)
        reconstruction_loss = self.vae_criterion(recon, acts)
        kl_divergence = -0.5 * (
            1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = reconstruction_loss + kl_divergence

        """
        QF Loss
        """
        with torch.no_grad():
            repeated_next_obs = torch.repeat_interleave(
                next_obs.unsqueeze(1), 10, 1
            )

            sampled_acts_next = self.decoder([repeated_next_obs])
            noise_next = self.target_pf([repeated_next_obs, sampled_acts_next])
            generated_acts = sampled_acts_next + self.phi * noise_next
            if self.tanh_action:
                generated_acts = torch.tanh(generated_acts)
            target_q1 = self.target_qf1([repeated_next_obs, generated_acts])
            target_q2 = self.target_qf2([repeated_next_obs, generated_acts])

            target_q = self.lmbda * torch.min(target_q1, target_q2) + \
                (1. - self.lmbda) * torch.max(target_q1, target_q2)

            target_q = target_q.max(1)[0]
            target_q = rews + self.discount * (1 - ters) * target_q

        q1_pred = self.qf1([obs, acts])
        q2_pred = self.qf2([obs, acts])

        assert q1_pred.shape == target_q.shape, \
            "q1_pred shape: {}, target_q shape: {}".format(
                q1_pred.shape, target_q.shape)
        assert q2_pred.shape == target_q.shape, \
            "q2_pred shape: {}, target_q shape: {}".format(
                q2_pred.shape, target_q.shape)
        qf1_loss = self.qf_criterion(q1_pred, target_q)
        qf2_loss = self.qf_criterion(q2_pred, target_q)

        sampled_action_current = self.decoder([obs])
        noise_current = self.pf([obs, sampled_action_current])

        sampled_action_current = sampled_action_current + \
            self.phi * noise_current
        if self.tanh_action:
            sampled_action_current = torch.tanh(sampled_action_current)
        policy_loss = - torch.min(
            self.qf1([obs, sampled_action_current]),
            self.qf1([obs, sampled_action_current])
        ).mean()

        """
        Update Networks
        """
        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        self.pf_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        self._update_target_functions()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rews.mean().item()

        info['Training/policy_loss'] = policy_loss.item()
        info['Training/vae_loss'] = vae_loss.item()
        info['Training/qf1_loss'] = qf1_loss.item()
        info['Training/qf2_loss'] = qf2_loss.item()

        info['mean/mean'] = mean.mean().item()
        info['mean/std'] = mean.std().item()
        info['mean/max'] = mean.max().item()
        info['mean/min'] = mean.min().item()

        return info

    @property
    def functions(self):
        return [
            self.pf,
            self.qf1,
            self.qf2,
            self.encoder,
            self.decoder,
            self.target_qf1,
            self.target_qf2
        ]

    @property
    def snapshot_functions(self):
        return [
            ["pf", self.pf],
            ["encoder", self.encoder],
            ["decoder", self.decoder],
            ["qf1", self.qf1],
            ["qf2", self.qf2]
        ]

    @property
    def target_functions(self):
        return [
            (self.qf1, self.target_qf1),
            (self.qf2, self.target_qf2),
            (self.pf, self.target_pf)
        ]
