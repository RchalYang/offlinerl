import torch
import torch.utils.data as tdata
import time
import os
import numpy as np


class Trainer:
    def __init__(
        self, env, dataset, agent, logger,
        training_epoch,
        batch_size,
        eval_interval,
        eval_episodes,
        save_interval
    ):
        self.env = env
        self.dataset = dataset
        self.agent = agent

        self.dataloader = tdata.DataLoader(
            dataset, batch_size=batch_size)

        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

        self.save_interval = save_interval

        self.training_epoch = training_epoch

        self.logger = logger
        self.snapshot_dir = os.path.join(self.logger.work_dir, "snapshot")

        self.total_update = 0

        self.update_timestamp = time.time()

    def eval(self):
        rewards = []
        lengths = []
        for _ in range(self.eval_episodes):
            ob = self.env.reset()
            done = False
            episode_reward = 0
            length = 0
            while not done:
                length += 1
                ob_tensor = torch.Tensor(ob).to(self.agent.device)
                act = self.agent.eval_act(ob_tensor)
                ob, r, done, infos = self.env.step(act)
                episode_reward += r
            rewards.append(episode_reward)
            lengths.append(length)
        return {
            "episode_rewards": np.mean(rewards),
            "episode_lengths": np.mean(lengths)
        }

    def train(self):
        for epoch in range(self.training_epoch):
            for i, batch in enumerate(self.dataloader):
                self.total_update += 1
                update_info = self.agent.update(batch)
                self.logger.add_update_info(update_info)

                if self.total_update % self.eval_interval == 0:
                    eval_info = self.eval()
                    time_consumed = time.time() - self.update_timestamp
                    print(eval_info)
                    self.logger.add_eval_info(
                        self.total_update, time_consumed, eval_info)

                if self.total_update % self.save_interval == 0:
                    self.agent.snapshot(self.snapshot_dir, self.total_update)
