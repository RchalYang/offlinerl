import os.path as osp
import torch
import offlinerl.agent.utils as atu
import pathlib


class Agent():
    def __init__(
            self, env, device, discount,
            use_soft_update=True, tau=0.001,
            target_hard_update_period=1000,
            **kwargs):
        self.env = env
        self.device = device
        self.discount = discount
        self.use_soft_update = use_soft_update
        self.tau = tau
        self.target_hard_update_period = target_hard_update_period
        self.training_update_num = 0

    @property
    def functions(self):
        return []

    @property
    def snapshot_functions(self):
        return []

    @property
    def target_functions(self):
        return []

    def snapshot(self, prefix, updates):
        pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
        for name, function in self.snapshot_functions:
            model_file_name = "model_{}_{}.pth".format(name, updates)
            model_path = osp.join(prefix, model_file_name)
            torch.save(function.state_dict(), model_path)

    def explore(self, ob):
        return self.pf.explore(ob)

    def eval_act(self, ob):
        return self.pf.eval_act(ob)

    def to(self, device):
        for net in self.functions:
            net.to(device)

    def _update_target_functions(self):
        if self.use_soft_update:
            for func, target_func in self.target_functions:
                atu.soft_update_from_to(func, target_func, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                for func, target_func in self.target_functions:
                    atu.copy_model_params_from_to(func, target_func)
