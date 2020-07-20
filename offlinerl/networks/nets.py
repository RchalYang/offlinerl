import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import offlinerl.networks.init as init


class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros(1)


class Net(nn.Module):
    def __init__(
            self,
            output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=nn.ReLU,
            **kwargs):
        super().__init__()
        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for next_shape in append_hidden_shapes:
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            self.append_fcs.append(self.activation_func())
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

        self.append_fcs.append(self.last)
        self.seq_append_fcs = nn.Sequential(*self.append_fcs)

    def forward(self, x):
        out = self.base(x)
        out = self.seq_append_fcs(out)
        return out


class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat(input, dim=-1)
        return super().forward(out)


class BootstrappedNet(nn.Module):
    def __init__(
            self,
            output_shape,
            base_type, head_num=10,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=nn.ReLU(),
            **kwargs):
        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func

        self.bootstrapped_heads = []

        append_input_shape = self.base.output_shape

        for idx in range(head_num):
            append_input_shape = self.base.output_shape
            append_fcs = []
            for next_shape in append_hidden_shapes:
                fc = nn.Linear(append_input_shape, next_shape)
                append_hidden_init_func(fc)
                append_fcs.append(fc)
                append_fcs.append(self.activation_func())
                # set attr for pytorch to track parameters( device )
                append_input_shape = next_shape

            last = nn.Linear(append_input_shape, output_shape)
            net_last_init_func(last)
            append_fcs.append(last)
            head = nn.Sequential(*append_fcs)
            self.__setattr__(
                "head{}".format(idx),
                head)
            self.bootstrapped_heads.append(head)

    def forward(self, x, head_idxs):
        output = []
        feature = self.base(x)
        for idx in head_idxs:
            output.append(self.bootstrapped_heads[idx](feature))
        return output


class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, head_idxs):
        out = torch.cat(input, dim=-1)
        return super().forward(out, head_idxs)


ENCODE_LOG_SIG_MAX = 15
ENCODE_LOG_SIG_MIN = -4


class Encoder(Net):
    def forward(self, input):
        output = super().forward(input)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = log_std.clamp(ENCODE_LOG_SIG_MIN, ENCODE_LOG_SIG_MAX)
        std = torch.exp(log_std)

        z = mean + std * torch.randn_like(std)

        return z, mean, std, log_std


class FlattenEncoder(Encoder):
    def forward(self, input):
        out = torch.cat(input, dim=-1)
        return super().forward(out)


class FlattenDecoder(Net):
    def __init__(self, input_shape, latent_shape, **kwargs):
        assert len(input_shape) == 1, "Current just support 1-dim input"
        assert len(latent_shape) == 1, "Current just support 1-dim input"
        self.latent_shape = latent_shape
        super().__init__(input_shape=input_shape+latent_shape, **kwargs)

    def forward(self, input):
        if len(input) == 1:
            input.append(
                torch.randn(
                    input.shape[:-1] + torch.Size(self.latent_shape)
                ).to(input.device).clamp(-0.5, 0.5)
            )
        out = torch.cat(input, dim=-1)
        return super().forward(out)
