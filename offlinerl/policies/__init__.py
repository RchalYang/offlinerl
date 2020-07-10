from .continuous_policies import *
from .discrete_policies import *
from .distribution import *
import offlinerl.networks as networks


def get_policy(input_shape, output_shape, policy_cls, policy_params):
    if len(input_shape) == 3:
        base_type = networks.CNNBase
    else:
        base_type = networks.MLPBase

    return policy_cls(
        input_shape=input_shape,
        output_shape=output_shape,
        base_type=base_type,
        **policy_params)
