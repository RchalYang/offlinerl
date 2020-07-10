from .nets import *
from .base import *
from .init import *


def get_network(input_shape, output_shape, network_cls, network_params):
    if len(input_shape) == 3:
        base_type = CNNBase
    else:
        base_type = MLPBase

    return network_cls(
        input_shape=input_shape,
        output_shape=output_shape,
        base_type=base_type,
        **network_params)
