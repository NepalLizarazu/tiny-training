import os, sys, os.path as osp
import functools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tvm
from tvm import relay

sys.path.append('../../algorithm')
sys.path.append('C:/Users/HP/Desktop/Master_Thesis/tiny-training')

#from .mcunetv3_wrapper import (
#    build_mcu_model,
#    configs,
#    load_config_from_file,
#    update_config_from_args,
#    update_config_from_unknown_args,
#    QuantizedConv2dDiff,
#    QuantizedMbBlockDiff,
#    ScaledLinear,
#    QuantizedAvgPoolDiff,
#)

from algorithm.core.model import build_mcu_model
from algorithm.core.utils.config import (
    configs,
    load_config_from_file,
    update_config_from_args,
    update_config_from_unknown_args,
)
from algorithm.quantize.quantized_ops_diff import (
    QuantizedConv2dDiff,
    QuantizedMbBlockDiff,
    ScaledLinear,
    QuantizedAvgPoolDiff,
)



def build_quantized_model(net_name="mbv2-w0.35", num_classes=10):
    load_config_from_file("C:/Users/HP/Desktop/Master_Thesis/tiny-training/algorithm/configs/transfer.yaml")
    configs["net_config"]["net_name"] = "dscnn_te_customized"
    configs["net_config"]["net_name"] = "proxyless-w0.3"
    print("Generating files for: ", net_name)
    configs["net_config"]["net_name"] = net_name
    configs["net_config"]["mcu_head_type"] = "quantized"

    subnet = build_mcu_model()
    subnet = nn.Sequential(*subnet[:5])
    resolution = 128
    last = subnet[-1]
    subnet[-1] = QuantizedConv2dDiff(
        last.in_channels,
        num_classes,
        kernel_size=last.kernel_size,
        stride=last.stride,
        zero_x=last.zero_x,
        zero_y=last.zero_y,
        effective_scale=last.effective_scale[:num_classes],
    )
    subnet[-1].y_scale = last.y_scale
    subnet[-1].x_scale = last.x_scale
    subnet[-1].weight.data = last.weight.data[:num_classes, :, :, :]
    return subnet, resolution


build_quantized_mcunet = functools.partial(
    build_quantized_model, net_name="mcunet-5fps"
)
build_quantized_mbv2 = functools.partial(build_quantized_model, net_name="mbv2-w0.35")
build_quantized_proxyless = functools.partial(
    build_quantized_model, net_name="proxyless-w0.3"
)
build_quantized_dscnn = functools.partial(build_quantized_model, net_name="dscnn_te_customized")

if __name__ == "__main__":
    net, rs = build_quantized_mbv2(num_classes=10)
    d = torch.randn(1, 3, rs, rs)
    net(d)
