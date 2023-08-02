import torch
import copy
from model import DSCNN_TE

def fuse_conv_bn_eval(conv, bn, transpose=False):
    #assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, transpose)

    return fused_conv

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    fused_conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
    fused_conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    
    return torch.nn.Parameter(fused_conv_w, conv_w.requires_grad), torch.nn.Parameter(fused_conv_b, conv_b.requires_grad)

if __name__ == '__main__':
    file_name = "DSCNN_BN_7x7_40_92.1875"
    param_path = "../kws-on-pulp/quantization/"+file_name+".pth"
    model = DSCNN_TE()
    model.load_state_dict(torch.load(param_path))
    model.eval()
    previous_module = None
    fused_modules = []
    pr = False
    for name, module in model.named_modules():
        if name[:2] == "bn":            
            fused_modules.append(fuse_conv_bn_eval(previous_module, module))

        previous_module = copy.deepcopy(module)

    i = 0
    for name, module in model.named_modules():
        if (name[:5] == "first") or (name[:5] == "point") or (name[:5] == "depth"):
            print("Fused: "+name)
            module.weight = fused_modules[i].weight
            module.bias = fused_modules[i].bias
            #print("Fused:", name)
            i+=1

        previous_module = module
    
    torch.save(model.state_dict(), file_name+"_fused.pkl")
    

    
