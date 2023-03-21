import torch
import torch.nn as nn
from model import DSCNN_TE
import numpy as np

def dict_from_conv (conv, name, weights, w_scales, bias, x_scale, x_zero, y_scale, y_zero):
    in_channel = conv.in_channels
    out_channel = conv.out_channels
    kernel_size = conv.kernel_size
    stride = conv.stride
    groups = conv.groups
    if name[0:5] == "depth":
        depthwise = True
    else:
        depthwise = False
    
    params = {
        'weight' : weights,
        'bias' : bias,
        'x_scale' : x_scale,
        'x_zero' : x_zero,
        'y_scale' : y_scale,
        'y_zero' : y_zero,
        'w_scales' : w_scales,
    }

    dictionary = {
        'in_channel' : in_channel,
        'in_shape' : None,
        'out_channel' : out_channel,
        'out_shape' : None,
        'kernel_size': kernel_size,
        'stride' : stride,
        'groups' : groups,
        'depthwise' : depthwise,
        'act' : None,
        'params' : params,
    }
    return dictionary

def dict_from_dwblock(depthw, pointw):
    dictionary = {
        'pointwise1' : None,
        'depthwise' : depthw,
        'pointwise2' : pointw,
        'residual' : None,
        'se' : None,
    }
    return dictionary

def dict_from_classifier (conv, weights, w_scales, bias, x_scale, x_zero, y_scale, y_zero):
    in_channel = conv.in_features
    out_channel = conv.out_features
    kernel_size = (1,1)
    stride = 1
    groups = 1
    
    
    params = {
        'weight' : weights,
        'bias' : bias,
        'x_scale' : x_scale,
        'x_zero' : x_zero,
        'y_scale' : y_scale,
        'y_zero' : y_zero,
        'w_scales' : w_scales,
    }

    dictionary = {
        'in_channel' : in_channel,
        'out_channel' : out_channel,
        'params' : params,
        'kernel_size': kernel_size,
        'stride' : stride,
        'groups' : groups,
    }
    return dictionary

def get_scale_from_min_max (min, max):
    if isinstance(min, torch.Tensor):
        try:
            scale =  [np.maximum(np.abs(min[i].cpu().data.float().numpy()), np.abs(max[i].cpu().data.float().numpy()))/127 for i in range(len(min))]
        except:
            scale =  np.maximum(np.abs(min.cpu().data.float().numpy()), np.abs(max.cpu().data.float().numpy()))/127
        return np.array(scale)
    
if __name__ == "__main__":

    cfg_path = "../kws-on-pulp/quantization/DSCNN_BN_7x7_40_92.1875_fused.pkl"
    cfg = torch.load(cfg_path)
    model = DSCNN_TE(use_bias=True)
    conv = model.first_conv

    #Gets the names of the layers
    names = []
    last = ""
    for name, _ in model.named_parameters():
        if not last == name.split('.')[0]:
            names.append(name.split('.')[0]) 
            last = name.split('.')[0]

    #Gets the convolutional modules out from the model
    modules = []
    for module in model.children():
        if (isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.linear.Linear)):
            modules.append(module)

    #Starts to build up the pkl format
    #Define the first convolution
    if (names[0] == "first_conv"):
        w_scales = get_scale_from_min_max(cfg['min_weights'][0], cfg['max_weights'][0])
        q_weights = np.array([(np.array(cfg['first_conv.weight'].cpu())[i]/(w_scales[i])).round() for i in range(modules[0].out_channels)])
        x_scale = float(get_scale_from_min_max(cfg['min_in_out'][0], cfg['max_in_out'][0]))
        b_scale = w_scales*x_scale
        q_bias = np.array([(cfg['first_conv.bias'].cpu()[i]/(b_scale[i])).round() for i in range(modules[0].out_channels)])
        y_scale = float(get_scale_from_min_max(cfg['min_in_out'][1], cfg['max_in_out'][1]))
        first_conv = dict_from_conv(conv = modules[0], name = names[0], weights = q_weights, 
                                    w_scales = w_scales, bias = q_bias, x_scale = x_scale, x_zero = 0, 
                                    y_scale = y_scale, y_zero = 0)
        
    #Define the Blocks
    nb_blocks = 0
    for i in range(len(names)):
        if(names[i][:5] == "depth"):
            if(names[i+1][:6] == "pointw"):
                nb_blocks += 1
    blocks = []
    for j in range(nb_blocks):
        ii = j*2
        iii = j*4
        #Define the DW Conv
        w_scales_dw = get_scale_from_min_max(cfg['min_weights'][ii+1], cfg['max_weights'][ii+1])
        q_weights_dw = np.array([(np.array(cfg[names[ii+1]+'.weight'].cpu())[i]/(w_scales_dw[i])).round() for i in range(modules[ii+1].out_channels)])
        x_scale_dw = float(get_scale_from_min_max(cfg['min_in_out'][iii+2], cfg['max_in_out'][iii+2]))
        b_scale_dw = w_scales_dw*x_scale_dw
        q_bias_dw = np.array([(cfg[names[ii+1]+'.bias'].cpu()[i]/(b_scale_dw[i])).round() for i in range(modules[ii+1].out_channels)])
        y_scale_dw = float(get_scale_from_min_max(cfg['min_in_out'][iii+3], cfg['max_in_out'][iii+3]))
        depthw = dict_from_conv(conv = modules[ii+1], name = names[ii+1], weights = q_weights_dw, 
                                    w_scales = w_scales_dw, bias = q_bias_dw, x_scale = x_scale_dw, x_zero = 0, 
                                    y_scale = y_scale_dw, y_zero = 0)
        
        #Define the Pointwise Conv
        w_scales_pw = get_scale_from_min_max(cfg['min_weights'][ii+2], cfg['max_weights'][ii+2])
        q_weights_pw = np.array([(np.array(cfg[names[ii+2]+'.weight'].cpu())[i]/(w_scales_pw[i])).round() for i in range(modules[ii+2].out_channels)])
        x_scale_pw = float(get_scale_from_min_max(cfg['min_in_out'][iii+4], cfg['max_in_out'][iii+4]))
        b_scale_pw = w_scales_pw*x_scale_pw
        q_bias_pw = np.array([(cfg[names[ii+2]+'.bias'].cpu()[i]/(b_scale_pw[i])).round() for i in range(modules[ii+2].out_channels)])
        y_scale_pw = float(get_scale_from_min_max(cfg['min_in_out'][iii+5], cfg['max_in_out'][iii+5]))
        pointw = dict_from_conv(conv = modules[ii+2], name = names[ii+2], weights = q_weights_pw, 
                                    w_scales = w_scales_pw, bias = q_bias_pw, x_scale = x_scale_pw, x_zero = 0, 
                                    y_scale = y_scale_pw, y_zero = 0)
        blocks.append(dict_from_dwblock(depthw=depthw, pointw=pointw))

    #Define Feature Mix
    feature_mix = None

    #Define Classifier

    w_scales_cl = get_scale_from_min_max(cfg['min_weights'][9], cfg['max_weights'][9])
    q_weights_cl = np.array([(np.array(cfg[names[9]+'.weight'].cpu())[i]/(w_scales_cl[i])).round() for i in range(modules[9].out_features)])
    x_scale_cl = float(get_scale_from_min_max(cfg['min_in_out'][18], cfg['max_in_out'][18]))
    b_scale_cl = w_scales_cl*x_scale_cl
    q_bias_cl = np.array([(cfg[names[9]+'.bias'].cpu()[i]/(b_scale_cl[i])).round() for i in range(modules[9].out_features)])
    y_scale_cl = float(get_scale_from_min_max(cfg['min_in_out'][19], cfg['max_in_out'][19]))
    classifier = dict_from_classifier(conv=modules[9], weights=q_weights_cl, w_scales=w_scales_cl, 
                                    bias=q_bias_cl, x_scale=x_scale_cl, x_zero=0, 
                                    y_scale=y_scale_cl, y_zero=0)

    dictionary = {
        'first_conv' : first_conv,
        'blocks' : blocks,
        'feature_mix' : feature_mix,
        'classifier' : classifier,
    }
    torch.save(dictionary, "DSCNN_BN_7x7_40_92.1875_fused_customized.pkl")