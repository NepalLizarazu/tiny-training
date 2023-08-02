import torch
import torch.nn as nn
import dataset
from utils2 import remove_txt, parameter_generation
from train import Train
from model import DSCNN_TE_3x3_49x10
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
        'x_zero' : -x_zero,
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
        'x_zero' : -x_zero,
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

def get_scale_from_min_max (min, max): #For activations considers the whole range for quantization
    if isinstance(min, torch.Tensor):
        try:
            scale =  [(max[i].cpu().data.float().numpy() - min[i].cpu().data.float().numpy())/255 for i in range(len(min))]
        except:
            scale =  (max.cpu().data.float().numpy() - min.cpu().data.float().numpy())/255
        return np.array(scale)
    
def get_scale_from_min_max_weights (min, max): #For weights the scale considers one side since the distribution of the weights is Gaussian
    if isinstance(min, torch.Tensor):
        try:
            scale =  [np.maximum(np.abs(min[i].cpu().data.float().numpy()), np.abs(max[i].cpu().data.float().numpy()))/127 for i in range(len(min))]
        except:
            scale =  np.maximum(np.abs(min.cpu().data.float().numpy()), np.abs(max.cpu().data.float().numpy()))/127
        return np.array(scale)

    
def get_max_min_weights_from_attr (attr, model):
    #Converts the column tensor into a matrix-like structure
    max_min_weights = []
    idx = 0
    for name, module in model.named_modules():
        if name[:5] == "first" or name[:5] == "depth" or name[:5] == "point":
            max_min_weights.append(attr[idx:idx+module.out_channels])
            idx += module.out_channels
        if name[:2] == "fc":
            max_min_weights.append(attr[idx:idx+module.out_features])
    return max_min_weights

def conv_quantized_w_and_x_zero (qw, zx):
    zx_m = torch.zeros((1,qw.shape[1],qw.shape[2],qw.shape[3]))+zx
    res = nn.functional.conv2d(input = zx_m, weight=torch.tensor(qw),bias=None, stride=1, padding=0)[0,:,0,0]
    return res 

    
if __name__ == "__main__":

    file_name = "dscnn_40_pw_7x7_49x10_param_94.53125_fused"
    file_name = "dscnn_40_3x3_7x7_49x10_param_90.625_fused"
    cfg_path = "/Users/HP/Desktop/Master_Thesis/tiny-training/compilation/"+file_name+".pkl"
    cfg = torch.load(cfg_path)
    model = DSCNN_TE_3x3_49x10(use_bias=True)

    device = torch.device('cpu')
    print (torch.version.__version__)
    print(device)
    # Dataset set up
    training_parameters, data_processing_parameters = parameter_generation()  # To be parametrized
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)
    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))
    trainining_environment = Train(audio_processor, training_parameters, model, device)

    model.load_state_dict(cfg)
    model.eval()
    print("Min input activation: ",cfg['min_in_out'][0])
    print("Max input activation: ",cfg['max_in_out'][0])
    conv = model.first_conv

    #Gets the names of the layers
    names = []
    last = ""
    for name, _ in model.named_modules():
        if name[:5] == "first" or name[:5] == "depth" or name[:5] == "point" or name[:2] == "fc":
            names.append(name) 

    #Gets the convolutional modules out from the model
    modules = []
    for module in model.children():
        if (isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.linear.Linear)):
            modules.append(module)


    scales_zeros = []
    #Starts to build up the pkl format
    #Define the first convolution
    first_conv = None
    if (names[0] == "first_conv"):
        w_scales = get_scale_from_min_max_weights(get_max_min_weights_from_attr(cfg['min_weights'], model)[0], get_max_min_weights_from_attr(cfg['max_weights'], model)[0])
        q_weights = np.array([(np.array(cfg['first_conv.weight'].cpu())[i]/(w_scales[i])).round() for i in range(modules[0].out_channels)])
        model.first_conv.weight.data = torch.tensor(q_weights)
        x_scale = float(get_scale_from_min_max(cfg['min_in_out'][0], cfg['max_in_out'][0]))
        x_zero = (-128 - cfg['min_in_out'][0]/x_scale).round()
        b_scale = w_scales*x_scale
        q_b = torch.tensor([cfg['first_conv.bias'].cpu()[i]/(b_scale[i]) for i in range(modules[0].out_channels)])
        #q_bias = np.array((q_b - conv_quantized_w_and_x_zero(q_weights, x_zero)).round())
        q_bias = q_b.round()
        model.first_conv.bias.data = torch.tensor(q_bias)
        y_scale = float(get_scale_from_min_max(cfg['min_in_out'][1], cfg['max_in_out'][1]))
        y_zero = (-128 - cfg['min_in_out'][1]/y_scale).round()
        first_conv = dict_from_conv(conv = modules[0], name = names[0], weights = q_weights, 
                                    w_scales = w_scales, bias = q_bias, x_scale = x_scale, x_zero = x_zero, 
                                    y_scale = y_scale, y_zero = y_zero)
        scales_zeros.append((torch.tensor(w_scales), x_scale, y_scale, x_zero, y_zero))
        
    #Define the Blocks
    nb_blocks = 0
    for i in range(len(names)):
        if(names[i][:5] == "depth"):
            if(names[i+1][:6] == "pointw"):
                nb_blocks += 1
                print("Block detected: ", names[i], names[i+1])
    blocks = []
    for j in range(nb_blocks):
        ii = j*2
        iii = j*4
        #Define the DW Conv
        w_scales_dw = get_scale_from_min_max_weights(get_max_min_weights_from_attr(cfg['min_weights'], model)[ii+1], get_max_min_weights_from_attr(cfg['max_weights'], model)[ii+1])
        q_weights_dw = np.array([(np.array(cfg[names[ii+1]+'.weight'].cpu())[i]/(w_scales_dw[i])).round() for i in range(modules[ii+1].out_channels)])
        modules[ii+1].weight.data = torch.tensor(q_weights_dw)
        x_scale_dw = float(get_scale_from_min_max(cfg['min_in_out'][iii+2], cfg['max_in_out'][iii+2]))
        x_zero_dw = (-128 - cfg['min_in_out'][iii+2]/x_scale_dw).round()
        b_scale_dw = w_scales_dw*x_scale_dw
        q_b_dw = torch.tensor([cfg[names[ii+1]+'.bias'].cpu()[i]/(b_scale_dw[i]) for i in range(modules[ii+1].out_channels)])
        #q_bias_dw = np.array((q_b_dw - conv_quantized_w_and_x_zero(q_weights_dw, x_zero_dw)).round())
        q_bias_dw = q_b_dw.round()
        modules[ii+1].bias.data = torch.tensor(q_bias_dw)
        y_scale_dw = float(get_scale_from_min_max(cfg['min_in_out'][iii+3], cfg['max_in_out'][iii+3]))
        y_zero_dw = (-128 - cfg['min_in_out'][iii+3]/y_scale_dw).round()
        depthw = dict_from_conv(conv = modules[ii+1], name = names[ii+1], weights = q_weights_dw, 
                                    w_scales = w_scales_dw, bias = q_bias_dw, x_scale = x_scale_dw, x_zero = x_zero_dw, 
                                    y_scale = y_scale_dw, y_zero = y_zero_dw)
        
        scales_zeros.append((torch.tensor(w_scales_dw), x_scale_dw, y_scale_dw, x_zero_dw, y_zero_dw))
        
        #Define the Pointwise Conv
        w_scales_pw = get_scale_from_min_max_weights(get_max_min_weights_from_attr(cfg['min_weights'], model)[ii+2], get_max_min_weights_from_attr(cfg['max_weights'], model)[ii+2])
        q_weights_pw = np.array([(np.array(cfg[names[ii+2]+'.weight'].cpu())[i]/(w_scales_pw[i])).round() for i in range(modules[ii+2].out_channels)])
        modules[ii+2].weight.data = torch.tensor(q_weights_pw)
        x_scale_pw = float(get_scale_from_min_max(cfg['min_in_out'][iii+4], cfg['max_in_out'][iii+4]))
        x_zero_pw = (-128 - cfg['min_in_out'][iii+4]/x_scale_pw).round()
        b_scale_pw = w_scales_pw*x_scale_pw
        q_b_pw = torch.tensor([cfg[names[ii+2]+'.bias'].cpu()[i]/(b_scale_pw[i]) for i in range(modules[ii+2].out_channels)])
        #q_bias_pw = np.array((q_b_pw - conv_quantized_w_and_x_zero(q_weights_pw, x_zero_pw)).round())
        q_bias_pw = q_b_pw.round()
        modules[ii+2].bias.data = torch.tensor(q_bias_pw)
        y_scale_pw = float(get_scale_from_min_max(cfg['min_in_out'][iii+5], cfg['max_in_out'][iii+5]))
        y_zero_pw = (-128 - cfg['min_in_out'][iii+5]/y_scale_pw).round()
        pointw = dict_from_conv(conv = modules[ii+2], name = names[ii+2], weights = q_weights_pw, 
                                    w_scales = w_scales_pw, bias = q_bias_pw, x_scale = x_scale_pw, x_zero = x_zero_pw, 
                                    y_scale = y_scale_pw, y_zero = y_zero_pw)
        
        scales_zeros.append((torch.tensor(w_scales_pw), x_scale_pw, y_scale_pw, x_zero_pw, y_zero_pw))
        blocks.append(dict_from_dwblock(depthw=depthw, pointw=pointw))

    #Define Feature Mix
    feature_mix = None

    #Define Classifier

    w_scales_cl = get_scale_from_min_max_weights(get_max_min_weights_from_attr(cfg['min_weights'], model)[9], get_max_min_weights_from_attr(cfg['max_weights'], model)[9])
    q_weights_cl = np.array([(np.array(cfg[names[9]+'.weight'].cpu())[i]/(w_scales_cl[i])).round() for i in range(modules[9].out_features)])
    modules[9].weight.data = torch.tensor(q_weights_cl)
    x_scale_cl = float(get_scale_from_min_max(cfg['min_in_out'][18], cfg['max_in_out'][18]))
    x_zero_cl = (-128 - cfg['min_in_out'][18]/x_scale_cl).round()
    b_scale_cl = w_scales_cl*x_scale_cl
    q_b_cl = torch.tensor([cfg[names[9]+'.bias'].cpu()[i]/(b_scale_cl[i]) for i in range(modules[9].out_features)])
    #q_bias_cl = np.array((q_b_cl - torch.matmul(torch.tensor(q_weights_cl), (torch.zeros(modules[9].in_features)+x_zero_cl))).round())
    q_bias_cl = q_b_cl.round()
    modules[9].bias.data = torch.tensor(q_bias_cl)
    y_scale_cl = float(get_scale_from_min_max(cfg['min_in_out'][19], cfg['max_in_out'][19]))
    y_zero_cl = (-128 - cfg['min_in_out'][19]/y_scale_cl).round()
    classifier = dict_from_classifier(conv=modules[9], weights=q_weights_cl, w_scales=w_scales_cl, 
                                    bias=q_bias_cl, x_scale=x_scale_cl, x_zero=x_zero_cl, 
                                    y_scale=y_scale_cl, y_zero=y_zero_cl)
    
    scales_zeros.append((torch.tensor(w_scales_cl), x_scale_cl, y_scale_cl, x_zero_cl, y_zero_cl))
    trainining_environment.validate(model, 'validation', 128, register_min_max=False, quantized=True, scales_zeros=scales_zeros)
    #trainining_environment.validate(model, 'validation', 128, register_min_max=True)

    dictionary = {
        'first_conv' : first_conv,
        'blocks' : blocks,
        'feature_mix' : feature_mix,
        'classifier' : classifier,
    }

    for key, value in dictionary.items(): #Replace the y_scale value for the x_scale value of the next layer, and same with the y_zero
        if key == 'first_conv':
            dictionary[key]['params']['y_scale'] = dictionary['blocks'][0]['depthwise']['params']['x_scale']
            dictionary[key]['params']['y_zero'] = -dictionary['blocks'][0]['depthwise']['params']['x_zero']
        if key == 'blocks':
            for i in range(len(dictionary['blocks'])):
                dictionary['blocks'][i]['depthwise']['params']['y_scale'] = dictionary['blocks'][i]['pointwise2']['params']['x_scale']
                dictionary['blocks'][i]['depthwise']['params']['y_zero'] = -dictionary['blocks'][i]['pointwise2']['params']['x_zero']
                if not(i == (len(dictionary['blocks'])-1) ):
                    dictionary['blocks'][i]['pointwise2']['params']['y_scale'] = dictionary['blocks'][i+1]['depthwise']['params']['x_scale']
                    dictionary['blocks'][i]['pointwise2']['params']['y_zero'] = -dictionary['blocks'][i+1]['depthwise']['params']['x_zero']
                else:
                    dictionary['blocks'][i]['pointwise2']['params']['y_scale'] = dictionary['classifier']['params']['x_scale']
                    dictionary['blocks'][i]['pointwise2']['params']['y_zero'] = -dictionary['classifier']['params']['x_zero']
    print("BIAS -----------------------\n",blocks[0]['depthwise']['params']['bias'])
    torch.save(dictionary, file_name+"_customized.pkl")