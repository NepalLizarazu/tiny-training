import torch
import copy
from utils2 import remove_txt, parameter_generation
import dataset
from train import Train

from model import DSCNN_TE_3x3_49x10 #This is the model used in pre-training

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

def set_max_min_weight_inst(model):
        i = 0
        for name, param in model.named_parameters():
            #Over the layers weights
            if (str(name)[-6:] == 'weight' and str(name)[:2] != 'bn'):
                out_ch = param.shape[0]
                #Over the output channels i.e. first dimension of the weight
                for j in range(out_ch):
                    model.max_weights[i+j] = torch.max(param[j]).data
                    model.min_weights[i+j] = torch.min(param[j]).data
                i += out_ch

def reset_min_max_activations(model):
        for i in range(model.max_in_out.shape[0]):
            model.max_in_out[i] = torch.tensor(-1000.)
            model.min_in_out[i] = torch.tensor(1000.)

if __name__ == '__main__':
    
    file_name = "dscnn_40_3x3_7x7_49x10_param_90.625"
    param_path = "../../kws-on-pulp/quantization/"+file_name+".pth" #The file can be a .pkl or .pth, make sure to change it in this line accordingly
    
    model = DSCNN_TE_3x3_49x10() #3x3 -> 7x7 // This model instance has to be the same used for pre-training

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

    model.load_state_dict(torch.load(param_path))
    model.eval()
    #trainining_environment.validate(model, 'validation', 128, register_min_max=False)
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
    set_max_min_weight_inst(model)

    reset_min_max_activations(model)
    trainining_environment.validate(model, 'validation', 128, register_min_max=True)
    torch.save(model.state_dict(), file_name+"_fused.pkl")
    print("File saved as: "+file_name+"_fused.pkl")
    

    
