# Copyright (C) 2021 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DSCNN_TE_3x3_49x10(torch.nn.Module):
    reg_count = 0
    def __init__(self, use_bias=True):
        super(DSCNN_TE_3x3_49x10, self).__init__()

        use_bias = True
        
        #Modify from here
        self.first_conv = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3, 3), stride = (2, 2), padding = (1,1),bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()

        self.depth1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (7, 7), stride = (1, 1), padding = 3, groups = 64, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.pointw1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()

        self.depth2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU()
        self.pointw2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(64)
        self.relu5 = torch.nn.ReLU()

        self.depth3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(64)
        self.relu6 = torch.nn.ReLU()
        self.pointw3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(64)
        self.relu7 = torch.nn.ReLU()

        self.depth4 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
        self.bn8   = torch.nn.BatchNorm2d(64)
        self.relu8 = torch.nn.ReLU()
        self.pointw4 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn9   = torch.nn.BatchNorm2d(64)
        self.relu9 = torch.nn.ReLU()

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.fc1   = torch.nn.Linear(64, 10, bias=use_bias)
        #Modify up to here

        max_weights = []
        min_weights = []
        max_in_out = []
        min_in_out = []

        for name, module in self.named_modules():
            if name[:5] == "first" or name[:5] == "depth" or name[:5] == "point":
                max_weights.append(torch.zeros(module.out_channels))
                min_weights.append(torch.zeros(module.out_channels))
                max_in_out.append(torch.tensor(-1000.))
                max_in_out.append(torch.tensor(-1000.))
                min_in_out.append(torch.tensor(1000.))
                min_in_out.append(torch.tensor(1000.))
            if name[:2] == "fc":
                max_weights.append(torch.zeros(module.out_features))
                min_weights.append(torch.zeros(module.out_features)) 
                max_in_out.append(torch.tensor(-1000.))
                max_in_out.append(torch.tensor(-1000.))
                min_in_out.append(torch.tensor(1000.))
                min_in_out.append(torch.tensor(1000.))  

        self.register_buffer('max_weights', torch.cat(max_weights))
        
        self.register_buffer('min_weights', torch.cat(min_weights))

        self.register_buffer('max_in_out', torch.stack(max_in_out))

        self.register_buffer('min_in_out', torch.stack(min_in_out))

        
    def forward(self, x, save = False, quantized = False, scales_zeros = None): 
        if not save and not quantized: #Normal forward (for pre-training)
            x = self.first_conv(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
            x = self.depth1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pointw1(x)
            x = self.bn3(x)
            x = self.relu3(x)
            
            x = self.depth2(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.pointw2(x)
            x = self.bn5(x)
            x = self.relu5(x)
            
            x = self.depth3(x)
            x = self.bn6(x)
            x = self.relu6(x)
            x = self.pointw3(x)
            x = self.bn7(x)
            x = self.relu7(x)
            
            x = self.depth4(x)
            x = self.bn8(x)
            x = self.relu8(x)   
            x = self.pointw4(x)
            x = self.bn9(x)
            x = self.relu9(x)   
            
            x = self.avg(x)
            x = torch.flatten(x, 1) 
            x = self.fc1(x)

        elif save and not quantized: #Forward without Batch Norm layers, for fusing and collecting quantization statistics
            self.set_max_min_in_out(x) #This function call saves statistics for quantization in every layer
            x = self.first_conv(x)
            self.set_max_min_in_out(x)
            x = self.relu1(x)
            
            self.set_max_min_in_out(x)
            x = self.depth1(x)
            self.set_max_min_in_out(x)
            x = self.relu2(x)
            self.set_max_min_in_out(x)
            x = self.pointw1(x)
            self.set_max_min_in_out(x)
            x = self.relu3(x)
            
            self.set_max_min_in_out(x)
            x = self.depth2(x)
            self.set_max_min_in_out(x)
            x = self.relu4(x)
            self.set_max_min_in_out(x)
            x = self.pointw2(x)
            self.set_max_min_in_out(x)
            x = self.relu5(x)
            
            self.set_max_min_in_out(x)
            x = self.depth3(x)
            self.set_max_min_in_out(x)
            x = self.relu6(x)
            self.set_max_min_in_out(x)
            x = self.pointw3(x)
            self.set_max_min_in_out(x)
            x = self.relu7(x)

            self.set_max_min_in_out(x)
            x = self.depth4(x)
            self.set_max_min_in_out(x)
            x = self.relu8(x)   
            self.set_max_min_in_out(x)
            x = self.pointw4(x)
            self.set_max_min_in_out(x)
            x = self.relu9(x)  
            
            x = self.avg(x)
            x = torch.flatten(x, 1) 
            self.set_max_min_in_out(x)
            x = self.fc1(x)
            self.set_max_min_in_out(x, True)
            

        elif not save and quantized: # Quantized forward for obtaining the post-quantization accuracy
            
            x = (x/scales_zeros[0][1] + scales_zeros[0][3]).round() #quantizing input

            print()
            print('*'*50)
            print("First input scaling factor is: ",scales_zeros[0][1])
            print("First input zero point is: ", -int(scales_zeros[0][3].item()))
            print('*'*50)
            print()
            
            #Modify from here:
            x = self.substract_zero(x,scales_zeros)
            x = self.first_conv(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)   
            x = self.apply_clamp(x)
              
            x = self.substract_zero(x,scales_zeros)
            x = self.depth1(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            x = self.apply_clamp(x)
            
            x = self.substract_zero(x,scales_zeros)
            x = self.pointw1(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros) 
            x = self.apply_clamp(x)
            
            x = self.substract_zero(x,scales_zeros)
            x = self.depth2(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            x = self.apply_clamp(x)
            
            x = self.substract_zero(x,scales_zeros)
            x = self.pointw2(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            x = self.apply_clamp(x)
            
            x = self.substract_zero(x,scales_zeros)
            x = self.depth3(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            x = self.apply_clamp(x)
           
            x = self.substract_zero(x,scales_zeros)
            x = self.pointw3(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            x = self.apply_clamp(x)

            x = self.substract_zero(x,scales_zeros)
            x = self.depth4(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            x = self.apply_clamp(x)

            x = self.substract_zero(x,scales_zeros)
            x = self.pointw4(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            x = self.apply_clamp(x)
            
            x = self.avg(x)
            
            x = torch.flatten(x, 1) 
            x = self.substract_zero(x,scales_zeros)
            x = self.fc1(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros,True) # Add True parameter if last layer
        return x 

    def set_max_min_in_out (self, x, last = False):
        x_v = x
        
        if (self.max_in_out[self.reg_count] < torch.max(x_v)):
            self.max_in_out[self.reg_count] = torch.max(x_v)
        if (self.min_in_out[self.reg_count] > torch.min(x_v)):
            self.min_in_out[self.reg_count] = torch.min(x_v)
        if last:
            self.reg_count = 0
        else:
            self.reg_count +=1 
    
    def scaleout_and_convert_into_next_input (self, x, scales_zeros, last = False): #scales_zeros: [w_scales, x_scale, y_scale, x_zero, y_zero]
        for i in range (x.shape[1]): #N_Channels assuming x has [B, Ch, H, W]
            x [:,i] = x [:,i] * (scales_zeros[self.reg_count][0][i]*(scales_zeros[self.reg_count][1]/scales_zeros[self.reg_count][2]))
        qy = (x + scales_zeros[self.reg_count][4]).round()
        qy = self.apply_clamp(qy)
    
        if not last: #Calculate next quantized input
            qx = scales_zeros[self.reg_count][2]/scales_zeros[self.reg_count+1][1]* F.relu(qy-scales_zeros[self.reg_count][4])+scales_zeros[self.reg_count+1][3]
            qx = self.apply_clamp(qx)
            qx = qx.round()
            self.reg_count +=1 
        else:
            qx = qy
            self.reg_count = 0
        
        return qx
    
    def substract_zero (self, x, scales_zeros):
        return x-scales_zeros[self.reg_count][3]
    
    def apply_clamp (self, x):
        y = torch.clamp(x,-128,127)
        return y
    
