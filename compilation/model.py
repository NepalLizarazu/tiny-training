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

#from utils import npy_to_txt


class DSCNN(torch.nn.Module):
    def __init__(self, use_bias=True):
        super(DSCNN, self).__init__()
        
        self.pad1  = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (10, 4), stride = (2, 2), bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()

        self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()

        self.pad4  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv4 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(64)
        self.relu5 = torch.nn.ReLU()

        self.pad6  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv6 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(64)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(64)
        self.relu7 = torch.nn.ReLU()

        self.pad8  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        self.conv8 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 64, bias = use_bias)
        self.bn8   = torch.nn.BatchNorm2d(64)
        self.relu8 = torch.nn.ReLU()
        self.conv9 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn9   = torch.nn.BatchNorm2d(64)
        self.relu9 = torch.nn.ReLU()

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.fc1   = torch.nn.Linear(64, 12, bias=use_bias)
        # self.soft  = torch.nn.Softmax(dim=1)
        # self.soft = F.log_softmax(x, dim=1)


        # CONV2D replacing Block1 for evaluation purposes
        # self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        # self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 1, bias = use_bias)
        # self.bn2   = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU()
        
    def forward(self, x, save = False):
        if (save):

            x = self.pad1 (x)
            x = self.conv1(x)
            x = self.bn1  (x)
            x = self.relu1(x)
            npy_to_txt(0, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))

            x = self.pad2 (x)
            x = self.conv2(x)
            x = self.bn2  (x)
            x = self.relu2(x)
            npy_to_txt(1, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))
            x = self.conv3(x)
            x = self.bn3  (x)
            x = self.relu3(x)
            npy_to_txt(2, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))

            x = self.pad4 (x)
            x = self.conv4(x)
            x = self.bn4  (x)
            x = self.relu4(x)
            npy_to_txt(3, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))
            x = self.conv5(x)
            x = self.bn5  (x)
            x = self.relu5(x)
            npy_to_txt(4, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))

            x = self.pad6 (x)
            x = self.conv6(x)
            x = self.bn6  (x)
            x = self.relu6(x)
            npy_to_txt(5, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))
            x = self.conv7(x)
            x = self.bn7  (x)
            x = self.relu7(x)
            npy_to_txt(6, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))

            x = self.pad8 (x)
            x = self.conv8(x)
            x = self.bn8  (x)
            x = self.relu8(x)   
            npy_to_txt(7, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))
            x = self.conv9(x)
            x = self.bn9  (x)
            x = self.relu9(x)   
            npy_to_txt(8, x.int().cpu().detach().numpy())
            print ("Sum: ", str(torch.sum(x.int())))

            x = self.avg(x)
            npy_to_txt(9, x.int().cpu().detach().numpy())
            x = torch.flatten(x, 1) 
            x = self.fc1(x)
            npy_to_txt(10, x.int().cpu().detach().numpy())

        else:

            x = self.pad1 (x)
            x = self.conv1(x)       
            x = self.bn1  (x)         
            x = self.relu1(x)
            
            x = self.pad2 (x)
            x = self.conv2(x)           
            x = self.bn2  (x)            
            x = self.relu2(x)            
            x = self.conv3(x)            
            x = self.bn3  (x)            
            x = self.relu3(x)
            
            x = self.pad4 (x)
            x = self.conv4(x)            
            x = self.bn4  (x)            
            x = self.relu4(x)            
            x = self.conv5(x)            
            x = self.bn5  (x)            
            x = self.relu5(x)            

            x = self.pad6 (x)
            x = self.conv6(x)          
            x = self.bn6  (x)            
            x = self.relu6(x)          
            x = self.conv7(x)            
            x = self.bn7  (x)            
            x = self.relu7(x)
            
            x = self.pad8 (x)            
            x = self.conv8(x)            
            x = self.bn8  (x)            
            x = self.relu8(x)            
            x = self.conv9(x)            
            x = self.bn9  (x)            
            x = self.relu9(x)          

            x = self.avg(x)            
            x = torch.flatten(x, 1) 
            x = self.fc1(x)
            
        return x # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1) 

class DSCNN_TE(torch.nn.Module):
    reg_count = 0
    def __init__(self, use_bias=True):
        super(DSCNN_TE, self).__init__()

        use_bias = True
        
        self.first_conv = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (7, 7), stride = (2, 2), padding = (3,3),bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()

        self.depth1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
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
        
        max_weights = []
        min_weights = []
        max_in_out = []
        min_in_out = []

        for name, module in self.named_modules():
            if name[:5] == "first" or name[:5] == "depth" or name[:5] == "point":
                max_weights.append(torch.zeros(module.out_channels))
                min_weights.append(torch.zeros(module.out_channels))
                max_in_out.append(torch.tensor(0))
                max_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))
            if name[:2] == "fc":
                max_weights.append(torch.zeros(module.out_features))
                min_weights.append(torch.zeros(module.out_features)) 
                max_in_out.append(torch.tensor(0))
                max_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))  

        self.register_buffer('max_weights', torch.cat(max_weights))
        
        self.register_buffer('min_weights', torch.cat(min_weights))

        self.register_buffer('max_in_out', torch.stack(max_in_out))

        self.register_buffer('min_in_out', torch.stack(min_in_out))

        
    def forward(self, x, save = False):
        if save:
            self.set_max_min_in_out(x)
            x = self.first_conv(x)
            x = self.bn1(x)
            self.set_max_min_in_out(x)
            x = self.relu1(x)
            
            self.set_max_min_in_out(x)
            x = self.depth1(x)
            x = self.bn2(x)
            self.set_max_min_in_out(x)
            x = self.relu2(x)
            self.set_max_min_in_out(x)
            x = self.pointw1(x)
            x = self.bn3(x)
            self.set_max_min_in_out(x)
            x = self.relu3(x)
            
            self.set_max_min_in_out(x)
            x = self.depth2(x)
            x = self.bn4(x)
            self.set_max_min_in_out(x)
            x = self.relu4(x)
            self.set_max_min_in_out(x)
            x = self.pointw2(x)
            x = self.bn5(x)
            self.set_max_min_in_out(x)
            x = self.relu5(x)
            
            self.set_max_min_in_out(x)
            x = self.depth3(x)
            x = self.bn6(x)
            self.set_max_min_in_out(x)
            x = self.relu6(x)
            self.set_max_min_in_out(x)
            x = self.pointw3(x)
            x = self.bn7(x)
            self.set_max_min_in_out(x)
            x = self.relu7(x)
            
            self.set_max_min_in_out(x)
            x = self.depth4(x)
            x = self.bn8(x)
            self.set_max_min_in_out(x)
            x = self.relu8(x)   
            self.set_max_min_in_out(x)
            x = self.pointw4(x)
            x = self.bn9(x)
            self.set_max_min_in_out(x)
            x = self.relu9(x)   
            
            x = self.avg(x)
            x = torch.flatten(x, 1) 
            self.set_max_min_in_out(x)
            x = self.fc1(x)
            self.set_max_min_in_out(x, True)

        else:
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
        
        return x # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1) 

    def set_max_min_in_out (self, x, last = False):
        if (self.max_in_out[self.reg_count] < torch.max(x)):
            self.max_in_out[self.reg_count] = torch.max(x)
        if (self.min_in_out[self.reg_count] > torch.min(x)):
            self.min_in_out[self.reg_count] = torch.min(x)
        if last:
            self.reg_count = 0
        else:
            self.reg_count +=1

class DSCNN_TE_3x3_49x10(torch.nn.Module):
    reg_count = 0
    def __init__(self, use_bias=True):
        super(DSCNN_TE_3x3_49x10, self).__init__()

        use_bias = True
        
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
        
        max_weights = []
        min_weights = []
        max_in_out = []
        min_in_out = []

        for name, module in self.named_modules():
            if name[:5] == "first" or name[:5] == "depth" or name[:5] == "point":
                max_weights.append(torch.zeros(module.out_channels))
                min_weights.append(torch.zeros(module.out_channels))
                max_in_out.append(torch.tensor(0))
                max_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))
            if name[:2] == "fc":
                max_weights.append(torch.zeros(module.out_features))
                min_weights.append(torch.zeros(module.out_features)) 
                max_in_out.append(torch.tensor(0))
                max_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(0))  

        self.register_buffer('max_weights', torch.cat(max_weights))
        
        self.register_buffer('min_weights', torch.cat(min_weights))

        self.register_buffer('max_in_out', torch.stack(max_in_out))

        self.register_buffer('min_in_out', torch.stack(min_in_out))

        
    def forward(self, x, save = False):
        if save:
            self.set_max_min_in_out(x)
            x = self.first_conv(x)
            x = self.bn1(x)
            self.set_max_min_in_out(x)
            x = self.relu1(x)
            
            self.set_max_min_in_out(x)
            x = self.depth1(x)
            x = self.bn2(x)
            self.set_max_min_in_out(x)
            x = self.relu2(x)
            self.set_max_min_in_out(x)
            x = self.pointw1(x)
            x = self.bn3(x)
            self.set_max_min_in_out(x)
            x = self.relu3(x)
            
            self.set_max_min_in_out(x)
            x = self.depth2(x)
            x = self.bn4(x)
            self.set_max_min_in_out(x)
            x = self.relu4(x)
            self.set_max_min_in_out(x)
            x = self.pointw2(x)
            x = self.bn5(x)
            self.set_max_min_in_out(x)
            x = self.relu5(x)
            
            self.set_max_min_in_out(x)
            x = self.depth3(x)
            x = self.bn6(x)
            self.set_max_min_in_out(x)
            x = self.relu6(x)
            self.set_max_min_in_out(x)
            x = self.pointw3(x)
            x = self.bn7(x)
            self.set_max_min_in_out(x)
            x = self.relu7(x)
            
            self.set_max_min_in_out(x)
            x = self.depth4(x)
            x = self.bn8(x)
            self.set_max_min_in_out(x)
            x = self.relu8(x)   
            self.set_max_min_in_out(x)
            x = self.pointw4(x)
            x = self.bn9(x)
            self.set_max_min_in_out(x)
            x = self.relu9(x)   
            
            x = self.avg(x)
            x = torch.flatten(x, 1) 
            self.set_max_min_in_out(x)
            x = self.fc1(x)
            self.set_max_min_in_out(x, True)

        else:
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
        
        return x # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1) 

    def set_max_min_in_out (self, x, last = False):
        if (self.max_in_out[self.reg_count] < torch.max(x)):
            self.max_in_out[self.reg_count] = torch.max(x)
        if (self.min_in_out[self.reg_count] > torch.min(x)):
            self.min_in_out[self.reg_count] = torch.min(x)
        if last:
            self.reg_count = 0
        else:
            self.reg_count +=1

class DSCNN_TE_NoBN_3x3(torch.nn.Module):
    def __init__(self, use_bias=True):
        super(DSCNN_TE_NoBN_3x3, self).__init__()

        use_bias = True
        max_weights = [torch.zeros(64) for i in range(10)]
        #max_weights[9] = torch.zeros(12)
        self.register_buffer('max_weights', torch.stack(max_weights))
        

        min_weights = [torch.zeros(64) for i in range(10)]
        #min_weights[9] = torch.zeros(12)
        self.register_buffer('min_weights', torch.stack(min_weights))

        max_in_out = [torch.tensor(0) for i in range(20)]
        self.register_buffer('max_in_out', torch.stack(max_in_out))

        min_in_out = [torch.tensor(0) for i in range(20)]
        self.register_buffer('min_in_out', torch.stack(min_in_out))


        self.first_conv = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3, 3), stride = (2, 2), padding = (1,1),bias = use_bias)
        self.relu1 = torch.nn.ReLU()

        self.depth1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
        self.relu2 = torch.nn.ReLU()
        self.pointw1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.relu3 = torch.nn.ReLU()

        self.depth2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
        self.relu4 = torch.nn.ReLU()
        self.pointw2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.relu5 = torch.nn.ReLU()

        self.depth3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
        self.relu6 = torch.nn.ReLU()
        self.pointw3 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.relu7 = torch.nn.ReLU()

        self.depth4 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 64, bias = use_bias)
        self.relu8 = torch.nn.ReLU()
        self.pointw4 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.relu9 = torch.nn.ReLU()

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.fc1   = torch.nn.Linear(64, 10, bias=use_bias)
        # self.soft  = torch.nn.Softmax(dim=1)
        # self.soft = F.log_softmax(x, dim=1)


        # CONV2D replacing Block1 for evaluation purposes
        # self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        # self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 1, bias = use_bias)
        # self.bn2   = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU()
        
    def forward(self, x, save = False):
        
        self.set_max_min_in_out(x, 0)
        x = self.first_conv(x)
        self.set_max_min_in_out(x, 1)
        x = self.relu1(x)
        
        self.set_max_min_in_out(x, 2)
        x = self.depth1(x)
        self.set_max_min_in_out(x, 3)
        x = self.relu2(x)
        self.set_max_min_in_out(x, 4)
        x = self.pointw1(x)
        self.set_max_min_in_out(x, 5)
        x = self.relu3(x)
        
        self.set_max_min_in_out(x, 6)
        x = self.depth2(x)
        self.set_max_min_in_out(x, 7)
        x = self.relu4(x)
        self.set_max_min_in_out(x, 8)
        x = self.pointw2(x)
        self.set_max_min_in_out(x, 9)
        x = self.relu5(x)
        
        self.set_max_min_in_out(x, 10)
        x = self.depth3(x)
        self.set_max_min_in_out(x, 11)
        x = self.relu6(x)
        self.set_max_min_in_out(x, 12)
        x = self.pointw3(x)
        self.set_max_min_in_out(x, 13)
        x = self.relu7(x)
        
        self.set_max_min_in_out(x, 14)
        x = self.depth4(x)
        self.set_max_min_in_out(x, 14)
        x = self.relu8(x)   
        self.set_max_min_in_out(x, 15)
        x = self.pointw4(x)
        self.set_max_min_in_out(x, 16)
        x = self.relu9(x)   
        
        self.set_max_min_in_out(x, 17)
        x = self.avg(x)
        x = torch.flatten(x, 1) 
        self.set_max_min_in_out(x, 18)
        x = self.fc1(x)
        self.set_max_min_in_out(x, 19)
            
        return x # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1) 

    def set_max_min_in_out (self, x, i):
        if (self.max_in_out[i] < torch.max(x)):
            self.max_in_out[i] = torch.max(x)
        if (self.min_in_out[i] > torch.min(x)):
            self.min_in_out[i] = torch.min(x)

class IBB_Block(torch.nn.Module):
    def __init__(self, module, res=True):
        super(IBB_Block, self).__init__()
        self.module = module
        self.res = res
    
    def forward(self, x):
        if self.res:
            return self.module(x)+x
        else:
            return self.module(x)

class MCUNet5FPS(torch.nn.Module):
    def __init__(self, use_bias=True):
        super(MCUNet5FPS, self).__init__()

        use_bias = True
        kernel_size = (1, 1)
        stride = 1
        padding = 1
        #max_weights = [torch.zeros(64) for i in range(10)]
        #max_weights[9] = torch.zeros(12)
        #self.register_buffer('max_weights', torch.stack(max_weights))
        

        #min_weights = [torch.zeros(64) for i in range(10)]
        #min_weights[9] = torch.zeros(12)
        #self.register_buffer('min_weights', torch.stack(min_weights))

        #max_in_out = [torch.tensor(0) for i in range(20)]
        #self.register_buffer('max_in_out', torch.stack(max_in_out))

        #min_in_out = [torch.tensor(0) for i in range(20)]
        #self.register_buffer('min_in_out', torch.stack(min_in_out))


        self.first_conv = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 3), stride = (2, 2), padding = (1,1),bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()

        self.depth1 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 16, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU()
        self.pointw1 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(8)
        self.relu3 = torch.nn.ReLU()
        self.ibb1 = IBB_Block(torch.nn.Sequential(self.depth1, self.bn2, self.relu2, self.pointw1, self.bn3, self.relu3), False)

        self.pointw2 = torch.nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(32)
        self.relu4 = torch.nn.ReLU()
        self.depth2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 32, bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(32)
        self.relu5 = torch.nn.ReLU()
        self.pointw3 = torch.nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(16)
        self.relu6 = torch.nn.ReLU()
        self.ibb2 = IBB_Block(torch.nn.Sequential(self.pointw2, self.bn4, self.relu4, self.depth2, self.bn5, self.relu5, self.pointw3, self.bn6, self.relu6), False)

        self.pointw4 = torch.nn.Conv2d(in_channels = 16, out_channels = 48, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(48)
        self.relu7 = torch.nn.ReLU()
        self.depth3 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 48, bias = use_bias)
        self.bn8   = torch.nn.BatchNorm2d(48)
        self.relu8 = torch.nn.ReLU()
        self.pointw5 = torch.nn.Conv2d(in_channels = 48, out_channels = 16, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn9   = torch.nn.BatchNorm2d(16)
        self.relu9 = torch.nn.ReLU()
        self.ibb3 = IBB_Block(torch.nn.Sequential(self.pointw4, self.bn7, self.relu7, self.depth3, self.bn8, self.relu8, self.pointw5, self.bn9, self.relu9), True)

        self.pointw6 = torch.nn.Conv2d(in_channels = 16, out_channels = 48, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn10   = torch.nn.BatchNorm2d(48)
        self.relu10 = torch.nn.ReLU()
        self.depth4 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = (7, 7), stride = (1, 1), padding = 3, groups = 48, bias = use_bias)
        self.bn11   = torch.nn.BatchNorm2d(48)
        self.relu11 = torch.nn.ReLU()
        self.pointw7 = torch.nn.Conv2d(in_channels = 48, out_channels = 24, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn12   = torch.nn.BatchNorm2d(24)
        self.relu12 = torch.nn.ReLU()
        self.ibb4 = IBB_Block(torch.nn.Sequential(self.pointw6, self.bn10, self.relu10, self.depth4, self.bn11, self.relu11, self.pointw7, self.bn12, self.relu12), False)

        self.pointw8 = torch.nn.Conv2d(in_channels = 24, out_channels = 120, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn13   = torch.nn.BatchNorm2d(120)
        self.relu13 = torch.nn.ReLU()
        self.depth5 = torch.nn.Conv2d(in_channels = 120, out_channels = 120, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 120, bias = use_bias)
        self.bn14   = torch.nn.BatchNorm2d(120)
        self.relu14 = torch.nn.ReLU()
        self.pointw9 = torch.nn.Conv2d(in_channels = 120, out_channels = 24, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn15   = torch.nn.BatchNorm2d(24)
        self.relu15 = torch.nn.ReLU()
        self.ibb5 = IBB_Block(torch.nn.Sequential(self.pointw8, self.bn13, self.relu13, self.depth5, self.bn14, self.relu14, self.pointw9, self.bn15, self.relu15), True)

        self.pointw10 = torch.nn.Conv2d(in_channels=24, out_channels=120, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn16   = torch.nn.BatchNorm2d(120)
        self.relu16 = torch.nn.ReLU()
        self.depth6 = torch.nn.Conv2d(in_channels=120, out_channels=120, kernel_size=(3, 3), stride=stride, padding=padding, groups=120, bias=use_bias)
        self.bn17   = torch.nn.BatchNorm2d(120)
        self.relu17 = torch.nn.ReLU()
        self.pointw11 = torch.nn.Conv2d(in_channels=120, out_channels=40, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn18   = torch.nn.BatchNorm2d(40)
        self.relu18 = torch.nn.ReLU()
        self.ibb6 = IBB_Block(torch.nn.Sequential(self.pointw10, self.bn16, self.relu16, self.depth6, self.bn17, self.relu17, self.pointw11, self.bn18, self.relu18), False)

        self.pointw12 = torch.nn.Conv2d(in_channels=40, out_channels=160, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn19   = torch.nn.BatchNorm2d(160)
        self.relu19 = torch.nn.ReLU()
        self.depth7 = torch.nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 7), stride=stride, padding=3, groups=160, bias=use_bias)
        self.bn20   = torch.nn.BatchNorm2d(160)
        self.relu20 = torch.nn.ReLU()
        self.pointw13 = torch.nn.Conv2d(in_channels=160, out_channels=40, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn21   = torch.nn.BatchNorm2d(40)
        self.relu21 = torch.nn.ReLU()
        self.ibb7 = IBB_Block(torch.nn.Sequential(self.pointw12, self.bn19, self.relu19, self.depth7, self.bn20, self.relu20, self.pointw13, self.bn21, self.relu21), False)

        self.pointw14 = torch.nn.Conv2d(in_channels=40, out_channels=160, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn22   = torch.nn.BatchNorm2d(160)
        self.relu22 = torch.nn.ReLU()
        self.depth8 = torch.nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(5, 5), stride=stride, padding=2, groups=160, bias=use_bias)
        self.bn23   = torch.nn.BatchNorm2d(160)
        self.relu23 = torch.nn.ReLU()
        self.pointw15 = torch.nn.Conv2d(in_channels=160, out_channels=48, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn24   = torch.nn.BatchNorm2d(48)
        self.relu24 = torch.nn.ReLU()
        self.ibb8 = IBB_Block(torch.nn.Sequential(self.pointw14, self.bn22, self.relu22, self.depth8, self.bn23, self.relu23, self.pointw15, self.bn24, self.relu24), False)

        self.pointw16 = torch.nn.Conv2d(in_channels=48, out_channels=144, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn25   = torch.nn.BatchNorm2d(144)
        self.relu25 = torch.nn.ReLU()
        self.depth9 = torch.nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=stride, padding=padding, groups=144, bias=use_bias)
        self.bn26   = torch.nn.BatchNorm2d(144)
        self.relu26 = torch.nn.ReLU()
        self.pointw17 = torch.nn.Conv2d(in_channels=144, out_channels=48, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn27   = torch.nn.BatchNorm2d(48)
        self.relu27 = torch.nn.ReLU()
        self.ibb9 = IBB_Block(torch.nn.Sequential(self.pointw16, self.bn25, self.relu25, self.depth9, self.bn26, self.relu26, self.pointw17, self.bn27, self.relu27), True)

        self.pointw18 = torch.nn.Conv2d(in_channels=48, out_channels=192, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn28   = torch.nn.BatchNorm2d(192)
        self.relu28 = torch.nn.ReLU()
        self.depth10 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=stride, padding=padding, groups=192, bias=use_bias)
        self.bn29   = torch.nn.BatchNorm2d(192)
        self.relu29 = torch.nn.ReLU()
        self.pointw19 = torch.nn.Conv2d(in_channels=192, out_channels=48, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn30   = torch.nn.BatchNorm2d(48)
        self.relu30 = torch.nn.ReLU()
        self.ibb10 = IBB_Block(torch.nn.Sequential(self.pointw18, self.bn28, self.relu28, self.depth10, self.bn29, self.relu29, self.pointw19, self.bn30, self.relu30), True)

        self.pointw20 = torch.nn.Conv2d(in_channels=48, out_channels=240, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn31   = torch.nn.BatchNorm2d(240)
        self.relu31 = torch.nn.ReLU()
        self.depth11 = torch.nn.Conv2d(in_channels=240, out_channels=240, kernel_size=(7, 7), stride=stride, padding=3, groups=240, bias=use_bias)
        self.bn32   = torch.nn.BatchNorm2d(240)
        self.relu32 = torch.nn.ReLU()
        self.pointw21 = torch.nn.Conv2d(in_channels=240, out_channels=96, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn33   = torch.nn.BatchNorm2d(96)
        self.relu33 = torch.nn.ReLU()
        self.ibb11 = IBB_Block(torch.nn.Sequential(self.pointw20, self.bn31, self.relu31, self.depth11, self.bn32, self.relu32, self.pointw21, self.bn33, self.relu33), False)

        self.pointw22 = torch.nn.Conv2d(in_channels=96, out_channels=384, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn34   = torch.nn.BatchNorm2d(384)
        self.relu34 = torch.nn.ReLU()
        self.depth12 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(5, 5), stride=stride, padding=2, groups=384, bias=use_bias)
        self.bn35   = torch.nn.BatchNorm2d(384)
        self.relu35 = torch.nn.ReLU()
        self.pointw23 = torch.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn36   = torch.nn.BatchNorm2d(96)
        self.relu36 = torch.nn.ReLU()
        self.ibb12 = IBB_Block(torch.nn.Sequential(self.pointw22, self.bn34, self.relu34, self.depth12, self.bn35, self.relu35, self.pointw23, self.bn36, self.relu36), True)

        self.pointw24 = torch.nn.Conv2d(in_channels=96, out_channels=384, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn37   = torch.nn.BatchNorm2d(384)
        self.relu37 = torch.nn.ReLU()
        self.depth13 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(5, 5), stride=stride, padding=2, groups=384, bias=use_bias)
        self.bn38   = torch.nn.BatchNorm2d(384)
        self.relu38 = torch.nn.ReLU()
        self.pointw25 = torch.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn39   = torch.nn.BatchNorm2d(96)
        self.relu39 = torch.nn.ReLU()
        self.ibb13 = IBB_Block(torch.nn.Sequential(self.pointw24, self.bn37, self.relu37, self.depth13, self.bn38, self.relu38, self.pointw25, self.bn39, self.relu39), False)

        self.pointw26 = torch.nn.Conv2d(in_channels=96, out_channels=576, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn40   = torch.nn.BatchNorm2d(576)
        self.relu40 = torch.nn.ReLU()
        self.depth14 = torch.nn.Conv2d(in_channels=576, out_channels=576, kernel_size=(3, 3), stride=stride, padding=padding, groups=576, bias=use_bias)
        self.bn41   = torch.nn.BatchNorm2d(576)
        self.relu41 = torch.nn.ReLU()
        self.pointw27 = torch.nn.Conv2d(in_channels=576, out_channels=160, kernel_size=kernel_size, stride=stride, bias=use_bias)
        self.bn42   = torch.nn.BatchNorm2d(160)
        self.relu42 = torch.nn.ReLU()
        self.ibb14 = IBB_Block(torch.nn.Sequential(self.pointw26, self.bn40, self.relu40, self.depth14, self.bn41, self.relu41, self.pointw27, self.bn42, self.relu42), False)

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.fc1   = torch.nn.Linear(160, 10, bias=use_bias)
        # self.soft  = torch.nn.Softmax(dim=1)
        # self.soft = F.log_softmax(x, dim=1)


        # CONV2D replacing Block1 for evaluation purposes
        # self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        # self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 1, bias = use_bias)
        # self.bn2   = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU()
        
    def forward(self, x, save = False):
        
        #self.set_max_min_in_out(x, 0)
        x = self.first_conv(x)
        x = self.bn1(x)
        #self.set_max_min_in_out(x, 1)
        x = self.relu1(x)
        
        x = self.ibb1(x)
        x = self.ibb2(x)
        x = self.ibb3(x)
        x = self.ibb4(x)
        x = self.ibb5(x)
        x = self.ibb6(x)
        x = self.ibb7(x)
        x = self.ibb8(x)
        x = self.ibb9(x)
        x = self.ibb10(x)
        x = self.ibb11(x)
        x = self.ibb12(x)
        x = self.ibb13(x)
        x = self.ibb14(x)
        
        #self.set_max_min_in_out(x, 84)
        x = self.avg(x)
        x = torch.flatten(x, 1) 
        #self.set_max_min_in_out(x, 85)
        x = self.fc1(x)
        #self.set_max_min_in_out(x, 86)
            
        return x # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1) 

    def set_max_min_in_out (self, x, i):
        if (self.max_in_out[i] < torch.max(x)):
            self.max_in_out[i] = torch.max(x)
        if (self.min_in_out[i] > torch.min(x)):
            self.min_in_out[i] = torch.min(x)

class MobileNetv2(torch.nn.Module):
    def __init__(self, use_bias=True):
        super(MobileNetv2, self).__init__()
        
        max_weights = [torch.zeros(64) for i in range(10)]
        #max_weights[9] = torch.zeros(12)
        self.register_buffer('max_weights', torch.stack(max_weights))
        

        min_weights = [torch.zeros(64) for i in range(10)]
        #min_weights[9] = torch.zeros(12)
        self.register_buffer('min_weights', torch.stack(min_weights))

        max_in_out = [torch.tensor(0) for i in range(20)]
        self.register_buffer('max_in_out', torch.stack(max_in_out))

        min_in_out = [torch.tensor(0) for i in range(20)]
        self.register_buffer('min_in_out', torch.stack(min_in_out))
        #TODO fix for quantizing
        self.first_conv = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 3), stride = (2, 2), padding = (1,1),bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()

        self.depth1 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 16, bias = use_bias)
        self.bn2   = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ReLU()
        self.pointw1 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn3   = torch.nn.BatchNorm2d(8)
        self.relu3 = torch.nn.ReLU()

        self.pointw2 = torch.nn.Conv2d(in_channels = 8, out_channels = 48, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn4   = torch.nn.BatchNorm2d(48)
        self.relu4 = torch.nn.ReLU()
        self.depth2 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 48, bias = use_bias)
        self.bn5   = torch.nn.BatchNorm2d(48)
        self.relu5 = torch.nn.ReLU()
        self.pointw3 = torch.nn.Conv2d(in_channels = 48, out_channels = 8, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn6   = torch.nn.BatchNorm2d(8)
        self.relu6 = torch.nn.ReLU()

        self.pointw4 = torch.nn.Conv2d(in_channels = 8, out_channels = 48, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn7   = torch.nn.BatchNorm2d(48)
        self.relu7 = torch.nn.ReLU()
        self.depth3 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 48, bias = use_bias)
        self.bn8   = torch.nn.BatchNorm2d(48)
        self.relu8 = torch.nn.ReLU()
        self.pointw5 = torch.nn.Conv2d(in_channels = 48, out_channels = 8, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn9   = torch.nn.BatchNorm2d(8)
        self.relu9 = torch.nn.ReLU()
        self.ibb1 = IBB_Block(nn.Sequential(self.pointw4, self.bn7, self.relu7, self.depth3, self.bn8, self.relu8, self.pointw5, self.bn9, self.relu9))

        self.pointw6 = torch.nn.Conv2d(in_channels = 8, out_channels = 48, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn10   = torch.nn.BatchNorm2d(48)
        self.relu10 = torch.nn.ReLU()
        self.depth4 = torch.nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 48, bias = use_bias)
        self.bn11   = torch.nn.BatchNorm2d(48)
        self.relu11 = torch.nn.ReLU()
        self.pointw7 = torch.nn.Conv2d(in_channels = 48, out_channels = 16, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn12   = torch.nn.BatchNorm2d(16)
        self.relu12 = torch.nn.ReLU()

        self.pointw8 = torch.nn.Conv2d(in_channels = 16, out_channels = 96, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn13   = torch.nn.BatchNorm2d(96)
        self.relu13 = torch.nn.ReLU()
        self.depth5 = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = (3, 3), stride = (1, 1), padding = 1, groups = 96, bias = use_bias)
        self.bn14   = torch.nn.BatchNorm2d(96)
        self.relu14 = torch.nn.ReLU()
        self.pointw9 = torch.nn.Conv2d(in_channels = 96, out_channels = 16, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn15   = torch.nn.BatchNorm2d(16)
        self.relu15 = torch.nn.ReLU()
        self.ibb2 = IBB_Block(nn.Sequential(self.pointw8, self.bn13, self.relu13, self.depth5, self.bn14, self.relu14, self.pointw9, self.bn15,  self.relu15))

        self.pointw10 = torch.nn.Conv2d(in_channels=16, out_channels=96, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn16   = torch.nn.BatchNorm2d(96)
        self.relu16 = torch.nn.ReLU()
        self.depth6 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=96, bias=use_bias)
        self.bn17   = torch.nn.BatchNorm2d(96)
        self.relu17 = torch.nn.ReLU()
        self.pointw11 = torch.nn.Conv2d(in_channels=96, out_channels=16, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn18   = torch.nn.BatchNorm2d(16)
        self.relu18 = torch.nn.ReLU()
        self.ibb3 = IBB_Block(nn.Sequential(self.pointw10, self.bn16, self.relu16, self.depth6, self.bn17, self.relu17, self.pointw11, self.bn18, self.relu18))

        self.pointw12 = torch.nn.Conv2d(in_channels=16, out_channels=96, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn19 = torch.nn.BatchNorm2d(96)
        self.relu19 = torch.nn.ReLU()
        self.depth7 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=96, bias=use_bias)
        self.bn20 = torch.nn.BatchNorm2d(96)
        self.relu20 = torch.nn.ReLU()
        self.pointw13 = torch.nn.Conv2d(in_channels=96, out_channels=24, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn21 = torch.nn.BatchNorm2d(24)
        self.relu21 = torch.nn.ReLU()

        self.pointw14 = torch.nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn22 = torch.nn.BatchNorm2d(144)
        self.relu22 = torch.nn.ReLU()
        self.depth8 = torch.nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=144, bias=use_bias)
        self.bn23 = torch.nn.BatchNorm2d(144)
        self.relu23 = torch.nn.ReLU()
        self.pointw15 = torch.nn.Conv2d(in_channels=144, out_channels=24, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn24 = torch.nn.BatchNorm2d(24)
        self.relu24 = torch.nn.ReLU()
        self.ibb4 = IBB_Block(nn.Sequential(self.pointw14, self.bn22, self.relu22, self.depth8, self.bn23, self.relu23, self.pointw15, self.bn24, self.relu24))

        self.pointw16 = torch.nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn25 = torch.nn.BatchNorm2d(144)
        self.relu25 = torch.nn.ReLU()
        self.depth9 = torch.nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=144, bias=use_bias)
        self.bn26 = torch.nn.BatchNorm2d(144)
        self.relu26 = torch.nn.ReLU()
        self.pointw17 = torch.nn.Conv2d(in_channels=144, out_channels=24, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn27 = torch.nn.BatchNorm2d(24)
        self.relu27 = torch.nn.ReLU()
        self.ibb5 = IBB_Block(nn.Sequential(self.pointw16, self.bn25, self.relu25, self.depth9, self.bn26, self.relu26, self.pointw17, self.bn27, self.relu27))

        self.pointw18 = torch.nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn28 = torch.nn.BatchNorm2d(144)
        self.relu28 = torch.nn.ReLU()
        self.depth10 = torch.nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=144, bias=use_bias)
        self.bn29 = torch.nn.BatchNorm2d(144)
        self.relu29 = torch.nn.ReLU()
        self.pointw19 = torch.nn.Conv2d(in_channels=144, out_channels=24, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn30 = torch.nn.BatchNorm2d(24)
        self.relu30 = torch.nn.ReLU()
        self.ibb6 = IBB_Block(nn.Sequential(self.pointw18, self.bn28, self.relu28, self.depth10, self.bn29, self.relu29, self.pointw19, self.bn30, self.relu30))

        self.pointw20 = torch.nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn31 = torch.nn.BatchNorm2d(144)
        self.relu31 = torch.nn.ReLU()
        self.depth11 = torch.nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=144, bias=use_bias)
        self.bn32 = torch.nn.BatchNorm2d(144)
        self.relu32 = torch.nn.ReLU()
        self.pointw21 = torch.nn.Conv2d(in_channels=144, out_channels=32, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn33 = torch.nn.BatchNorm2d(32)
        self.relu33 = torch.nn.ReLU()

        self.pointw22 = torch.nn.Conv2d(in_channels=32, out_channels=192, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn34 = torch.nn.BatchNorm2d(192)
        self.relu34 = torch.nn.ReLU()
        self.depth12 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=192, bias=use_bias)
        self.bn35 = torch.nn.BatchNorm2d(192)
        self.relu35 = torch.nn.ReLU()
        self.pointw23 = torch.nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn36 = torch.nn.BatchNorm2d(32)
        self.relu36 = torch.nn.ReLU()
        self.ibb7 = IBB_Block(nn.Sequential(self.pointw22, self.bn34, self.relu34, self.depth12, self.bn35, self.relu35, self.pointw23, self.bn36, self.relu36))

        self.pointw24 = torch.nn.Conv2d(in_channels=32, out_channels=192, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn37 = torch.nn.BatchNorm2d(192)
        self.relu37 = torch.nn.ReLU()
        self.depth13 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=192, bias=use_bias)
        self.bn38 = torch.nn.BatchNorm2d(192)
        self.relu38 = torch.nn.ReLU()
        self.pointw25 = torch.nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn39 = torch.nn.BatchNorm2d(32)
        self.relu39 = torch.nn.ReLU()
        self.ibb8 = IBB_Block(nn.Sequential(self.pointw24, self.relu37, self.depth13, self.relu38, self.pointw25, self.relu39))

        self.pointw26 = torch.nn.Conv2d(in_channels=32, out_channels=192, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn40 = torch.nn.BatchNorm2d(192)
        self.relu40 = torch.nn.ReLU()
        self.depth14 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=192, bias=use_bias)
        self.bn41 = torch.nn.BatchNorm2d(192)
        self.relu41 = torch.nn.ReLU()
        self.pointw27 = torch.nn.Conv2d(in_channels=192, out_channels=56, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn42 = torch.nn.BatchNorm2d(56)
        self.relu42 = torch.nn.ReLU()

        self.pointw28 = torch.nn.Conv2d(in_channels=56, out_channels=336, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn43 = torch.nn.BatchNorm2d(336)
        self.relu43 = torch.nn.ReLU()
        self.depth15 = torch.nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=336, bias=use_bias)
        self.bn44 = torch.nn.BatchNorm2d(336)
        self.relu44 = torch.nn.ReLU()
        self.pointw29 = torch.nn.Conv2d(in_channels=336, out_channels=56, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn45 = torch.nn.BatchNorm2d(56)
        self.relu45 = torch.nn.ReLU()
        self.ibb9 = IBB_Block(nn.Sequential(self.pointw28, self.bn43, self.relu43, self.depth15, self.bn44, self.relu44, self.pointw29, self.bn45, self.relu45))

        self.pointw30 = torch.nn.Conv2d(in_channels=56, out_channels=336, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn46 = torch.nn.BatchNorm2d(336)
        self.relu46 = torch.nn.ReLU()
        self.depth16 = torch.nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=336, bias=use_bias)
        self.bn47 = torch.nn.BatchNorm2d(336)
        self.relu47 = torch.nn.ReLU()
        self.pointw31 = torch.nn.Conv2d(in_channels=336, out_channels=56, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn48 = torch.nn.BatchNorm2d(56)
        self.relu48 = torch.nn.ReLU()
        self.ibb10 = IBB_Block(nn.Sequential(self.pointw30, self.bn46, self.relu46, self.depth16, self.bn47, self.relu47, self.pointw31, self.bn48, self.relu48))

        self.pointw32 = torch.nn.Conv2d(in_channels=56, out_channels=336, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn49 = torch.nn.BatchNorm2d(336)
        self.relu49 = torch.nn.ReLU()
        self.depth17 = torch.nn.Conv2d(in_channels=336, out_channels=336, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=336, bias=use_bias)
        self.bn50 = torch.nn.BatchNorm2d(336)
        self.relu50 = torch.nn.ReLU()
        self.pointw33 = torch.nn.Conv2d(in_channels=336, out_channels=112, kernel_size=(1, 1), stride=(1, 1), bias=use_bias)
        self.bn51 = torch.nn.BatchNorm2d(112)
        self.relu51 = torch.nn.ReLU()

        self.avg   = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.fc1   = torch.nn.Linear(112, 10, bias=use_bias)
        # self.soft  = torch.nn.Softmax(dim=1)
        # self.soft = F.log_softmax(x, dim=1)


        # CONV2D replacing Block1 for evaluation purposes
        # self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        # self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 1, bias = use_bias)
        # self.bn2   = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU()
        
    def forward(self, x, save = False):
        
        #self.set_max_min_in_out(x, 0)
        x = self.first_conv(x)
        #self.set_max_min_in_out(x, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        
        #self.set_max_min_in_out(x, 2)
        x = self.depth1(x)
        #self.set_max_min_in_out(x, 3)
        x = self.bn2(x)
        x = self.relu2(x)
        #self.set_max_min_in_out(x, 4)
        x = self.pointw1(x)
        #self.set_max_min_in_out(x, 5)
        x = self.bn3(x)
        x = self.relu3(x)
        
        #self.set_max_min_in_out(x, 6)
        x = self.pointw2(x)
        #self.set_max_min_in_out(x, 7)
        x = self.bn4(x)
        x = self.relu4(x)
        #self.set_max_min_in_out(x, 8)
        x = self.depth2(x)
        #self.set_max_min_in_out(x, 9)
        x = self.bn5(x)
        x = self.relu5(x)
        #self.set_max_min_in_out(x, 10)
        x = self.pointw3(x)
        #self.set_max_min_in_out(x, 11)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.ibb1(x)

        #self.set_max_min_in_out(x, 18)
        x = self.pointw6(x)
        #self.set_max_min_in_out(x, 19)
        x = self.bn10(x)
        x = self.relu10(x)
        #self.set_max_min_in_out(x, 20)
        x = self.depth4(x)
        #self.set_max_min_in_out(x, 21)
        x = self.bn11(x)
        x = self.relu11(x)
        #self.set_max_min_in_out(x, 22)
        x = self.pointw7(x)
        #self.set_max_min_in_out(x, 23)
        x = self.bn12(x)
        x = self.relu12(x)

        x = self.ibb2(x)
        x = self.ibb3(x)

        #self.set_max_min_in_out(x, 36)
        x = self.pointw12(x)
        #self.set_max_min_in_out(x, 37)
        x = self.bn19(x)
        x = self.relu19(x)
        #self.set_max_min_in_out(x, 38)
        x = self.depth7(x)
        #self.set_max_min_in_out(x, 39)
        x = self.bn20(x)
        x = self.relu20(x)
        #self.set_max_min_in_out(x, 40)
        x = self.pointw13(x)
        #self.set_max_min_in_out(x, 41)
        x = self.bn21(x)
        x = self.relu21(x)

        x = self.ibb4(x)

        x = self.ibb5(x)

        x = self.ibb6(x)

        #self.set_max_min_in_out(x, 60)
        x = self.pointw20(x)
        #self.set_max_min_in_out(x, 61)
        x = self.bn31(x)
        x = self.relu31(x)
        #self.set_max_min_in_out(x, 62)
        x = self.depth11(x)
        #self.set_max_min_in_out(x, 63)
        x = self.bn32(x)
        x = self.relu32(x)
        #self.set_max_min_in_out(x, 64)
        x = self.pointw21(x)
        #self.set_max_min_in_out(x, 65)
        x = self.bn33(x)
        x = self.relu33(x)

        
        x = self.ibb7(x)

        x = self.ibb8(x)

        #self.set_max_min_in_out(x, 78)
        x = self.pointw26(x)
        #self.set_max_min_in_out(x, 79)
        x = self.bn40(x)
        x = self.relu40(x)
        #self.set_max_min_in_out(x, 80)
        x = self.depth14(x)
        #self.set_max_min_in_out(x, 81)
        x = self.bn41(x)
        x = self.relu41(x)
        #self.set_max_min_in_out(x, 82)
        x = self.pointw27(x)
        #self.set_max_min_in_out(x, 83)
        x = self.bn42(x)
        x = self.relu42(x)

        x = self.ibb9(x)

        x = self.ibb10(x)

        #self.set_max_min_in_out(x, 96)
        x = self.pointw32(x)
        #self.set_max_min_in_out(x, 97)
        x = self.bn49(x)
        x = self.relu49(x)
        #self.set_max_min_in_out(x, 98)
        x = self.depth17(x)
        #self.set_max_min_in_out(x, 99)
        x = self.bn50(x)
        x = self.relu50(x)
        #self.set_max_min_in_out(x, 100)
        x = self.pointw33(x)
        #self.set_max_min_in_out(x, 101)
        x = self.bn51(x)
        x = self.relu51(x)

        #self.set_max_min_in_out(x, 102)
        x = self.avg(x)
        x = torch.flatten(x, 1) 
        #self.set_max_min_in_out(x, 103)
        x = self.fc1(x)
        #self.set_max_min_in_out(x, 104)
            
        return x # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1) 

    def set_max_min_in_out (self, x, i):
        if (self.max_in_out[i] < torch.max(x)):
            self.max_in_out[i] = torch.max(x)
        if (self.min_in_out[i] > torch.min(x)):
            self.min_in_out[i] = torch.min(x)