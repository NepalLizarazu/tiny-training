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
    test = 0
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

        
    def forward(self, x, save = False, quantized = False, scales_zeros = None): #Only in this PC BN are commented, in the server the BN must remain for training
        if save and not quantized:
            if self.test == 0:
                print("Input: ",x[0,0])
                x[0,0] = torch.tensor([[130.38089,131.43936,129.67656,128.64754,129.50114,129.33595,129.139,129.07536,129.11879,128.89668],[132.14569,131.96643,130.01714,128.8141,129.21088,127.99666,128.78825,128.42905,128.0869,127.49982],[128.35165,129.80748,129.42358,129.08359,128.1454,128.36932,127.58107,126.86836,127.245834,128.79285],[128.81448,130.77385,129.60121,129.37752,128.90532,127.70365,127.40197,127.98097,128.34013,129.19919],[131.13123,131.54993,128.80214,129.41241,127.74812,126.53058,127.69071,128.06276,128.24432,128.8354],[131.53786,130.96332,129.11359,129.21504,129.23364,128.54529,128.88281,128.34442,128.06316,128.1965],[128.05711,130.13252,128.70697,128.45973,130.10385,130.82455,129.74408,128.42085,128.2472,128.26],[128.1312,129.08397,128.58223,128.35168,126.587906,127.375854,128.19754,129.10364,128.17268,127.70088],[129.35075,130.01837,128.35463,128.83736,127.761284,128.48302,129.09694,128.60547,128.76706,127.85829],[128.41386,129.86687,129.89742,128.80344,128.53549,127.90915,128.96712,128.32274,127.7653,128.52193],[129.22514,130.7849,129.99434,128.1142,128.86398,128.30019,129.2305,127.81578,128.22368,129.3372],[131.29575,132.11177,128.73656,129.18912,129.13423,128.92043,128.02274,127.06772,127.450516,128.05367],[131.26366,131.32031,129.42886,128.97224,128.7604,128.23688,128.22533,128.84297,127.66184,128.86665],[129.78667,130.57726,129.54733,130.1221,130.67233,128.01291,128.61336,128.22899,127.80336,127.01785],[130.9817,132.0546,131.24615,128.04715,128.80269,129.11694,127.64418,128.29086,129.36479,128.03162],[129.22012,131.18294,129.45885,130.34448,129.68335,129.2046,128.4853,128.04555,127.959206,126.60313],[127.382545,129.05104,128.95412,129.24907,128.70264,129.36572,128.29893,128.0956,127.88728,128.08362],[129.18799,129.7071,129.47694,129.65155,127.47723,127.28498,128.89212,128.68385,127.88265,127.60673],[130.55862,131.02905,129.46361,129.56233,129.17644,128.35544,127.36579,126.99023,128.1641,128.65002],[128.68651,131.86043,130.59073,128.96101,129.91628,130.42805,128.9516,127.41191,128.76907,128.9656],[127.94517,129.92342,130.31711,129.21799,128.2734,127.32934,126.97719,127.454056,128.48653,129.25325],[129.73463,131.57393,130.32227,129.3245,128.01846,127.99733,128.88972,126.95246,127.71982,126.68717],[127.62111,128.41125,128.66315,129.29437,128.96425,127.355896,128.52422,128.45409,127.9129,128.27483],[128.9328,131.4582,130.66963,131.0656,127.86458,128.3115,127.93027,129.05913,128.71132,128.61723],[131.08109,131.3194,130.62816,130.07663,127.38624,126.660934,128.65234,127.46814,127.96135,126.84312],[130.06282,131.89987,129.82843,130.67426,128.76277,129.59354,128.89397,127.37855,128.3633,128.8769],[131.58694,130.07455,128.51755,128.88132,128.03683,127.99954,127.41392,128.2849,127.8067,128.33197],[129.27559,130.74522,129.21829,128.32835,126.95609,127.24045,128.20753,126.89045,128.31631,128.98364],[128.7202,131.31871,130.45091,129.52812,127.71568,129.55162,129.5272,127.751015,128.2281,129.27512],[130.69963,131.48055,128.8031,128.81729,128.98569,128.74188,127.21871,127.81623,128.79608,129.49234],[131.4759,132.3811,128.9594,129.81427,129.59752,128.54005,128.2609,127.82605,127.52885,127.02463],[136.96071,136.26843,134.33806,130.36679,127.8647,126.15627,125.26397,127.50848,127.49243,127.31677],[138.6253,138.53914,132.72159,130.753,126.59739,125.84584,126.521164,126.90665,126.73005,127.42356],[140.52264,139.45036,134.37659,131.32686,124.94601,125.45653,124.89712,126.12991,128.16473,129.07458],[148.95009,138.87361,134.4575,130.02313,120.327194,125.77954,127.348145,127.073166,130.22685,130.053],[153.97717,138.07811,131.6171,126.93567,120.50964,125.472,129.12758,127.01421,129.44717,130.13475],[155.64886,136.95386,129.2486,127.93064,120.6698,125.86978,130.99641,128.45712,127.98936,128.72221],[152.3936,137.04018,130.2789,128.15215,120.36439,126.085464,131.00851,128.74878,127.70343,127.96544],[151.36586,138.36502,128.6467,129.6366,121.04358,127.08364,130.16263,128.35088,126.82116,128.82199],[148.57884,137.67598,129.87448,129.32774,122.97951,126.99945,129.42673,128.7358,127.69691,129.41473],[144.38457,137.38373,132.09015,128.16415,125.35038,126.8733,129.64491,127.877785,126.27705,129.31264],[135.10799,134.86197,132.13268,131.6621,128.5219,125.90094,128.14346,127.42562,127.67985,127.807945],[132.03072,130.99577,131.74574,128.6959,129.6752,127.69522,129.13075,128.14706,128.35507,127.58883],[132.12108,130.70503,133.01549,129.62593,131.80594,127.78311,127.12216,126.63043,127.85326,126.451294],[130.61545,128.98962,129.47429,126.898964,129.6931,126.51575,129.66504,128.00993,129.37575,129.62807],[132.88889,128.60167,130.20937,127.926544,129.88683,127.5016,128.17296,127.75982,130.34598,129.13078],[132.36478,129.33015,130.3931,127.55827,130.1992,126.867424,128.5663,127.50715,129.65144,128.53305],[134.66039,129.95078,130.85008,127.76645,129.91913,126.509766,128.99881,127.584015,128.7468,126.994576],[132.15672,128.27806,130.55762,126.81696,129.64821,127.999954,129.85884,128.74342,128.96924,127.750565]])
            self.set_max_min_in_out(x)
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
            if self.test == 0:
                print("Final output: ",x[0])
                self.test = 1

        elif not save and not quantized:
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

        elif not save and quantized: #scales_zeros: [w_scales, x_scale, y_scale, x_zero, y_zero]
            if self.test == 0:
                x[0,0] = torch.tensor([[130.38089,131.43936,129.67656,128.64754,129.50114,129.33595,129.139,129.07536,129.11879,128.89668],[132.14569,131.96643,130.01714,128.8141,129.21088,127.99666,128.78825,128.42905,128.0869,127.49982],[128.35165,129.80748,129.42358,129.08359,128.1454,128.36932,127.58107,126.86836,127.245834,128.79285],[128.81448,130.77385,129.60121,129.37752,128.90532,127.70365,127.40197,127.98097,128.34013,129.19919],[131.13123,131.54993,128.80214,129.41241,127.74812,126.53058,127.69071,128.06276,128.24432,128.8354],[131.53786,130.96332,129.11359,129.21504,129.23364,128.54529,128.88281,128.34442,128.06316,128.1965],[128.05711,130.13252,128.70697,128.45973,130.10385,130.82455,129.74408,128.42085,128.2472,128.26],[128.1312,129.08397,128.58223,128.35168,126.587906,127.375854,128.19754,129.10364,128.17268,127.70088],[129.35075,130.01837,128.35463,128.83736,127.761284,128.48302,129.09694,128.60547,128.76706,127.85829],[128.41386,129.86687,129.89742,128.80344,128.53549,127.90915,128.96712,128.32274,127.7653,128.52193],[129.22514,130.7849,129.99434,128.1142,128.86398,128.30019,129.2305,127.81578,128.22368,129.3372],[131.29575,132.11177,128.73656,129.18912,129.13423,128.92043,128.02274,127.06772,127.450516,128.05367],[131.26366,131.32031,129.42886,128.97224,128.7604,128.23688,128.22533,128.84297,127.66184,128.86665],[129.78667,130.57726,129.54733,130.1221,130.67233,128.01291,128.61336,128.22899,127.80336,127.01785],[130.9817,132.0546,131.24615,128.04715,128.80269,129.11694,127.64418,128.29086,129.36479,128.03162],[129.22012,131.18294,129.45885,130.34448,129.68335,129.2046,128.4853,128.04555,127.959206,126.60313],[127.382545,129.05104,128.95412,129.24907,128.70264,129.36572,128.29893,128.0956,127.88728,128.08362],[129.18799,129.7071,129.47694,129.65155,127.47723,127.28498,128.89212,128.68385,127.88265,127.60673],[130.55862,131.02905,129.46361,129.56233,129.17644,128.35544,127.36579,126.99023,128.1641,128.65002],[128.68651,131.86043,130.59073,128.96101,129.91628,130.42805,128.9516,127.41191,128.76907,128.9656],[127.94517,129.92342,130.31711,129.21799,128.2734,127.32934,126.97719,127.454056,128.48653,129.25325],[129.73463,131.57393,130.32227,129.3245,128.01846,127.99733,128.88972,126.95246,127.71982,126.68717],[127.62111,128.41125,128.66315,129.29437,128.96425,127.355896,128.52422,128.45409,127.9129,128.27483],[128.9328,131.4582,130.66963,131.0656,127.86458,128.3115,127.93027,129.05913,128.71132,128.61723],[131.08109,131.3194,130.62816,130.07663,127.38624,126.660934,128.65234,127.46814,127.96135,126.84312],[130.06282,131.89987,129.82843,130.67426,128.76277,129.59354,128.89397,127.37855,128.3633,128.8769],[131.58694,130.07455,128.51755,128.88132,128.03683,127.99954,127.41392,128.2849,127.8067,128.33197],[129.27559,130.74522,129.21829,128.32835,126.95609,127.24045,128.20753,126.89045,128.31631,128.98364],[128.7202,131.31871,130.45091,129.52812,127.71568,129.55162,129.5272,127.751015,128.2281,129.27512],[130.69963,131.48055,128.8031,128.81729,128.98569,128.74188,127.21871,127.81623,128.79608,129.49234],[131.4759,132.3811,128.9594,129.81427,129.59752,128.54005,128.2609,127.82605,127.52885,127.02463],[136.96071,136.26843,134.33806,130.36679,127.8647,126.15627,125.26397,127.50848,127.49243,127.31677],[138.6253,138.53914,132.72159,130.753,126.59739,125.84584,126.521164,126.90665,126.73005,127.42356],[140.52264,139.45036,134.37659,131.32686,124.94601,125.45653,124.89712,126.12991,128.16473,129.07458],[148.95009,138.87361,134.4575,130.02313,120.327194,125.77954,127.348145,127.073166,130.22685,130.053],[153.97717,138.07811,131.6171,126.93567,120.50964,125.472,129.12758,127.01421,129.44717,130.13475],[155.64886,136.95386,129.2486,127.93064,120.6698,125.86978,130.99641,128.45712,127.98936,128.72221],[152.3936,137.04018,130.2789,128.15215,120.36439,126.085464,131.00851,128.74878,127.70343,127.96544],[151.36586,138.36502,128.6467,129.6366,121.04358,127.08364,130.16263,128.35088,126.82116,128.82199],[148.57884,137.67598,129.87448,129.32774,122.97951,126.99945,129.42673,128.7358,127.69691,129.41473],[144.38457,137.38373,132.09015,128.16415,125.35038,126.8733,129.64491,127.877785,126.27705,129.31264],[135.10799,134.86197,132.13268,131.6621,128.5219,125.90094,128.14346,127.42562,127.67985,127.807945],[132.03072,130.99577,131.74574,128.6959,129.6752,127.69522,129.13075,128.14706,128.35507,127.58883],[132.12108,130.70503,133.01549,129.62593,131.80594,127.78311,127.12216,126.63043,127.85326,126.451294],[130.61545,128.98962,129.47429,126.898964,129.6931,126.51575,129.66504,128.00993,129.37575,129.62807],[132.88889,128.60167,130.20937,127.926544,129.88683,127.5016,128.17296,127.75982,130.34598,129.13078],[132.36478,129.33015,130.3931,127.55827,130.1992,126.867424,128.5663,127.50715,129.65144,128.53305],[134.66039,129.95078,130.85008,127.76645,129.91913,126.509766,128.99881,127.584015,128.7468,126.994576],[132.15672,128.27806,130.55762,126.81696,129.64821,127.999954,129.85884,128.74342,128.96924,127.750565]])
                #print("Input: ",x[0,0]) 
            x = (x/scales_zeros[0][1] + scales_zeros[0][3]).round() #quantize input

            print("Using the quantized model")
                #print("Input Zero: ", scales_zeros[0][3])
            x = self.substract_zero(x,scales_zeros)
            if self.test == 0:
                print("Quantized input: ", x[0,0])
            x = self.first_conv(x)
            #if self.test == 0:
            #    print("First conv quantized output: ", x[0,0,0,:])
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu1(x)     
                   
            x = self.apply_clamp(x)
            if self.test == 0:
                print("Input for 2nd layer: ", x[0,:,0,0])
                print("Weights: ", self.first_conv.weight[:5,:,:,:])
                print("Biases: ",self.first_conv.bias)
            #if self.test == 0:
            #    print("First depth input: ", x[0,0,0,:])   
              
            x = self.substract_zero(x,scales_zeros)
            x = self.depth1(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu2(x)
            x = self.apply_clamp(x)
            if self.test == 0:
                 print("Input for 3rd layer: ", x[0,:,0,0])
                 print("Weights: ", self.depth1.weight[:5,:,:,:])
                 print("Biases: ",self.depth1.bias)
            #    print("First depthwise output: ", x[0,0,0,:])
            x = self.substract_zero(x,scales_zeros)
            x = self.pointw1(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu3(x)
            
            x = self.apply_clamp(x)
            if self.test == 0:
                print("Input for 4th layer: ", x[0,:,0,0])
                #print("Weights: ", self.pointw1.weight[:5,:,:,:])
                #print("Biases: ",self.pointw1.bias)
            #    print("First pointwise output: ", x[0,0,0,:])
            x = self.substract_zero(x,scales_zeros)
            x = self.depth2(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu4(x)
            x = self.apply_clamp(x)
            if self.test == 0:
                print("Input for 5th layer: ", x[0,:,0,0])
            x = self.substract_zero(x,scales_zeros)
            x = self.pointw2(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu5(x)
            
            x = self.apply_clamp(x)
            x = self.substract_zero(x,scales_zeros)
            x = self.depth3(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu6(x)
            x = self.apply_clamp(x)
            x = self.substract_zero(x,scales_zeros)
            x = self.pointw3(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu7(x)
            
            x = self.apply_clamp(x)
            x = self.substract_zero(x,scales_zeros)
            x = self.depth4(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu8(x)
            x = self.apply_clamp(x)
            x = self.substract_zero(x,scales_zeros)
            x = self.pointw4(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros)
            #x = self.relu9(x)   
            
            x = self.apply_clamp(x)
            if self.test == 0:
                print("Input for FC: ", x[0,:,0,0])
            x = self.avg(x)
            if self.test == 0:
                print("Output of average: ", x[0])
                print("Output of average shape: ", x.shape)
            x = torch.flatten(x, 1) 
            x = self.substract_zero(x,scales_zeros)
            x = self.fc1(x)
            x = self.scaleout_and_convert_into_next_input(x, scales_zeros,True)
            #x = torch.clamp(x,-128,127)
            if self.test == 0:
                print("Final output shape: ", x.shape)
                print("Final output: ",x[0])
                print("Weights: ", self.fc1.weight[:5,:])
                print("Biases: ",self.fc1.bias)
                print("Scales: ", scales_zeros[9][0]*scales_zeros[9][1]/scales_zeros[9][2])
                self.test = 1
        return x # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1) 

    def set_max_min_in_out (self, x, last = False):
        x_v = x
        #if (self.reg_count == 0):
        #    x_v = x[:,:,:,1:]
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
    
class DSCNN_TE_7x7_49x10(torch.nn.Module):
    reg_count = 0
    test = 0
    def __init__(self, use_bias=True):
        super(DSCNN_TE_7x7_49x10, self).__init__()

        use_bias = True
        
        #self.first_conv = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3, 3), stride = (2, 2), padding = (1,1),bias = use_bias)
        self.first_conv = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (1, 1), stride = (1, 1), bias = use_bias)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()
        self.depth1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (7, 7), stride = (2, 2), padding = 3, groups = 64, bias = use_bias)
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
                min_in_out.append(torch.tensor(1000))
                min_in_out.append(torch.tensor(1000))
            if name[:2] == "fc":
                max_weights.append(torch.zeros(module.out_features))
                min_weights.append(torch.zeros(module.out_features)) 
                max_in_out.append(torch.tensor(0))
                max_in_out.append(torch.tensor(0))
                min_in_out.append(torch.tensor(1000))
                min_in_out.append(torch.tensor(1000))  

        self.register_buffer('max_weights', torch.cat(max_weights))
        
        self.register_buffer('min_weights', torch.cat(min_weights))

        self.register_buffer('max_in_out', torch.stack(max_in_out))

        self.register_buffer('min_in_out', torch.stack(min_in_out))

        
    def forward(self, x, save = False, quantized = False, scales_zeros = None, x_scale_zero = None):
        if save:
            if (self.test ==0):
                self.test =1
                print("Max: ",torch.max(x))
                print("Min: ", torch.min(x))
            self.set_max_min_in_out(x)
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

        else:
            #x = self.first_conv(x)
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