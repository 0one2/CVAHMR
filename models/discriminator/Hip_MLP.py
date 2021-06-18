import torch 
import torchvision.models as models 
import torch.nn as nn
import numpy as np

import time 
import pdb 

from .preprocess.hip_normalized import normalize_hip
from utils.data_loader import load_mean_theta
from utils.geometry import rot6d_to_rotmat


class Hip_Discriminator(nn.Module):
    def __init__(self, num_kps):
        super(Hip_Discriminator, self).__init__()
        num_kps = int(num_kps)
        self.fc1 = nn.Linear(num_kps * 2, 1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.leaky = nn.LeakyReLU()
        self.fc4 = nn.Linear(1024,1)
                
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0.0, std = 0.02)

    def forward(self, j2d, vis_idxz):
        '''
        Arguments: 
            j2d (batch x num_kps x 2): Input 2D normalized joints
        Returns: 
            output (batch x 1) 
        '''
        x = normalize_hip(j2d)
        x1 = self.leaky(self.fc1(x))
        x2 = self.leaky(self.fc2(x1))
        x3 = self.leaky(self.fc3(x2)+ x1)
        output = self.fc4(x3)
        return output
        
        
class Hip_Discriminator_Cycle(nn.Module):
    def __init__(self, num_kps):
        super(Hip_Discriminator_Cycle, self).__init__()
        num_kps = int(num_kps)
        self.fc1 = nn.Linear(num_kps, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)    
        self.fc4 = nn.Linear(1024, 1024)
        self.leaky = nn.LeakyReLU()

        self.disc1 = nn.Linear(1024, 512)
        self.disc2 = nn.Linear(512, 1)

        self.params_blocks = self.fcblocks(1024, relu=True)

        npose = 24 *6
        self.fc_1 = nn.Linear(512 * 2 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc_2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # according to paper, BN is not effective
        init_pose, init_shape, init_cam = load_mean_theta()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def fcblocks(self, dims, relu = True):
        layers = [nn.Linear(dims + 85, dims), \
                        nn.Linear(dims, dims), nn.Linear(dims, dims *2)]
        if relu:
            layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, j2d):
        '''
        Arguments: 
            j2d (batch x num_kps x 2) : Input 2D normalized joint position (-1 ~ 1)
        Returns: 
            output (batch x 1) : output value for Wgan loss 
            pred_rotmat (batch x 24 x 3 x 3) : rotation matrix of SMPL parameters  
            pred_shape (batch x 10) : shape parameters of SMPL parameters
            pred_cam (batch x 3) : camera parameters of SMPL parameters
        '''
        batch_size = j2d.size(0)

        x = normalize_hip(j2d)
        x1 = self.leaky(self.fc1(x))
        x2 = self.leaky(self.fc2(x1))
        x3 = self.leaky(self.fc3(x2) + x1)
        x4 = self.fc4(x3)

        ### for R/F prob
        rf0 = self.disc1(x4)
        probability = self.disc2(rf0)

        n_iter = 3

        pred_pose = self.init_pose.repeat(batch_size, 1)
        pred_shape = self.init_shape.repeat(batch_size, 1)
        pred_cam = self.init_cam.repeat(batch_size, 1)
        for i in range(n_iter):
            pdb.set_trace()
            xc = torch.cat([x4, pred_pose, pred_shape, pred_cam], dim = 1)
            xc = self.fc_1(xc)
            xc = self.drop1(xc)
            xc = self.fc_2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return probability, pred_rotmat, pred_shape, pred_cam


if __name__ == '__main__':
    num_kps = 17
    hip = Hip_Discriminator(num_kps).cuda()     
    for i in range(5):                    
        input = torch.rand(5, num_kps, 2).cuda()
        start = time.time()                                                          
        output = hip(input)           
        print(output)