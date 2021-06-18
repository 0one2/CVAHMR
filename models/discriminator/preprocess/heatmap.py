import math
import numbers
import torch

from torch import nn
from torch.nn import functional as F

import time
import numpy as np
from scipy.ndimage import gaussian_filter

import pdb

class GaussianSmoothing(nn.Module):
    def __init__(self, num_kps):
        super(GaussianSmoothing, self).__init__()
        num_kps = int(num_kps)
        self.heatmaps = nn.Sequential(
            nn.Conv2d(num_kps, num_kps, kernel_size = 5, padding=2, bias=None)
        )
        self.weights_init()

    def weights_init(self):
        n= np.zeros((5,5))
        n[0, 0], n[0, 1], n[0, 2], n[0, 3], n[0, 4] = 1, 4, 7, 4, 1
        n[1, 0], n[1, 1], n[1, 2], n[1, 3], n[1, 4] = 4, 16, 26, 16, 4
        n[2, 0], n[2, 1], n[2, 2], n[2, 3], n[2, 4] = 7, 26, 41, 26, 7
        n[3, 0], n[3, 1], n[3, 2], n[3, 3], n[3, 4] = 4, 16, 26, 16, 4
        n[4, 0], n[4, 1], n[4, 2], n[4, 3], n[4, 4] = 1, 4, 7, 4, 1

        n /= 41

        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(n))

    def make_sparseHeat(self, j2d, vis_idx, heatmapSize = 64):
        """
        Arguments:
            j2d (Torch.Tensor | batch x num_kps x 2): Input 2D joint normalized coordinate (-1 ~ 1)
            heatmapSize (Torch.Tensor | 1,): Heatmap size, default = 224
        Returns:
            sparseHeatmap (Torch.Tensor | batch x 14 x ):
        """
        batch = j2d.size(0)
        num_kps = j2d.size(1)

        # 1. rescale the normalized keypoints by image resolution (-1 ~ 1) --> (0 ~ 223)
        j2d_resized = torch.round((j2d + 1) / 2 * (heatmapSize - 1)).long()
       
        # 2. make the heatmap using the indices of rescaled keypoints (batch x 14 x 2) --> heatmap: (batch x 14 x 224 x 224)
        heatmap = torch.zeros((batch, num_kps, heatmapSize, heatmapSize)).cuda()

        # There are some GT kps whose value is equal or exceeds the 224. We don't make any heatmap on that kps. 
        batch_over_idx, kps_over_idx, _ = torch.where(j2d_resized >= heatmapSize)
        batch_under_idx, kps_under_idx, _ = torch.where(j2d_resized < 0)

        # make values to be zero. 
        j2d_resized[torch.cat((batch_over_idx, batch_under_idx)), torch.cat((kps_over_idx, kps_under_idx)), :] = 0

        batch_idx, kps_idx= torch.where(j2d_resized[:, :, 0])
        heatmap[batch_idx, kps_idx, j2d_resized[batch_idx, kps_idx, 1], j2d_resized[batch_idx, kps_idx, 0]] = 1

        # 3. make the entire values of invisible keypoint channel to be zeros 
        heatmap[torch.cat((batch_over_idx, batch_under_idx)), torch.cat((kps_over_idx, kps_under_idx)), :, :] = 0

        if vis_idx:
            vis_batch, vis_kps = vis_idx
            heatmap[vis_batch, vis_kps, :, :] = 0

        return heatmap

    def forward(self, x, vis_idx):
        sparseHeat = self.make_sparseHeat(x, vis_idx)
        heatmap = self.heatmaps(sparseHeat)    
        return heatmap


if __name__ == '__main__':
    smoothing = GaussianSmoothing(4).cuda()      
    for i in range(5):                    
        input = torch.rand(3, 4, 2).cuda()
        input[0, 0, 0] += 2
        input[2, 0, 1] += 2
        input[1, 0, 0] += 2
        input[1, 0, 1] += 2
        start = time.time()                                                          
        output = smoothing(input, None)          
        print(f'time {time.time() - start}')