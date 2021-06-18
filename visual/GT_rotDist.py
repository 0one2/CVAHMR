import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math 
import pdb 

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
# from utils.renderer import Renderer
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to network checkpoint')
parser.add_argument('--dataset', default='mpi_inf_3dhp_train', choices=['h36m_valid_protocol1', 'h36m-p2', 'hr-lspet_train', '3dpw', 'mpi_inf_3dhp_train','coco'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--variance', default=1 ,help='Variance of noise applied to joint parameters')
parser.add_argument('--view_angle', default=12, help='How would you have angels divided?')
parser.add_argument('--name', default='experimental',help='Experiment Name')
parser.add_argument('--output_path', default='logs/rotation_distribution', help='Path to network checkpoint')
parser.add_argument('--sample_no', default=3000, help='Number of samples')
parser.add_argument('--mode', default='global', choices=['global','pose'], help = 'Which rotation to check?')


def plot_rot_dist(data):


    directory = args.output_path + '/' + args.name
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory ' +  directory)


    x_rotation, y_rotation, z_rotation, x_translation, y_translation, scale = [],[],[],[],[],[]

    data_rot = torch.tensor(np.array(data['pose']).reshape(-1, 24, 3)[:, 0, :]).cuda()

    pred_rotmat = batch_rodrigues(data_rot)

    step_x = torch.atan2(pred_rotmat[:, 2, 1],pred_rotmat[:, 2, 2])
    step_y = torch.atan2(-pred_rotmat[:, 2, 0],torch.sqrt(torch.square(pred_rotmat[:, 2, 1])+torch.square(pred_rotmat[:, 2, 2])))
    step_z = torch.atan2(pred_rotmat[:, 1, 0],pred_rotmat[:, 0, 0])

    print(f'torch.max x,, y, z {torch.max(step_x)} | {torch.max(step_y)} | {torch.max(step_z)}')
    
    result = pd.DataFrame({'1. X-axis Rotation':step_x.tolist(),
                        '2. Y-axis Rotation':step_y.tolist(),
                        '3. Z-axis Rotation':step_z.tolist()})

    result.to_csv((directory +'/xyz_rotation_list.csv'),index=False)


    for key in result:
        plt.xlim(-math.pi, math.pi)
        subject = sns.displot(result[key], color='blue', bins=300)
        subject.savefig(directory + '/' + key + '.png')
        plt.clf()


if __name__ == '__main__':
    args = parser.parse_args()
    # summary_writer = SummaryWriter('./experiments')

    root_path = 'data/dataset_extras/'
    dataset = f'{args.dataset}.npz'
    dataset_path = os.path.join(root_path, dataset)
    
    data = np.load(dataset_path)
    print(data.files)
    raise ValueError
    plot_rot_dist(data)
    