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
parser.add_argument('--dataset', default='coco', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp','coco'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--variance', default=1 ,help='Variance of noise applied to joint parameters')
parser.add_argument('--view_angle', default=12, help='How would you have angels divided?')
parser.add_argument('--name', default='experimental',help='Experiment Name')
parser.add_argument('--output_path', default='logs/rotation_distribution', help='Path to network checkpoint')
parser.add_argument('--sample_no', default=3000, help='Number of samples')
parser.add_argument('--mode', default='global', choices=['global','pose'], help = 'Which rotation to check?')


def run_evaluation( dataset_name, dataset, result_file, model,
                   batch_size=16, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50, pose_variance=0.1, camera_variance=0.2, view_angle=12):
    """Run evaluation on the datasets and metrics we report in the paper. """

    # Create Directory

    directory = args.output_path + '/' + args.name

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory ' +  directory)

    focal_length = 5000
    idx_COCO = [i for i in range(25, 37)] + [j for j in range(44, 49)] 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(device)
    
    # renderer = Renderer(focal_length=focal_length, img_res=img_res, faces=smpl.faces)

    # Regressor for coco
    # J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)).float()
    
    # Disable shuffling if you want to save the results
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #######################################

    if args.mode == 'global':

        x_rotation, y_rotation, z_rotation, x_translation, y_translation, scale = [],[],[],[],[],[]

        for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            # Get ground truth annotations from the batch
            images = batch['img'].to(device)

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = model(images)
                pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices
                pred_keypoints_3d = pred_output.joints[:,25:,:]

                pred_cam_t = torch.stack([pred_camera[:,1],
                                    pred_camera[:,2],
                                    2 * focal_length/(img_res * pred_camera[:,0] +1e-9)],dim=-1)

                step_x = torch.atan2(pred_rotmat[:, 0, 2, 1],pred_rotmat[:, 0, 2, 2])
                step_y = torch.atan2(-pred_rotmat[:, 0, 2, 0],torch.sqrt(torch.square(pred_rotmat[:, 0, 2, 1])+torch.square(pred_rotmat[:,0,2,2])))
                step_z = torch.atan2(pred_rotmat[:, 0, 1, 0],pred_rotmat[:, 0, 0, 0])

                x_rotation += list(step_x.cpu().numpy())
                y_rotation += list(step_y.cpu().numpy())
                z_rotation += list(step_z.cpu().numpy())
                x_translation += list(pred_cam_t[:,0].cpu().numpy())
                y_translation += list(pred_cam_t[:,1].cpu().numpy())
                scale += list(pred_camera[:,0].cpu().numpy())
        
            if step * args.batch_size >= int(args.sample_no):
                break
        
        result = pd.DataFrame({'1. X-axis Rotation':x_rotation,
                            '2. Y-axis Rotation':y_rotation,
                            '3. Z-axis Rotation':z_rotation,
                            '4. X Translation': x_translation,
                            '5. Y Translation': y_translation,
                            '6. Scale': scale})
        
        result.to_csv((directory +'/xyz_rotation_list.csv'),index=False)
    

    # X,Y,Z Rotation Distributions for each joint

    elif args.mode == 'pose':

        rotations = [[],[],[]]
        # 3 by 23 rotation list to contain x, y, z rotations for each joint
        for axis in range(3):
            rotations[axis] = [[] for i in range(23)]

        for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            # Get ground truth annotations from the batch
            images = batch['img'].to(device)

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = model(images)

                for i in range(1 , pred_rotmat.shape[1]):
                    step_x = torch.atan2(pred_rotmat[:,i,2,1],pred_rotmat[:,i,2,2])
                    step_y = torch.atan2(-pred_rotmat[:,i,2,0],torch.sqrt(torch.square(pred_rotmat[:,i,2,1])+torch.square(pred_rotmat[:,i,2,2])))
                    step_z = torch.atan2(pred_rotmat[:,i,1,0],pred_rotmat[:,i,0,0])

                    rotations[0][i-1] += list(step_x.cpu().numpy())
                    rotations[1][i-1] += list(step_y.cpu().numpy())
                    rotations[2][i-1] += list(step_z.cpu().numpy())

        
            if step * args.batch_size >= int(args.sample_no):
                break
    
        rotation_dict = {}
        axis_list = ['x','y','z']

        for joint in range(1,24):
            axis_idx = 0
            for axis in axis_list:
                key = str(joint) + axis
                rotation_dict[key] =  rotations[axis_idx][joint-1]
                axis_idx += 1

        result = pd.DataFrame(rotation_dict)
        
        result.to_csv((directory +'/joint_rotation_list.csv'),index=False)


    for key in result:
        plt.xlim(-3.141592, 3.141592)
        subject = sns.distplot(result[key], color='blue', bins=300)
        fig = subject.get_figure()
        fig.savefig(directory + '/' + key + '.png')
        plt.clf()

        
if __name__ == '__main__':
    args = parser.parse_args()
    # summary_writer = SummaryWriter('./experiments')

    model = hmr(config.SMPL_MEAN_PARAMS).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation( args.dataset, dataset, args.result_file, model,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
