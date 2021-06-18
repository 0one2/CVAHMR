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

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
from utils.renderer import Renderer

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco-test', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

def run_evaluation(spin, eft, dataset_name, dataset, result_file, summary_writer,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    focal_length = 5000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    spin.to(device)
    eft.to(device)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(device)
    
    renderer = Renderer(focal_length=focal_length, img_res=img_res, faces=smpl.faces)

    # Regressor for coco
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)).float()
    
    # Disable shuffling if you want to save the results
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    gt_3d_joints = []

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_rotmat = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_cam = batch['cam'].to(device)
        gt_cam_t = torch.stack([gt_cam[:,1],
                                  gt_cam[:,2],
                                  2*focal_length/(img_res * gt_cam[:,0] +1e-9)],dim=-1)

        gt_output = smpl(betas=gt_betas, body_pose=gt_rotmat[:,1:], global_orient=gt_rotmat[:,0].unsqueeze(1), pose2rot=False)
        gt_vertices = gt_output.vertices
        gt_joints = gt_output.joints

        gt_3d_joints.append(gt_joints[:,25:,:].clone().cpu().numpy())


        images = batch['img'].to(device)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = spin(images)

            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

            pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*focal_length/(img_res * pred_camera[:,0] +1e-9)],dim=-1)

            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

            images_spin = renderer.visualize_tb(pred_vertices, pred_cam_t, images)
            summary_writer.add_image('SPIN', images_spin, step)
        
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = eft(images)

            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

            pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*focal_length/(img_res * pred_camera[:,0] +1e-9)],dim=-1)

            images_eft = renderer.visualize_tb(pred_vertices, pred_cam_t, images)
            summary_writer.add_image('EFT', images_eft, step)
    
        
if __name__ == '__main__':
    args = parser.parse_args()
    spin = hmr(config.SMPL_MEAN_PARAMS)
    eft = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint_spin = torch.load('./data/model_checkpoint.pt')
    checkpoint_eft = torch.load('./data/EFT.pt')
    spin.load_state_dict(checkpoint_spin['model'], strict=False)
    eft.load_state_dict(checkpoint_eft['model'], strict=False)
    spin.eval()
    eft.eval()
    summary_writer = SummaryWriter('./preresearch')

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(spin,eft, args.dataset, dataset, args.result_file,summary_writer,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
