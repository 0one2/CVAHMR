import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# from utils.renderer import Renderer
# from utils.imutils import plot_kps
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='eval', help='Path to network checkpoint')
parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to network checkpoint')
parser.add_argument('--output_path', default='./eval', help='Path to network checkpoint')
parser.add_argument('--step_size', default=10000, help='step size')
parser.add_argument('--dataset', default='coco-test', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=4, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

def run_evaluation(model, dataset_name, dataset, result_file, checkpoint_dir, step_size, summary_writer, last,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    focal_length = 5000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(device)
    
    # renderer = Renderer(focal_length=5000, img_res=img_res, faces=smpl.faces)
    
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'coco-test':
        eval_pose = True

    joint_mapper_gt = [i for i in range(0, 12)] + [j for j in range(19, 24)]
    # Iterate over the entire dataset

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_rotmat = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_cam = batch['cam'].to(device)
        gt_vertices = smpl(betas=gt_betas, body_pose=gt_rotmat[:,1:], global_orient=gt_rotmat[:,0].unsqueeze(1), pose2rot=False).vertices
        images = batch['img'].to(device)
        gt_keypoints_2d = batch['keypoints'].to(device)
        curr_batch_size = images.shape[0]
        

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_keypoints_3d = pred_output.joints[:,25:,:]
            pred_joints = pred_output.joints


            if last == 1:
                pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*focal_length/(img_res * pred_camera[:,0] +1e-9)],dim=-1)
                camera_center = torch.zeros(curr_batch_size, 2, device=device)

                pred_keypoints_2d = perspective_projection(pred_joints,
                                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                                            translation=pred_cam_t,
                                                            focal_length=5000,
                                                            camera_center=camera_center)

                pred_keypoints_2d = pred_keypoints_2d / (img_res / 2.)
                
                images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
                images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

            
        # 3D pose evaluation
        if eval_pose:

            # Get 14 ground truth joints
            if 'coco-test' in dataset_name or 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
                
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


            # Get 14 predicted joints from the mesh

            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_gt, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print()

    # Print final results during evaluation
    print('*** '+str(step_size)+' Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print()
    
    return (1000 * mpjpe.mean()), (1000 * recon_err.mean())

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = BaseDataset(None, args.dataset, is_train=False)
    step_size = 0
    last = 0
    mpjpe = []
    recon_err = []
    step = []
    if not os.path.exists(args.output_path+'/'+args.name):
        os.makedirs(args.output_path+'/'+args.name)
    summary_writer = SummaryWriter(args.output_path+'/'+args.name)
    checkpoints = os.listdir(args.checkpoint)
    checkpoint_list = sorted(checkpoints, key=lambda x: int(x.split('_')[-1][2:].split('.')[0]))
    last_checkpoint = os.listdir(args.checkpoint)[-1]

    for checkpoint_dir in checkpoint_list:
        if last_checkpoint == checkpoint_dir:
            last=1

        step_size = step_size + int(args.step_size)
        checkpoint_dir = args.checkpoint +'/'+checkpoint_dir
        print('-------------------------------------------------------------------------')
        print(f'loaded from checkpoint======================= {checkpoint_dir}')
        model = hmr(config.SMPL_MEAN_PARAMS)
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()        # make the status of evaluation --> changes in dropout, batchnorm layer

        # Run evaluation
        step_mpjpe, step_recon_err = run_evaluation(model, args.dataset, dataset, args.result_file, checkpoint_dir, step_size, summary_writer, last,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    log_freq=args.log_freq)