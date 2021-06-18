import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
import pdb

from datasets import MixedDataset
from models import hmr, SMPL
from models.loss import GradientPaneltyLoss
from utils.geometry import perspective_projection, estimate_translation
from utils.renderer import Renderer
from utils import BaseTrainer

import config
import constants

class Trainer_Cycle(BaseTrainer):
    
    def init_fn(self):
        self.discriminator = self.discriminator.cuda()
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.GP_func = GradientPaneltyLoss().cuda()

        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def set_requires_grad(self, nets, requires_grad=False) :
        if not isinstance(nets, list) :
            nets = [nets]

        for net in nets:
            if net is not None :
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_step(self, input_batch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)
        
        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        pred_rotmat = pred_rotmat.reshape(batch_size, -1)
        
        # BigTheta = a set of SMPML parameters(cam, shape, pose) to make human mesh  
        pred_BigTheta = torch.cat((pred_betas, pred_camera, pred_rotmat), dim = 1)

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)

        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        idx_COCO = [i for i in range(25, 37)] + [j for j in range(44, 49)]

        # set kps value to zero if visibility of certain kps is 0 
        gt_keypoints_2d[gt_keypoints_2d[:, :, 2] == 0] = 0
        pred_keypoints_2d[gt_keypoints_2d[:, :, 2] == 0] = 0

        # get kps of COCO indexes (batch, 49, 2) -> (batch, 17, 2)
        gt_kps = gt_keypoints_2d[:, idx_COCO, :2].clone() 
        pred_kps = pred_keypoints_2d[:, idx_COCO].clone() 


        if self.options.train == 'Cycle':
            loss_discriminator = self.Base_GAN_SwapCam_Cycle_loss(pred_cam_t, pred_joints, pred_BigTheta, gt_kps, pred_kps)
        else:
            return NameError(f'{self.options.train} not implemented yet!')
               
        # Discriminator Update 
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad()
        loss_discriminator.backward(retain_graph=True)
        self.optimizer_d.step()


        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        ## For Generator Update
        if self.options.train == 'Cycle':
            loss_gan_generator = self.Base_GAN_SwapCam_Cycle_loss(pred_cam_t, pred_joints, pred_BigTheta, gt_kps, pred_kps, 'generator')
            loss_generator = self.options.loss_generator_weight * loss_gan_generator +\
                                 self.options.loss_keypoint_weight * loss_keypoints
        else:
            return NameError(f'{self.options.train} not implemented yet!')
        
        loss_generator *= 60

        # Do backprop
        self.set_requires_grad(self.discriminator, False)
        self.optimizer.zero_grad()
        loss_generator.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'pred_cam_t': pred_cam_t.detach()}

        if self.options.train == 'Cycle':
            losses = {'loss': loss_generator.detach().item(),
                        'loss_keypoints': loss_keypoints.detach().item(),
                        'loss_discriminator': loss_discriminator.detach().item(),
                        'loss_generator': loss_gan_generator.detach().item()}
        else:
            return NameError(f'{self.options.train} not implemented yet!')

        return output, losses


    def Base_GAN_SwapCam_Cycle_loss(self, pred_cam_t, pred_joints, pred_BigTheta, gt_kps, pred_kps, mode = 'discriminator'):
        batch_size = pred_cam_t.size(0)

        pred_shape, pred_cam, pred_rotmat = pred_BigTheta[:, :10], pred_BigTheta[:, 10:13], pred_BigTheta[:, 13:]
        swap_cam = torch.zeros_like(pred_cam)
        swap_cam_t = torch.zeros_like(pred_cam_t)

        for i in range(batch_size):
            swap_cam[i, :] = pred_cam[batch_size - i-1, :].clone()
            swap_cam_t[i, :] = pred_cam_t[batch_size - i-1, :].clone()

        camera_center = torch.zeros(batch_size, 2, device=self.device)
        swap_keypoints_2d = perspective_projection(pred_joints,
                                                    rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                    translation=swap_cam_t,
                                                    focal_length=self.focal_length,
                                                    camera_center=camera_center)    
        
        swap_keypoints_2d = swap_keypoints_2d / (self.options.img_res / 2.) 
        
        idx_COCO = [i for i in range(25, 37)] + [j for j in range(44, 49)]

        swap_kps = swap_keypoints_2d[:, idx_COCO].clone()
        
        # Discriminator predicts i) gan probability, ii) SMPL parameters(pose, shape), iii) camera used for projection
        probability_pred, d_pred_rotmat, d_pred_shape, d_pred_cam = self.discriminator(pred_kps)
        probability_swap, d_swap_rotmat, d_swap_shape, d_swap_cam = self.discriminator(swap_kps)

        # loss_discriminator when Base_GAN
        if mode == 'discriminator':
            # make swap_BigTheta by swapping camera within batch 
            swap_BigTheta = torch.cat((pred_shape, swap_cam, pred_rotmat.reshape(batch_size, -1)), dim = 1)
            
            d_pred_BigTheta = torch.cat((d_pred_shape, d_pred_cam, d_pred_rotmat.reshape(batch_size, -1)), dim = 1)
            d_swap_BigTheta = torch.cat((d_swap_shape, d_swap_cam, d_swap_rotmat.reshape(batch_size, -1)), dim = 1)

            probability_real, _, _, _ = self.discriminator(gt_kps)

            loss_D_real = torch.mean(probability_real)
            loss_D_pred = - torch.mean(probability_pred)
            loss_D_swap = - torch.mean(probability_swap)

            alpha = torch.rand(gt_kps.size(0), 1, 1).cuda()
            output_ = (alpha * gt_kps + (1 - alpha) * pred_kps.detach()).requires_grad_(True)
            src_out_, _, _, _ = self.discriminator(output_)
            loss_D_GP = self.GP_func(src_out_, output_)

            loss_D = 0.5 * (loss_D_real +\
                        (1 - self.options.loss_swap_weight) * loss_D_pred +\
                        self.options.loss_swap_weight * loss_D_swap) \
                    + loss_D_GP

            loss_smpl = self.criterion_regr(pred_BigTheta, d_pred_BigTheta) +\
                            self.criterion_regr(swap_BigTheta, d_swap_BigTheta)
            
            loss = self.options.loss_D_weight * loss_D +\
                         self.options.loss_smpl_regression_weight * loss_smpl

        # loss_generator when Base_GAN
        else:
            loss_G_pred = torch.mean(probability_pred)
            loss_G_swap = torch.mean(probability_swap)
            
            loss = (1 - self.options.loss_swap_weight) * loss_G_pred +\
                        self.options.loss_swap_weight * loss_G_swap 

        return loss


    def train_summaries(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        pred_cam_t = output['pred_cam_t']
        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
