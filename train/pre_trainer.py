import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
import pdb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from datasets import MixedDataset
from models import hmr, SMPL
from models.loss import GradientPaneltyLoss, Vanilla, Wgan, WGANGP, calculate_accuracy, calculate_rot_err
from utils.geometry import perspective_projection, estimate_translation, rot6d_to_rotmat
from utils.renderer import Renderer
from utils import BaseTrainer
from utils.imutils import plot_kps

import config
import constants


class Pre_Trainer(BaseTrainer):
    
    def init_fn(self):
        self.discriminator = self.discriminator.to(self.device)
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        # self.optimizer = torch.optim.Adam(params=self.model.parameters(),
          #                                 lr=self.options.lr,
            #                               weight_decay=0)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(),
                                          lr=self.options.lr*self.options.lr_rate,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        
        self.mean_params = np.load(config.SMPL_MEAN_PARAMS)

        self.init_pose = torch.from_numpy(self.mean_params['pose'][:]).repeat(self.options.batch_size).view(self.options.batch_size,-1).to(self.device)
        self.init_betas = torch.from_numpy(self.mean_params['shape'][:].astype('float32')).repeat(self.options.batch_size).view(self.options.batch_size,-1).to(self.device)
        self.init_rotmat = rot6d_to_rotmat(self.init_pose).view(self.options.batch_size, 24, 3, 3).to(self.device)
        
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        # Loss for GAN BCE LOSS 
        self.criterion_BCELogitsLoss = nn.BCEWithLogitsLoss() # combines a sigmoid layer + BCELoss in one single class. more stable than a plain sigmoid followed BCELoss 
        self.GP_func = WGANGP().cuda()
        self.models_dict = {'discriminator' : self.discriminator}
        self.optimizers_dict = {'optimizer_d' : self.optimizer_d}

        print(f'models saved {self.models_dict.keys()}')
        print(f'optimizer saved {self.optimizers_dict.keys()}')


        self.focal_length = constants.FOCAL_LENGTH
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)
        
        if self.options.gan_loss == 'vanilla':
            self.gan_loss = Vanilla().cuda()
        elif self.options.gan_loss == 'wgan':
            self.gan_loss = Wgan().cuda()
        else:
            raise NameError(f'{self.options.gan_loss} not implemented Yet!')

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

    def train_step(self, input_batch):
        self.model.train()      # effects on certain modeuls, like Dropout, BatchNorm etc 

        # Get data from the batch
        images = input_batch['img']# input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        img_name = input_batch['imgname']
        gt_rotmat = input_batch['pose']

        batch_size = images.shape[0]
        mean_betas = self.init_betas
        mean_rotmat = self.init_rotmat

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)

        pred_output = self.smpl(betas=mean_betas, body_pose=mean_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

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

        # idx_COCO = [i for i in range(25, 37)] + [j for j in range(44, 49)]      # follow the SMPL index order 


        # get kps of COCO indexes (batch, 49, 3) -> (batch, 17, 3)
        # gt_kps = gt_keypoints_2d[:, idx_COCO].clone() 
        # pred_kps = pred_keypoints_2d[:, idx_COCO].clone() 

        gt_kps = gt_keypoints_2d[:, 25:].clone()
        pred_kps = pred_keypoints_2d[:, 25:].clone()


        # set kps value to zero if visibility of certain kps is 0 
        vis_idx = torch.where(gt_kps[:, :, 2] == 0)
        vis_batch, vis_kps = vis_idx


        gt_kps[vis_batch, vis_kps] = 0
        gt_kps = gt_kps[:, :, :2]
        pred_kps[vis_batch, vis_kps] = 0 
        

        # Discriminator Update

        if self.options.train == 'Base':
            pass
        else:
            if self.options.train == 'Base_GAN':
                loss_D, loss_D_real, loss_D_fake, real_acc, fake_acc = self.BaseGAN_loss(gt_kps, pred_kps, vis_idx)

            elif self.options.train == 'RandomCam':
                # Calculate RandomCam
                std = torch.ones(1).cuda() * self.options.rand_std
                rot_angle = torch.normal(0,std).cuda()

                y_rotation = torch.FloatTensor([[torch.cos(rot_angle),0,torch.sin(rot_angle)],
                                                [0,1,0],
                                                [-torch.sin(rot_angle),0,torch.cos(rot_angle)]]).cuda()

                # Random y rotation
                if self.options.random_type == 'Shift':
                    rand_rotmat = torch.matmul(y_rotation, pred_rotmat)

                elif self.options.random_type == 'Direct':
                    rand_rotmat = y_rotation

                loss_D, loss_D_real, loss_D_fake, real_acc, fake_acc = self.DoubleCam_loss(pred_rotmat, rand_rotmat, pred_betas, pred_cam_t, gt_kps, pred_kps, vis_idx)

            elif self.options.train == 'SwapCam':
                swap_rotmat = torch.flip(pred_rotmat, dims = [0])
                loss_D, loss_D_real, loss_D_fake, real_acc, fake_acc = self.DoubleCam_loss(pred_rotmat, swap_rotmat, pred_betas, pred_cam_t, gt_kps, pred_kps, vis_idx)

            else:
                return NameError(f'{self.options.train} not implemented yet!')

            # Discriminator Update 
            if (self.step_count+1) % self.options.update_freq == 0 :
                self.optimizer_d.zero_grad()            # set the grads of the previous epoch to be zero, to prevent the accumulation of grads of previous step | equal to model.zero_grad() if optimizer contains params of model 
                loss_D.backward(retain_graph=True)      # calculate the gradient at each layer | retain computational graph for the update of generator(if not, calculating grads and then disappear computation graph)
                self.optimizer_d.step()            # update the layer weight of related params, here 'discriminator parameters'

        ## For Generator Update

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.camera_fitting_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)
        # if dataset_name == 'mpi-inf-3dhp':
        rot_err = calculate_rot_err(pred_rotmat, gt_rotmat)

        if self.options.train == 'Base':
            loss_G = self.options.loss_kps_weight * loss_keypoints
            g_real_acc, g_fake_acc = 0.5, 0.5

        else: 
            loss_G = self.options.loss_kps_weight * loss_keypoints

            if self.options.train == 'Base_GAN':
                loss_gan_generator, g_real_acc, g_fake_acc = self.BaseGAN_loss(gt_kps,pred_kps,vis_idx,'genrator')
                loss_G += self.options.loss_G_weight * loss_gan_generator

            elif self.options.train == 'RandomCam':
                loss_gan_generator, g_real_acc, g_fake_acc = self.DoubleCam_loss(pred_rotmat, rand_rotmat, pred_betas, pred_cam_t, gt_kps, pred_kps, vis_idx, mode = 'generator')
                loss_G += self.options.loss_G_weight * loss_gan_generator

            elif self.options.train == 'SwapCam':
                swap_rotmat = torch.flip(pred_rotmat, dims = [0])

                loss_gan_generator, g_real_acc, g_fake_acc = self.DoubleCam_loss(pred_rotmat, swap_rotmat, pred_betas, pred_cam_t, gt_kps, pred_kps, vis_idx, mode = 'generator')
                loss_G += self.options.loss_G_weight * loss_gan_generator

            else:
                return NameError(f'{self.options.train} not implemented yet!')

        # self.optimizer.zero_grad()  # same as self.discriminator.zero_grad()    
        # loss_G.backward()           # calculate gradient 
        # self.optimizer.step()

        # Pack output arguments for tensorboard logging

        if self.options.train == 'Base':
            output = {'pred_vertices': pred_vertices.detach(),
                        'pred_cam_t': pred_cam_t.detach(),
                        'pred_kps': pred_kps.detach(),
                        'gt_kps': gt_kps.detach()}

            losses = {'loss': loss_G.detach().item(),
                        'rot_err': sum(rot_err) / self.options.batch_size,
                        'loss_keypoints': loss_keypoints.detach().item()}
        else:
            output = {'pred_vertices': pred_vertices.detach(),
                        'pred_cam_t': pred_cam_t.detach(),
                        'pred_kps': pred_kps.detach(),
                        'gt_kps': gt_kps.detach()}

            losses = {'loss': loss_G.detach().item(),
                        'loss_keypoints': loss_keypoints.detach().item(),
                        'loss_discriminator': loss_D.detach().item(),
                        'loss_generator': loss_gan_generator.detach().item(),
                        'fake_acc': fake_acc/self.options.batch_size,
                        'real_acc': real_acc/self.options.batch_size,
                        'fake_loss': loss_D_fake.detach().item(),
                        'real_loss': loss_D_real.detach().item(),
                        'rot_err': sum(rot_err) / self.options.batch_size,
                        'g_fake_acc' : 1-(g_fake_acc/self.options.batch_size)}


        return output, losses
    

    def camera_fitting_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
    # """ Compute 2D reprojection loss for selected keypoints.
    # The loss is weighted by the confidence.
    # The available keypoints are different for each dataset."""
       
        # Extremities 
        op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder','OP RAnkle','OP LAnkle','OP RWrist','OP LWrist']
        op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
        gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder','Right Ankle','Left Ankle','Right Wrist','Left Wrist']
        gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]

        # Elbows and Knees
        # op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder','OP RKnee','OP LKnee','OP RElbow','OP LElbow']
        # op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
        # gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder','Right Knee','Left Knee','Right Elbow','Left Elbow']
        # gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf[:,gt_joints_ind] * self.criterion_keypoints(pred_keypoints_2d[:,gt_joints_ind], gt_keypoints_2d[:, gt_joints_ind, :-1])).mean()
        return loss

    def BaseGAN_loss(self, gt_kps, pred_kps, vis_idx, mode = 'discriminator'):
        # loss_discriminator when Base_GAN
        probability_real = self.discriminator(gt_kps, vis_idx)

        if mode == 'discriminator':
            probability_pred = self.discriminator(pred_kps.detach(), vis_idx).cuda()
            loss, loss_D_real, loss_D_fake = self.gan_loss(probability_real, probability_pred, None, numCam = 'single', mode='discriminator')

            if self.options.gan_loss == 'wgan':
                loss_D_GP = self.GP_func(self.discriminator, gt_kps, pred_kps, None, vis_idx, numCam = 'single')
                loss += loss_D_GP    

            correct_real, correct_fake = calculate_accuracy(probability_pred, probability_real, None, numCam = 'single')
            return loss, loss_D_real, loss_D_fake, correct_real, correct_fake

        # loss_generator when Base_GAN
        else :
            probability_pred = self.discriminator(pred_kps, vis_idx).cuda()
            loss = self.gan_loss(None, probability_pred, None,  numCam= 'single', mode='generator')            
            correct_real, correct_fake = calculate_accuracy(probability_pred, probability_real, None, numCam = 'single')
            return loss, correct_real, correct_fake


    def DoubleCam_loss(self, pred_rotmat, new_rotmat, pred_betas, pred_cam_t, gt_kps, pred_kps, vis_idx, mode = 'discriminator'):
        new_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=new_rotmat[:,0].unsqueeze(1), pose2rot=False)
        new_vertices = new_output.vertices
        new_joints = new_output.joints

        camera_center = torch.zeros(self.options.batch_size, 2, device=self.device)
        new_keypoints_2d = perspective_projection(new_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(self.options.batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center) 

        new_keypoints_2d = new_keypoints_2d / (self.options.img_res / 2.) 
        new_kps = new_keypoints_2d[:, 25:]


        # loss_discriminator when Base_GAN
        probability_real = self.discriminator(gt_kps, vis_idx)

        if mode == 'discriminator':
            probability_pred = self.discriminator(pred_kps.detach(), vis_idx)
            probability_new = self.discriminator(new_kps.detach(), vis_idx)

            loss, loss_D_real, loss_D_fake = self.gan_loss(probability_real, probability_pred, probability_new, numCam = 'double', mode='discriminator')
            
            if self.options.gan_loss == 'wgan':
                loss_D_GP = self.GP_func(self.discriminator, gt_kps, pred_kps, new_kps, vis_idx, numCam = 'double')
                loss += loss_D_GP
            correct_real, correct_fake = calculate_accuracy(probability_pred, probability_real, probability_new, numCam = 'double')
            return loss, loss_D_real, loss_D_fake, correct_real, correct_fake
            

        # loss_generator when Base_GAN
        else:
            probability_pred = self.discriminator(pred_kps, vis_idx)
            probability_new = self.discriminator(new_kps, vis_idx)

            loss = self.gan_loss(None, probability_pred, probability_new, numCam = 'double', mode='generator')
            correct_real, correct_fake = calculate_accuracy(probability_pred, probability_real, probability_new, numCam = 'double')
            return loss, correct_real, correct_fake


    def train_summaries(self, mode, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        pred_cam_t = output['pred_cam_t']

        pred_kps = output['pred_kps']
        gt_kps = output['gt_kps']

        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        figure = plot_kps(images,gt_kps,pred_kps)
        
        if mode == 'train':
            self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
            self.summary_writer.add_mesh('pred_mesh', vertices = pred_vertices, global_step=self.step_count)
            self.summary_writer.add_figure('pred_kps', figure, self.step_count) 
            for loss_name, val in losses.items():
                self.summary_writer.add_scalar(loss_name, val, self.step_count)

        # to see the result of the most-challenging-results of samples   
        else:
            self.summary_writer_sample.add_image('pred_shape', images_pred, self.step_count)
            self.summary_writer_sample.add_mesh('pred_mesh', vertices = pred_vertices, global_step=self.step_count)            
            for loss_name, val in losses.items():
                self.summary_writer_sample.add_scalar(loss_name, val, self.step_count)