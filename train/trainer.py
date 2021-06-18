import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
import pdb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import csv

from datasets import MixedDataset
from models import hmr, SMPL, Bottleneck
from models.loss import GradientPaneltyLoss, Vanilla, Wgan, WGANGP, calculate_accuracy, calculate_rot_err
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from utils.renderer import Renderer
from utils.imutils import plot_kps
from utils import BaseTrainer


import config
import constants


class Trainer(BaseTrainer):
    
    def init_fn(self):
        self.discriminator = self.discriminator.cuda()
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)

        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)

        if self.options.stage == 3 : # JY temporary code
            self.model_pretrained = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
            

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(),
                                          lr=self.options.lr*self.options.lr_rate,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)

        # ####################################################3
        # self.mean_params = np.load(config.SMPL_MEAN_PARAMS)
        # self.init_betas = torch.from_numpy(self.mean_params['shape'][:].astype('float32')).repeat(self.options.batch_size).view(self.options.batch_size,-1).to(self.device)
        # #####################################################

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
        self.models_dict = {'model' : self.model, 'discriminator': self.discriminator}
        self.optimizers_dict = {'optimizer' : self.optimizer}

        self.focal_length = constants.FOCAL_LENGTH
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

        if self.options.gan_loss == 'vanilla':
            self.gan_loss = Vanilla().cuda()
        elif self.options.gan_loss == 'wgan':
            self.gan_loss = Wgan().cuda()
        else:
            raise NameError(f'{self.options.gan_loss} not implemented Yet!')

        #Load the csv file for pose priors 
        if self.options.pose_rg == True:
            with open(self.options.poseprior_directory,'r',encoding='utf-16') as data:
                writer = csv.DictReader(data)
                self.row_dictionary = []
                for rg_param in writer:
                    self.row_dictionary.append(rg_param)


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
        gt_betas = input_batch['betas']

        batch_size = images.shape[0]

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)

        if self.options.stage == 3 :
            pseudo_pred_rotmat, pseudo_pred_betas, pseudo_pred_camera = self.model_pretrained(images)

        # ######################### overwritten with mean betas
        # pred_betas = self.init_betas
        # #########################

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        gt_output = self.smpl(betas=gt_betas, body_pose=gt_rotmat[:,1:], global_orient=gt_rotmat[:,0].unsqueeze(1), pose2rot=False)
        gt_vertices = gt_output.vertices
        gt_joints = gt_output.joints


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

        # get kps of COCO indexes (batch, 49, 3) -> (batch, 17, 3)
        gt_kps = gt_keypoints_2d[:, 25:].clone() 
        pred_kps = pred_keypoints_2d[:, 25:].clone() 

        # set kps value to zero if visibility of certain kps is 0 
        vis_idx = torch.where(gt_kps[:, :, 2] == 0)
        vis_batch, vis_kps = vis_idx
        
        gt_kps[vis_batch, vis_kps] = 0
        gt_kps = gt_kps[:, :, :2]
        pred_kps[vis_batch, vis_kps] = 0        

        if self.options.train == 'Base':
            pass
        else:
            if self.options.train == 'Base_GAN':
                loss_D, loss_D_real, loss_D_fake, real_acc, fake_acc = self.BaseGAN_loss(gt_kps, pred_kps, vis_idx)

            elif self.options.train == 'RandomCam':
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

            # # Discriminator Update 
            if (self.step_count+1) % self.options.update_freq == 0 :
                self.optimizer_d.zero_grad()            # set the grads of the previous epoch to be zero, to prevent the accumulation of grads of previous step | equal to model.zero_grad() if optimizer contains params of model 
                loss_D.backward(retain_graph=True)      # calculate the gradient at each layer | retain computational graph for the update of generator(if not, calculating grads and then disappear computation graph)
                self.optimizer_d.step()            # update the layer weight of related params, here 'discriminator parameters'


        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                              self.options.gt_train_weight)

        rot_err = calculate_rot_err(pred_rotmat, gt_rotmat)

        ## For Generator Updates

        if self.options.train == 'Base':
            loss_G = self.options.loss_kps_weight * loss_keypoints
            g_real_acc, g_fake_acc = 0.5, 0.5
            
        else: 

            if self.options.train == 'Base_GAN':
                loss_gan_generator, g_real_acc, g_fake_acc = self.BaseGAN_loss(gt_kps, pred_kps, vis_idx, 'generator')

            elif self.options.train == 'RandomCam':
                # Random y rotation
                loss_gan_generator, g_real_acc, g_fake_acc = self.DoubleCam_loss(pred_rotmat, rand_rotmat, pred_betas, pred_cam_t, gt_kps, pred_kps, vis_idx, mode = 'generator')

            elif self.options.train == 'SwapCam':
                loss_gan_generator, g_real_acc, g_fake_acc = self.DoubleCam_loss(pred_rotmat, swap_rotmat, pred_betas, pred_cam_t, gt_kps, pred_kps, vis_idx, mode = 'generator')

            else:
                return NameError(f'{self.options.train} not implemented yet!')

            if self.options.close_D == True:
                loss_G = self.options.loss_kps_weight * loss_keypoints

            else:
                loss_G = self.options.loss_G_weight * loss_gan_generator +\
                                    self.options.loss_kps_weight * loss_keypoints


        # Applying axis-wise pose regression loss

        if self.options.pose_rg == True:
            loss_regr_poses_total = 0
            loss_regr_poses = self.pose_rg_loss(pred_rotmat)
            for i in range(len(loss_regr_poses)):
                loss_G += loss_regr_poses[i]
                loss_regr_poses_total += loss_regr_poses[i]

        if self.options.beta_rg == True:
            loss_regr_betas = torch.mean(pred_betas**2)
            loss_G += self.options.betas_rg_weight * loss_regr_betas

        if self.options.stage == 3 :
            loss_G += self.options.pseudo_weight * (torch.mean((pred_betas - pseudo_pred_betas.detach())**2))
            loss_G += self.options.pseudo_weight * 0.5 * (torch.mean((pred_rotmat[:,0].unsqueeze(1) - pseudo_pred_rotmat[:,0].unsqueeze(1))**2))
            # new_rotmat

            
        # Generator update 

        if self.epoch < 3:
            self.optimizer.zero_grad()  # same as self.discriminator.zero_grad()    
            loss_G.backward()           # calculate gradient 
            self.optimizer.step()

        else:
            pass

        # Pack output arguments for tensorboard logging

        if self.options.train == 'Base':
            output = {'pred_vertices': pred_vertices.detach(),
                      'gt_vertices': gt_vertices.detach(),
                        'pred_cam_t': pred_cam_t.detach(),
                        'gt_kps': gt_kps.detach(),
                        'pred_kps': pred_kps.detach(),
                        'images': images.detach()}

            losses = {'loss': loss_G.detach().item(),
                        'rot_err': sum(rot_err) / self.options.batch_size,
                        'loss_keypoints': loss_keypoints.detach().item()}
        else:
            output = {'pred_vertices': pred_vertices.detach(),
                      'gt_vertices': gt_vertices.detach(),
                        'pred_cam_t': pred_cam_t.detach(),
                        'gt_kps': gt_kps.detach(),
                        'pred_kps': pred_kps.detach(),
                        'images': images.detach()} 

            # print(pred_rotmat[:,0])

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
        
        if self.options.beta_rg == True:
            losses.update({'loss_regr_betas' : loss_regr_betas})
        if self.options.pose_rg == True:
            losses.update({'loss_regr_poses' : loss_regr_poses_total})

        if self.options.stage == 3 :
            losses.update({'loss_pseudo_betas' : torch.mean((pred_betas - pseudo_pred_betas)**2).detach().item()    })
            losses.update({'loss_pseudo_rot' : torch.mean((pred_rotmat[:,0].unsqueeze(1) - pseudo_pred_rotmat[:,0].unsqueeze(1))**2).detach().item()})
            # self.options.pseudo_weight * (torch.mean((pred_betas - pseudo_pred_betas)**2))
            # loss_G += self.options.pseudo_weight * 0.5 * (torch.mean((pred_rotmat[:,0].unsqueeze(1) - pseudo_pred_rotmat[:,0].unsqueeze(1))**2))

        return output, losses


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


    # Per-pose regularization loss
    def pose_rg_loss(self, pred_rotmat):

        # Sort which joints will have regularization applied
        rg_joint_list = []
        for joint in self.row_dictionary:
            if joint['reg_applied'] == 'yes':
                rg_joint_list.append((joint['Joint_no']))

        # Calculate axis-wise joint rotations
        losses = []

        for idx in map(int,rg_joint_list):
            angle_loss = []
            axis_angles = []
            axis_angles.append(torch.atan2(pred_rotmat[:,idx,2,1],pred_rotmat[:,idx,2,2]))
            axis_angles.append(torch.atan2(-pred_rotmat[:,idx,2,0],torch.sqrt(torch.square(pred_rotmat[:,idx,2,1])+torch.square(pred_rotmat[:,idx,2,2]))))
            axis_angles.append(torch.atan2(pred_rotmat[:,idx,1,0],pred_rotmat[:,idx,0,0]))
        
            # Apply desired Regularization
            joint_dict = self.row_dictionary[idx-1]

            axis_reg = [joint_dict['X_axis'], joint_dict['Y_axis'], joint_dict['Z_axis']]
            axis_weights = [joint_dict['X_weight'], joint_dict['Y_weight'], joint_dict['Z_weight']]

            for i in range(3):
                if axis_reg[i] == 'exp':
                    angle_loss.append(float(axis_weights[i]) * (torch.exp(torch.abs(axis_angles[i])-1)))
                elif axis_reg[i] == 'l2':
                    angle_loss.append(float(axis_weights[i]) * axis_angles[i]**2)
                else:
                    angle_loss.append(0)
            
            whole_angle_loss = torch.mean(angle_loss[0] + angle_loss[1] + angle_loss[2])
            losses.append(whole_angle_loss)
        
        return losses


    def train_summaries(self, mode, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        gt_vertices = output['gt_vertices']

        pred_cam_t = output['pred_cam_t']

        pred_kps = output['pred_kps']
        gt_kps = output['gt_kps']

        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

        figure = plot_kps(images,gt_kps,pred_kps)
        
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_mesh('pred_mesh', vertices = pred_vertices, global_step=self.step_count)
        self.summary_writer.add_figure('pred_kps', figure, self.step_count)            
        
        # # to see the result of the most-challenging-results of samples   
        # else:
        #     # self.summary_writer_sample.add_image('pred_shape', images_pred, self.step_count)
        #     # self.summary_writer_sample.add_mesh('pred_mesh', vertices = pred_vertices, global_step=self.step_count)            
        #     for loss_name, val in losses.items():
        #         self.summary_writer_sample.add_scalar(loss_name, val, self.step_count)