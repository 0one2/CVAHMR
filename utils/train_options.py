import os
import json
import argparse
import numpy as np
from collections import namedtuple

class TrainOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--discriminator', required=True, choices = ['Hip_MLP', 'Basic_MLP', 'Geo_CNN'], help='Type of the discriminator')
        req.add_argument('--train', required=True, choices = ['Base', 'Base_GAN', 'RandomCam', 'SwapCam', 'Cycle'], help='Version of losses that affect training')
        req.add_argument('--name', required=True, help='Name of the experiment')
        req.add_argument('--gan_loss', default='vanilla', type=str, help='kind of GAN loss')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=2, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained checkpoint at the beginning training') 


        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=50, help='Total number of training epochs')
        train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        train.add_argument("--lr_rate", type=float, default=1, help="Learning rate")
        train.add_argument("--update_freq", type=float, default=1, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=16, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--test_steps', type=int, default=1000, help='Testing frequency during training')            #####
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Checkpoint saving frequency')           #####
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network') 
        train.add_argument('--rot_factor', type=float, default=30, help='Random rotation in the range [-rot_factor, rot_factor]') #####
        train.add_argument('--noise_factor', type=float, default=0.4, help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]') #####
        train.add_argument('--scale_factor', type=float, default=0.25, help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]') #####
        train.add_argument('--ignore_3d', default=True, action='store_true', help='Ignore GT 3D data (for unpaired experiments')#####
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training') #####
        train.add_argument('--openpose_train_weight', default=0., help='Weight for OpenPose keypoints during training') #####
        train.add_argument('--num_kps', type=int, default=24., help='Number of 2D joints to detect, default is COCO available 24') 
        train.add_argument('--rand_std', default=1, type=float, help='Degree of std when RandomCam') #####
        train.add_argument('--loss_G_weight', default=1, type=float, help='Weight of generator loss over total loss for generator update') 
        train.add_argument('--loss_kps_weight', default=60., type=float, help='Weight of 2D and 3D keypoint loss') 
        train.add_argument('--pose_rg', default=False, type=bool, help='Add L2 loss of pose')
        train.add_argument('--beta_rg', default=False, type=bool, help='beta regression do or not')
        train.add_argument('--betas_rg_weight', default=1, type=float, help='beta regression weight ')
        train.add_argument('--poseprior_directory', default = 'poseprior.csv', help ='Directory for csv file with pose regularization configurations')
        train.add_argument('--pretrain', default = False, type=bool, help ='to choose whethere to pretrain or not')
        train.add_argument('--loss_rot_err', default = 0.1, type = float, help ='weight of rotation error')
        train.add_argument('--random_type', default='Shift', choices=['Shift','Direct'], help='Which RandomCam to use?') #####
        train.add_argument('--close_D', default=False, type = bool, help='To pretrain Discriminator') #####

        train.add_argument('--stage', default=2, type = int, help='stage') #####

        train.add_argument('--pseudo_weight', default=1, type = int, help='stage') #####


        # For Base_GAN_SwapCam
        train.add_argument('--loss_swap_weight', default=0.5, type=float, help='Weight of swap probability loss over (pred prob + swap prob) when Base_GAN_SwapCam') 

        # For Cycle 
        train.add_argument('--loss_D_weight', default=0.5, type=float, help='Weight of discriminator loss over total loss when Cycle Consistency') 
        train.add_argument('--loss_smpl_regression_weight', default=0.5, type=float, help='Weight of smpl regression loss over discriminator loss when when Cycle Consistency') 
        train.add_argument('--pretrain_D', default=False, type=bool, help='pretrain discriminator when using pretrained_encoder') 


        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)
        return 

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = '/home/ubuntu/project/CAM_PRET'
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            self.args.summary_dir_sample = os.path.join(self.args.log_dir, 'tensorboard_sample')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return

