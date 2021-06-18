from __future__ import division
import sys
import time
import pdb 

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.utils.tensorboard import SummaryWriter
import numpy as np 

from utils import CheckpointDataLoader, CheckpointSaver
from models.discriminator import Hip_Discriminator, Mlp_Discriminator, Geo_Discriminator
from models.discriminator import Hip_Discriminator_Cycle, Mlp_Discriminator_Cycle, Geo_Discriminator_Cycle
# from datasets.base_dataset import BaseDataset

class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)
        self.summary_writer_sample = SummaryWriter(self.options.summary_dir_sample)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']


        if self.options.discriminator == 'Hip_MLP':
            self.discriminator = Hip_Discriminator(self.options.num_kps)
        elif self.options.discriminator == 'Basic_MLP':
            self.discriminator = Mlp_Discriminator(self.options.num_kps)
        elif self.options.discriminator == 'Geo_CNN':
            self.discriminator = Geo_Discriminator()
        else:
            raise ValueError(f'Discriminator version {self.options.discriminator} not implemented yet!')


        train_version = ['Base', 'Base_GAN', 'RandomCam', 'SwapCam', 'Cycle']
        if self.options.train not in train_version:
            raise ValueError(f'Train version {self.options.train} not implemented yet!')

        elif self.options.train == 'Cycle':
            if self.options.discriminator == 'Hip_MLP':
                self.discriminator = Hip_Discriminator_Cycle(self.options.num_kps)
            elif self.options.discriminator == 'Basic_MLP':
                self.discriminator = Mlp_Discriminator_Cycle(self.options.num_kps)
            elif self.options.discriminator == 'Geo_CNN':
                self.discriminator = Geo_Discriminator_Cycle()
            else:
                raise ValueError(f'Discriminator version {self.options.discriminator} not implemented yet!')
        self.init_fn()


    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    print(model)
                    self.models_dict[model].load_state_dict(checkpoint[model], strict=True)
                    print('Checkpoint loaded')

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
                
        data_path = 'data/dataset_extras/coco_2014_train.npz'
        data = np.load(data_path)

        samples = {k: v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k,v in data.items()}

        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            self.epoch = epoch
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)
            print(f'shuffle is {self.options.shuffle_train}')
            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                    
                if time.time() < self.endtime:

                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    
                    out = self.train_step(batch)
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries('train', batch, *out)

                        # # for sample data visualization
                        # sample_data = BaseDataset(self.options, 'coco-sample')
                        # sample_batch = {}

                        # for data_idx, data in enumerate(sample_data):    
                        #     for (k, v) in data.items():
                        #         if data_idx == 0:
                        #             sample_batch[k] = []
                        #         sample_batch[k].append(v)

                        
                        # sample_batch['img'] = torch.cat(sample_batch['img'], dim = 0).reshape(self.options.batch_size, 3, self.options.img_res, self.options.img_res)
                        # sample_batch['keypoints'] = torch.cat(sample_batch['keypoints'], dim = 0).reshape(self.options.batch_size, -1, 3)
                        
                        # sample_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in sample_batch.items()}

                        # out_samples = self.train_step(sample_batch)
                        # self.train_summaries('sample', sample_batch, *out_samples)
                        # # end
                        

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.finalize()
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)
            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count)
        return
                                                                                

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        pass