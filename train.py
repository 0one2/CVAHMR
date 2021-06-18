import torch

from utils import TrainOptions
from train import Trainer, Pre_Trainer, Trainer_Cycle

if __name__ == '__main__':
    torch.set_num_threads(6)

    options = TrainOptions().parse_args()
    train_version = ['Base', 'Base_GAN', 'RandomCam', 'SwapCam', 'Cycle']
    print(f'options pretrain is {options.pretrain}')

    if options.train in train_version:
        if options.train == 'Cycle':
            trainer = Trainer_Cycle(options)
        elif options.pretrain:
            trainer = Pre_Trainer(options)
            print('pretrained!!!!')
        else:
            trainer = Trainer(options)
        trainer.train()
    else:
        raise NameError(f'{options.train} not implemented!')
