## pretrain for discriminator 

# zz
# python train.py --name HipGanF_D_5 --discriminator Hip_MLP --lr_rate 5 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint  --train Base_GAN --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name HipGanF_5 --discriminator Hip_MLP --lr_rate 5 --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint  --train Base_GAN --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500



# python train.py --name HipGan_3rd_wgan --discriminator Hip_MLP --lr_rate 5 --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint  --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name GeoGan_3rd_wgan --discriminator Geo_CNN --lr_rate 5 --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint  --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name HipGan_3rd_vanilla_midweight --discriminator Hip_MLP --lr_rate 0.1 --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint /home/ubuntu/Desktop/HipGanF_D_5/checkpoints/2021_03_12-03_22_106500.pt --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name GeoGan_3rd_vanilla_lastweight_ganfreeze --discriminator Geo_CNN --pose_rg True --beta_rg True --lr 5e-6 --lr_rate 1 --betas_rg_weight 0.3 --pretrained_checkpoint /home/ubuntu/Desktop/GeoGanF_D_5/checkpoints/2021_03_12-00_13_4018000.pt --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name MANG --discriminator Geo_CNN --pose_rg True --beta_rg True --lr_rate 1 --betas_rg_weight 0.3 --pretrained_checkpoint /home/ubuntu/Desktop/GeoGanF_D_5/checkpoints/2021_03_12-00_13_4018000.pt --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name MANG_Geo_W --discriminator Geo_CNN --pose_rg True --beta_rg True --lr_rate 1 --betas_rg_weight 0.3 --pretrained_checkpoint /home/ubuntu/Desktop/GeoGanF_D_5/checkpoints/2021_03_12-00_13_4018000.pt --train SwapCam --num_epochs 3 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name MANG_Hip_W --discriminator Hip_MLP --pose_rg True --beta_rg True --lr_rate 1 --betas_rg_weight 0.3 --pretrained_checkpoint /home/ubuntu/Desktop/GeoGanF_D_5/checkpoints/2021_03_12-00_13_4018000.pt --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 500

# python train.py --name GeoGanF_D_1_V --discriminator Geo_CNN --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train Base_GAN --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name GeoSwapF_D_1_V --discriminator Geo_CNN --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name HipGanF_D_1_V --discriminator Hip_MLP --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train Base_GAN --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name HipSwapF_D_1_V --discriminator Hip_MLP --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train SwapCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name GeoGanF_D_1_W --discriminator Geo_CNN --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train Base_GAN --num_epochs 3 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name GeoSwapF_D_1_W --discriminator Geo_CNN --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train SwapCam --num_epochs 3 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name HipGanF_D_1_W --discriminator Hip_MLP --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train Base_GAN --num_epochs 3 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name HipSwapF_D_1_W --discriminator Hip_MLP --lr_rate 1 --close_D True --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint logs/mpi_pretrain/checkpoints/2021_03_03-15_33_04108000.pt --train SwapCam --num_epochs 3 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750

# python train.py --name GeoGanF_D_1_W_3rd_6000 --discriminator Geo_CNN --lr_rate 1 --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint GeoGanF_D_1_W/checkpoints/2021_03_14-07_06_096000.pt --train Base_GAN --num_epochs 1 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 --stage 3

# python train.py --name GeoSwapF_D_1_W_3rd_6000 --discriminator Geo_CNN --lr_rate 1 --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint GeoSwapF_D_1_W/checkpoints/2021_03_14-09_10_376000.pt --train SwapCam --num_epochs 1 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 --stage 3

#python train.py --name GeoGanF_D_1_W_2nd_6000 --discriminator Geo_CNN --close_D True --lr_rate 1 --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint checkpoints/Enc_pret.pt --train Base_GAN --num_epochs 3 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 

# python train.py --name GeoSwapF_DD_1_W_2nd_6000 --discriminator Geo_CNN --`close_D` True --lr_rate 1 --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint checkpoints/Enc_pret.pt --train SwapCam --num_epochs 1 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 


# python train.py --name GeoSwapF_D_1_W_3rd_6000 --discriminator Geo_CNN --lr_rate 1 --pose_rg True --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint GeoSwapF_D_1_W/checkpoints/2021_03_14-09_10_376000.pt --train SwapCam --num_epochs 10 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 --stage 3

# D: at stage 2 whether the discriminator loss affected generator (if D: No GAN loss to generator at 2nd stage) 
# number after D: relative learning rate of discriminator compared to the generator 
# W or V: Gan Loss used to train discriminator, W: wgan V: vanilla 


# python train.py --name Geo_Random_2stage_epoch10_V --discriminator Geo_CNN --close_D True --pose_rg True --lr_rate 3 --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint checkpoints/mpi_pretrain.pt --train RandomCam --num_epochs 10 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 750 

# python train.py --name Geo_Random_3stage_epoch3_V --discriminator Geo_CNN --pose_rg True --lr_rate 3 --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint Geo_Random_2stage_epoch5_V/checkpoints/2021_03_20-17_40_0830000.pt --train RandomCam --num_epochs 3 --gan_loss vanilla --loss_kps_weight 300 --checkpoint_steps 750 --stage 3
# python train.py --name Geo_Random_2stage_epoch10_W --discriminator Geo_CNN --close_D True --pose_rg True --lr_rate 3 --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint checkpoints/mpi_pretrain.pt --train RandomCam --num_epochs 10 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 

# python train.py --name Geo_Random_3stage_epoch5_W --discriminator Geo_CNN --pose_rg True --lr_rate 3 --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint Geo_Random_2stage_epoch10_W/checkpoints/2021_03_21-00_36_0360250.pt --train RandomCam --num_epochs 5 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 --stage 3


python train.py --name Basic_Swap_2stage_epoch7_W --discriminator Basic_MLP --close_D True --pose_rg True --lr_rate 3 --beta_rg True --betas_rg_weight 0.3 --pretrained_checkpoint checkpoints/mpi_pretrain.pt --train SwapCam --num_epochs 7 --gan_loss wgan --loss_kps_weight 300 --checkpoint_steps 750 