import torch
import torch.nn as nn
import pdb 
import math 

from utils.geometry import batch_rodrigues


class GradientPaneltyLoss(nn.Module):
    def __init__(self):
        super(GradientPaneltyLoss, self).__init__()

    def forward(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)          # batch x num_kps*2
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

class Vanilla(nn.Module):
    def __init__(self):
        super(Vanilla, self).__init__()
        self.criterion_BCELogitsLoss = nn.BCEWithLogitsLoss() # combines a sigmoid layer + BCELoss in one single class. more stable than a plain sigmoid followed BCELoss 

    def forward(self, prob_real, prob_pred, prob_new, numCam = 'double', mode = 'discriminator'):
        if mode == 'discriminator':
            loss_D_real = self.criterion_BCELogitsLoss(prob_real, torch.ones_like(prob_real))
            loss_D_pred = self.criterion_BCELogitsLoss(prob_pred, torch.zeros_like(prob_pred))
            
            if numCam == 'double':
                loss_D_new = self.criterion_BCELogitsLoss(prob_new, torch.zeros_like(prob_new))
                # loss_D_fake = 0.5 * (loss_D_pred + loss_D_new)
                loss_D_fake = loss_D_new

            else:     
                loss_D_fake = loss_D_pred

            loss = 0.5 * (loss_D_real + loss_D_fake)

            return loss, loss_D_real, loss_D_fake              
            
        else:          
            loss_G_pred = self.criterion_BCELogitsLoss(prob_pred, torch.ones_like(prob_pred))

            if numCam == 'double':
                loss_G_new = self.criterion_BCELogitsLoss(prob_new, torch.ones_like(prob_new))
                # loss = 0.5 * (loss_G_pred + loss_G_new)
                loss = loss_G_new

            else:       
                loss = loss_G_pred                
        
            return loss


class Wgan(nn.Module):
    def __init__(self):
        super(Wgan, self).__init__()
    
    def forward(self, prob_real, prob_pred, prob_new, numCam = 'double', mode = 'discriminator'):
        if mode == 'discriminator':
            loss_D_real = torch.mean(prob_real)
            loss_D_pred = -torch.mean(prob_pred)

            if numCam == 'double':
                loss_D_new = -torch.mean(prob_new)
                loss_D_fake = loss_D_new

            else:
                loss_D_fake = loss_D_pred
            loss = loss_D_real + loss_D_fake
            return loss, loss_D_real, loss_D_fake              

        else:
            loss_G_pred = torch.mean(prob_pred)

            if numCam == 'double':
                loss_G_new = torch.mean(prob_new)
                loss = loss_G_new

            else:
                loss = loss_G_pred

        return loss


class WGANGP(nn.Module):
    def __init__(self):
        super(WGANGP, self).__init__()
        self.GP = GradientPaneltyLoss().cuda()

    def forward(self, model, gt_kps, pred_kps, new_kps, vis_idx, numCam = 'single'):
        batch_size = gt_kps.size(0)
        alpha = torch.rand(batch_size, 1, 1).cuda()
        if numCam == 'single':
            fake_kps = pred_kps
        else:
            # fake_kps = torch.cat((pred_kps, new_kps), dim = 0)
            fake_kps = new_kps
            # gt_kps = gt_kps.repeat(2, 1, 1)
            # alpha = alpha.repeat(2, 1, 1)

        output = (alpha * gt_kps + (1 - alpha) * fake_kps.detach()).requires_grad_(True)
        src_out = model(output, vis_idx)
        loss_D_GP = self.GP(src_out, output)
        return loss_D_GP


def calculate_accuracy(prob_pred, prob_real, prob_new, numCam = 'double'):
    class_pred = torch.round(torch.sigmoid(prob_pred))
    class_real = torch.round(torch.sigmoid(prob_real))
    correct_pred = (class_pred == torch.zeros_like(class_pred)).sum().item()
    correct_real = (class_real == torch.ones_like(class_real)).sum().item()
    
    if numCam == 'single':
        correct_fake = correct_pred

    else :
        class_new = torch.round(torch.sigmoid(prob_new))
        correct_new = (class_new == torch.zeros_like(class_new)).sum().item()
        correct_fake = correct_new


    return correct_real, correct_fake            

def calculate_rot_err(pred_rot24, gt_rot24):
    pred_rot = pred_rot24[:, 0, :, :]
    gt_rot = gt_rot24[:, 0, :, :]
  
    angle_dist = []

    for i in range(gt_rot.size(0)):
        a = torch.matmul(pred_rot[i], torch.ones(3,1).cuda()/torch.norm(torch.ones(3,1).cuda()))
        b = torch.matmul(gt_rot[i], torch.ones(3,1).cuda()/torch.norm(torch.ones(3,1).cuda()))

        angle = 180/math.pi * torch.asin(torch.norm(torch.cross(a,b)))
        angle_dist.append(angle)
        
    return angle_dist

