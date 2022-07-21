from lib.loss_interface import Loss, LossInterface

import torch
import torch.nn.functional as F

class Wav2LipLoss(LossInterface):
    def get_loss_G(self, dict):
        L_G = 0.0
        
        # TODO: SyncNet Loss

        # Perceptual Loss
        if self.args.W_perc:
            d_feats = dict["d_feats"]
            L_perc = F.binary_cross_entropy(d_feats, torch.ones((len(d_feats), 1)).cuda())
            L_G += self.args.W_perc * L_perc
            self.loss_dict["L_perc"] = round(L_perc.item(), 4)    

        # Reconstruction Loss
        W_recon = 1. - self.args.W_sync - self.args.W_perc
        if W_recon:
            L_recon = F.l1_loss(dict['lip_syncing_faces'], dict['gt_faces'])
            L_G += W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)

        return L_G

    def get_loss_D(self, dict):
        L_real = F.binary_cross_entropy(dict["d_real"], torch.ones((len(dict["d_real"]), 1)).cuda())
        L_fake = F.binary_cross_entropy(dict["d_fake"], torch.zeros((len(dict["d_fake"]), 1)).cuda())

        L_D = L_real + L_fake
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        