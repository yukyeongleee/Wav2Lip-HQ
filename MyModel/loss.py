from lib.loss_interface import Loss, LossInterface
import torch.nn.functional as F

class MyModelLoss(LossInterface):
    def get_loss_G(self, dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = (-dict["d_adv"]).mean()
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)

        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(dict["id_source"], dict["id_swapped"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)
            
        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L1_loss(dict["I_target"]*dict["same_person"], dict["I_swapped"]*dict["same_person"])
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # Cycle loss
        if self.args.W_cycle:
            L_cycle = Loss.get_L2_loss(dict["I_target"], dict["I_cycle"])
            L_G += self.args.W_cycle * L_cycle
            self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)

        # feat loss
        if self.args.W_feat:
            L_feat = Loss.get_L1_loss(dict["feat_fake"]["3"], dict["feat_real"]["3"])
            L_G += self.args.W_feat * L_feat
            self.loss_dict["L_feat"] = round(L_feat.item(), 4)
            
        # LPIPS loss
        if self.args.W_lpips:
            L_lpips = Loss.get_lpips_loss(dict["I_swapped"], dict["I_target"])
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)

        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, dict):
        L_real = (F.relu(1 - dict["d_real"])).mean()
        L_fake = (F.relu(1 + dict["d_fake"])).mean()
        L_D = L_real + L_fake
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        