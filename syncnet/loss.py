from lib.loss_interface import Loss, LossInterface

import time

import torch
import torch.nn.functional as F

class SyncNetLoss(LossInterface):
    def get_loss_G(self, dict):
        return 0.0

    def get_loss_D(self, dict):
        return 0.0

    
    def get_loss_S(self, dict):
        
        # Cosine Loss
        dist = F.cosine_similarity(dict['audio_embedding'], dict['face_embedding'])
        L_S = F.binary_cross_entropy(dist.unsqueeze(1), dict['label'])

        self.loss_dict["L_S"] = round(L_S.item(), 4)

        return L_S

    def print_loss(self, global_step):
        """
        Print discriminator and generator loss and formatted elapsed time.
        """
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossS: {self.loss_dict["L_S"]}')