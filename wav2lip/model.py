import torch
from torch.utils.data import DataLoader

from lib import utils
from lib.model_interface import ModelInterface

from wav2lip.dataset import AudioVisualDataset
from wav2lip.loss import Wav2LipLoss
from wav2lip.nets import LipGenerator, LipDiscriminator


class Wav2Lip(ModelInterface):
    """
        In the __init__ function of the ModelInterface class, the SetupModel function is called
        The SetupModel function helps initializing nets, optimizers, datasets, data_iterator, validation and losses.

        Override the functions called by the SetupModel()
    """
    def set_networks(self):
        self.G = LipGenerator().cuda(self.gpu).train()
        self.D = LipDiscriminator().cuda(self.gpu).train()
        # self.D.feature_network.eval()
        # self.D.feature_network.requires_grad_(False)

    def set_dataset(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        self.train_dataset = AudioVisualDataset(
            self.args.train_dataset_root, 
            self.args.video_step_size, 
            self.args.mel_step_size, 
            self.args.img_size, 
            self.args.fps, 
            "train", 
            self.args.isMaster
        )
        
        if self.args.use_validation:
            self.valid_dataset = AudioVisualDataset(
                self.args.valid_dataset_root, 
                self.args.video_step_size, 
                self.args.mel_step_size, 
                self.args.img_size, 
                self.args.fps, 
                "val", 
                self.args.isMaster
            )

    def set_validation(self):
        if self.args.use_validation:
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=5, num_workers=8, drop_last=True)
            faces, concatenated_mels, mel, gt_faces = next(iter(self.valid_dataloader))
            self.valid_faces, self.valid_concatenated_mels, self.valid_gt_faces = faces.to(self.gpu), concatenated_mels.to(self.gpu), gt_faces.to(self.gpu)    


    def set_loss_collector(self):
        self._loss_collector = Wav2LipLoss(self.args)

    def go_step(self, global_step):
        # load batch
        faces, concatenated_mels, mel, gt_faces = self.load_next_batch()
        self.dict["faces"] = faces
        self.dict["concatenated_mels"] = concatenated_mels
        self.dict["mel"] = mel
        self.dict["gt_faces"] = gt_faces # [32, 3, 5, 96, 96]

        # run G
        self.run_G()

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.opt_G, loss_G)

        # run D
        self.run_D()

        # update D
        loss_D = self.loss_collector.get_loss_D(self.dict)
        utils.update_net(self.opt_D, loss_D)

        # print images
        self.train_images = [
            self.dict["gt_faces"], 
            self.dict["lip_syncing_faces"]
        ]

    def run_G(self):
        lip_syncing_faces = self.G(self.dict["concatenated_mels"], self.dict["faces"])    
        d_feats = self.D.perceptual_forward(lip_syncing_faces)

        self.dict["lip_syncing_faces"] = lip_syncing_faces  # [32, 3, 5, 96, 96]
        self.dict["d_feats"] = d_feats

    def run_D(self):
        d_real = self.D(self.dict["gt_faces"])
        d_fake = self.D(self.dict["lip_syncing_faces"].detach())

        self.dict["d_real"] = d_real
        self.dict["d_fake"] = d_fake

    def do_validation(self, step):
        with torch.no_grad():
            lip_syncing_faces = self.G(self.valid_concatenated_mels, self.valid_faces)
            self.valid_images = [
                self.valid_gt_faces,
                lip_syncing_faces
            ]

    @property
    def loss_collector(self):
        return self._loss_collector

    """
    Override
    """
    def load_next_batch(self):      
        """
        Load next batch of source image, target image, and boolean values that denote 
        if source and target are identical.
        """
        try:
            faces, concatenated_mels, mel, gt_faces = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            faces, concatenated_mels, mel, gt_faces = next(self.train_iterator)
        faces, concatenated_mels, mel, gt_faces = faces.to(self.gpu), concatenated_mels.to(self.gpu), mel.to(self.gpu), gt_faces.to(self.gpu)
        return faces, concatenated_mels, mel, gt_faces
