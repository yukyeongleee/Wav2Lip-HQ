import torch
from torch.utils.data import DataLoader

from lib import utils, checkpoint
from lib.model_interface import ModelInterface

from syncnet.dataset import AudioVisualDataset
from syncnet.loss import SyncNetLoss
from syncnet.nets import SyncNet


class SyncNetColor(ModelInterface):
    """
        In the __init__ function of the ModelInterface class, the SetupModel function is called
        The SetupModel function helps initializing nets, optimizers, datasets, data_iterator, validation and losses.

        Override the functions called by the SetupModel()
    """
    def set_networks(self):
        self.S = SyncNet(self.args.tighter_box).cuda(self.gpu).train()
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
            self.args.isMaster,
            self.args.tighter_box
        )
        
        if self.args.use_validation:
            self.valid_dataset = AudioVisualDataset(
                self.args.valid_dataset_root, 
                self.args.video_step_size, 
                self.args.mel_step_size, 
                self.args.img_size, 
                self.args.fps, 
                "val", 
                self.args.isMaster,
                self.args.tighter_box
            )

    def set_validation(self):
        if self.args.use_validation:
            sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.args.use_mGPU else None
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.args.batch_per_gpu, sampler=sampler, num_workers=8, drop_last=True)
            faces, mel, label = next(iter(self.valid_dataloader))
            self.valid_faces, self.valid_mel, self.valid_label = faces.to(self.gpu), mel.to(self.gpu), label.to(self.gpu)    
            # sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.args.use_mGPU else None
            # self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=5, pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
            # self.valid_iterator = iter(self.valid_dataloader)


    def set_loss_collector(self):
        self._loss_collector = SyncNetLoss(self.args)

    def go_step(self, global_step):
        # load batch
        faces, mel, label = self.load_next_batch()
        self.dict["faces"] = faces
        self.dict["mel"] = mel
        self.dict["label"] = label

        # run S
        self.run_S()

        # update S
        loss_S = self.loss_collector.get_loss_S(self.dict)
        utils.update_net(self.opt_S, loss_S) 


    def run_S(self):
        audio_embedding, face_embedding = self.S(self.dict['mel'], self.dict['faces'])

        self.dict["audio_embedding"] = audio_embedding
        self.dict["face_embedding"] = face_embedding


    def do_validation(self, step):
        with torch.no_grad():
            audio_embedding, face_embedding = self.S(self.valid_mel, self.valid_faces)
            
            valid_dict = {}
            valid_dict["audio_embedding"] = audio_embedding
            valid_dict["face_embedding"] = face_embedding
            valid_dict["label"] = self.valid_label

            # update S
            loss_S = self.loss_collector.get_loss_S(valid_dict, True)


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
            faces, mel, label = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            faces, mel, label = next(self.train_iterator)
        faces, mel, label = faces.to(self.gpu), mel.to(self.gpu), label.to(self.gpu)
        return faces, mel, label

    def set_optimizers(self):
        self.opt_S = torch.optim.Adam(self.S.parameters(), lr=self.args.lr_S, betas=self.args.betas)

    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        self.S = torch.nn.parallel.DistributedDataParallel(self.S, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module

    def save_checkpoint(self, global_step):
        """
        Save model and optimizer parameters.
        """

        checkpoint.save_checkpoint(self.args, self.S, self.opt_S, name='S', global_step=global_step)

        if self.args.isMaster:
            print(f"\nCheckpoints are succesively saved in {self.args.save_root}/{self.args.run_id}/ckpt/\n")
    
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """

        self.args.global_step = \
        checkpoint.load_checkpoint(self.args, self.S, self.opt_S, "S")

        if self.args.isMaster:
            print(f"Pretrained parameters are succesively loaded from {self.args.save_root}/{self.args.ckpt_id}/ckpt/")