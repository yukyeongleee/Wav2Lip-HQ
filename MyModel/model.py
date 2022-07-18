import torch
from lib import utils
from lib.model_interface import ModelInterface
from MyModel.loss import MyModelLoss
from MyModel.nets import MyGenerator
from pg_modules.projected_discriminator import ProjectedDiscriminator


class MyModel(ModelInterface):
    def set_networks(self):
        self.G = MyGenerator().cuda(self.gpu).train()
        self.D = ProjectedDiscriminator().cuda(self.gpu).train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)

    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.args)

    def go_step(self, global_step):
        # load batch

        I_source, I_target, same_person = self.load_next_batch()
        same_person = same_person.reshape(-1, 1, 1, 1).repeat(1, 3, 256, 256)

        self.dict["I_source"] = I_source
        self.dict["I_target"] = I_target
        self.dict["same_person"] = same_person

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
            self.dict["I_source"], 
            self.dict["I_target"], 
            self.dict["I_swapped"]
            ]

    def run_G(self):
        I_swapped, id_source = self.G(self.dict["I_source"], self.dict["I_target"])
        I_cycle, _ = self.G(self.dict["I_target"], I_swapped)
        id_swapped = self.G.get_id(I_swapped)
        d_adv, feat_fake = self.D(I_swapped, None)
        feat_real = self.D.get_feature(self.dict["I_target"])

        self.dict["I_swapped"] = I_swapped
        self.dict["I_cycle"] = I_cycle
        self.dict["id_source"] = id_source
        self.dict["id_swapped"] = id_swapped
        self.dict["d_adv"] = d_adv
        self.dict["feat_real"] = feat_real
        self.dict["feat_fake"] = feat_fake

    def run_D(self):
        d_real, _ = self.D(self.dict["I_source"], None)
        d_fake, _ = self.D(self.dict["I_swapped"].detach(), None)

        self.dict["d_real"] = d_real
        self.dict["d_fake"] = d_fake

    def do_validation(self, step):
        with torch.no_grad():
            result_images = self.G(self.valid_source, self.valid_target)[0]
        self.valid_images = [
            self.valid_source, 
            self.valid_target, 
            result_images
            ]

    @property
    def loss_collector(self):
        return self._loss_collector
        