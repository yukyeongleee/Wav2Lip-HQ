import os
import sys
sys.path.append("./")
sys.path.append("./packages")

# from lib.utils import save_image
from lib.config import Config
from syncnet.model import SyncNetColor

import torch
import torchvision

import cv2
import wandb


def train(gpu, args): 
    torch.cuda.set_device(gpu)

    # convert dictionary to class
    args = Config(args)    
    model = SyncNetColor(args, gpu)

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster and args.use_wandb:
        wandb.init(project=args.model_id, name=args.run_id)

    # Training loop
    global_step = args.global_step if args.load_ckpt else 0
    while global_step < args.max_step:

        # go one step
        model.go_step(global_step)

        if args.isMaster:
            # Save and print loss
            if global_step % args.loss_cycle == 0:
                model.loss_collector.print_loss(global_step)

                if args.use_wandb:
                    wandb.log(model.loss_collector.loss_dict)
                
            # Save image
            if global_step % args.test_cycle == 0:
                # save_image(model.args, global_step, "train_imgs", model.train_images)

                if args.use_validation:
                    model.do_validation(global_step) 
                    # save_image(model.args, global_step, "valid_imgs", model.valid_images)

            # Save checkpoint parameters 
            if global_step % args.ckpt_cycle == 0:
                model.save_checkpoint(global_step)

        global_step += 1


if __name__ == "__main__":

    # load config
    config_path = "configs/train_configs_syncnet.yaml"
    args = Config.from_yaml(config_path)
    
    # update configs
    args.run_id = sys.argv[1] # command line: python train.py {run_id}
    args.gpu_num = torch.cuda.device_count()
    
    # save config
    os.makedirs(f"{args.save_root}/{args.run_id}", exist_ok=True)
    args.save_yaml()

    # Set up multi-GPU training
    if args.use_mGPU:  
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args.__dict__, ))

    # Set up single GPU training
    else:
        train(gpu=0, args=args.__dict__)
