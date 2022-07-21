import os
import sys
sys.path.append("./")
sys.path.append("./packages")

# from lib.utils import save_image
from lib.config import Config
from wav2lip.model import Wav2Lip

import torch
import torchvision

import cv2
import wandb


def train(gpu, args): 
    torch.cuda.set_device(gpu)

    # convert dictionary to class
    args = Config(args)    
    model = Wav2Lip(args, gpu)

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
                save_image(model.args, global_step, "train_imgs", model.train_images)

                if args.use_validation:
                    model.do_validation(global_step) 
                    save_image(model.args, global_step, "valid_imgs", model.valid_images)

            # Save checkpoint parameters 
            if global_step % args.ckpt_cycle == 0:
                model.save_checkpoint(global_step)

        global_step += 1


"""
Helper functions
"""
def make_grid_image(images_list):   
    grid_rows = []

    gt_images = images_list[0]
    gen_images = images_list[1] # [32, 3, 5, 96, 96]

    for i in range(5):
        gt = gt_images[i].squeeze(0)
        gt = gt.transpose(1, 0) # [5, 3, 96, 96]
           
        gt_grid_row = torchvision.utils.make_grid(gt, nrow=gt.shape[0]) # * 0.5 + 0.5
        grid_rows.append(gt_grid_row)

        gen = gen_images[i].squeeze(0)
        gen = gen.transpose(1, 0) 
        gen_grid_row = torchvision.utils.make_grid(gen, nrow=gen.shape[0]) # * 0.5 + 0.5
        grid_rows.append(gen_grid_row)

    grid = torch.cat(grid_rows, dim=1)
    return grid    

def save_image(args, global_step, dir, images):
    dir_path = f'train_result/{args.run_id}/{dir}'
    os.makedirs(dir_path, exist_ok=True)
    
    sample_image = make_grid_image(images).detach().cpu().numpy().transpose([1,2,0]) * 255
    cv2.imwrite(f'{dir_path}/{str(global_step).zfill(8)}.jpg', sample_image[:,:,::-1])


if __name__ == "__main__":

    # load config
    config_path = "configs/train_configs.yaml"
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
