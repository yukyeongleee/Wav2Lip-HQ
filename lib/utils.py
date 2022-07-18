import torch
import torch.nn as nn
import torchvision
import cv2
import os
import glob

def get_all_images(dataset_root_list):
    image_path_list = []
    image_num_list = []

    for dataset_root in dataset_root_list:
        imgpaths_in_root = glob.glob(f'{dataset_root}/*.*g')

        for root, dirs, _ in os.walk(dataset_root):
            for dir in dirs:
                imgpaths_in_root += glob.glob(f'{root}/{dir}/*.*g')

        image_path_list.append(imgpaths_in_root)
        image_num_list.append(len(imgpaths_in_root))

    return image_path_list, image_num_list

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
        
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def update_net(optimizer, loss):
    optimizer.zero_grad()  
    loss.backward()   
    optimizer.step()  

def setup_ddp(gpu, ngpus_per_node):
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)

def save_image(args, global_step, dir, images):
    dir_path = f'train_result/{args.run_id}/{dir}'
    os.makedirs(dir_path, exist_ok=True)
    
    sample_image = make_grid_image(images).detach().cpu().numpy().transpose([1,2,0]) * 255
    cv2.imwrite(f'{dir_path}/{str(global_step).zfill(8)}.jpg', sample_image[:,:,::-1])

def make_grid_image(images_list):
    grid_rows = []

    for images in images_list:
        images = images[:8] # Drop images if there are more than 8 images in the list
        grid_row = torchvision.utils.make_grid(images, nrow=images.shape[0]) * 0.5 + 0.5
        grid_rows.append(grid_row)

    grid = torch.cat(grid_rows, dim=1)
    return grid

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag