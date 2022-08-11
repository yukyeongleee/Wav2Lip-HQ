import torch
import os

        
def load_checkpoint(args, model, optimizer, name):
    ckpt_step = "latest" if args.ckpt_step is None else args.ckpt_step
    ckpt_path = f'{args.save_root}/{args.ckpt_id}/ckpt/{name}_{ckpt_step}.pt'
    
    ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt_dict['model'], strict=False)
    optimizer.load_state_dict(ckpt_dict['optimizer'])

    return ckpt_dict['global_step']

def save_checkpoint(args, model, optimizer, name, global_step):
    
    ckpt_dict = {}
    ckpt_dict['global_step'] = global_step
    ckpt_dict['model'] = model.state_dict()
    ckpt_dict['optimizer'] = optimizer.state_dict()

    dir_path = f'./{args.save_root}/{args.run_id}/ckpt'
    os.makedirs(dir_path, exist_ok=True)
    
    ckpt_path = f'{dir_path}/{name}_{global_step}.pt'
    torch.save(ckpt_dict, ckpt_path)

    latest_ckpt_path = f'{dir_path}/{name}_latest.pt'
    torch.save(ckpt_dict, latest_ckpt_path)
        