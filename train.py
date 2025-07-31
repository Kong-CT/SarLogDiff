import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
from collections import defaultdict
import torch
from torch import nn
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from Scheduler import GradualWarmupScheduler
from dataloader_cifar import load_data, transback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from torch import Tensor
from PIL import Image
import numpy as np
from resize_right import resize
# 设置训练和验证数据集目录

data_dir = ''
val_data_dir = ''  

def train(params: argparse.Namespace):
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0, 'please re-set your genbatch!!!'
     
    # 初始化分布式进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4018'
    init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)  
    
    local_rank = get_rank()
    device = torch.device("cuda", local_rank)
    

    dataloader, sampler = load_data(params.batchsize, params.numworkers, data_dir)
    val_dataloader, _ = load_data(params.batchsize, params.numworkers, val_data_dir)
    
    first_batch = next(iter(dataloader))
    images, labels = first_batch


    print(f"Training Dataset size: {len(dataloader.dataset)}, Batch size: {params.batchsize}, Total steps: {len(dataloader)}")
    print(f"Validation Dataset size: {len(val_dataloader.dataset)}, Batch size: {params.batchsize}, Total steps: {len(val_dataloader)}")
    
 
    net = Unet(
        in_ch=params.inch,
        mod_ch=params.modch,
        out_ch=params.outch,
        ch_mul=params.chmul,
        num_res_blocks=params.numres,
        cdim=params.cdim,
        use_conv=params.useconv,
        droprate=params.droprate,
        dtype=params.dtype
    )
    
    cemblayer = ConditionalEmbedding(5, params.cdim, params.cdim).to(device)
    
    lastpath = os.path.join(params.moddir, 'last_epoch.pt')
    if os.path.exists(lastpath):
        lastepc = torch.load(lastpath)['last_epoch']
        checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
    
    betas = get_named_beta_schedule(num_diffusion_timesteps=params.T)
    diffusion = GaussianDiffusion(
        dtype=params.dtype,
        model=net,
        betas=betas,
        w=params.w,
        v=params.v,
        device=device, 
     
    )
    
    diffusion.model = DDP(diffusion.model, device_ids=[local_rank],output_device=local_rank)
    cemblayer = DDP(cemblayer, device_ids=[local_rank], output_device=local_rank )
    
    optimizer = torch.optim.AdamW(
        itertools.chain(diffusion.model.parameters(), cemblayer.parameters()),
        lr=params.lr,
        weight_decay=1e-4
    )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=params.epoch,
        eta_min=0,
        last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=params.multiplier,
        warm_epoch=params.epoch // 5,
        after_scheduler=cosineScheduler,
        last_epoch=lastepc
    )
    
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    
    def prepare_conditions(dataset_path, clsnum, genbatch_per_cls, device):
        print(f"Preparing conditions for classes: {clsnum}")
        cond_loader, _ = load_data(1, 0, dataset_path)
        dataset = cond_loader.dataset
        
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        
        print(f"Class indices: {class_indices}")
        conditions = []
        labels = []
        
        for cls in range(clsnum):
            indices = class_indices.get(cls, [])
            if not indices:
                raise ValueError(f"Class {cls} not found in training set")
            print(f"Selected indices for class {cls}: {indices[:5]}")  
            for _ in range(genbatch_per_cls):
                rand_idx = np.random.choice(indices)
                sample, _ = dataset[rand_idx]
                sample = sample.unsqueeze(0).to(device)
                print(f"Class {cls} - Selected image index: {rand_idx}")
                with torch.no_grad():
                    downsampled = resize(sample, out_shape=(32, 32))
                    resized = resize(downsampled, out_shape=(64, 64))
                conditions.append(resized)
                labels.append(cls)

        return torch.cat(conditions, dim=0), torch.tensor(labels).to(device)
    
    cnt = torch.cuda.device_count()
    
    for epc in range(lastepc, params.epoch):
        diffusion.model.train()
        cemblayer.train()
        sampler.set_epoch(epc)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        
        with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            for img, lab in tqdmDataLoader:
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                lab = lab.to(device)
                
                with torch.no_grad():
                    downsampled = resize(x_0, out_shape=(32, 32))
                    resized = resize(downsampled, out_shape=(64, 64))

                cemb = cemblayer(lab, resized )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                cemb[np.where(np.random.rand(b) < params.threshold)] = 0
                loss = diffusion.trainloss(x_0, cemb=cemb)
                loss.backward()
                optimizer.step()
                
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss": loss.item(),
                        "batch per device": x_0.shape[0],
                        "img shape": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
        
        warmUpScheduler.step()

        if (epc + 1) % params.interval == 0:
            diffusion.model.eval()
            cemblayer.eval()
            all_samples = []
            each_device_batch = params.genbatch // cnt
            val_loss = 0
            conditions, lab = prepare_conditions(
                dataset_path=data_dir,
                clsnum=params.clsnum,
                genbatch_per_cls=params.genbatch // params.clsnum,
                device=device
            )
        
            with torch.no_grad():
                lab = lab.to(device)
                cemb = cemblayer(lab, conditions)

                genshape = (each_device_batch, 1, 64, 64)
                if params.ddim:
                    generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb=cemb)
                else:
                    generated = diffusion.sample(genshape, cemb=cemb)

           
            
                img = transback(generated)
                
            
                img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 1, 64, 64).contiguous()

                gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]
                all_gather(gathered_samples, img)
                all_samples.extend([img for img in gathered_samples])
                samples = torch.concat(all_samples, dim=1).reshape(params.genbatch, 1, 64, 64)

                if local_rank == 0:
                    epoch_dir = os.path.join(params.samdir, f'epoch_{epc + 1}')
                    os.makedirs(epoch_dir, exist_ok=True)
                    for class_idx in range(params.clsnum):
                        start_idx = class_idx * (params.genbatch // params.clsnum)
                        end_idx = (class_idx + 1) * (params.genbatch // params.clsnum)
                        class_samples = samples[start_idx:end_idx]
                        for i, sample in enumerate(class_samples):
                            sample_np = sample.squeeze().cpu().numpy().astype(np.uint8) 
                            img_pil = Image.fromarray(sample_np, mode='L') 
                            print(f"Saving image for class {class_idx} at {epoch_dir}/{class_idx}_{i + 1}.png")
                            img_pil.save(os.path.join(epoch_dir, f'{class_idx}_{i + 1}.png'))
                
            checkpoint = {
                'net': diffusion.model.module.state_dict(),
                'cemblayer': cemblayer.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': warmUpScheduler.state_dict()
            }
            torch.save({'last_epoch': epc + 1}, os.path.join(params.moddir, 'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc + 1}_checkpoint.pt'))
        
        torch.cuda.empty_cache()
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize', type=int, default=20, help='batch size per device for training Unet model')
    parser.add_argument('--numworkers', type=int, default=2, help='num workers for training Unet model')
    parser.add_argument('--inch', type=int, default=1, help='input channels for Unet model') 
    parser.add_argument('--modch', type=int, default=64, help='model channels for Unet model')
    parser.add_argument('--T', type=int, default=1000, help='timesteps for Unet model')
    parser.add_argument('--outch', type=int, default=1, help='output channels for Unet model')
    parser.add_argument('--chmul', type=list, default=[1, 2, 2, 2], help='architecture parameters training Unet model')
    parser.add_argument('--numres', type=int, default=2, help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim', type=int, default=6, help='dimension of conditional embedding')
    parser.add_argument('--useconv', type=bool, default=True, help='whether use convlution in downsample')
    parser.add_argument('--droprate', type=float, default=0.1, help='dropout rate for model')
    parser.add_argument('--dtype', default=torch.float32, help='data type for model')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--w', type=float, default=3, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v', type=float, default=0.3, help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch', type=int, default=1500, help='epochs for training')
    parser.add_argument('--multiplier', type=float, default=2.5, help='multiplier for warmup')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for classifier-free guidance')
    parser.add_argument('--interval', type=int, default=5, help='epoch interval between two evaluations')
    parser.add_argument('--moddir', type=str, default='model', help='model addresses')
    parser.add_argument('--samdir', type=str, default='sample', help='sample addresses')
    parser.add_argument('--genbatch', type=int, default=5, help='batch size for sampling process')
    parser.add_argument('--clsnum', type=int, default=5, help='num of label classes')
    parser.add_argument('--num_steps', type=int, default=50, help='sampling steps for DDIM')
    parser.add_argument('--eta', type=float, default=0, help='eta for variance during DDIM sampling process')
    parser.add_argument('--select', type=str, default='linear', help='selection strategies for DDIM')
    parser.add_argument('--ddim', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help='whether to use ddim')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
 