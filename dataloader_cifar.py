
import math
import numpy as np
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
import torch
from torch import nn
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler

def log_transform(tensor: Tensor, eps: float = 1e-6) -> Tensor:
    # 确保张量的范围在 [0, 255]，并应用 log(x + 1) 变换
    return torch.log(tensor + 1.0)  # 应用 log(x + 1)，值将限制在 [0, 5.545]

def global_normalize(tensor) :
    # 归一化到 [0, 1]
    """全局归一化到 [0, 1]"""
    min_val = torch.log(torch.tensor(1.0))  # 结果为 0.0
    max_val = torch.log(torch.tensor(256.0))
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized 

class CustomTransform:
    def __init__(self, batchsize):
        self.count = 0
        self.max_prints = batchsize
        
        self.base_transform = transforms.Compose([
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS),      
            transforms.Grayscale(num_output_channels=1)  
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def __call__(self, img):
     
        if self.count < self.max_prints:
            img_np = np.array(img)
            print(f"\nCustomTransform: 图像编号 {self.count}")
            print("原始图像像素范围:", img_np.min(), "-", img_np.max())
            self.count += 1


        img_transformed = self.base_transform(img)

      
        img_tensor = torch.from_numpy(np.array(img_transformed)).float()  
        img_tensor = img_tensor.unsqueeze(0) 

       
        img_log = log_transform(img_tensor)


        img_normalized = global_normalize(img_log)

        return img_normalized  # 输出格式 [C, H, W]，保持单通道输出

def load_data(batchsize: int, numworkers: int, data_dir: str) -> tuple[DataLoader, DistributedSampler]:
 
    debug_transformer = CustomTransform(batchsize)
 
    trans = transforms.Compose([
        transforms.Lambda(lambda img: debug_transformer(img)),
        
    ])

    data_train = ImageFolder(
        root=data_dir,
        transform=trans
    )

    sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
        data_train,
        batch_size=batchsize,
        num_workers=0,
        sampler=sampler,
        drop_last=True
    )
    return trainloader, sampler


def transback(normalized_tensor) -> Tensor:


   max_val = torch.log(torch.tensor(256.0))
 
   y = normalized_tensor * max_val
   
   restored_tensor = torch.exp(y) - 1
   return restored_tensor
