import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from PIL import Image
from glob import glob
from scipy.linalg import sqrtm
from tqdm import tqdm

# 计算两个多元高斯分布之间的Frechet距离
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Calculate the Frechet Distance between two multivariate Gaussians."""
    covmean = sqrtm(sigma1 @ sigma2)


    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob(f"{image_dir}/*.jpg")+ glob(f"{image_dir}/*.tif")  
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  
        if self.transform:
            image = self.transform(image)
        return image, 0  


def compute_statistics_of_images(images, model, batch_size=32):
    """Compute the statistics (mean, covariance) of the activations of the images."""
    model.eval()
    act = []
    with torch.no_grad():
        for img_batch in tqdm(DataLoader(images, batch_size=batch_size)):
            img_batch = img_batch[0].to(device)
            pred = model(img_batch)
            if pred.ndim == 4: 
                pred = pred.view(pred.size(0), -1)  
            elif pred.ndim == 2:  
                pred = pred
            else:
                raise ValueError(f"Unexpected shape of predictions: {pred.shape}")
            act.append(pred.cpu().numpy())
    
    act = np.concatenate(act, axis=0)
    print(f"Shape of the activations: {act.shape}")  


    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    
    print(f"Shape of mean: {mu.shape}, Shape of covariance: {sigma.shape}")  
    return mu, sigma




def calculate_fid(real_images, generated_images, batch_size=32):
    """Calculate FID score between real and generated images."""

    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = nn.Identity()  


    mu1, sigma1 = compute_statistics_of_images(real_images, model, batch_size)
    mu2, sigma2 = compute_statistics_of_images(generated_images, model, batch_size)
    
    # 计算FID分数
    fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_score

# 主函数
if __name__ == "__main__":
 
    transform = Compose([
        Resize((299, 299)), 
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])

  
    real_images = CustomImageDataset(image_dir='data/real_images', transform=transform)
    generated_images = CustomImageDataset(image_dir='data/generated_images', transform=transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fid_score = calculate_fid(real_images, generated_images)
    print(f"FID score: {fid_score}")
