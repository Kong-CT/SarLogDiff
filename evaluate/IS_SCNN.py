



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from scipy.stats import entropy
import numpy as np
import os
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
class MSSNet(nn.Module):
    def __init__(self):
        super(MSSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 2, 256)  
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

      
        spp_1_1 = F.adaptive_avg_pool2d(x, 1)
        power_2_1 = F.adaptive_avg_pool2d(x ** 2, 1)
        power_1_1 = spp_1_1 ** 2

       
        sub_1 = power_2_1 - power_1_1

  
        concat1 = torch.cat((sub_1, spp_1_1), dim=1)

   
        concat1 = concat1.view(concat1.size(0), -1)  
        x = F.relu(self.fc1(concat1))
        x = self.fc2(x)
        return x

def inception_score(imgs, model, batch_size=32, splits=10):
    """计算给定图像数据集的Inception Score."""
    N = len(imgs)

    dataloader = DataLoader(imgs, batch_size=batch_size)
    
    preds = np.zeros((N, 7))#classnums

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            preds[i * batch_size:i * batch_size + pred.size(0)] = F.softmax(pred, dim=1).cpu().numpy()

   
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))  
        split_scores.append(np.exp(np.mean(scores)))

 
    return np.mean(split_scores), np.std(split_scores)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSSNet().to(device)
model.load_state_dict(torch.load('/SCNN.pth'))
model.eval()


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class GeneratedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"The directory {root_dir} does not exist.")
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.tif')]  # 假设生成的图像是PNG格式
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

generated_images = GeneratedImageDataset(root_dir='data/generated_images', transform=transform)


mean_is, std_is = inception_score(generated_images, model, batch_size=32, splits=10)

print(f"Inception Score: {mean_is:.2f} ± {std_is:.2f}")
