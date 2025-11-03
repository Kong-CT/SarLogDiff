import torch
from torch import nn

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        super().__init__()
        self.label_emb = nn.Sequential(
            nn.Embedding(num_labels + 1, dim),
            nn.Linear(dim, dim),
            nn.SiLU()
        )
        
        self.img_emb = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, padding=1),  
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim // 4, dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim, dim),  
            nn.SiLU()
        )

    def forward(self, labels, img_features):
        
        if img_features.dim() == 3:  
            img_features = img_features.unsqueeze(1)  
        
        label_emb = self.label_emb(labels)
        img_emb = self.img_emb(img_features)
        combined = torch.cat([label_emb, img_emb], dim=1)
        return self.fusion(combined)
