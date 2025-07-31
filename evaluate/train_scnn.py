
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# export CUDA_VISIBLE_DEVICES=0

# 定义模型
class MSSNet(nn.Module):
    def __init__(self):
        super(MSSNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 2, 256) 
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # SPP 
        spp_1_1 = F.adaptive_avg_pool2d(x, 1)
        power_2_1 = F.adaptive_avg_pool2d(x ** 2, 1)
        power_1_1 = spp_1_1 ** 2

        # Subtraction 
        sub_1 = power_2_1 - power_1_1

        # Concat 
        concat1 = torch.cat((sub_1, spp_1_1), dim=1)

        
        concat1 = concat1.view(concat1.size(0), -1)  
        x = F.relu(self.fc1(concat1))
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        for filename in os.listdir(root_dir):
            if filename.endswith('.tif'):
                self.files.append(filename)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
      
        label = int(self.files[idx].split('_')[0])  
        return image, label


train_dataset = CustomDataset(root_dir='dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

val_dataset = CustomDataset(root_dir='/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()



num_runs = 10
num_epochs = 500
all_train_losses = []
all_val_accuracies = []

for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")
    model = MSSNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.01 * epoch))
     
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        print(f"Run {run+1}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
    
        scheduler.step()

       
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        save_dir = 'scnn_model'
        csv_file_path = os.path.join(save_dir, f'train_val_run_{run+1}.csv')
    
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Validation Accuracy'])
            for i in range(len(train_losses)):
                writer.writerow([i+1, train_losses[i], val_accuracies[i]])

   
        os.makedirs(save_dir, exist_ok=True)  
        if (epoch + 1) % 500 == 0:
          torch.save(model.state_dict(), os.path.join(save_dir, f'mssnet_model_weight_run_{run+1}_epoch_{epoch+1}.pth'))


    all_train_losses.append(train_losses)
    all_val_accuracies.append(val_accuracies)


mean_train_losses = np.mean(all_train_losses, axis=0)
mean_val_accuracies = np.mean(all_val_accuracies, axis=0)


sns.set(style="dark")

fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('Training Epoch')
ax1.set_ylabel('Average Training Loss', color=color)
ax1.plot(range(1, num_epochs + 1), mean_train_losses, label='Average Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Validation Accuracy', color=color)
ax2.plot(range(1, num_epochs + 1), mean_val_accuracies, label='Average Accuracy', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

fig.tight_layout()
plt.title('Average Training Loss and Validation Accuracy over 10 Runs')


model.eval()
correct = 0
total = 0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
results = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        for i in range(len(labels)):
            label = labels[i].item()
            prediction = predicted[i].item()
            results.append([label, prediction])
            class_correct[label] += (prediction == label)
            class_total[label] += 1

# 保存预测结果到CSV文件
with open('pre.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Actual', 'Predicted'])
    writer.writerows(results)

# 总体准确度
accuracy = 100 * correct / total
print(f'Overall Accuracy: {accuracy:.2f}%')

# 每类的准确度
text_str = f'Overall Accuracy: {accuracy:.2f}%\n'
for i in range(5):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        text_str += f'Accuracy of class {i}: {class_acc:.2f}%\n'
        print(f'Accuracy of class {i}: {class_acc:.2f}%')
    else:
        text_str += f'No samples for class {i}\n'
        print(f'No samples for class {i}')


plt.gcf().text(0.15, -0.3 , text_str, fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.8))


fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.4))


fig.savefig('scnn_model/model_average_training_validation_curve_10_runs_with_text.pdf',bbox_inches='tight')

plt.show()

