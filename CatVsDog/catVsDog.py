# https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/submissions
# 1. data
# 2. model
# 3. train
# 4. evaluate

import torch    # pytorch
import numpy as np  # 数据处理
import os   # 文件操作

from torch.utils.data import Dataset, DataLoader    # 数据集
import torchvision  # pytorch 图像
import torch.nn as nn   # 神经网络
import torchvision.transforms as transforms # 数据增强
import matplotlib.pyplot as plt # 绘图
from PIL import Image   # 图像处理

## 准备工作
# device config
# GPU是否可用，后续可以据此判断是否可以加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
# 学习用的超参数
num_epoches = 1
batch_size = 64
learning_rate = 0.01

## dataset
# 自定义的数据集类，用作处理读入数据，必须实现以下三个方法，因为生成dataloader要用，你想想，如果没有这三个方法能够做batchsize、shuffle等操作吗
# 初步解释：https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# 深入解释dataloader：https://pytorch.org/docs/stable/data.html
# __init__()用于初始化，先调用父类的__init__()
# __len__()返回数据集中的样本个数
# __getitem__()根据索引返回一个样本（包括特征向量和标签）
class CDDataset(Dataset):
    def __init__(self, dir, mode='train', transforms=None) -> None:
        super().__init__()
        
        self.dir = dir
        self.files = os.listdir(dir) # list of files' name
        self.transforms = transforms
        self.mode = mode
        
        self.labels = []
        if mode == "train":
            for img in self.files:
                # dog->1, cat->0
                if "dog" in img:
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dir, self.files[index])) # Open index_th image
        if self.transforms:
            img = self.transforms(img)
        
        if self.mode == 'train':
            return img, self.labels[index]
        else:
            return img

# 数据增强，mean和std是归一化是三个通道用的均值和标准差，
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])
data_transforms = {
    'train': transforms.Compose([ # 图片操作在ToTensor()之前
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# 实例化数据集
train_dir = "./train"
test_dir = './test'
train_dataset = CDDataset(dir=train_dir, mode='train', transforms=data_transforms['train'])
# 数据集划分train和test
# 注意本来是：train、valuation、test三个，train作训练、valuation做超参数和训练过程中评估（有时没有）、test做最终评估
# generator保证每固定随机种子
# https://pytorch.org/docs/stable/data.html
# attention: np.int32 for length
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [np.int32(0.85*len(train_dataset)), np.int32(0.15*len(train_dataset))], generator=torch.Generator().manual_seed(0))

# 生成dataloader，如果没有实现三种dataset三种方法，则会报错
# test不必做shuffle，train才有必要做shuffle
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## model
# fine tuning pytorch tutorial:https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
class model(nn.Module):
    def __init__(self, num_classes=2):
        super(model, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        # set params freezed
        for param in self.vgg16.parameters():
            param.requires_grad = False
        in_features = self.vgg16.classifier[6].in_features
        # redefine the 6th classifier
        self.vgg16.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes)
        
    def forward(self, x):
        x.to(device)
        x = self.vgg16(x)
        # 这里不用加softmax，因为在交叉熵损失中会自动用softmax
        return x

## train
model = model().to(device)	# 很重要
print(model)
criterion = nn.CrossEntropyLoss()	# 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)	# adam优化算法

# 每次epoch需要多少次iteration
n_total_steps = len(train_loader)
print("Begin to train")
for epoch in range(num_epoches):
    for i, (imgs, labels) in enumerate(train_loader):
		# CPU和GPU上的tensor不能做运算，GPU只能算显存内的数据
        imgs, labels = imgs.to(device), labels.to(device)
		
		# 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)
		
		# 反向传播
		# zero_grad()必须，否则会把每次迭代的梯度都加起来
        optimizer.zero_grad()
        loss.backward()		# 逻辑反向
        optimizer.step()	# 代数反向
		
		# 训练时打印epoch、iteration、loss信息
        if (i+1)%100 == 0:
            print(f'Epoch {epoch+1}, Step {i+1}/{n_total_steps}, Loss:{loss.item():.4f}')

print("Finished Training")

with torch.no_grad():	# 不计算梯度
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
	
	# 使用testloader中的batch_size数据评估
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predictions = torch.max(outputs, 1)	# 返回value和index
        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()
    
    acc = 100.0 * n_correct / n_samples
    print(f"accuracy = {acc}")