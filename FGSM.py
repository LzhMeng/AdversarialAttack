import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

# 加载数据集
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# 加载模型
model = resnet18(pretrained=False)
model = model.to('cuda')
model.load_state_dict(torch.load('resnet18.pth'))
model.eval()

# 选择攻击样本
x, y = next(iter(trainloader))
x = x.cuda()
y = y.cuda()

# 生成对抗样本
epsilon = 0.01
x_adv = x + epsilon * torch.randn_like(x)

# 验证对抗样本的有效性
with torch.no_grad():
    output = model(x_adv)
    pred = output.argmax(dim=1)

print(f'正确答案：{y}, 预测结果：{pred}')
