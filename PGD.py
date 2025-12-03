import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import numpy as np
from PIL import ImageDraw, ImageFont, Image

# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)
model.eval()  # 设置为评估模式

# 定义图像预处理流程
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载并预处理图像
img = Image.open('./images/ILSVRC2012_val_00000248.png').convert('RGB')
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # 增加batch维度

# 使用GPU（如果可用）
if torch.cuda.is_available():
    model = model.cuda()
    input_batch = input_batch.cuda()
def get_model_out(model, input_batch):
    # 禁用梯度计算以提高效率
    with torch.no_grad():
        output = model(input_batch)
    
    # 获取预测结果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # 加载ImageNet类别标签
    with open("imagenet_class_index.json", "r") as f:
        labels = json.load(f)
    # 输出前5个预测结果
    for i in range(5):
        print(f"\tTop {i+1}: {labels[str(top5_catid[i].item())][1]}: {top5_prob[i].item():.4f}")
    return top5_catid[0],labels[str(top5_catid[0].item())][1],top5_prob[0].item()

# 定义 PGD 攻击函数
def pgd_attack(model, images, labels, criterion,epsilon=8/255, alpha=2/255,iterations=10,random_start=True):
    """
    pgd 攻击函数
    :param model: 受害者模型
    :param image: 原始输入图像
    :param labels: 标签
    :param criterion: 损失函数
    :param epsilon: 扰动强度
    :param alpha: 单步迭代步长
    :param iterations: 迭代次数(步数)
    :param random_start: 是否随机初始化扰动
    :return: 对抗样本
    """
    # 克隆输入
    perturbed_images = images.clone()

     # 是否初始化对抗样本
    if random_start:
        # 随机初始化扰动（均匀分布）
        perturbed_images =  perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
        # perturbed_images = torch.clamp(perturbed_images, 0, 1).detach() #  # 约束到合法像素范围[0,1]

    # 多次迭代生成对抗样本
    for _ in range(iterations):
        perturbed_images.requires_grad = True # 计算输入样本的梯度
        
        # 计算损失，反向传播
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

         # 生成对抗样本
        with torch.no_grad():
            data_grads = perturbed_images.grad.data  # 获取到关于样本的梯度
            perturbed_images = perturbed_images + alpha * data_grads.sign()

             # 对抗样本投影到ε邻域内
            perturbed_images = torch.clamp(perturbed_images, images - epsilon, images + epsilon)
    
            
            
    return perturbed_images.detach() # 返回不包含梯度信息的对抗样本

def save_adv(x_adv):
    # 保存对抗样本为图像文件（反归一化并保存）
    inv_mean = [0.485, 0.456, 0.406]
    inv_std = [0.229, 0.224, 0.225]
    adv_img = x_adv.detach().cpu().squeeze(0).clone()  # CHW
    for c in range(3):
        adv_img[c] = adv_img[c] * inv_std[c] + inv_mean[c]
    adv_img = torch.clamp(adv_img, 0.0, 1.0)
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(adv_img)
    pil_img.save("adv_image.png")
    
def save_image(input_batch,x_adv,label_o,top1_o_prob,label_a,top1_a_prob):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    to_pil = transforms.ToPILImage()
    def tensor_to_pil(tensor):
        t = tensor.clone().cpu().squeeze(0)
        for c in range(3):
            t[c] = t[c] * std[c] + mean[c]
        t = t.clamp(0, 1)
        return to_pil(t)
    
    orig_pil = tensor_to_pil(input_batch)
    adv_pil = tensor_to_pil(x_adv)
    
    # 3) 创建并列图像，在下方写上第一个标签和置信度
    w, h = orig_pil.size
    padding_bottom = 30
    canvas = Image.new("RGB", (w * 2, h + padding_bottom), "white")
    canvas.paste(orig_pil, (0, 0))
    canvas.paste(adv_pil, (w, 0))
    
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    
    text_left = f"{label_o}: {top1_o_prob:.4f}"
    text_right = f"{label_a}: {top1_a_prob:.4f}"
    
    left, top, right, bottom = font.getbbox(text_left)
    tw_left, th = right - left, bottom - top
    
    left, top, right, bottom = font.getbbox(text_right)
    tw_right, _ = right - left, bottom - top
    
    draw.text(((w - tw_left) / 2, h + 5), text_left, fill="black", font=font)
    draw.text((w + (w - tw_right) / 2, h + 5), text_right, fill="black", font=font)
    
    # 4) 保存图像
    out_path = "orig_vs_adv.png"
    canvas.save(out_path)
    print(f"Saved comparison to {out_path}")
print(f"Ori label:")
label,tex_o,prob_o = get_model_out(model, input_batch)
x_adv = pgd_attack(model, input_batch, label.unsqueeze(0), nn.CrossEntropyLoss())

save_adv(x_adv)
print(f"After Attack label:")
label_a,tex_a,prob_a = get_model_out(model, x_adv)

save_image(input_batch,x_adv,tex_o,prob_o,tex_a,prob_a)
