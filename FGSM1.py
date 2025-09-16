import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet import ResNet18
import numpy as np
import cv2

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = ResNet18().to(device)
model.load_state_dict(torch.load('D:/Resnet-18/model/net_031.pth'))
model.eval()  # 设置为评估模式

# CIFAR-10 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 预处理函数
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载并预处理图片
original_image = Image.open('D:/Resnet-18/dog1.jpg').convert('RGB')
image_path = 'D:/Resnet-18/dog1.jpg'  # 替换为你的图片路径
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度并移动到设备

# 设置目标标签
target_label = torch.tensor([5]).to(device)  # 目标标签（例如，类“狗”）

# 计算损失并生成对抗样本
image.requires_grad = True  # 允许对输入图像计算梯度

# 获取当前预测
output = model(image)
loss = torch.nn.CrossEntropyLoss()(output, target_label)

# 清除之前的梯度
model.zero_grad()

# 反向传播计算梯度
loss.backward()

# 获取输入图像的梯度
data_grad = image.grad.data

# 设置扰动强度（epsilon）
epsilon = 0.02  # 调整 epsilon 值以生成明显的对抗样本

# 生成对抗样本
adversarial_image = image + epsilon * data_grad.sign()
# 确保对抗样本在合法范围内
adversarial_image = torch.clamp(adversarial_image, 0, 1)

# 预测对抗样本
with torch.no_grad():
    adv_output = model(adversarial_image)
    _, adv_predicted = torch.max(adv_output.data, 1)

# 保存对抗样本
adversarial_image = epsilon * data_grad.sign()

# 将对抗样本数据从 GPU 移动到 CPU，并转换为 NumPy 数组
adversarial_image_np = adversarial_image.squeeze().cpu().detach().numpy()

#将原始图像保存为 NumPy 数组
original_array = np.array(original_image)

# 反归一化
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
adversarial_image_np = adversarial_image_np * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
adversarial_image_np = (adversarial_image_np * 255).astype('uint8')

# 确保数组的形状是 (32, 32, 3)
if adversarial_image_np.shape[0] == 3:
    adversarial_image_np = np.transpose(adversarial_image_np, (1, 2, 0))

# 创建 PIL 图像
adversarial_image_pil = Image.fromarray(adversarial_image_np)

# 调整对抗样本大小以匹配原始图像
original_image = Image.open(image_path).convert('RGB')
adversarial_image_resized = adversarial_image_pil.resize(original_image.size, Image.LANCZOS)
adversarial_array = np.array(adversarial_image_resized)

# 将扰动添加到原始图像
result_array = adversarial_array

# 确保结果在合法范围内
result_array = np.clip(result_array, 0, 255).astype(np.uint8)
#original_image = np.array(original_image)
#result_array = result_array + original_image

# 将结果数组转换回 PIL 图像
result_image = Image.fromarray(result_array)

# 保存结果图像
result_image.save('D:/Resnet-18/result_image.jpg')

# 保存调整后的对抗样本
#adversarial_image_resized.save('D:/Resnet-18/adversarial_image1.jpg')


# 输出结果
print(f'原始标签: {target_label.item()}')
print(f'对抗样本预测类别: {adv_predicted.item()}')