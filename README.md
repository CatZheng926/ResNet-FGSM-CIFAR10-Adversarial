# ResNet-FGSM-CIFAR10-Adversarial
An exploration of adversarial attacks and defenses on the CIFAR-10 dataset using the ResNet architecture and the Fast Gradient Sign Method (FGSM).


-----

这是一个利用 ResNet-18 和 FGSM 算法生成对抗样本的 Python 项目。本项目旨在探索**对抗攻击的原理**，并展示如何通过微小的扰动使深度学习模型产生错误的预测。

-----

### ✨ 主要特性

* **模型实现**：包含一个在 CIFAR-10 上预训练的 ResNet-18 模型。
* **对抗攻击**：实现了经典的 **FGSM** 算法来生成对抗样本。
* **可视化对比**：代码结构清晰，方便你直观地对比原始图像、对抗扰动和最终的对抗样本。
* **可解释性**：通过可视化扰动，可以更好地理解模型对哪些特征敏感，从而提高模型的可解释性。

### 🚀 快速开始

#### 1\. 克隆仓库

打开你的终端或 Git Bash，运行以下命令：
```bash
git clone 
````

#### 2\. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3\. 运行模型

  * **生成对抗样本并评估**：

    ```bash
    python FGSM1.py
    ```
    
    ```bash
    python test_one.py
    ```

    此脚本将自动加载预训练的 ResNet-18 模型，对 CIFAR-10 数据集中的图像生成对抗样本，并评估攻击的成功率。

### 🖼️ 结果展示



