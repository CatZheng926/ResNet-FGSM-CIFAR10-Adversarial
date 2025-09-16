# ResNet-FGSM-CIFAR10-Adversarial
An exploration of adversarial attacks and defenses on the CIFAR-10 dataset using the ResNet architecture and the Fast Gradient Sign Method (FGSM).
# Adversarial-Sample-Generation

A **ResNet-18** model and a **Fast Gradient Sign Method (FGSM)** implementation in **PyTorch** for generating adversarial examples on the **CIFAR-10** dataset.

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
git clone [https://github.com/YourGitHubUsername/adversarial-sample-generation.git](https://github.com/YourGitHubUsername/adversarial-sample-generation.git)
cd adversarial-sample-generation
````

#### 2\. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3\. 运行模型

  * **生成对抗样本并评估**：

    ```bash
    python main.py
    ```

    此脚本将自动加载预训练的 ResNet-18 模型，对 CIFAR-10 数据集中的图像生成对抗样本，并评估攻击的成功率。

### 🖼️ 结果展示

#### 原始图像、扰动和对抗样本对比

\<p align="center"\>
\<img src="https://www.google.com/search?q=https://i.imgur.com/example\_original.png" alt="Original Image" width="200"/\>
\<img src="https://www.google.com/search?q=https://i.imgur.com/example\_noise.png" alt="Adversarial Noise" width="200"/\>
\<img src="https://www.google.com/search?q=https://i.imgur.com/example\_adversarial.png" alt="Adversarial Sample" width="200"/\>
\</p\>
\<p align="center"\>
\<sub\>从左至右：原始图像，对抗扰动（放大），对抗样本\</sub\>
\</p\>

-----

#### 示例

| 原始图像 (标签: Cat) | 对抗样本 (预测: Dog) |
|:---:|:---:|
| \<img src="https://www.google.com/search?q=https://i.imgur.com/example\_cat.png" width="300" alt="Original Image of a Cat"\> | \<img src="https://www.google.com/search?q=https://i.imgur.com/example\_adversarial\_cat.png" width="300" alt="Adversarial Image predicted as a Dog"\> |
| 原始图像标签: 7 (Cat) | 模型预测标签: 5 (Dog) |

```
```
