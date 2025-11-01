---

# Complex-RAN-NN

> **Complex-RAN-NN** 是一个基于复数域 (complex-valued) 神经网络的高性能图像恢复框架。
> 核心模块 **ComplexRestormer** 将复数表示与 Transformer 架构融合，用于处理具有幅度与相位信息的复杂视觉信号。

---

## 项目简介

传统图像恢复网络大多基于**实值（real-valued）**神经网络，只能捕捉强度（amplitude）信息，忽略了信号中的**相位（phase）**特征。而在成像、通信、声学、雷达、SAR 等场景中，输入数据往往为复数形式，包含丰富的相位与振幅信息。

**Complex-RAN-NN** 旨在建立一个能够**直接处理复数数据（`dtype=complex64`）**的网络架构，借助复数卷积、复数激活和复数注意力机制，实现端到端的图像/信号恢复任务。

其中，核心模型 **ComplexRestormer** 继承自 Restormer 架构（CVPR 2022, Zamir et al.），并将其推广到复数域，显著增强了模型在频域、干涉、散射等物理相关任务中的表达能力。

---

## 功能与特性

* **复数域建模**：
  支持原生 `complex64` 张量输入，网络层可执行复数卷积、复数激活与复数归一化操作。

* **ComplexRestormer 架构**：
  在 Restormer 的基础上，引入复数版多头注意力（Complex Multi-Dconv Head Attention）与复数门控前馈网络（Complex GDFN）。

* **线性复杂度注意力**：
  通过通道维度注意力（Channel-wise Attention）与深度可分卷积实现高效的特征交互，可支持高分辨率输入。

* **多尺度结构 + 残差连接**：
  采用 Encoder-Decoder 式架构，结合多层 skip 连接与复数残差路径，促进浅层与深层特征融合。

* **通用性强**：
  可应用于复数图像去噪、去模糊、重建、雷达图像增强、声场恢复等多种任务。

---

## 代码结构

```
complex-RAN-NN/
├── complex_layer.py         # 定义复数卷积层、激活、归一化等
├── complex_restormer.py     # ComplexRestormer 模型主体
├── utils/                   # 训练与数据处理辅助模块（如有）
└── README.md
```

---

## ComplexRestormer 模型详解

### 背景：Restormer

Restormer（CVPR 2022 – Zamir et al.）是一个专为高分辨率图像恢复任务设计的高效 Transformer 模型，其核心思想包括：

* **多 Dconv 头转置注意力 (MDTA)**：
  在通道维度上执行注意力计算，避免空间维度 O(N²) 复杂度。

* **门控深度卷积前馈网络 (GDFN)**：
  在前馈层中引入门控机制与深度可分卷积，实现空间特征增强。

* **U-Net 风格结构**：
  通过下采样与上采样模块构建多尺度特征流，结合跳接连接 (skip connections) 提高重建质量。

ComplexRestormer 在此基础上，将所有关键运算拓展至**复数域**，从而保留信号的完整物理特性。

---

### ComplexRestormer：复数域扩展

#### a) 复数输入与特征流

ComplexRestormer 直接接受复数类型输入：

```python
# 输入形状
(B, C, H, W)  # dtype=torch.complex64
```

输入张量的每个通道包含复数特征（实部与虚部在底层内存中一体存储），网络中的所有运算（卷积、注意力、激活、残差）均定义在复数域上。

#### b) 复数卷积与激活

`complex_layer.py` 定义了复数卷积 (ComplexConv2d)：
[
y = (W_r * x_r - W_i * x_i) + j (W_r * x_i + W_i * x_r)
]
同时提供复数激活（如 ComplexReLU 或 ComplexPReLU），在保持幅度连续性的同时避免信息丢失。

#### c) 复数注意力机制 (Complex Attention)

在每个 Transformer 块中，复数注意力机制将 Q、K、V 扩展至复数空间：
[
Q = Q_r + jQ_i, \quad K = K_r + jK_i, \quad V = V_r + jV_i
]
注意力权重计算与特征融合通过复数乘法完成，以捕捉幅度与相位依赖关系：
[
A = \text{Softmax}(\text{Re}(QK^H)) ,\quad Y = A \cdot V
]
这种设计允许模型在**相位变化**敏感的场景下保持稳定性能。

#### d) 复数 Gated-Dconv FFN

前馈网络 (Feed-Forward Network, FFN) 部分引入复数门控机制：
[
\text{FFN}(x) = \text{DConv}(x) \odot \sigma(\text{DConv}(x))
]
其中 DConv 为复数深度卷积层，用以增强局部结构信息。

#### e) 模型结构概览

```
Input (complex64)
   ↓
Shallow ComplexConv
   ↓
[Complex Transformer Block × N]
   ↓
Complex Reconstruction Head
   ↓
Output (complex64)
```

可根据任务自由调整块数、通道数与嵌入维度。

---

### 多尺度与残差连接

ComplexRestormer 沿用 Restormer 的 U-Net 多尺度架构，通过复数下采样/上采样模块获取多层级特征表示，并在解码阶段通过复数跳接连接融合浅层与深层信息。

复数残差连接定义为：
[
y = x + f(x)
]
其中 (x, f(x)) 均为复数特征张量，确保相位与幅度在传播中保持一致。

---

### 模型优点

* **复数域信息保持**：无需分离实部与虚部，直接在 `complex64` 上操作，保留信号的相位一致性。
* **高效 Transformer 架构**：继承 Restormer 的线性复杂度与强大表示能力。
* **跨领域适用性**：适合处理雷达成像 (SAR)、MRI、超声、声场建模、无线通信信道重建等复数数据。
* **可扩展性强**：可作为通用复数 Transformer 骨干网络，用于后续复数神经网络研究。

---

## 📘 关键文件说明

| 文件                     | 功能                                      |
| ---------------------- | --------------------------------------- |
| `complex_layer.py`     | 定义复数卷积层、复数激活、复数批归一化与复数残差模块。             |
| `complex_restormer.py` | 核心模型 ComplexRestormer：复数版 Restormer 实现。 |
| `utils/` *(可选)*        | 数据加载、指标计算、图像可视化工具等。                     |
| `README.md`            | 项目说明文件。                                 |

---

## 示例伪代码

```python
import torch
from complex_restormer import ComplexRestormer

# 输入为复数类型
x = torch.randn(1, 3, 256, 256, dtype=torch.complex64).cuda()

model = ComplexRestormer(in_channels=3, out_channels=3, embed_dim=64, num_blocks=8).cuda()
y = model(x)

print(y.dtype)  # torch.complex64
print(y.shape)  # (1, 3, 256, 256)
```

---

