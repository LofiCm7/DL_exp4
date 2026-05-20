# 基于 CIFAKE 的 AIGC 图像检测 Baseline 代码讲义

## 超参数总览与修改方法

这一节放在文档最前面，作为实验时的“参数速查表”。当前项目中你可以调整的参数分为两类：

1. **命令行参数**
   - 不需要改源码。
   - 直接在运行 `train.py` 或 `test.py` 时通过 `--参数名 参数值` 修改。

2. **代码内固定参数**
   - 当前没有暴露为命令行参数。
   - 需要直接修改 Python 文件中的常量或函数实现。

下面按这两类分别说明。

### A. `train.py` 中可直接设置的超参数

训练脚本入口是：

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python train.py [参数]
```

当前支持的训练参数如下。

| 参数名 | 默认值 | 作用 | 典型修改方式 |
| --- | --- | --- | --- |
| `--data-root` | `data4` | 数据集根目录。训练脚本只会读取 `data_root/train`，并在内部再划分成训练集和验证集 | 当数据集目录变化时修改，例如 `--data-root ./data4` |
| `--output-dir` | `output/train_val` | 训练阶段结果输出目录，用于保存训练/验证曲线、最优模型、验证集混淆矩阵和训练摘要 | 建议每组实验单独设置，例如 `--output-dir output/train_val/resnet18_ratio10` |
| `--model` | `resnet18` | 选择 backbone 网络结构 | 可改为 `--model mobilenet_v2` |
| `--labeled-ratio` | `0.1` | 在训练划分中保留多少比例的带标签样本参与训练 | `1%` 用 `--labeled-ratio 0.01`，`10%` 用 `--labeled-ratio 0.1` |
| `--val-ratio` | `0.2` | 从原始 `data4/train` 中划出多少比例作为验证集 | 例如 `--val-ratio 0.2` 表示 80% 训练 / 20% 验证 |
| `--image-size` | `32` | 输入图像尺寸 | 当前数据本身是 `32x32`，通常不改；若后面换数据集可调整 |
| `--epochs` | `10` | 训练轮数 | 想让模型训练更久可设置更大，例如 `--epochs 30` |
| `--batch-size` | `128` | 每个 batch 中包含的样本数 | 显存够可增大，例如 `256`；显存不够可减小，例如 `64` |
| `--num-workers` | `4` | DataLoader 读取数据时使用的子进程数量 | Windows 或某些环境下卡住时可改成 `0`；Linux 下一般 `2~8` 都常见 |
| `--lr` | `1e-3` | 学习率，决定参数更新步长 | 模型不收敛时可尝试 `1e-4` 或 `5e-4` |
| `--weight-decay` | `1e-4` | 权重衰减，起到一定正则化作用，帮助抑制过拟合 | 过拟合明显时可增大，如 `5e-4`；欠拟合时可减小 |
| `--seed` | `42` | 随机种子，影响训练/验证划分、训练样本抽样、参数初始化和部分随机增强 | 为了公平比较实验，建议固定不变；也可换种子做鲁棒性测试 |
| `--device` | `cuda` | 指定训练设备；当前默认值为 `cuda`，即默认优先在 GPU 上运行 | 若需要改为 CPU，可手动写 `--device cpu` |

#### A.1 最常改的几个核心超参数

如果你只想先做最基础实验，最常需要改的是下面几个：

1. `--labeled-ratio`
   - 控制训练集使用比例。
   - 对应作业中的 `1% / 10%` 对比实验。

2. `--val-ratio`
   - 控制从原始 `train` 中留多少比例做验证集。
   - 这是现在训练流程中的新关键参数。

3. `--model`
   - 控制使用哪种 backbone。
   - 当前可选 `resnet18`、`mobilenet_v2`。

4. `--epochs`
   - 控制训练时长。
   - 轮数太少可能欠拟合，太多可能过拟合。

5. `--batch-size`
   - 控制每次梯度更新所看的样本数。
   - 更大 batch 通常更稳定，但更占显存。

6. `--lr`
   - 控制优化步长。
   - 如果 loss 不下降，往往优先怀疑学习率是否不合适。

#### A.2 训练命令示例

`10%` 标注数据，`20%` 验证划分：

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python train.py \
  --data-root data4 \
  --model resnet18 \
  --labeled-ratio 0.1 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --lr 1e-3 \
  --output-dir output/train_val/resnet18_ratio10
```

`1%` 标注数据：

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python train.py \
  --data-root data4 \
  --model resnet18 \
  --labeled-ratio 0.01 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --lr 1e-3 \
  --output-dir output/train_val/resnet18_ratio01
```

改成 `MobileNetV2`：

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python train.py \
  --data-root data4 \
  --model mobilenet_v2 \
  --labeled-ratio 0.1 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir output/train_val/mobilenet_ratio10
```

### B. `test.py` 中可直接设置的参数

测试脚本入口是：

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python test.py [参数]
```

当前支持的测试参数如下。

| 参数名 | 默认值 | 作用 | 典型修改方式 |
| --- | --- | --- | --- |
| `--data-root` | `data4` | 最终测试数据根目录。该脚本会读取 `data_root/test` | 数据集位置变化时修改 |
| `--checkpoint` | 无，必须填写 | 指定要加载的模型权重文件 | 例如 `--checkpoint output/train_val/resnet18_ratio10/best_model.pt` |
| `--model` | `resnet18` | 指定模型结构，必须与训练时保存的模型结构一致 | 若训练用的是 `mobilenet_v2`，测试时也要写 `--model mobilenet_v2` |
| `--batch-size` | `256` | 测试时的 batch 大小 | 显存足够时可以调大来提速 |
| `--num-workers` | `4` | 测试集读取进程数 | 环境兼容性有问题时可调成 `0` |
| `--image-size` | `32` | 测试输入尺寸 | 应与训练预处理保持一致 |
| `--device` | `cuda` | 指定测试设备；当前默认值为 `cuda` | 若需要改为 CPU，可手动写 `--device cpu` |
| `--output-dir` | `output/final_test` | 最终测试结果输出目录，保存测试指标 JSON 和测试混淆矩阵 | 例如 `--output-dir output/final_test/resnet18_ratio10` |

#### B.1 测试命令示例

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python test.py \
  --data-root data4 \
  --checkpoint output/train_val/resnet18_ratio10/best_model.pt \
  --model resnet18 \
  --batch-size 256 \
  --output-dir output/final_test/resnet18_ratio10
```

### C. 当前写在代码里的“固定超参数”

除了命令行参数外，还有一些设置目前写死在源码里。它们也会影响实验结果，但当前不能直接通过命令行修改。

#### C.1 数据增强参数

位置：`data_loader.py` 中的 `build_transforms()`

当前训练增强是：

```python
transforms.Resize((image_size, image_size))
transforms.RandomCrop(image_size, padding=4)
transforms.RandomHorizontalFlip()
transforms.ColorJitter(
    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
)
transforms.ToTensor()
transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
```

这些参数的意义如下：

| 位置 | 当前值 | 作用 | 如何修改 |
| --- | --- | --- | --- |
| `RandomCrop(..., padding=4)` | `padding=4` | 先补边再随机裁剪，增强平移鲁棒性 | 直接改 `data_loader.py` 中的 `padding` 数值 |
| `RandomHorizontalFlip()` | 默认概率 `0.5` | 随机水平翻转 | 若想显式控制概率，可写成 `RandomHorizontalFlip(p=0.5)` 或改为其他值 |
| `ColorJitter` | `brightness=0.2` 等 | 控制颜色扰动强度 | 直接修改四个参数的数值 |
| `Normalize(CIFAR_MEAN, CIFAR_STD)` | CIFAR 常用统计量 | 让输入分布更稳定 | 若换数据集，可自行改均值和方差 |

如果你想调数据增强强度，需要直接修改 [data_loader.py](/home/lofi_linux/Machine_Learning/DeepLearning/experiment_4/DL_exp4/data_loader.py:12)。

#### C.2 归一化统计量

位置：`data_loader.py` 顶部

```python
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
```

意义：

- 这是输入归一化使用的通道均值与标准差。
- 当前采用的是 CIFAR 风格设置。

修改方式：

- 直接编辑 [data_loader.py](/home/lofi_linux/Machine_Learning/DeepLearning/experiment_4/DL_exp4/data_loader.py:8) 里的两个元组。

#### C.3 模型内部结构参数

位置：`net.py`

虽然你可以通过 `--model` 选择 `resnet18` 或 `mobilenet_v2`，但一些更细的结构参数目前写死在代码里，例如：

| 位置 | 当前值 | 作用 | 如何修改 |
| --- | --- | --- | --- |
| ResNet 第一层卷积 | `kernel_size=3, stride=1, padding=1` | 适配 `32x32` 小图输入 | 修改 [net.py](/home/lofi_linux/Machine_Learning/DeepLearning/experiment_4/DL_exp4/net.py:5) 中 `_build_resnet18()` |
| ResNet 最大池化层 | `nn.Identity()` | 避免过早下采样 | 若想恢复池化，改回原层或换成 `MaxPool2d` |
| MobileNet 第一层卷积 | `stride=1` | 保留更多空间信息 | 修改 [net.py](/home/lofi_linux/Machine_Learning/DeepLearning/experiment_4/DL_exp4/net.py:15) 中 `_build_mobilenet_v2()` |
| 分类头输出维度 | `num_classes=2` | 对应 `FAKE/REAL` 二分类 | 若任务类别数变化，需要同时改数据和模型输出 |

#### C.4 选择“最佳模型”的标准

位置：`train.py`

当前逻辑是：

- 每个 epoch 在验证集上评估。
- 按 `val accuracy` 最高来保存 `best_model.pt`。

如果你想改成按 `F1` 保存最佳模型，需要修改 `train.py` 中保存最佳模型的判断逻辑。

#### C.5 优化器类型

位置：`train.py`

当前固定为：

```python
optimizer = Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
```

也就是说：

- 学习率和权重衰减可以通过命令行改；
- 但“优化器是 Adam 还是 SGD”目前不能直接通过命令行改。

如果你想换优化器，需要直接修改 [train.py](/home/lofi_linux/Machine_Learning/DeepLearning/experiment_4/DL_exp4/train.py:93) 附近的实现。

### D. 推荐你优先控制的实验变量

如果你准备做课程作业实验，而不是纯工程调参，建议优先把下面几个变量作为主实验轴：

1. `labeled_ratio`
   - `0.01` vs `0.1`
   - 对应作业要求中的 `1% / 10%`

2. `val_ratio`
   - 控制验证集比例
   - 关系到训练样本量和验证稳定性之间的平衡

3. `model`
   - `resnet18` vs `mobilenet_v2`
   - 对应不同 encoder 架构比较

4. `epochs`
   - 控制训练是否充分

5. `lr`
   - 学习率对收敛速度和稳定性影响很大

6. 数据增强强度
   - 后续进入 SimCLR 阶段时尤其重要

### E. 一个最直接的理解方式

如果你想知道“某个参数该去哪里改”，可以记住下面这条规则：

- **训练过程中的高层参数**：去 `train.py` 命令行里改。
- **最终测试过程中的高层参数**：去 `test.py` 命令行里改。
- **数据增强细节**：去 `data_loader.py` 里改。
- **网络结构细节**：去 `net.py` 里改。
- **保存最优模型的规则**：去 `train.py` 里改。

## 1. 文档定位

这份 `README.md` 不是一句话的“如何运行说明”，而是一份偏讲义风格的代码说明文档。目标是把当前仓库中已经实现的 **baseline 版本** 讲清楚，包括：

1. 这个 baseline 在算法上做了什么。
2. 每个 Python 文件承担什么职责。
3. 每个核心函数为什么这样写。
4. 代码中用到的 PyTorch / Python 语法分别是什么意思。
5. 训练、测试、画图、保存结果是如何串起来的。

当前仓库实现的是：

- 一个 **监督学习 baseline**。
- 任务是二分类：`REAL` vs `FAKE`。
- 支持按作业要求只取 `1%` / `10%` 的带标签训练样本来训练。
- 支持把原始 `train` 目录按类别分层切分为训练集和验证集。
- 支持两种 backbone：`resnet18` 与 `mobilenet_v2`。
- 支持训练、验证、最终测试、记录指标、保存最优模型、画训练曲线和混淆矩阵。

当前仓库 **还没有实现**：

- SimCLR 的双视图增强。
- Encoder + Projection Head 的自监督预训练。
- NT-Xent / InfoNCE 对比损失。
- 冻结 Encoder 后的线性评估。

因此，这份文档解释的是“**当前基线代码**”，而不是作业最终完整版本。

---

## 2. 项目目标与 baseline 的算法含义

### 2.1 作业整体目标

作业主题是“基于 SimCLR 的 AIGC 生成图像检测”。最终标准流程应该是：

1. 先做 **无监督对比学习预训练**。
2. 再冻结编码器，用少量标注数据训练线性分类器。
3. 最后和“直接监督训练的 baseline”做比较。

### 2.2 当前 baseline 的意义

在真正写 SimCLR 之前，先做一个可跑通的监督式 baseline 是非常必要的，因为它有三个作用：

1. 验证数据管道是否正确。
2. 验证网络是否适配 `32x32` 输入。
3. 给后续 SimCLR 结果提供对照组。

### 2.3 当前 baseline 的算法流程

当前实现的 baseline 可以概括为：

1. 从 `data4/train` 读取原始训练图像。
2. 将原始训练集按类别分层切分为训练集和验证集。
3. 仅在训练划分中按比例抽取一部分有标签样本，例如 `1%` 或 `10%`。
4. 用标准图像增强处理训练图像；验证集使用确定性预处理。
5. 构建一个卷积神经网络，例如 `ResNet-18`。
6. 用交叉熵损失 `CrossEntropyLoss` 做二分类监督训练。
7. 每个 epoch 后只在验证集上评估：
   - `Loss`
   - `Accuracy`
   - `F1 Score`
8. 保存验证准确率最高的模型。
9. 将训练过程画成曲线图，并绘制最优验证结果对应的混淆矩阵。
10. 只有在你显式运行 `test.py` 时，才会读取 `data4/test` 做最终测试。

这本质上是一个标准的 **有监督图像分类任务**，而不是自监督学习。

---

## 3. 目录结构与模块职责

当前主要文件如下：

```text
DL_exp4/
├── data4/                # 数据集目录
│   ├── train/
│   │   ├── FAKE/
│   │   └── REAL/
│   └── test/
│       ├── FAKE/
│       └── REAL/
├── data_loader.py        # 数据读取、train/val 划分与抽样
├── net.py                # 网络构建
├── train.py              # 训练入口
├── test.py               # 测试入口与评估函数
├── plot.py               # 画图模块
├── utils.py              # 工具函数
├── README.pdf            # 作业要求 PDF
└── README.md             # 当前这份说明文档
```

从职责上看，整个工程是典型的模块化设计：

- `data_loader.py`：解决“数据从哪里来、如何切分、如何增强、如何抽样”的问题。
- `net.py`：解决“模型长什么样”的问题。
- `train.py`：解决“如何训练、如何保存结果”的问题。
- `test.py`：解决“如何评估一个训练好的模型”的问题。
- `plot.py`：解决“如何可视化训练结果”的问题。
- `utils.py`：解决“随机种子、目录创建、JSON 保存、设备选择”这些通用问题。

这种拆分方式的好处是：

1. 逻辑分层清晰。
2. 后续加 SimCLR 时不需要推翻现有结构。
3. 每个文件只负责一类问题，便于调试与维护。

---

## 4. 代码总调用链

训练时的调用链如下：

```text
train.py
  ├── parse_args()                 解析命令行参数
  ├── seed_everything()            固定随机种子
  ├── get_device()                 选择 cpu / cuda
  ├── build_train_val_dataloaders() 构建训练与验证 DataLoader
  │     └── build_train_val_datasets()
  │           ├── build_transforms()
  │           ├── _split_train_val_indices()
  │           └── _sample_labeled_indices()
  ├── build_model()                构建 backbone
  ├── train_one_epoch()            训练一轮
  ├── evaluate()                   在验证集上评估
  ├── torch.save()                 保存最佳模型
  ├── save_json()                  保存训练摘要
  ├── plot_training_history()      绘制损失/指标曲线
  └── plot_confusion_matrix()      绘制验证集混淆矩阵
```

测试时的调用链如下：

```text
test.py
  ├── parse_args()                 解析命令行参数
  ├── get_device()                 选择 cpu / cuda
  ├── build_test_dataloader()      构建最终测试 DataLoader
  ├── build_model()                构建模型骨干
  ├── torch.load()                 加载 checkpoint
  ├── model.load_state_dict()      恢复参数
  ├── evaluate()                   计算 Loss/Accuracy/F1
  ├── save_json()                  保存测试指标
  └── plot_confusion_matrix()      保存测试混淆矩阵
```

---

## 5. `data_loader.py` 详解

文件：[`data_loader.py`](./data_loader.py)

这个模块负责四件事情：

1. 定义图像预处理和数据增强。
2. 从目录中读取图像数据。
3. 将原始 `train` 目录分层切分为训练集和验证集。
4. 按照 `1%` / `10%` 的比例抽样带标签训练数据。

### 5.1 导入部分

```python
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
```

这几行导入了本文件需要用到的工具：

- `defaultdict`
  - 来自 Python 标准库 `collections`
  - 作用是构造“默认值自动初始化”的字典
  - 这里用来把同一类别的样本索引聚合到一起

- `torch`
  - PyTorch 核心库
  - 这里主要用来做随机打乱和张量操作

- `DataLoader`
  - PyTorch 中最常用的数据迭代器
  - 负责把数据集按 batch 提供给训练循环

- `Subset`
  - 表示“原始数据集的一个子集”
  - 这里用它来表示按比例抽样后的训练集

- `datasets.ImageFolder`
  - `torchvision` 中经典的目录式数据集读取器
  - 目录结构形如：
    - `train/FAKE/*.jpg`
    - `train/REAL/*.jpg`
  - 它会自动根据文件夹名分配类别标签

- `transforms`
  - 图像增强与预处理模块

### 5.2 均值与标准差

```python
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
```

这是对 CIFAR 风格数据常用的归一化参数。

算法意义：

- 归一化可以让不同通道的数据尺度更稳定。
- 有助于优化器更快收敛。
- 因为本作业数据来自 CIFAKE，而 CIFAKE 与 CIFAR-10 分辨率、统计特征接近，所以使用这组均值与标准差是合理的 baseline 选择。

语法层面：

- 这里定义的是 Python 元组 `tuple`。
- 三个数字分别对应 RGB 三个通道。

### 5.3 `build_transforms`

```python
def build_transforms(image_size: int = 32):
```

这是一个函数定义：

- `def` 表示定义函数。
- `image_size: int = 32`
  - `: int` 是类型注解，表示参数期望是整数。
  - `= 32` 是默认值，表示调用时可以不传，默认使用 `32`。

#### 5.3.1 训练增强

```python
train_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)
```

这里使用了 `transforms.Compose(...)` 把多个处理步骤串成一个流水线。

`Compose` 的语义是：

- 输入一张图像。
- 依次执行列表中的每一个变换。
- 输出最终处理后的张量。

各步骤含义如下：

1. `Resize((image_size, image_size))`
   - 把图像统一调整到 `32x32`
   - 虽然 CIFAKE 通常已经是 `32x32`，但显式写出来能增强稳健性

2. `RandomCrop(image_size, padding=4)`
   - 先在图像边缘补 4 个像素，再随机裁回 `32x32`
   - 算法意义是提升平移与局部扰动鲁棒性

3. `RandomHorizontalFlip()`
   - 以默认概率 `0.5` 进行水平翻转
   - 算法意义是增强左右对称场景的泛化能力

4. `ColorJitter(...)`
   - 随机改变亮度、对比度、饱和度、色调
   - 算法意义是减少模型对颜色细节的过拟合

5. `ToTensor()`
   - 把 PIL 图像或 NumPy 数组转为 PyTorch 张量
   - 同时把像素值从 `[0, 255]` 缩放到 `[0, 1]`

6. `Normalize(CIFAR_MEAN, CIFAR_STD)`
   - 对每个通道做标准化
   - 公式为：
     \[
     x' = \frac{x - \mu}{\sigma}
     \]

#### 5.3.2 验证/测试预处理

```python
test_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)
```

验证和最终测试时不做随机增强，只做确定性的预处理。原因是：

- 验证集和测试集指标都应尽量稳定。
- 如果使用随机裁剪、随机翻转，会让每次测试输入不同，影响结果可重复性。

### 5.4 `_sample_labeled_indices`

```python
def _sample_labeled_indices(targets, labeled_ratio: float, seed: int):
```

这个函数是当前 baseline 中最关键的“作业约束适配器”之一。

它的作用是：

- 输入完整训练集的标签列表 `targets`。
- 按比例从每个类别中采样样本索引。
- 返回采样后的索引列表。

这样就能模拟“只使用 1% / 10% 标注数据”的训练场景。

#### 5.4.1 比例合法性检查

```python
if not 0 < labeled_ratio <= 1:
    raise ValueError("labeled_ratio must be in the range (0, 1].")
```

语法说明：

- `not 0 < labeled_ratio <= 1` 是 Python 的链式比较写法。
- 如果条件不满足，就抛出 `ValueError`。

算法意义：

- `labeled_ratio=0.1` 表示使用 10% 标注样本。
- `labeled_ratio=0.01` 表示使用 1% 标注样本。
- 不能传 0，也不能超过 1。

#### 5.4.2 固定随机性

```python
generator = torch.Generator().manual_seed(seed)
```

这里创建了一个 PyTorch 随机数生成器，并固定其种子。

作用：

- 保证每次抽样结果可复现。
- 同一 `seed` 下，多次运行训练，抽到的是同一批样本。

#### 5.4.3 按类别收集索引

```python
class_to_indices = defaultdict(list)
for index, label in enumerate(targets):
    class_to_indices[label].append(index)
```

解释：

- `enumerate(targets)` 会同时给出：
  - 索引 `index`
  - 标签 `label`
- `defaultdict(list)` 表示如果某个键不存在，就自动创建一个空列表

最终结果类似于：

```python
{
    0: [0, 3, 8, 10, ...],   # 某一类的样本索引
    1: [1, 2, 4, 5, ...]
}
```

算法意义：

- 这是 **分层抽样** 的前置步骤。
- 不是从整个训练集直接按比例随机抽，而是先按类别分组再抽。
- 这样能防止小比例抽样时类别分布失衡。

#### 5.4.4 每类内部随机打乱并采样

```python
for label in sorted(class_to_indices):
    indices = torch.tensor(class_to_indices[label], dtype=torch.long)
    perm = torch.randperm(len(indices), generator=generator)
    shuffled = indices[perm]
    sample_count = max(1, int(len(shuffled) * labeled_ratio))
    sampled_indices.extend(shuffled[:sample_count].tolist())
```

这一段非常值得理解。

逐行解释：

- `sorted(class_to_indices)`
  - 以排序后的类别顺序遍历
  - 让行为更稳定、更可预测

- `torch.tensor(..., dtype=torch.long)`
  - 把 Python 列表转成整型张量
  - 因为索引在 PyTorch 中通常使用 `long`

- `torch.randperm(len(indices), generator=generator)`
  - 生成 `0 ~ len(indices)-1` 的随机排列
  - 相当于“打乱顺序”

- `shuffled = indices[perm]`
  - 通过索引重排张量

- `sample_count = max(1, int(len(shuffled) * labeled_ratio))`
  - 计算当前类别应该采多少个样本
  - `max(1, ...)` 保证每类至少采到 1 张图

- `sampled_indices.extend(...)`
  - 把本类别采到的索引并入总列表

算法意义：

- 这是一个简单但实用的 **按类等比例抽样**。
- 它保证：
  - 小比例训练仍能覆盖所有类别
  - 类别分布大体与原数据一致

### 5.5 `_split_train_val_indices`

```python
def _split_train_val_indices(targets, val_ratio: float, seed: int):
```

作用：

- 按类别分层切分原始 `train` 目录中的样本索引。
- 返回 `train_indices` 和 `val_indices`。

这个函数的核心思想是：

1. 先把每个类别的样本索引分别收集起来。
2. 在每个类别内部随机打乱。
3. 按 `val_ratio` 从每个类别中切一部分进入验证集。
4. 剩余部分进入训练集。

这样做的意义是：

- 训练集和验证集的类别分布更稳定。
- 不会出现验证集只偏向某一类的情况。
- 这比简单的整体随机切分更符合分类任务的常规做法。

### 5.6 `build_train_val_datasets`

```python
def build_train_val_datasets(
    data_root: str,
    labeled_ratio: float,
    val_ratio: float,
    image_size: int,
    seed: int,
):
```

作用：

- 读取原始 `data4/train`。
- 生成训练集和验证集。
- 只在训练划分中进一步做 `labeled_ratio` 抽样。

#### 5.6.1 为什么要建立两个 `ImageFolder`

代码里会分别建立：

- 一个带训练增强的 `ImageFolder`
- 一个带验证/测试预处理的 `ImageFolder`

这是因为：

- 训练集需要随机增强。
- 验证集不应该使用随机增强。
- 但它们都来自同一个原始 `data4/train` 目录。

#### 5.6.2 训练集为什么还要再做一次抽样

新流程是：

1. 先从原始 `train` 中切出验证集。
2. 再从剩下的训练划分中按 `1% / 10%` 比例抽取真正参与训练的样本。

这样设计的好处是：

- 验证集始终是固定的“留出集”。
- 训练样本量变化不会污染验证流程。

#### 5.6.3 `Subset` 的作用

`Subset` 表示：

- 不复制原始数据。
- 只用索引引用其中一部分样本。

这让“划分训练集/验证集”和“抽样训练子集”都能高效完成。

#### 5.6.4 元信息 `meta`

当前 `meta` 中主要包含：

- `class_names`
- `num_train_samples`
- `num_val_samples`
- `num_full_train_samples`
- `labeled_ratio`
- `val_ratio`

这些信息会被 `train.py` 用来保存训练摘要。

### 5.7 `build_train_val_dataloaders`

```python
def build_train_val_dataloaders(...):
```

作用：

- 用 `Dataset` 构建可迭代的训练 DataLoader 和验证 DataLoader。

#### 5.6.1 `loader_kwargs`

```python
loader_kwargs = {
    "batch_size": batch_size,
    "num_workers": num_workers,
    "pin_memory": torch.cuda.is_available(),
}
```

这里用了一个 Python 字典来统一管理 DataLoader 共有参数。

语法重点：

- 后面调用 `DataLoader(..., **loader_kwargs)` 时，`**` 的作用是“把字典展开为关键字参数”。

例如：

```python
DataLoader(dataset, **loader_kwargs)
```

等价于：

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
```

#### 5.6.2 `pin_memory`

如果使用 GPU，`pin_memory=True` 往往能让 CPU 到 GPU 的数据拷贝更高效。

#### 5.7.3 训练与验证 DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    drop_last=False,
    **loader_kwargs,
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)
```

这里的差异是：

- 训练集 `shuffle=True`
  - 每轮打乱样本顺序
  - 减少训练顺序偏差

- 验证集 `shuffle=False`
  - 保持顺序稳定
  - 便于复现评估结果

`drop_last=False` 表示最后一个不足一个 batch 的小批次也保留。

### 5.8 `build_test_dataloader`

这个函数只服务于最终测试阶段。

作用：

- 读取 `data4/test`
- 构建最终测试的 DataLoader

重要的是：

- 训练脚本不会调用它。
- 只有你显式运行 `test.py`，它才会被使用。

---

## 6. `net.py` 详解

文件：[`net.py`](./net.py)

这个模块负责“网络构建”。

### 6.1 为什么单独拆成 `net.py`

原因有三个：

1. 训练逻辑与模型结构解耦。
2. 后续切换 backbone 更方便。
3. 将来加入 SimCLR 时，可以把“编码器构建”复用起来。

### 6.2 导入

```python
import torch.nn as nn
from torchvision import models
```

- `torch.nn as nn`
  - 是 PyTorch 神经网络模块的惯用写法
  - `nn.Conv2d`、`nn.Linear`、`nn.Identity` 都在这里

- `torchvision.models`
  - 提供常见视觉模型结构
  - 如 `resnet18`、`mobilenet_v2`

### 6.3 `_build_resnet18`

```python
def _build_resnet18(num_classes: int):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
```

#### 6.3.1 为什么不用默认 ResNet-18

标准 ImageNet 版 ResNet-18 的输入习惯是 `224x224`，其前几层通常是：

- `7x7` 卷积
- `stride=2`
- 后接 `maxpool`

对 `32x32` 小图来说，这种下采样过于激进，会导致空间信息很快丢失。

所以这里做了两个关键修改：

1. 第一层卷积改成 `3x3, stride=1, padding=1`
2. 取消最大池化层

这是一种非常常见的 “CIFAR 风格 ResNet 改造”。

#### 6.3.2 `weights=None`

表示不加载预训练权重，采用随机初始化。

这与 baseline 设定一致，因为当前就是“从头监督训练”的对照组。

#### 6.3.3 `model.conv1 = ...`

这里不是“新建一个模型”，而是：

- 先创建标准结构
- 再把第一层替换掉

这体现了 PyTorch 的一个重要特点：

- 模型本质上是 Python 对象
- 其子模块可以被直接赋值替换

#### 6.3.4 `nn.Identity()`

`Identity` 表示恒等映射，即输出等于输入。

把 `maxpool` 换成 `Identity` 的效果是：

- 保留该层位置
- 但不做任何运算

这是一个比“删代码”更安全、更清晰的写法。

#### 6.3.5 输出层改造

```python
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

因为原始 ResNet-18 的最后全连接层类别数是 ImageNet 的 `1000` 类，而本任务只有 2 类，所以要替换成新的线性层。

### 6.4 `_build_mobilenet_v2`

```python
def _build_mobilenet_v2(num_classes: int):
    model = models.mobilenet_v2(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        3,
        first_conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
```

这里与 ResNet 的思路相同，也是为了适配 `32x32` 输入。

`MobileNetV2` 的结构稍微复杂一点，所以替换第一层时要通过索引定位：

- `model.features[0][0]`

这表示：

- `features` 是一个大的特征提取模块
- 第 `0` 个子模块里还有一个卷积层
- 再取这个卷积层本体

语法上，这里连续用了“容器索引”。

### 6.5 `build_model`

```python
def build_model(name: str = "resnet18", num_classes: int = 2):
```

这是一种“工厂函数”写法。

其核心思想是：

- 调用方只需要提供名字，例如 `"resnet18"`
- 函数内部决定具体调用哪个构造器

实现方式：

```python
builders = {
    "resnet18": _build_resnet18,
    "mobilenet_v2": _build_mobilenet_v2,
}
```

这里字典的值不是“结果”，而是“函数对象”。

然后：

```python
return builders[name](num_classes=num_classes)
```

表示：

1. 先取出对应的函数。
2. 再调用这个函数。

这是一种很典型的 Python 动态分发方式。

---

## 7. `utils.py` 详解

文件：[`utils.py`](./utils.py)

这个模块放的是多个文件都可能用到的小工具函数。

### 7.1 `seed_everything`

```python
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

作用：

- 固定 Python、NumPy、PyTorch 的随机种子。

为什么要固定：

1. 训练集抽样要可复现。
2. 参数初始化要尽量可复现。
3. 数据增强的随机性也要尽量可控。

这里的 `-> None` 是返回值类型注解，表示函数不返回任何值。

### 7.2 `ensure_dir`

```python
def ensure_dir(path: str | Path) -> Path:
```

这里用到了 Python 3.10+ 的联合类型语法：

- `str | Path`
  - 表示参数可以是字符串，也可以是 `Path` 对象

函数体：

```python
directory = Path(path)
directory.mkdir(parents=True, exist_ok=True)
return directory
```

作用：

- 把输入路径统一转成 `Path` 对象。
- 如果目录不存在，就递归创建。

语法说明：

- `parents=True`
  - 若父目录不存在，也一并创建
- `exist_ok=True`
  - 若目录已经存在，不报错

### 7.3 `save_json`

```python
def save_json(data: dict, path: str | Path) -> None:
```

作用：

- 把实验结果保存成 JSON 文件。

代码中先调用：

```python
ensure_dir(output_path.parent)
```

这是一个很好的工程习惯：

- 先保证父目录存在，再写文件
- 避免因目录不存在导致写入失败

然后：

```python
with output_path.open("w", encoding="utf-8") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)
```

语法说明：

- `with ... as file`
  - 上下文管理器写法
  - 好处是文件用完会自动关闭

- `indent=2`
  - 让 JSON 缩进更美观

- `ensure_ascii=False`
  - 允许直接保存中文，而不是转义成 `\uXXXX`

### 7.4 `get_device`

```python
def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

作用：

- 统一确定训练/测试使用的设备。

逻辑是：

1. 如果用户显式传了 `--device`，优先使用用户指定值。
2. 否则自动判断：
   - 有 GPU 就用 `cuda`
   - 没有就用 `cpu`

这里的语法重点：

- `str | None`
  - 表示参数类型可以是字符串，也可以是 `None`

- `"cuda" if torch.cuda.is_available() else "cpu"`
  - 这是 Python 的条件表达式
  - 类似三元运算符

---

## 8. `test.py` 详解

文件：[`test.py`](./test.py)

这个模块既包含 **评估函数**，也包含 **独立测试脚本入口**。

### 8.1 为什么 `evaluate()` 写在 `test.py`

因为“测试”本来就是它的核心职责，而 `train.py` 在每个 epoch 后也需要评估，所以直接从 `test.py` 导入 `evaluate`，避免重复写一份验证逻辑。

### 8.2 `@torch.no_grad()`

```python
@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
```

这是一个装饰器。

算法意义：

- 测试时不需要反向传播。
- 不需要存储梯度。
- 可以减少显存/内存消耗，提升速度。

语法意义：

- `@装饰器` 会在函数定义时对函数做包装。
- 相当于把该函数放在“关闭梯度记录”的环境中执行。

### 8.3 `model.eval()`

```python
model.eval()
```

作用：

- 将模型切换到评估模式。

对某些层很重要，例如：

- `BatchNorm`
- `Dropout`

如果不切换，测试结果会不稳定或不正确。

### 8.4 测试循环

```python
for images, labels in data_loader:
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    logits = model(images)
    loss = criterion(logits, labels)
```

解释：

- `for images, labels in data_loader`
  - 每次从 DataLoader 取一个 batch

- `.to(device, non_blocking=True)`
  - 将张量移动到目标设备上
  - 若启用 pinned memory，则 `non_blocking=True` 可以提升传输效率

- `logits = model(images)`
  - 前向传播
  - 输出是每一类的原始得分，不是概率

- `loss = criterion(logits, labels)`
  - 计算损失
  - 当前 criterion 是交叉熵损失

### 8.5 预测类别

```python
preds = logits.argmax(dim=1)
```

算法意义：

- `logits` 的形状通常是 `[batch_size, num_classes]`
- `dim=1` 表示沿类别维度取最大值位置
- 得到的就是预测类别编号

语法意义：

- `argmax` 返回最大元素的索引，而不是最大值本身

### 8.6 收集所有预测与标签

```python
all_preds.extend(preds.cpu().tolist())
all_labels.extend(labels.cpu().tolist())
```

解释：

- `.cpu()`
  - 把张量移回 CPU

- `.tolist()`
  - 把张量变成 Python 列表

- `extend(...)`
  - 将列表内容逐个追加到现有列表
  - 区别于 `append(...)`，`append` 会把整个列表当作一个元素加入

### 8.7 指标计算

```python
avg_loss = total_loss / len(data_loader.dataset)
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="binary")
```

这里用到了 `sklearn.metrics`。

指标解释：

- `Loss`
  - 反映模型输出与真实标签之间的整体差异

- `Accuracy`
  - 正确分类数 / 总样本数

- `F1 Score`
  - 精确率与召回率的调和平均
  - 对类别不平衡问题更有参考价值

当前是二分类，所以 `average="binary"` 是合理的。

### 8.8 `parse_args`

`argparse` 的作用是把命令行输入解析成 Python 对象。

例如：

```bash
python test.py --checkpoint output/train_val/best_model.pt --model resnet18
```

解析后可通过：

```python
args.checkpoint
args.model
```

来访问参数值。

### 8.9 `main()` 的流程

`main()` 里做了以下步骤：

1. 解析参数。
2. 确定设备。
3. 构建最终测试 DataLoader。
4. 构建模型结构。
5. 加载 checkpoint。
6. 恢复模型参数。
7. 评估指标。
8. 打印结果。
9. 将测试指标和测试混淆矩阵保存到最终测试输出目录。

### 8.10 `torch.load()` 与 `load_state_dict()`

```python
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

含义：

- `torch.load(...)`
  - 从磁盘读取 checkpoint 字典

- `map_location=device`
  - 即使模型原先是在 GPU 保存的，也能映射到当前设备

- `load_state_dict(...)`
  - 把参数真正加载进模型对象

---

## 9. `train.py` 详解

文件：[`train.py`](./train.py)

这是整个 baseline 的主入口文件。

### 9.1 导入结构

```python
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
```

这里既有标准库，也有 PyTorch 组件。

`Adam` 是优化器，负责根据梯度更新参数。

### 9.2 `tqdm` 的兼容写法

```python
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
```

这是为了适配当前环境中可能没有安装 `tqdm` 的情况。

语法解释：

- `try ... except ...`
  - 捕获异常
  - 如果导入失败，就退化成不显示进度条

工程意义：

- 保证脚本不会因为少一个“非核心依赖”而完全无法运行。
- 又不违反“不能修改环境”的约束。

### 9.3 `train_one_epoch`

```python
def train_one_epoch(model, data_loader, criterion, optimizer, device):
```

顾名思义，这个函数只负责训练一个 epoch。

为什么单独拆函数：

1. 主流程更简洁。
2. 后续可以方便替换训练策略。
3. 如果以后有预训练阶段和线性评估阶段，各自都能复用类似结构。

#### 9.3.1 `model.train()`

```python
model.train()
```

作用：

- 将模型置于训练模式。
- 与 `eval()` 对应。

#### 9.3.2 累积变量

```python
running_loss = 0.0
correct = 0
total = 0
```

这些变量用来累计整轮训练的统计量：

- `running_loss`
- `correct`
- `total`

#### 9.3.3 `iterator = data_loader`

```python
iterator = data_loader
if tqdm is not None:
    iterator = tqdm(data_loader, desc="Train", leave=False)
```

这是一个很典型的“优雅降级”写法：

- 如果有 `tqdm`，就用带进度条的迭代器。
- 否则直接用原始 `data_loader`。

#### 9.3.4 训练核心五步

训练循环中的核心步骤如下：

1. 取一个 batch
2. 清空旧梯度
3. 前向传播
4. 反向传播
5. 优化器更新参数

对应代码：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

逐项解释：

- `zero_grad(...)`
  - PyTorch 默认会累积梯度
  - 所以每个 batch 前都要清零

- `set_to_none=True`
  - 一种更高效的清零方式

- `loss.backward()`
  - 自动求导
  - 计算所有参数对损失的梯度

- `optimizer.step()`
  - 根据梯度更新参数

#### 9.3.5 统计训练准确率

```python
preds = logits.argmax(dim=1)
correct += (preds == labels).sum().item()
total += labels.size(0)
```

这里用 `(preds == labels)` 得到一个布尔张量：

- 预测正确位置为 `True`
- 预测错误位置为 `False`

`.sum()` 会把 `True` 当作 `1`，`False` 当作 `0`，因此得到正确样本数。

`.item()` 把单元素张量转成 Python 标量。

#### 9.3.6 进度条后缀

```python
iterator.set_postfix(
    loss=f"{running_loss / total:.4f}",
    acc=f"{correct / total:.4f}",
)
```

这是 `tqdm` 的界面更新功能，用于动态显示当前平均 loss 和准确率。

### 9.4 `parse_args()`

训练脚本的参数包括：

- `--data-root`
- `--output-dir`
- `--model`
- `--labeled-ratio`
- `--image-size`
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--lr`
- `--weight-decay`
- `--seed`
- `--device`

这些参数的意义分别是：

- `data-root`
  - 数据集根目录

- `output-dir`
  - 输出目录，用来保存模型和图

- `model`
  - backbone 类型

- `labeled-ratio`
  - 标注训练集采样比例，适配作业的 `1%/10%`

- `epochs`
  - 训练轮数

- `lr`
  - 学习率

- `weight-decay`
  - 权重衰减，等价于一种 L2 正则化

### 9.5 `main()` 的总体流程

#### 9.5.1 固定随机种子与设备

```python
seed_everything(args.seed)
device = get_device(args.device)
output_dir = ensure_dir(args.output_dir)
```

先保证可复现，再准备输出目录。

#### 9.5.2 构建数据加载器

```python
train_loader, val_loader, meta = build_train_val_dataloaders(...)
```

这里一次性拿到：

- 训练集加载器
- 验证集加载器
- 元信息

#### 9.5.3 构建模型

```python
model = build_model(name=args.model, num_classes=len(meta["class_names"]))
model.to(device)
```

`model.to(device)` 的作用是把模型参数整体移动到目标设备。

#### 9.5.4 定义损失函数与优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
```

算法解释：

- `CrossEntropyLoss`
  - 适用于单标签多分类任务
  - 二分类也完全适用
  - 输入是 logits，标签是类别编号

- `Adam`
  - 是一种自适应学习率优化器
  - 通常作为 baseline 很稳妥

#### 9.5.5 `history` 字典

```python
history = {
    "epoch": [],
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": [],
}
```

这个结构本质上是“训练日志缓存”。

为什么用列表：

- 方便每个 epoch 追加一个值。
- 方便后续直接画折线图。

#### 9.5.6 最优模型记录

```python
best_accuracy = -1.0
best_metrics = None
best_checkpoint_path = output_dir / "best_model.pt"
```

这里采用“按验证准确率保存最佳模型”的策略。

#### 9.5.7 epoch 循环

```python
for epoch in range(1, args.epochs + 1):
```

`range(1, args.epochs + 1)` 的写法让 epoch 从 1 开始计数，更符合人类阅读习惯。

每一轮都执行：

1. 调用 `train_one_epoch`
2. 调用 `evaluate`
3. 记录结果
4. 打印日志
5. 若准确率更高则保存 checkpoint

#### 9.5.8 保存 checkpoint

```python
torch.save(
    {
        "epoch": epoch,
        "model": args.model,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "class_names": meta["class_names"],
    },
    best_checkpoint_path,
)
```

这里保存的不是单一张量，而是一个字典。

这样做的好处是：

- 不仅保存模型参数
- 还保存了优化器状态和实验元信息

以后如果要继续训练，或者分析结果来源，会更方便。

#### 9.5.9 保存训练摘要

```python
save_json(...)
```

输出的 `training_summary.json` 一般会包含：

- 命令行参数
- 数据集元信息
- 每个 epoch 的历史指标
- 最优测试指标
- 最优 checkpoint 路径

#### 9.5.10 画图

```python
plot_training_history(history, output_dir)
plot_confusion_matrix(...)
```

这一步是为了把实验结果转成更适合报告展示的可视化形式。

---

## 10. `plot.py` 详解

文件：[`plot.py`](./plot.py)

这个模块负责可视化。

### 10.1 为什么一开始要设置环境变量

```python
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
```

这是为了解决当前环境中 `matplotlib` 默认缓存目录不可写的问题。

含义：

- 如果这些环境变量尚未设置，就为它们设置默认值。
- 这样 `matplotlib` 的缓存可以写到 `/tmp`。

这属于 **运行时兼容处理**，不涉及安装任何包，也不修改环境本身。

### 10.2 无界面后端

```python
matplotlib.use("Agg")
```

含义：

- 使用非交互式绘图后端
- 适合服务器、终端、无图形界面的环境

如果不用 `Agg`，在某些无桌面环境中可能会出错。

### 10.3 `plot_training_history`

这个函数负责绘制训练曲线。

#### 10.3.1 创建子图

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
```

表示：

- 创建一个 1 行 2 列的子图布局
- `fig` 是整张图
- `axes` 是两个子图坐标轴对象

#### 10.3.2 左图：Loss 曲线

```python
axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
axes[0].plot(epochs, history["val_loss"], label="val_loss", linewidth=2)
```

用途：

- 同时展示训练损失和验证损失
- 便于观察是否收敛、是否过拟合

#### 10.3.3 右图：Accuracy / F1 曲线

```python
axes[1].plot(epochs, history["train_acc"], label="train_acc", linewidth=2)
axes[1].plot(epochs, history["test_acc"], label="test_acc", linewidth=2)
axes[1].plot(epochs, history["test_f1"], label="test_f1", linewidth=2)
```

这样报告里会更直观：

- 训练准确率
- 测试准确率
- 测试 F1

三条线放在一起，便于综合比较。

#### 10.3.4 保存图像

```python
fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
```

说明：

- `dpi=200` 提高输出分辨率
- `bbox_inches="tight"` 自动裁掉多余边距
- `plt.close(fig)` 释放内存，防止多次绘图时资源积压

### 10.4 `plot_confusion_matrix`

这个函数负责绘制混淆矩阵。

算法意义：

- 混淆矩阵可以直观看到：
  - `FAKE` 被错分成 `REAL` 的数量
  - `REAL` 被错分成 `FAKE` 的数量

相比单一 `accuracy`，它能更具体地展示错误分布。

---

## 11. 各模块之间的“算法协作关系”

从工程角度看，这套代码不是“几个互不相关的脚本”，而是一个有清晰数据流的系统。

### 11.1 数据流

```text
图像文件
  -> ImageFolder
  -> Dataset / Subset
  -> DataLoader
  -> 模型前向传播
  -> Loss / 预测标签
  -> 指标统计
  -> JSON / checkpoint / 可视化图像
```

### 11.2 训练阶段的张量流

单个 batch 的张量流可理解为：

```text
images: [B, 3, 32, 32]
   -> model(images)
logits: [B, 2]
   -> CrossEntropyLoss(logits, labels)
loss: 标量
   -> backward()
gradients
   -> optimizer.step()
updated parameters
```

其中：

- `B` 表示 batch size
- `3` 表示 RGB 三通道
- `32 x 32` 表示图像大小
- `2` 表示二分类输出

### 11.3 测试阶段的张量流

测试时没有反向传播，因此流程更简单：

```text
images
  -> model(images)
  -> logits
  -> argmax(dim=1)
  -> preds
  -> 与 labels 对比
  -> Accuracy / F1
```

---

## 12. 当前 baseline 的优点与局限

### 12.1 优点

1. 结构清晰，模块边界明确。
2. 已适配作业要求中的 `1%/10%` 小样本监督训练设置。
3. 已适配 `32x32` 输入，不直接照搬 ImageNet 结构。
4. 已具备实验最基本需要的输出：
   - 最优模型
   - Accuracy
   - F1
   - 曲线图
   - 混淆矩阵
5. 后续扩展 SimCLR 时，可以较少改动原有基础设施。

### 12.2 局限

1. 仍然只是监督学习 baseline。
2. 没有实现作业核心的 SimCLR 自监督阶段。
3. 数据增强还不是 SimCLR 所需的“双分支强增强”。
4. 还没有 Projection Head。
5. 还没有 NT-Xent 对比损失。
6. 还没有冻结评估与 linear probe。

---

## 13. 如何运行当前 baseline

根据你的执行约束，运行时必须显式使用 `ml_env` 的 Python 解释器。当前脚本中 `--device` 的默认值已经设置为 `cuda`，也就是说在不额外传参时，会默认按 GPU 方式运行。

### 13.1 训练 10% baseline

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python train.py \
  --data-root data4 \
  --model resnet18 \
  --labeled-ratio 0.1 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir output/train_val/resnet18_ratio10
```

### 13.2 训练 1% baseline

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python train.py \
  --data-root data4 \
  --model resnet18 \
  --labeled-ratio 0.01 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir output/train_val/resnet18_ratio01
```

### 13.3 测试模型

```bash
/home/lofi_linux/anaconda3/envs/ml_env/bin/python test.py \
  --data-root data4 \
  --checkpoint output/train_val/resnet18_ratio10/best_model.pt \
  --model resnet18 \
  --output-dir output/final_test/resnet18_ratio10
```

---

## 14. 训练完成后会生成什么

训练结束后，训练/验证阶段的结果一般会在输出目录下看到：

```text
output/train_val/resnet18_ratio10/
├── best_model.pt
├── best_val_confusion_matrix.png
├── training_curves.png
└── training_summary.json
```

它们分别表示：

- `best_model.pt`
  - 验证准确率最高时对应的模型参数和实验信息

- `best_val_confusion_matrix.png`
  - 最优验证结果对应的混淆矩阵

- `training_curves.png`
  - 训练/验证 loss、accuracy、F1 曲线

- `training_summary.json`
  - 实验参数与历史指标摘要

如果你之后显式运行 `test.py`，最终测试阶段的结果会放在另一套目录下，例如：

```text
output/final_test/resnet18_ratio10/
├── test_confusion_matrix.png
└── test_metrics.json
```

如果运行 `test.py` 并指定 `--output-dir`，还会生成：

```text
test_metrics.json
confusion_matrix.png
```

---

## 15. 从“代码层面”到“报告层面”应如何理解这份 baseline

你在写实验报告时，可以把当前代码描述为：

> 本实验首先实现了一个监督学习 baseline。我们直接使用带标签的 CIFAKE 图像进行二分类训练，模型 backbone 采用适配 `32x32` 输入的轻量卷积网络，以交叉熵损失进行优化，并在测试集上使用 Accuracy 和 F1 Score 进行评价。为了模拟小样本监督场景，训练集中仅按类别均衡地抽取 `1%` 或 `10%` 的样本参与训练。该 baseline 作为后续 SimCLR 自监督预训练 + 线性评估方法的对照组。

这个描述与当前代码是严格一致的。

---

## 16. 后续扩展到 SimCLR 时，哪些模块可以复用

未来如果继续扩展完整作业，可以沿着下面的思路改：

### 16.1 可复用部分

- `utils.py`
- `plot.py`
- `net.py` 中的 backbone 构建思想
- `data_loader.py` 的目录读取和抽样逻辑
- `test.py` 的监督评估逻辑

### 16.2 需要新增或修改的部分

1. 在 `data_loader.py` 中新增 SimCLR 双视图增强。
2. 在 `net.py` 中新增：
   - encoder 抽取
   - projection head
3. 新增 `losses.py`
   - 实现 NT-Xent / InfoNCE
4. 新增预训练脚本，例如 `pretrain_simclr.py`
5. 新增线性评估脚本，例如 `linear_eval.py`
6. 调整 `README.md`，将 baseline 和 SimCLR 主体区分开来

---

## 17. 总结

当前这套代码已经完成了一个干净、可扩展的 baseline 框架，核心特点是：

1. 面向作业场景，支持 `1%/10%` 带标签训练比例。
2. 使用适配 `32x32` 图像的 `ResNet-18` / `MobileNetV2`。
3. 实现了完整的训练、测试、保存和画图链路。
4. 代码组织为独立模块，便于后续继续加入 SimCLR。

如果把这份代码比作整份作业的“地基”，那么它已经把最基础的工程骨架搭好了。后面继续做自监督部分时，重点不再是“如何从零开始组织项目”，而是“如何在现有骨架上加入 SimCLR 的算法模块”。
