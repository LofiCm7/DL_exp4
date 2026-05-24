# 基于 CIFAKE 的 AIGC 图像检测项目说明书

本文档对应当前 `DL_exp4` 目录下的实际代码实现。当前代码已经完整覆盖附加实验之前的主实验链路，并在此基础上补充了两套可对比的 SimCLR 增强方案、统一的曲线数据保存，以及独立的增强可视化脚本。

当前已实现内容：

1. 监督学习 baseline 训练与最终测试
2. SimCLR 无监督预训练
3. 冻结 Encoder 的线性评估
4. baseline 与 linear probe 两条路线的最终测试
5. 两套可切换的 SimCLR 增强方案
6. 训练曲线图片与对应 CSV 数据保存
7. 独立的增强可视化脚本

## 超参数总览与修改方法

参数分为两类：

1. 命令行参数
   直接在运行脚本时通过 `--参数名 参数值` 修改。
2. 代码内固定参数
   需要在 Python 文件中修改，例如具体增强算子的组合方式。

### 0. 统一执行规范

所有脚本都必须显式使用以下解释器：

```bash
/data/sczli/conda_env/pytorch_env/bin/python
```

下面所有命令都默认在仓库根目录 `/home/sczli/Programs/DeepLearning/02_DLexp4` 下执行。

原因：

1. 当前脚本默认路径都写成了 `DL_exp4/...`
2. 显式指定解释器可以避免系统默认环境和自动激活环境带来的歧义
3. 当前项目遵守只读环境约束，不依赖任何安装新包的操作

### A. `train.py` 参数

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/train.py [参数]
```

| 参数名 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-root` | `DL_exp4/data4` | 数据集根目录，读取 `train/` |
| `--output-dir` | `DL_exp4/output/train_val` | baseline 训练输出目录 |
| `--model` | `resnet18` | backbone，支持 `resnet18` 和 `mobilenet_v2` |
| `--labeled-ratio` | `0.1` | 训练划分中实际使用的有标签比例 |
| `--val-ratio` | `0.2` | 从训练集切出的验证集比例 |
| `--image-size` | `32` | 输入尺寸 |
| `--epochs` | `10` | 训练轮数 |
| `--batch-size` | `128` | batch 大小 |
| `--num-workers` | `4` | DataLoader 子进程数 |
| `--lr` | `1e-3` | 学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--seed` | `42` | 随机种子 |
| `--device` | `cuda` | 训练设备 |

### B. `test.py` 参数

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/test.py [参数]
```

| 参数名 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-root` | `DL_exp4/data4` | 测试集根目录 |
| `--checkpoint` | 必填 | baseline 最优权重 |
| `--model` | `resnet18` | 模型结构 |
| `--batch-size` | `256` | 测试 batch 大小 |
| `--num-workers` | `4` | DataLoader 子进程数 |
| `--image-size` | `32` | 输入尺寸 |
| `--device` | `cuda` | 测试设备 |
| `--output-dir` | `DL_exp4/output/final_test` | baseline 测试输出目录 |

### C. `pretrain_simclr.py` 参数

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/pretrain_simclr.py [参数]
```

| 参数名 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-root` | `DL_exp4/data4` | 数据集根目录，读取 `train/` |
| `--output-dir` | `DL_exp4/output/simclr_pretrain` | SimCLR 预训练输出根目录 |
| `--model` | `resnet18` | Encoder backbone |
| `--augment` | `simclr_v1` | SimCLR 增强方案，可选 `simclr_v1`、`simclr_v2` |
| `--image-size` | `32` | 输入尺寸 |
| `--epochs` | `20` | 预训练轮数 |
| `--batch-size` | `256` | 预训练 batch 大小 |
| `--num-workers` | `4` | DataLoader 子进程数 |
| `--lr` | `1e-3` | 学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--temperature` | `0.5` | NT-Xent 温度系数 |
| `--projection-dim` | `64` | 投影维度 |
| `--projection-hidden-dim` | `None` | 投影头隐层维度 |
| `--seed` | `42` | 随机种子 |
| `--device` | `cuda` | 训练设备 |

### D. `linear_probe.py` 参数

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/linear_probe.py [参数]
```

| 参数名 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-root` | `DL_exp4/data4` | 数据集根目录 |
| `--pretrained-checkpoint` | `DL_exp4/output/simclr_pretrain/simclr_v1/weights/best_pretrain.pt` | SimCLR 预训练权重 |
| `--output-dir` | `DL_exp4/output/linear_probe` | 线性评估输出根目录 |
| `--model` | `None` | 可选，手动指定 encoder 结构 |
| `--augment` | `simclr_v1` | 输出目录所归属的增强方案标签 |
| `--labeled-ratio` | `0.1` | 有标签比例 |
| `--val-ratio` | `0.2` | 验证集比例 |
| `--image-size` | `32` | 输入尺寸 |
| `--epochs` | `10` | 训练轮数 |
| `--batch-size` | `128` | batch 大小 |
| `--num-workers` | `4` | DataLoader 子进程数 |
| `--lr` | `1e-3` | 学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--seed` | `42` | 随机种子 |
| `--device` | `cuda` | 训练设备 |

### E. `test_linear_probe.py` 参数

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/test_linear_probe.py [参数]
```

| 参数名 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-root` | `DL_exp4/data4` | 测试集根目录 |
| `--checkpoint` | `DL_exp4/output/linear_probe/simclr_v1/weights/best_linear_probe.pt` | 线性评估最优权重 |
| `--model` | `None` | 可选，手动指定 encoder 结构 |
| `--augment` | `simclr_v1` | 输出目录所归属的增强方案标签 |
| `--batch-size` | `256` | 测试 batch 大小 |
| `--num-workers` | `4` | DataLoader 子进程数 |
| `--image-size` | `32` | 输入尺寸 |
| `--device` | `cuda` | 测试设备 |
| `--output-dir` | `DL_exp4/output/final_test_linear_probe` | 线性评估测试输出根目录 |

### F. `visualize_augmentations.py` 参数

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/visualize_augmentations.py [参数]
```

| 参数名 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-root` | `DL_exp4/data4` | 数据集根目录，读取 `train/` |
| `--output-dir` | `DL_exp4/output/augmentation_views` | 增强可视化输出根目录 |
| `--augmentations` | `simclr_v1 simclr_v2` | 需要展示的增强方案列表 |
| `--image-size` | `32` | 可视化时统一缩放尺寸 |
| `--num-images` | `4` | 抽取多少张原图做展示 |
| `--num-pairs` | `2` | 每种增强方案下，为每张图生成多少组 view 对 |
| `--seed` | `42` | 抽样随机种子 |

### G. 当前固定在代码中的关键设置

1. `data_loader.py` 中 baseline 增强为：
   `Resize + RandomCrop + HorizontalFlip + ColorJitter + Normalize`
2. `data_loader.py` 中 SimCLR 提供两套增强方案：
   `simclr_v1 = RandomResizedCrop + HorizontalFlip + Rotation + ColorJitter + Grayscale + GaussianBlur + Normalize`
3. `data_loader.py` 中 `simclr_v2` 为明显不同的增强方案：
   `RandomResizedCrop + HorizontalFlip + RandomAffine + Autocontrast + Equalize + Posterize + Solarize + Normalize`
4. `net.py` 中对 `32x32` 输入做了 backbone 改造
5. `simclr.py` 中 Projection Head 固定为：
   `Linear -> ReLU -> Linear`
6. 最优模型保存标准：
   baseline 与 linear probe 按验证集 `accuracy` 保存最佳权重；SimCLR 按训练 `loss` 最低保存最佳权重

## 从开始到结束的实验操作流程

### Step 0. 进入仓库根目录并固定解释器

```bash
cd /home/sczli/Programs/DeepLearning/02_DLexp4
```

后续统一使用：

```bash
/data/sczli/conda_env/pytorch_env/bin/python
```

### Step 1. 训练监督 baseline

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/train.py \
  --data-root DL_exp4/data4 \
  --model resnet18 \
  --labeled-ratio 0.1 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir DL_exp4/output/train_val/resnet18_ratio10
```

为什么先做：

1. baseline 是对照组
2. 先验证监督训练链路正常
3. 后续所有 SimCLR 结果都要与它比较

### Step 2. 对 baseline 做最终测试

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/test.py \
  --data-root DL_exp4/data4 \
  --checkpoint DL_exp4/output/train_val/resnet18_ratio10/weights/best_model.pt \
  --model resnet18 \
  --output-dir DL_exp4/output/final_test/resnet18_ratio10
```

为什么单独测试：

1. 验证集只用于选模型
2. 最终报告必须使用测试集结果

### Step 3. 先可视化不同增强方式下的 view

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/visualize_augmentations.py \
  --data-root DL_exp4/data4 \
  --augmentations simclr_v1 simclr_v2 \
  --num-images 4 \
  --num-pairs 2 \
  --output-dir DL_exp4/output/augmentation_views
```

为什么建议先做：

1. 先确认两种增强确实有明显区别
2. 便于在报告中展示不同 view 的效果
3. 能避免训练后才发现增强差异不明显

### Step 4. 用第一种增强做 SimCLR 预训练

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/pretrain_simclr.py \
  --data-root DL_exp4/data4 \
  --model resnet18 \
  --augment simclr_v1 \
  --epochs 20 \
  --batch-size 256 \
  --temperature 0.5 \
  --projection-dim 64 \
  --output-dir DL_exp4/output/simclr_pretrain
```

为什么这样做：

1. SimCLR 先学习无监督表示
2. `simclr_v1` 是第一套增强方案的完整预训练结果
3. 输出会自动写入 `DL_exp4/output/simclr_pretrain/simclr_v1/`

### Step 5. 用第二种增强做 SimCLR 预训练

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/pretrain_simclr.py \
  --data-root DL_exp4/data4 \
  --model resnet18 \
  --augment simclr_v2 \
  --epochs 20 \
  --batch-size 256 \
  --temperature 0.5 \
  --projection-dim 64 \
  --output-dir DL_exp4/output/simclr_pretrain
```

为什么要单独再跑一遍：

1. 对比实验必须让两种增强各自产生独立预训练权重
2. 输出会自动写入 `DL_exp4/output/simclr_pretrain/simclr_v2/`
3. 两种增强的目录结构完全一致，便于后期横向比较

### Step 6. 用 `simclr_v1` 权重做冻结线性评估

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/linear_probe.py \
  --data-root DL_exp4/data4 \
  --pretrained-checkpoint DL_exp4/output/simclr_pretrain/simclr_v1/weights/best_pretrain.pt \
  --augment simclr_v1 \
  --labeled-ratio 0.1 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir DL_exp4/output/linear_probe
```

为什么这里也要带 `--augment`：

1. 线性评估结果也要按增强方案分文件夹保存
2. 输出会自动写入 `DL_exp4/output/linear_probe/simclr_v1/`

### Step 7. 用 `simclr_v2` 权重做冻结线性评估

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/linear_probe.py \
  --data-root DL_exp4/data4 \
  --pretrained-checkpoint DL_exp4/output/simclr_pretrain/simclr_v2/weights/best_pretrain.pt \
  --augment simclr_v2 \
  --labeled-ratio 0.1 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir DL_exp4/output/linear_probe
```

### Step 8. 对 `simclr_v1` 线性评估模型做最终测试

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/test_linear_probe.py \
  --data-root DL_exp4/data4 \
  --checkpoint DL_exp4/output/linear_probe/simclr_v1/weights/best_linear_probe.pt \
  --augment simclr_v1 \
  --output-dir DL_exp4/output/final_test_linear_probe
```

### Step 9. 对 `simclr_v2` 线性评估模型做最终测试

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/test_linear_probe.py \
  --data-root DL_exp4/data4 \
  --checkpoint DL_exp4/output/linear_probe/simclr_v2/weights/best_linear_probe.pt \
  --augment simclr_v2 \
  --output-dir DL_exp4/output/final_test_linear_probe
```

### Step 10. 做最终对比整理

建议对比以下结果：

1. baseline 与 linear probe 的测试集 `accuracy / f1`
2. `simclr_v1` 与 `simclr_v2` 的预训练 loss 曲线
3. `simclr_v1` 与 `simclr_v2` 的线性评估训练曲线
4. 不同增强方式下的 view 可视化图
5. 不同模型或不同标签比例下导出的 CSV 曲线数据

## 项目结构与脚本职责

```text
DL_exp4/
├── data4/
│   ├── train/
│   └── test/
├── data_loader.py
├── net.py
├── simclr.py
├── train.py
├── test.py
├── pretrain_simclr.py
├── linear_probe.py
├── test_linear_probe.py
├── visualize_augmentations.py
├── plot.py
├── utils.py
├── README.pdf
└── README.md
```

各文件职责：

1. `data_loader.py`
   数据增强、分层划分、小比例抽样、SimCLR 双视图 DataLoader
2. `net.py`
   backbone 构建，区分分类模型和纯 encoder
3. `simclr.py`
   NT-Xent 损失、Projection Head、SimCLRModel、预训练恢复、线性探针恢复
4. `train.py`
   监督 baseline 训练
5. `test.py`
   监督 baseline 最终测试，并提供通用 `evaluate()`
6. `pretrain_simclr.py`
   SimCLR 无监督预训练
7. `linear_probe.py`
   冻结 Encoder 的线性评估
8. `test_linear_probe.py`
   线性评估模型最终测试
9. `visualize_augmentations.py`
   独立展示不同增强方案下的 view 可视化结果
10. `plot.py`
   训练曲线与混淆矩阵绘图
11. `utils.py`
   公共工具函数

## 代码详解

### 1. `data_loader.py`

这个模块负责整个项目的数据准备。

#### 1.1 baseline 数据增强

`build_transforms()` 返回两套变换：

1. 训练增强
2. 评估增强

训练增强用于监督 baseline 和线性评估训练；评估增强用于验证和测试，保证结果稳定。

#### 1.2 SimCLR 双视图增强

当前 SimCLR 路线拆成了两个层次：

1. `build_simclr_view_transform()`
   生成单个 view 的增强流水线
2. `build_simclr_transform()`
   用 `TwoCropTransform` 对同一张图像连续调用两次单 view 变换，得到 `view1` 与 `view2`

这意味着：

1. 同一次训练中，`view1` 和 `view2` 使用同一套增强规则
2. 但每次调用时采样到的随机参数不同
3. 因此同一张图会得到两个不同但相关的随机视图

#### 1.3 两套 SimCLR 增强方案

当前提供两套可直接对比的增强：

1. `simclr_v1`
   以颜色抖动、灰度化、模糊和轻旋转为主，偏向颜色与局部纹理扰动
2. `simclr_v2`
   以仿射变换、自动对比度、均衡化、色阶压缩和太阳化为主，偏向几何和强烈色调重映射

#### 1.4 小比例标签抽样

`_sample_labeled_indices()` 按类别分层抽样 `1%` 或 `10%` 标签样本，避免极小比例时某一类被完全抽空。

#### 1.5 训练/验证划分

`_split_train_val_indices()` 按类别分层切分 `train` 与 `val`，保证验证集类别分布稳定。

#### 1.6 三类 DataLoader

当前保留三类 DataLoader 构建函数：

1. `build_train_val_dataloaders()`
   服务于 baseline 和 linear probe
2. `build_simclr_pretrain_dataloader()`
   服务于 SimCLR 预训练，并通过 `augment` 切换增强方案
3. `build_test_dataloader()`
   服务于两条测试链路

### 2. `net.py`

这个模块统一管理 backbone。

#### 2.1 为什么既要 `build_model()` 又要 `build_encoder()`

1. `build_model()` 用于监督分类，返回完整分类器
2. `build_encoder()` 用于 SimCLR 和 linear probe，返回纯特征提取器

#### 2.2 为什么要改 ResNet-18 和 MobileNetV2

因为当前任务输入是 `32x32`，原始 ImageNet 结构下采样过猛，因此需要：

1. 调整 ResNet 第一层卷积并去掉最大池化
2. 调整 MobileNetV2 第一层卷积步长

### 3. `simclr.py`

这个模块包含 SimCLR 路线的核心代码。

#### 3.1 `nt_xent_loss()`

这是手写的 NT-Xent / InfoNCE 损失。它把两个视图的投影特征拼接成一个 `2N` 大小的 batch，构建对比 logits，再把配对视图作为正样本，其余样本作为负样本。

#### 3.2 `ProjectionHead`

当前 Projection Head 是两层 MLP：

1. `Linear`
2. `ReLU`
3. `Linear`

#### 3.3 `SimCLRModel`

这个类把：

1. Encoder
2. Projection Head

组装成一个完整的 SimCLR 训练模型。

#### 3.4 `load_pretrained_encoder()`

这个函数负责从 SimCLR 预训练 checkpoint 中恢复 encoder，供 `linear_probe.py` 使用。

#### 3.5 `LinearProbeModel`

这个类把冻结的 encoder 和一个线性分类头组合起来。前向传播中始终让 encoder 保持 `eval()` 并关闭梯度，只训练线性层。

#### 3.6 `load_linear_probe_model()`

这个函数负责从 `best_linear_probe.pt` 中恢复完整线性探针模型，供 `test_linear_probe.py` 使用。

### 4. `train.py`

这是监督 baseline 训练入口。

它负责：

1. 构建 `train/val` DataLoader
2. 创建分类模型
3. 每轮做训练与验证
4. 按验证集 `accuracy` 保存最佳权重
5. 保存训练曲线图片
6. 保存训练曲线原始 CSV 数据
7. 保存最佳验证混淆矩阵

### 5. `test.py`

这是 baseline 最终测试入口，同时 `evaluate()` 被 baseline 和 linear probe 共用。

它负责：

1. 构建测试集 DataLoader
2. 恢复 baseline 最优权重
3. 计算测试集 `loss / accuracy / f1`
4. 保存测试结果与混淆矩阵

### 6. `pretrain_simclr.py`

这是 SimCLR 无监督预训练入口。

每个 batch 的流程是：

1. 取同一张图像的两个增强视图
2. 送入同一个 Encoder + Projection Head
3. 用 NT-Xent 计算对比损失
4. 反向传播更新参数
5. 按增强方案自动写入独立输出子目录

同时它会保存：

1. 最佳预训练权重
2. loss 曲线图片
3. loss 曲线 CSV 数据

### 7. `linear_probe.py`

这是冻结线性评估入口。

它负责：

1. 加载 SimCLR 预训练 encoder
2. 冻结 encoder
3. 只训练线性分类头
4. 在验证集上选最佳模型
5. 保存训练曲线图片
6. 保存训练曲线 CSV 数据
7. 保存最佳验证混淆矩阵
8. 按增强方案分目录输出

### 8. `test_linear_probe.py`

这是线性评估模型最终测试入口。

它负责：

1. 构建测试集 DataLoader
2. 恢复 `best_linear_probe.pt`
3. 计算测试集指标
4. 保存测试结果与混淆矩阵
5. 按增强方案分目录输出

### 9. `visualize_augmentations.py`

这是一个与训练流程完全独立的脚本。

它负责：

1. 从训练集抽取若干原图
2. 对每种增强方案分别生成多组 `view1/view2`
3. 将 `original + pair1_view1 + pair1_view2 + ...` 拼成网格图
4. 按增强方案分别保存

这个脚本的作用是直接展示：

1. 不同增强方案之间的整体差别
2. 同一种增强方案下，view1 与 view2 的随机变化

### 10. `plot.py`

这个模块只保留三个绘图函数：

1. `plot_training_history()`
2. `plot_pretraining_history()`
3. `plot_confusion_matrix()`

### 11. `utils.py`

这个模块保留了当前仍然必要的公共函数：

1. `seed_everything()`
2. `ensure_dir()`
3. `save_json()`
4. `save_history_csv()`
5. `get_device()`

## 输出目录分类说明

为了避免输出混乱，当前所有脚本都按功能分类保存结果。

### 1. baseline 训练输出

示例：`DL_exp4/output/train_val/resnet18_ratio10/`

```text
weights/
curves/
matrices/
```

其中：

1. `weights/best_model.pt`
2. `curves/training_curves.png`
3. `curves/training_history.csv`
4. `matrices/best_val_confusion_matrix.png`

### 2. baseline 测试输出

示例：`DL_exp4/output/final_test/resnet18_ratio10/`

```text
results/
matrices/
```

其中：

1. `results/test_metrics.json`
2. `matrices/test_confusion_matrix.png`

### 3. SimCLR 预训练输出

两种增强方式的目录结构完全一致：

```text
DL_exp4/output/simclr_pretrain/
├── simclr_v1/
└── simclr_v2/
```

每个增强子目录内部为：

```text
weights/
curves/
```

其中：

1. `weights/best_pretrain.pt`
2. `curves/pretraining_loss_curve.png`
3. `curves/pretraining_history.csv`

### 4. 线性评估训练输出

两种增强方式的目录结构完全一致：

```text
DL_exp4/output/linear_probe/
├── simclr_v1/
└── simclr_v2/
```

每个增强子目录内部为：

```text
weights/
curves/
matrices/
```

其中：

1. `weights/best_linear_probe.pt`
2. `curves/training_curves.png`
3. `curves/training_history.csv`
4. `matrices/best_val_confusion_matrix.png`

### 5. 线性评估测试输出

两种增强方式的目录结构完全一致：

```text
DL_exp4/output/final_test_linear_probe/
├── simclr_v1/
└── simclr_v2/
```

每个增强子目录内部为：

```text
results/
matrices/
```

其中：

1. `results/test_metrics.json`
2. `matrices/test_confusion_matrix.png`

### 6. 增强可视化输出

```text
DL_exp4/output/augmentation_views/
├── simclr_v1/
└── simclr_v2/
```

每个增强子目录内部为：

1. `augmentation_views.png`

## 曲线数据如何用于后续对比

当前 `train.py`、`pretrain_simclr.py`、`linear_probe.py` 都会在 `curves/` 下额外保存 CSV 文件。

使用方式：

1. 不同实验各自训练后，保留各自目录下的 CSV
2. 后续用任意脚本读取多个 CSV
3. 按相同横轴 `epoch` 将多个实验结果画在同一张图上

例如：

1. 对比不同 backbone 的 baseline 曲线
2. 对比 `simclr_v1` 与 `simclr_v2` 的预训练 loss 曲线
3. 对比不同标签比例下的 linear probe 曲线

## 当前代码保留了哪些作业主要求

当前代码完整保留了附加实验之前的所有主要要求：

1. 使用 CIFAKE 数据集做 `REAL/FAKE` 分类
2. 使用 `1% / 10%` 标签比例做对比
3. 提供监督 baseline
4. 提供 SimCLR 双视图增强
5. 提供 Encoder + Projection Head
6. 提供自写 NT-Xent / InfoNCE 损失
7. 提供无监督预训练脚本
8. 提供冻结 Encoder 的线性评估脚本
9. 提供 baseline 与 linear probe 两条路线的最终测试脚本
10. 提供训练曲线、测试结果与混淆矩阵输出
11. 提供不同增强方案的独立结果目录
12. 提供不同增强方案下 view 的独立可视化脚本

当前仍未实现的是附加实验本身，例如：

1. 其他对比学习 loss 的系统化对比
2. Projection Head 结构分析的多版本实验
3. 温度系数、batch size 等附加实验的批量化管理
4. 更完整的附加实验结果汇总脚本
