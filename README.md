# 基于 CIFAKE 的 AIGC 图像检测项目说明书

这份文档说明当前项目的完整作业实现、脚本用途、运行顺序、参数含义以及代码结构。当前代码已经完整覆盖附加实验之前的全部主要求：

1. 监督学习 baseline
2. SimCLR 无监督预训练
3. 冻结 Encoder 的线性评估
4. baseline 与线性评估两条路线的最终测试

## 超参数总览与修改方法

当前项目的参数分两类：

1. 命令行参数
   直接在运行脚本时通过 `--参数名 参数值` 指定。
2. 代码内固定参数
   需要直接修改 Python 文件中的实现。

### 0. 统一执行规范

所有脚本都应显式使用以下解释器：

```bash
/data/sczli/conda_env/pytorch_env/bin/python
```

下面所有命令都默认在仓库根目录 `/home/sczli/Programs/DeepLearning/02_DLexp4` 下执行。

### A. `train.py` 参数

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/train.py [参数]
```

| 参数名 | 默认值 | 作用 |
| --- | --- | --- |
| `--data-root` | `DL_exp4/data4` | 数据集根目录，读取 `train/` |
| `--output-dir` | `DL_exp4/output/train_val` | baseline 训练输出目录 |
| `--model` | `resnet18` | backbone，支持 `resnet18` 和 `mobilenet_v2` |
| `--labeled-ratio` | `0.1` | 训练划分中使用的标签比例 |
| `--val-ratio` | `0.2` | 验证集比例 |
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
| `--output-dir` | `DL_exp4/output/simclr_pretrain` | SimCLR 预训练输出目录 |
| `--model` | `resnet18` | Encoder backbone |
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
| `--pretrained-checkpoint` | `DL_exp4/output/simclr_pretrain/best_pretrain.pt` | SimCLR 预训练权重 |
| `--output-dir` | `DL_exp4/output/linear_probe` | 线性评估输出目录 |
| `--model` | `None` | 可选，手动指定 encoder 结构 |
| `--labeled-ratio` | `0.1` | 标签比例 |
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
| `--checkpoint` | 必填 | `best_linear_probe.pt` 路径 |
| `--model` | `None` | 可选，手动指定 encoder 结构 |
| `--batch-size` | `256` | 测试 batch 大小 |
| `--num-workers` | `4` | DataLoader 子进程数 |
| `--image-size` | `32` | 输入尺寸 |
| `--device` | `cuda` | 测试设备 |
| `--output-dir` | `DL_exp4/output/final_test_linear_probe` | 线性评估测试输出目录 |

### F. 当前固定在代码中的关键设置

1. `data_loader.py` 中的 baseline 增强
   `Resize + RandomCrop + HorizontalFlip + ColorJitter + Normalize`
2. `data_loader.py` 中的 SimCLR 强增强
   `RandomResizedCrop + HorizontalFlip + RandomRotation + ColorJitter + RandomGrayscale + GaussianBlur + Normalize`
3. `net.py` 中对 `32x32` 输入的 backbone 改造
4. `simclr.py` 中的 Projection Head 结构
   `Linear -> ReLU -> Linear`
5. 最优模型保存标准
   baseline 和 linear probe 按验证集 `accuracy` 保存最佳权重；SimCLR 预训练按训练 `loss` 最低保存最佳权重

## 从开始到结束的实验操作流程

### Step 0. 进入仓库根目录并固定解释器

```bash
cd /home/sczli/Programs/DeepLearning/02_DLexp4
```

后续统一使用：

```bash
/data/sczli/conda_env/pytorch_env/bin/python
```

原因：

1. 当前脚本默认路径都以 `DL_exp4/...` 为根
2. 显式指定解释器可以避免环境歧义

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
2. 先验证监督训练与评估链路是否正常
3. 后续需要和 SimCLR 结果比较

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
2. 最终报告应使用测试集结果

### Step 3. 做 SimCLR 无监督预训练

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/pretrain_simclr.py \
  --data-root DL_exp4/data4 \
  --model resnet18 \
  --epochs 20 \
  --batch-size 256 \
  --temperature 0.5 \
  --projection-dim 64 \
  --output-dir DL_exp4/output/simclr_pretrain/resnet18
```

为什么这样做：

1. SimCLR 先学无监督表示
2. 后续线性评估要复用这里的 Encoder

### Step 4. 做冻结线性评估

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/linear_probe.py \
  --data-root DL_exp4/data4 \
  --pretrained-checkpoint DL_exp4/output/simclr_pretrain/resnet18/weights/best_pretrain.pt \
  --labeled-ratio 0.1 \
  --val-ratio 0.2 \
  --epochs 10 \
  --batch-size 128 \
  --output-dir DL_exp4/output/linear_probe/resnet18_ratio10
```

为什么冻结 Encoder：

1. 这是 SimCLR 标准线性评估方式
2. 它衡量预训练表示本身的可分性

### Step 5. 对线性评估模型做最终测试

```bash
/data/sczli/conda_env/pytorch_env/bin/python DL_exp4/test_linear_probe.py \
  --data-root DL_exp4/data4 \
  --checkpoint DL_exp4/output/linear_probe/resnet18_ratio10/weights/best_linear_probe.pt \
  --output-dir DL_exp4/output/final_test_linear_probe/resnet18_ratio10
```

为什么还要测试：

1. 线性评估也必须像 baseline 一样单独在测试集上评估
2. 最终报告比较必须来自同一层级的结果

### Step 6. 扩展到完整实验矩阵

建议至少完成：

1. `baseline + resnet18 + 1%`
2. `baseline + resnet18 + 10%`
3. `linear probe + resnet18 + 1%`
4. `linear probe + resnet18 + 10%`
5. `baseline + mobilenet_v2 + 1%`
6. `baseline + mobilenet_v2 + 10%`
7. `linear probe + mobilenet_v2 + 1%`
8. `linear probe + mobilenet_v2 + 10%`

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
9. `plot.py`
   训练曲线与混淆矩阵绘图
10. `utils.py`
   公共工具函数

## 代码详解

### 1. `data_loader.py`

这个模块负责整个项目的数据准备。

#### 1.1 baseline 数据增强

`build_transforms()` 返回两套变换：

1. 训练增强
2. 评估增强

训练增强比评估增强更强，因为训练需要泛化，验证和测试需要稳定。

#### 1.2 SimCLR 双视图增强

`build_simclr_transform()` 构造更强的随机增强，并用 `TwoCropTransform` 对同一张图像生成两个视图。这是 SimCLR 的核心数据形式。

#### 1.3 小比例标签抽样

`_sample_labeled_indices()` 按类别分层抽样 `1%` 或 `10%` 标签样本，避免极小比例时某一类被完全抽空。

#### 1.4 训练/验证划分

`_split_train_val_indices()` 按类别分层切分 `train` 与 `val`，保证验证集类别分布稳定。

#### 1.5 三类 DataLoader

当前保留三类 DataLoader 构建函数：

1. `build_train_val_dataloaders()`
   服务于 baseline 和 linear probe
2. `build_simclr_pretrain_dataloader()`
   服务于 SimCLR 预训练
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

当前 projection head 是最经典的两层 MLP：

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

这个类把冻结的 encoder 和一个线性分类头组合起来。它在前向传播中始终让 encoder 保持 `eval()` 并关闭梯度，只训练线性层。

#### 3.6 `load_linear_probe_model()`

这个函数负责从 `best_linear_probe.pt` 中恢复完整线性探针模型，供 `test_linear_probe.py` 使用。

### 4. `train.py`

这是监督 baseline 训练入口。

1. 构建 `train/val` DataLoader
2. 创建分类模型
3. 每轮做训练与验证
4. 按验证集 `accuracy` 保存最佳权重
5. 保存训练曲线和最佳验证混淆矩阵

当前训练输出只保留三类文件：

1. `weights/best_model.pt`
2. `curves/training_curves.png`
3. `matrices/best_val_confusion_matrix.png`

### 5. `test.py`

这是 baseline 最终测试入口，同时 `evaluate()` 被 baseline 和 linear probe 共用。

它负责：

1. 构建测试集 DataLoader
2. 恢复 baseline 最优权重
3. 计算测试集 `loss / accuracy / f1`
4. 输出测试结果与混淆矩阵

测试输出只保留两类文件：

1. `results/test_metrics.json`
2. `matrices/test_confusion_matrix.png`

### 6. `pretrain_simclr.py`

这是 SimCLR 无监督预训练入口。

每个 batch 的流程是：

1. 取同一张图像的两个增强视图
2. 送入同一个 Encoder + Projection Head
3. 用 NT-Xent 计算对比损失
4. 反向传播更新参数

它只保留必须的输出：

1. `weights/best_pretrain.pt`
2. `curves/pretraining_loss_curve.png`

### 7. `linear_probe.py`

这是冻结线性评估入口。

1. 加载 SimCLR 预训练 encoder
2. 冻结 encoder
3. 只训练线性分类头
4. 在验证集上选最佳模型
5. 保存曲线与混淆矩阵

当前输出只保留：

1. `weights/best_linear_probe.pt`
2. `curves/training_curves.png`
3. `matrices/best_val_confusion_matrix.png`

### 8. `test_linear_probe.py`

这是线性评估模型最终测试入口。

1. 构建测试集 DataLoader
2. 恢复 `best_linear_probe.pt`
3. 计算测试集指标
4. 保存测试结果与混淆矩阵

输出只保留：

1. `results/test_metrics.json`
2. `matrices/test_confusion_matrix.png`

### 9. `plot.py`

这个模块只保留三个绘图函数：

1. `plot_training_history()`
2. `plot_pretraining_history()`
3. `plot_confusion_matrix()`

它们都直接接收输出路径，便于按 `weights / curves / matrices / results` 分类保存。

### 10. `utils.py`

这个模块保留了当前仍然必要的公共函数：

1. `seed_everything()`
2. `ensure_dir()`
3. `save_json()`
4. `get_device()`

## 输出目录分类说明

为了避免输出混乱，当前所有脚本都按功能分类保存结果。

### 训练脚本输出分类

训练脚本包括：

1. `train.py`
2. `pretrain_simclr.py`
3. `linear_probe.py`

它们的输出分类为：

1. `weights/`
   保存最佳模型权重
2. `curves/`
   保存训练曲线
3. `matrices/`
   保存验证阶段混淆矩阵

注意：

1. `pretrain_simclr.py` 没有验证集和混淆矩阵，所以只会产生 `weights/` 和 `curves/`
2. `train.py` 和 `linear_probe.py` 会完整产生 `weights/`、`curves/`、`matrices/`

### 测试脚本输出分类

测试脚本包括：

1. `test.py`
2. `test_linear_probe.py`

它们的输出分类为：

1. `results/`
   保存测试指标 JSON
2. `matrices/`
   保存测试集混淆矩阵

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

当前还没有实现的是附加实验本身，例如：

1. 其他对比学习 loss
2. Projection Head 结构分析的多版本实现
3. 温度系数、batch size 等附加实验专门脚本化管理

这些部分可以在当前主链路稳定后继续扩展。
