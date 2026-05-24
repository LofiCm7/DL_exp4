！！！本README.md文档由ai生成！！！
实验报告见文件内pdf

# DL_exp4 使用说明

本项目用于完成 CIFAKE 数据集上的 AIGC 图像检测实验。本文档只介绍各个模块的作用，以及如何运行对应脚本。

## 1. 目录说明

```text
DL_exp4/
├── train.py                       # 监督学习 baseline 训练
├── test.py                        # 监督学习 baseline 测试
├── pretrain_simclr.py             # SimCLR 无监督预训练
├── linear_probe.py                # 冻结 encoder 的线性评估训练
├── test_linear_probe.py           # 线性评估模型测试
├── additional_experiment_1.py     # 附加实验 1：不同对比学习损失比较
├── additional_experiment_2.py     # 附加实验 2：不同 projection head 比较
├── data_loader.py                 # 数据读取与数据增强定义
├── net.py                         # 模型构建
├── simclr.py                      # SimCLR 相关模块
├── plot.py                        # 曲线与混淆矩阵绘图
└── utils.py                       # 通用工具函数
```

## 2. 数据集目录格式

项目默认数据集目录为：

```text
DL_exp4/data4/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

## 3. 统一运行方式

建议在项目根目录下运行所有脚本：

```bash
cd <project_root>
```

其中 `<project_root>` 指当前仓库所在目录，目录内应包含 `DL_exp4/` 文件夹。

运行 Python 脚本时，请使用你本地已经配置好的 Python 环境或虚拟环境解释器。例如：

```bash
python
```

或者：

```bash
/path/to/your/python
```

下面的示例命令统一使用 `python` 表示解释器，你可以根据自己的环境替换成实际可用的 Python 路径。

## 4. 各模块作用与运行方法

### 4.1 监督学习 baseline 训练

作用：训练一个不经过 SimCLR 预训练的监督学习分类模型。

运行示例：

```bash
python DL_exp4/train.py \
  --data-root DL_exp4/data4 \
  --model resnet18 \
  --labeled-ratio 0.1
```

### 4.2 监督学习 baseline 测试

作用：使用训练好的 baseline 权重在测试集上做最终测试。

运行示例：

```bash
python DL_exp4/test.py \
  --data-root DL_exp4/data4 \
  --checkpoint <baseline_checkpoint.pt> \
  --model resnet18
```

### 4.3 SimCLR 无监督预训练

作用：对 encoder 做 SimCLR 预训练。

运行示例：

```bash
python DL_exp4/pretrain_simclr.py \
  --data-root DL_exp4/data4 \
  --model resnet18 \
  --augment simclr_v1
```

如果需要切换增强方式，可将 `--augment` 改为 `simclr_v2`。

### 4.4 线性评估训练

作用：加载 SimCLR 预训练好的 encoder，冻结 encoder，只训练线性分类器。

运行示例：

```bash
python DL_exp4/linear_probe.py \
  --data-root DL_exp4/data4 \
  --pretrained-checkpoint <pretrain_checkpoint.pt> \
  --labeled-ratio 0.1
```

### 4.5 线性评估测试

作用：对训练好的 linear probe 模型在测试集上做最终测试。

运行示例：

```bash
python DL_exp4/test_linear_probe.py \
  --data-root DL_exp4/data4 \
  --checkpoint <linear_probe_checkpoint.pt>
```

### 4.6 附加实验 1

作用：比较不同对比学习损失函数的效果。

运行示例：

```bash
python DL_exp4/additional_experiment_1.py
```

### 4.7 附加实验 2

作用：比较不同 projection head 结构的效果。

运行示例：

```bash
python DL_exp4/additional_experiment_2.py
```

## 5. 输出目录说明

所有结果默认保存在 `DL_exp4/output/` 下。

常见输出内容包括：

1. `weights/`：最佳模型权重
2. `curves/`：训练曲线图片与 CSV 数据
3. `matrices/`：混淆矩阵
4. `results/`：测试指标结果
