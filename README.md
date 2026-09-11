# 基于 CIFAR-10 子集的表示构建与聚类分析

课堂作业项目。依据 [实验要求.pdf](实验要求.pdf) 完成表示构建、PCA、k-means 聚类、可视化、指标评价与后续 Plus 实验。

## 项目入口

- [AGENTS.md](AGENTS.md)：项目范围、长期协作规则、已确认方案与实施策略。
- [项目进度.md](项目进度.md)：当前阶段、分析判断、实验结果和下一步。
- [实验要求.pdf](实验要求.pdf)：原始作业材料。
- [`src/run_basic.py`](src/run_basic.py)：Basic 主实验脚本。

## 当前 Basic 方案

使用 CIFAR-10 `airplane`、`automobile`、`frog`、`cat`、`dog` 五类，每类 600 张；分别构建 Raw Pixel、HSV Histogram、HoG 三种表示，各自做 PCA-50D 用于聚类、PCA-2D 用于可视化。k-means 扫描 `k=2~12`，根据 SSE 与 Silhouette 曲线确定各表示的最终 k，再计算 Silhouette、CH、DB、单次聚类时间与内存并输出代表样本。

## 环境安装

建议在项目根目录创建独立 Python 环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

CIFAR-10 数据首次运行时由 `torchvision` 自动下载到本地 `data/`；该目录已被 `.gitignore` 排除。

## 运行方式

### 1. 首次扫描 k

```bash
python src/run_basic.py
```

脚本会固定随机种子 42，从 CIFAR-10 train split 中抽取五类样本，导出 `results/sample_list.csv`，并为三种表示分别生成：

```text
results/
├─ raw_pixel/
│  ├─ k_scan.csv
│  └─ k_scan.png
├─ hsv_histogram/
│  ├─ k_scan.csv
│  └─ k_scan.png
└─ hog/
   ├─ k_scan.csv
   └─ k_scan.png
```

先结合每种表示的 SSE 肘部曲线和 Silhouette-k 曲线确定最终 k，不因为真实类别数为 5 而直接固定 `k=5`。

### 2. 最终聚类

例如人工判断 Raw / HSV / HoG 的最终 k 分别为 4、5、5 时：

```bash
python src/run_basic.py --final-k-raw 4 --final-k-hsv 5 --final-k-hog 5
```

三个 `--final-k-*` 参数必须同时提供。此阶段会额外输出：

- `results/final_metrics.csv`：三种表示的最终指标、聚类时间和峰值内存增量。
- `pca2_clusters.png`：PCA-2D 聚类散点图。
- `representatives.png` / `representatives.csv`：每簇最接近中心的 3 个代表样本。
- `cluster_composition.csv`：真实类别组成，仅用于聚类后的辅助解释。

## 目录约定

正式代码放入 `src/`，必要结果表与图放入 `results/`，报告放入 `report/`。原始数据放入本地 `data/`，交付文件覆盖更新前的备份放入本地 `历史版本/`。

GitHub 仓库：[mengmcy1/cifar10-clustering](https://github.com/mengmcy1/cifar10-clustering)。
