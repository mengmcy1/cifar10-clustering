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

本机优先复用 Conda `general` 环境（Python 3.13），先确认解释器和已有依赖，仅补装缺失包。其他机器可参考完整依赖清单：

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

### 3. 全部 k 的二维图与簇组成统计

已有扫描结果后，在 `general` 环境执行：

```bash
python src/visualize_k_scan.py --results-dir results/n3000
```

复用原实现和样本清单，重新拟合各 k 并核对 SSE，为每种表示补充 `pca2_all_k.png`（k=2～12 共 11 个子图）和 `all_k_cluster_composition.csv`（各簇样本数、占比、五类数量）。同一表示使用固定二维坐标和坐标范围；不同 k 的相同颜色不代表同一个簇。此命令不重写原扫描或最终聚类文件；重复覆盖补充图表前仍按项目规则备份。

### 4. 嵌套样本量实验

在 `general` 环境、项目根目录依次执行，每条命令使用独立 Python 进程：

```bash
python src/run_scaling.py --samples-per-class 200
python src/run_scaling.py --samples-per-class 400
python src/run_scaling.py --samples-per-class 600
```

以 `results/n3000/sample_list.csv` 为母集，按母集行顺序每类取前 200/400/600 张，形成 1000⊂2000⊂3000 的嵌套子集。三个规模分别拟合 PCA-50，固定 k=2，不扫描 k、不生成二维图。使用本地 CIFAR-10，不自动下载。

每个 `results/n{N}/` 输出 `scaling_sample_list.csv`（保留母集 sample_id）和 `scaling_metrics.csv`。计时仅覆盖最终一次 `KMeans.fit`，指标计算不计入；内存记录为 10 ms 采样的进程 RSS 峰值增量，受内存复用影响，不能等同算法总内存。`n_iter` 仅为最终保留解的迭代次数，不是 n_init=20 所有初始化的迭代总和。规模实验的资源记录与原先扫描后的 `final_metrics.csv` 分开保留，避免混用测量条件。重复覆盖规模结果表前按项目规则备份。

## 目录约定

正式代码放入 `src/`，必要结果表与图放入 `results/`，报告放入 `report/`。原始数据放入本地 `data/`，交付文件覆盖更新前的备份放入本地 `历史版本/`。

GitHub 仓库：[mengmcy1/cifar10-clustering](https://github.com/mengmcy1/cifar10-clustering)。
