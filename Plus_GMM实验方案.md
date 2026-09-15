# Plus：GMM 替代聚类实验方案

> 状态：方案已确认，待实现与正式运行。
>
> 本文件只记录当前已确认的 GMM 替代聚类实验设计、统计字段、绘图方式和结论边界。真实结果产生后再补充结果分析，不预设 GMM 优于 k-means。

## 1. 实验目的

Basic 已完成 Raw Pixel、HSV Histogram、HoG 三种表示在 PCA-50D 空间中的 k-means 聚类、`k=2~12` 扫描、最终指标、二维可视化、逐簇统计和规模实验。

Plus 第一项采用高斯混合模型（Gaussian Mixture Model，GMM）作为替代聚类方法，核心问题是：

> 在保持样本、表示方法、PCA 降维和随机种子等条件一致时，将 k-means 换成 GMM 后，内部聚类指标、簇大小、真实类别组成和二维结构会发生怎样的变化？

本实验不预设 GMM 更好，也不把结果变化归因于单一因素。

## 2. 输入与公平比较条件

使用与 Basic 主实验相同的 3000 张 CIFAR-10 图片：

- `airplane`
- `automobile`
- `frog`
- `cat`
- `dog`

每类 600 张，共 3000 张，使用现有 `results/n3000/sample_list.csv` 对应的同一批样本。

三种表示保持 Basic 完全一致：

- Raw Pixel：3072D，像素缩放到 `[0,1]`。
- HSV Histogram：H/S/V 各 32 bins，共 96D，各通道分别 L1 归一化。
- HoG：灰度图，`orientations=9`、`pixels_per_cell=(4,4)`、`cells_per_block=(2,2)`、`block_norm='L2-Hys'`。

每种表示继续独立拟合：

- PCA-50D：用于正式聚类。
- PCA-2D：只用于可视化。

GMM 不额外使用 `StandardScaler`、PCA whitening、特征拼接，也不重新选择 PCA 维数。

因此主要比较关系为：

```text
同一批 3000 张样本
        ↓
Raw / HSV / HoG
        ↓
各自 PCA-50D
        ↓
┌──────────────────┐
│                  │
k-means          GMM
│                  │
└──────────────────┘
```

真实类别标签不参与特征构建、PCA 或 GMM 拟合，只用于聚类完成后的类别组成解释和辅助统计。

## 3. GMM 核心设置

第一轮 GMM 使用对角协方差：

```python
GaussianMixture(
    n_components=K,
    covariance_type="diag",
    init_params="kmeans",
    n_init=10,
    random_state=42,
    max_iter=300,
    tol=1e-3,
    reg_covar=1e-6,
)
```

其中只有 `n_components=K` 在扫描过程中变化，其余参数固定。

### 3.1 协方差类型

第一轮固定：

```text
covariance_type = "diag"
```

即每个 GMM 分量分别学习 PCA-50D 各方向上的方差，但不估计分量内部不同维度之间的非对角协方差。

本轮不同时扫描 `spherical / tied / full`，避免把替代聚类实验扩展成 GMM 大规模超参数搜索。若实际结果暴露出明确需要，再单独讨论是否增加补充实验。

### 3.2 初始化与收敛

- `init_params="kmeans"`
- `n_init=10`
- `random_state=42`
- `max_iter=300`
- `tol=1e-3`
- `reg_covar=1e-6`

正式结果必须检查 GMM 是否收敛，并记录：

- `converged_`
- `n_iter_`
- `lower_bound_`

未收敛结果不得直接作为正常结果与 k-means 下结论，应先标记并检查。

## 4. K 扫描范围

GMM 与现有 k-means 一样扫描：

```text
K = 2, 3, ..., 12
```

共 11 个分量数量设置，每种表示分别独立扫描。

这样除了可以进行同 `K` 的算法公平比较，还可以观察：

> GMM 自身对分量数量的偏好是否与 k-means 的最终 `k=2` 一致。

作业原文明确要求 k-means 扫描 `k=2~12`；GMM 扫描同一区间属于本项目为了保持比较完整性而增加的实验设计，不应写成作业原文的强制要求。

## 5. 每个 K 记录的统计量

对每个：

```text
representation × K
```

至少记录以下字段：

| 字段 | 含义 |
|---|---|
| `representation` | Raw Pixel / HSV Histogram / HoG |
| `k` | GMM 分量数 |
| `silhouette` | 平均轮廓系数，通常越高越好 |
| `calinski_harabasz` | CH 指数，通常越高越好 |
| `davies_bouldin` | DB 指数，通常越低越好 |
| `bic` | GMM 的 BIC，通常越低越好 |
| `fit_time_sec` | GMM `fit` 耗时 |
| `converged` | 是否收敛 |
| `n_iter` | EM 实际迭代次数 |
| `lower_bound` | 最终对数似然下界 |
| `min_component_size` | 硬标签后最小分量样本数 |
| `max_component_size` | 硬标签后最大分量样本数 |

如实现方便，可额外保存 GMM 的 `weights_`，但分量权重与 `predict()` 后的硬分组样本数需要区分，不混为同一统计量。

## 6. K 的选择依据

GMM 扫描后主要结合：

- BIC：越低通常越好。
- Silhouette：越高通常越好。
- CH、DB：辅助评价。
- `converged_`：必须正常收敛。
- 分量大小：检查是否出现极端或退化分量。
- PCA-2D 展示：辅助观察结构。

真实类别组成只用于聚类后的解释，不参与最终 K 的选择。

不规定“BIC 最低的 K 自动就是最终 K”，而是综合 BIC、Silhouette、收敛情况及分量结构判断，并如实记录指标不一致的情况。

### 6.1 为什么加入 BIC

既然 GMM 扫描 `K=2~12`，需要一个与 GMM 概率模型相匹配的模型选择指标。BIC 会同时考虑模型拟合和模型复杂度，因此可以作为 GMM 内部选择分量数量的重要依据。

BIC 只用于 GMM 内部比较，不能把 GMM 的 BIC 数值与 k-means 的 SSE 数值直接比较。

## 7. 类别组成统计

对每个 GMM 的 K，使用：

```python
labels = gmm.predict(X_pca50)
```

得到硬标签，再统计：

```text
k
component
size
fraction
airplane
automobile
frog
cat
dog
```

统计格式尽量与现有 k-means 的 `all_k_cluster_composition.csv` 保持一致，便于同 K 横向比较。

真实标签统计属于事后辅助分析，不是监督分类准确率，也不用于训练或模型选择。

## 8. 绘图方案

### 8.1 GMM K 扫描图

三种表示分别输出 GMM 扫描图。

主图优先使用上下两个子图：

```text
上：BIC vs K
下：Silhouette vs K
```

不优先把 BIC 和 Silhouette 放在同一双 y 轴中，避免量纲差异影响阅读。

CH、DB 全部保存到 CSV；如真实结果显示它们与 BIC / Silhouette 存在有价值的差异，再决定是否增加辅助曲线，不为凑图提前生成大量重复图。

### 8.2 全 K PCA-2D 可视化

与 k-means 的 `pca2_all_k.png` 保持相同逻辑：

```text
原始表示 → 独立 PCA-50D → GMM 拟合
原始表示 → 独立 PCA-2D  → 显示 GMM hard label
```

GMM 不在 PCA-2D 上拟合。

每种表示输出 `K=2~12` 的二维对比图，并固定该表示内部的二维坐标与坐标范围，便于观察不同 K 下的划分变化。

不同 K 的 component 编号和颜色不表示跨 K 的分量一一对应。

### 8.3 k-means 与 GMM 指标对比图

至少比较：

- Silhouette
- CH
- DB

对同一种表示、同一个 K，将 k-means 与 GMM 的指标放在同一比较图中，从而回答：

> 在输入表示和 K 相同的条件下，仅更换聚类方法后，内部指标发生了怎样的变化？

BIC 不与 k-means 绘制同值比较，因为 k-means 没有对应 BIC 指标。

## 9. 两层正式比较

最终分析分成两层。

### 第一层：相同 K 的公平比较

Basic 正式主结果三种表示均使用：

```text
k-means, k=2
```

因此必须保留：

```text
k-means(k=2) vs GMM(K=2)
```

这层比较中样本、表示、PCA-50D 和 K 都一致，主要变化因素是聚类算法。

重点检查：

- 三项内部指标如何变化。
- HSV 原有的极端大小不均衡划分是否改变。
- HoG 原有动物 / 交通工具粗类别对应是否保留。
- 簇 / 分量大小是否发生明显变化。
- 真实类别组成如何变化。

### 第二层：各自模型选择后的结果比较

k-means 保留 Basic 已确定的最终结果；GMM 根据自己的 BIC、Silhouette、收敛情况和分量结构选择最终 K。

比较：

```text
k-means 的最终配置
vs
GMM 自己选择的最终配置
```

这层用于回答：

> 两种聚类方法在各自模型选择规则下，最终表现和结构有什么差异？

不得把这层结果解释成严格的“只改变算法”单因素实验，因为两边最终 K 可能不同。

## 10. 输出文件建议

建议将 GMM Plus 结果独立放在：

```text
results/plus/gmm/
```

例如：

```text
results/plus/gmm/
├─ gmm_scan_metrics.csv
├─ kmeans_vs_gmm_metrics.csv
├─ raw_pixel/
│  ├─ gmm_k_scan.png
│  ├─ pca2_all_k.png
│  └─ all_k_component_composition.csv
├─ hsv_histogram/
│  ├─ gmm_k_scan.png
│  ├─ pca2_all_k.png
│  └─ all_k_component_composition.csv
└─ hog/
   ├─ gmm_k_scan.png
   ├─ pca2_all_k.png
   └─ all_k_component_composition.csv
```

实际实现时如现有代码结构更适合略微调整文件名，可以保持语义等价，不需要为完全复制目录样式增加无必要抽象。

## 11. 结论边界

正式分析必须遵守以下边界：

- 不预设 GMM 优于 k-means。
- 只能比较当前样本、当前三种表示、PCA-50D 和固定参数下的结果。
- GMM 与 k-means 的差异不能自动归因于某一个具体协方差方向或某一种视觉因素。
- `diag` GMM 只代表当前选定的替代聚类 baseline，不代表所有 GMM 配置。
- 真实标签只用于事后解释，不参与模型选择。
- PCA-2D 只负责展示，不能代替 PCA-50D 中的定量评价。
- 若 BIC、Silhouette、CH、DB 或类别组成给出不同倾向，应如实报告，不强行统一成单一“最佳”结论。
- 若 GMM 某个 K 未收敛或形成无效分组，必须单独记录，不把它当成正常结果比较。

## 12. 后续顺序

当前 Plus 执行顺序保持：

1. GMM 替代聚类。
2. 多随机种子稳定性。
3. 轻量自编码器表示。
4. 根据真实结果整理正式报告。

本文件目前只冻结第 1 项 GMM 的方案；稳定性和自编码器的具体设置等轮到对应阶段再讨论，不提前锁死。
