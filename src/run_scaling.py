"""固定 k=2，在已有样本清单的嵌套子集上测量单次聚类。"""

import argparse                             # 读取母集清单、样本规模及文件路径参数。
from pathlib import Path                    # 组织本地数据与结果目录。

import numpy as np                          # 按清单索引读取图像，核对标签并统计簇大小。
import pandas as pd                         # 读取母集、筛选嵌套子集并保存指标表。
from sklearn.decomposition import PCA       # 为每个规模独立拟合 50 维特征。
from sklearn.metrics import (
    calinski_harabasz_score,                # CH：簇间与簇内离散程度的比较，通常越高越好。
    davies_bouldin_score,                   # DB：簇内松散程度相对簇间距离的比较，通常越低越好。
    silhouette_score,                       # 轮廓系数：样本相对本簇及其他簇的接近程度。
)
from torchvision.datasets import CIFAR10    # 读取已下载的数据集，本脚本不自动下载。

from run_basic import (                     # 复用主流程，避免改变表示和聚类的实现口径。
    RANDOM_STATE,                           # int：固定随机种子 42。
    REPRESENTATIONS,                        # dict[str, str]：表示简称到完整名称的映射。
    SELECTED_CLASSES,                       # tuple[str, ...]：母集必须包含的五个类别。
    build_representations,                  # 为当前子集构建三种特征矩阵。
    fit_with_resource_measurement,          # 聚类并返回模型、fit 秒数及 RSS 增量（MiB）。
)


def main() -> None:
    """运行一个指定样本规模的实验，固定 k=2，不扫描 k 或绘制二维图。

    输入：无函数参数；读取命令行选项：
        --data-dir（Path）：已下载的 CIFAR-10 目录，默认 data。
        --mother-list（Path）：五类各 600 张的原始样本清单，
            默认 results/n3000/sample_list.csv。
        --results-root（Path）：结果根目录，默认 results。
        --samples-per-class（int）：必填，200/400/600，对应 N=1000/2000/3000。
    输出：None；在 results-root/n{N} 中保存 scaling_sample_list.csv
        和 scaling_metrics.csv，后者每种表示一行，包含质量指标、fit
        耗时（秒）、RSS 增量（MiB）、最终解迭代次数及大小簇样本数。
    说明：按母集行顺序逐类取前若干张，保留母集 sample_id；每次调用只做
        一个规模，三个规模分别启动进程。真实标签用于清单校验，不参与拟合。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))  # 已有数据目录。
    parser.add_argument("--mother-list", type=Path, default=Path("results/n3000/sample_list.csv"))  # 固定母集。
    parser.add_argument("--results-root", type=Path, default=Path("results"))  # 自动追加 n{N} 子目录。
    parser.add_argument("--samples-per-class", type=int, choices=(200, 400, 600), required=True)  # 每类取样数。
    args = parser.parse_args()

    mother = pd.read_csv(args.mother_list)
    counts = mother["class_name"].value_counts()
    if (set(counts.index) != set(SELECTED_CLASSES) or not counts.eq(600).all()
            or mother["dataset_index"].duplicated().any()
            or not mother["cifar10_split"].eq("train").all()):
        raise ValueError("母集必须是训练集指定五类各 600 张且无重复索引的清单。")
    # cumcount 给每类按原行顺序编号，从而让小规模样本严格包含于大规模样本。
    samples = mother.loc[mother.groupby("class_name").cumcount() < args.samples_per_class].copy()
    dataset = CIFAR10(root=str(args.data_dir), train=True, download=False)
    indices = samples["dataset_index"].to_numpy()
    labels = np.asarray(dataset.targets)[indices]
    if (not np.array_equal(labels, samples["class_id"].to_numpy())
            or [dataset.classes[i] for i in labels] != samples["class_name"].tolist()):
        raise ValueError("清单标签与本地 CIFAR-10 不一致。")
    images = dataset.data[indices].copy()
    del dataset
    output = args.results_root / f"n{len(samples)}"
    output.mkdir(parents=True, exist_ok=True)
    # 保留母集 sample_id，便于直接检查嵌套关系；不重写主实验清单。
    samples.to_csv(output / "scaling_sample_list.csv", index=False, encoding="utf-8-sig")

    rows = []
    for key, features in build_representations(images).items():
        pca = PCA(n_components=50, whiten=False, random_state=RANDOM_STATE)  # 保留 50 维，不做白化。
        x50 = pca.fit_transform(features)
        model, elapsed, memory = fit_with_resource_measurement(x50, 2)  # 固定两簇，只测 fit，不计下方指标计算。
        row = {
            "n_samples": len(samples),
            "samples_per_class": args.samples_per_class,
            "representation": REPRESENTATIONS[key],
            "feature_dim": features.shape[1],
            "random_state": RANDOM_STATE,
            "final_k": 2,
            "n_init": model.n_init,
            "pca50_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "sse": float(model.inertia_),
            "silhouette": float(silhouette_score(x50, model.labels_)),
            "calinski_harabasz": float(calinski_harabasz_score(x50, model.labels_)),
            "davies_bouldin": float(davies_bouldin_score(x50, model.labels_)),
            "kmeans_time_sec": elapsed,  # float：一次 fit 的秒数，含模型设置的多次初始化。
            "kmeans_peak_memory_delta_mb": memory,  # float：观测到的 RSS 增量，实际单位 MiB。
            "n_iter": model.n_iter_,  # int：最终保留解的迭代数，不是所有初始化的总和。
            "min_cluster_size": int(np.bincount(model.labels_).min()),
            "max_cluster_size": int(np.bincount(model.labels_).max()),
        }
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(output / "scaling_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
