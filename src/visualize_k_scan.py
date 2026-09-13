"""补充已有 k 扫描的二维对比图与逐簇类别统计。"""

import argparse               # 读取数据目录和已有结果目录。
from pathlib import Path      # 组织样本清单、扫描表和输出图片的路径。

from run_basic import (       # 复用主实验实现，保证样本和参数一致。
    K_VALUES,                 # tuple[int, ...]：需要展示的 k=2～12。
    REPRESENTATIONS,          # dict[str, str]：表示简称到子目录名称的映射。
    SELECTED_CLASSES,         # tuple[str, ...]：指定的五个真实类别。
    build_representations,    # 构建同一批图像的 Raw、HSV、HoG 特征。
    fit_pca,                  # 从原特征独立得到 50 维聚类特征和 2 维绘图坐标。
    load_subset,              # 按原实验随机种子与类别数量重新读取样本。
    make_kmeans,              # 构造使用统一参数的 k-means 模型。
    np,                       # NumPy：数组筛选、计数及数值一致性比较。
    pd,                       # pandas：读取原结果并保存逐簇统计表。
    plt,                      # Matplotlib：使用主脚本已配置的非交互后端绘图。
)


def main() -> None:
    """为三种表示补充全部 k 的二维对比图和逐簇类别统计。

    输入：无函数参数；读取命令行选项：
        --data-dir（Path）：CIFAR-10 数据目录，默认 data。
        --results-dir（Path）：已有 sample_list.csv 和各表示 k_scan.csv
            的结果目录，默认 results/n3000；每类数量由清单读取。
    输出：None；每个表示目录保存 pca2_all_k.png（11 个 k 子图）和
        all_k_cluster_composition.csv（k、簇编号、总数、占比、五类数量）。
    说明：核对样本与原清单一致、SSE 与原扫描接近；固定每种表示的二维
        坐标和轴范围，不用颜色追踪跨 k 的同一簇，也不重写原主结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))  # 本地数据目录。
    parser.add_argument("--results-dir", type=Path, default=Path("results/n3000"))  # 原结果与补充输出目录。
    args = parser.parse_args()
    saved_samples = pd.read_csv(args.results_dir / "sample_list.csv")
    counts = saved_samples["class_name"].value_counts()
    if set(counts.index) != set(SELECTED_CLASSES) or counts.nunique() != 1:
        raise ValueError("样本清单必须包含指定五类且每类数量相同。")
    images, labels, samples, class_names = load_subset(args.data_dir, int(counts.iloc[0]))
    pd.testing.assert_frame_equal(samples, saved_samples)  # 防止补充图使用了不同样本或顺序。
    true_names = np.array([class_names[label] for label in labels])
    colors = plt.get_cmap("tab20").colors[:12]  # 每张子图最多 12 簇，一簇一种颜色。

    for key, features in build_representations(images).items():
        name = REPRESENTATIONS[key]
        rep_dir = args.results_dir / name
        scans = pd.read_csv(rep_dir / "k_scan.csv").set_index("k")
        x50, x2, _ = fit_pca(features)
        fig, axes = plt.subplots(4, 3, figsize=(16, 18), sharex=True, sharey=True)  # 11 图+1 图例，共用坐标范围。
        rows = []
        for ax, k in zip(axes.flat, K_VALUES):
            model = make_kmeans(k).fit(x50)
            if not np.isclose(model.inertia_, scans.loc[k, "sse"], rtol=1e-5):  # 相对容差允许微小数值误差。
                raise ValueError(f"{name}, k={k}: 本次 SSE 与已有扫描不一致，请核对环境和参数。")
            for cluster in range(k):
                mask = model.labels_ == cluster
                size = int(mask.sum())
                ax.scatter(x2[mask, 0], x2[mask, 1], s=5, alpha=0.55,
                           color=colors[cluster], linewidths=0)
                row = {"k": k, "cluster": cluster, "size": size,
                       "fraction": size / len(images)}  # 占全部样本的比例，不是某个类别在簇内的占比。
                row.update({c: int(np.sum(true_names[mask] == c)) for c in SELECTED_CLASSES})
                rows.append(row)
            ax.set_title(f"k={k} | Silhouette={scans.loc[k, 'silhouette']:.4f}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.tick_params(labelbottom=True, labelleft=True)
            print(f"{name}: k={k}, cluster sizes={[r['size'] for r in rows if r['k'] == k]}", flush=True)
        legend_ax = axes.flat[-1]
        legend_ax.axis("off")
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[c],
                              label=f"cluster {c}") for c in range(12)]
        legend_ax.legend(handles=handles, loc="center", ncol=3, title="Cluster IDs (within each k)")
        fig.suptitle(f"{name}: k-means in PCA-50D, displayed in PCA-2D\n"
                     "Same samples and axes; colors do not track clusters across k", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(rep_dir / "pca2_all_k.png", dpi=180)
        plt.close(fig)
        pd.DataFrame(rows).to_csv(rep_dir / "all_k_cluster_composition.csv", index=False,
                                 encoding="utf-8-sig")
    print("已生成三张全 k 对比图及三份逐簇统计表。")


if __name__ == "__main__":
    main()
