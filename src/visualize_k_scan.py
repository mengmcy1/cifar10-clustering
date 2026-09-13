"""补充已有 k 扫描的二维对比图与逐簇类别统计。"""

import argparse
from pathlib import Path

from run_basic import (
    K_VALUES,
    REPRESENTATIONS,
    SELECTED_CLASSES,
    build_representations,
    fit_pca,
    load_subset,
    make_kmeans,
    np,
    pd,
    plt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/n3000"))
    args = parser.parse_args()
    saved_samples = pd.read_csv(args.results_dir / "sample_list.csv")
    counts = saved_samples["class_name"].value_counts()
    if set(counts.index) != set(SELECTED_CLASSES) or counts.nunique() != 1:
        raise ValueError("样本清单必须包含指定五类且每类数量相同。")
    images, labels, samples, class_names = load_subset(args.data_dir, int(counts.iloc[0]))
    pd.testing.assert_frame_equal(samples, saved_samples)
    true_names = np.array([class_names[label] for label in labels])
    colors = plt.get_cmap("tab20").colors[:12]

    for key, features in build_representations(images).items():
        name = REPRESENTATIONS[key]
        rep_dir = args.results_dir / name
        scans = pd.read_csv(rep_dir / "k_scan.csv").set_index("k")
        x50, x2, _ = fit_pca(features)
        fig, axes = plt.subplots(4, 3, figsize=(16, 18), sharex=True, sharey=True)
        rows = []
        for ax, k in zip(axes.flat, K_VALUES):
            model = make_kmeans(k).fit(x50)
            if not np.isclose(model.inertia_, scans.loc[k, "sse"], rtol=1e-5):
                raise ValueError(f"{name}, k={k}: 本次 SSE 与已有扫描不一致，请核对环境和参数。")
            for cluster in range(k):
                mask = model.labels_ == cluster
                size = int(mask.sum())
                ax.scatter(x2[mask, 0], x2[mask, 1], s=5, alpha=0.55,
                           color=colors[cluster], linewidths=0)
                row = {"k": k, "cluster": cluster, "size": size,
                       "fraction": size / len(images)}
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
