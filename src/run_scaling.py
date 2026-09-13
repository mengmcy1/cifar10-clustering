"""固定 k=2，在已有样本清单的嵌套子集上测量单次聚类。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from torchvision.datasets import CIFAR10

from run_basic import (
    RANDOM_STATE,
    REPRESENTATIONS,
    SELECTED_CLASSES,
    build_representations,
    fit_with_resource_measurement,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mother-list", type=Path, default=Path("results/n3000/sample_list.csv"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--samples-per-class", type=int, choices=(200, 400, 600), required=True)
    args = parser.parse_args()

    mother = pd.read_csv(args.mother_list)
    counts = mother["class_name"].value_counts()
    if (set(counts.index) != set(SELECTED_CLASSES) or not counts.eq(600).all()
            or mother["dataset_index"].duplicated().any()
            or not mother["cifar10_split"].eq("train").all()):
        raise ValueError("母集必须是训练集指定五类各 600 张且无重复索引的清单。")
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
        pca = PCA(n_components=50, whiten=False, random_state=RANDOM_STATE)
        x50 = pca.fit_transform(features)
        model, elapsed, memory = fit_with_resource_measurement(x50, 2)
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
            "kmeans_time_sec": elapsed,
            "kmeans_peak_memory_delta_mb": memory,
            "n_iter": model.n_iter_,
            "min_cluster_size": int(np.bincount(model.labels_).min()),
            "max_cluster_size": int(np.bincount(model.labels_).max()),
        }
        rows.append(row)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(output / "scaling_metrics.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
