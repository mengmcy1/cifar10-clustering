from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from skimage import color
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from torchvision.datasets import CIFAR10


RANDOM_STATE = 42
SELECTED_CLASSES = ("airplane", "automobile", "frog", "cat", "dog")
K_VALUES = tuple(range(2, 13))
N_INIT = 20

REPRESENTATIONS = {
    "raw": "raw_pixel",
    "hsv": "hsv_histogram",
    "hog": "hog",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 子集表示构建与 k-means 聚类 Basic 实验"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--samples-per-class", type=int, default=600)
    parser.add_argument("--final-k-raw", type=int, choices=K_VALUES, default=None)
    parser.add_argument("--final-k-hsv", type=int, choices=K_VALUES, default=None)
    parser.add_argument("--final-k-hog", type=int, choices=K_VALUES, default=None)
    args = parser.parse_args()

    if not 10 <= args.samples_per_class <= 600:
        parser.error("--samples-per-class 必须在 10~600 之间，以保证 PCA-50 可计算。")

    final_ks = (args.final_k_raw, args.final_k_hsv, args.final_k_hog)
    if any(k is not None for k in final_ks) and not all(k is not None for k in final_ks):
        parser.error(
            "最终阶段请同时提供 --final-k-raw、--final-k-hsv、--final-k-hog；"
            "首次扫描阶段则三个都不要提供。"
        )
    return args


def load_subset(
    data_dir: Path,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    dataset = CIFAR10(root=str(data_dir), train=True, download=True)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(RANDOM_STATE)

    selected_indices: list[int] = []
    for class_name in SELECTED_CLASSES:
        class_id = dataset.class_to_idx[class_name]
        class_indices = np.flatnonzero(targets == class_id)
        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen.tolist())

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    rng.shuffle(selected_indices)

    images = dataset.data[selected_indices].copy()
    labels = targets[selected_indices].copy()
    sample_table = pd.DataFrame(
        {
            "sample_id": np.arange(len(selected_indices)),
            "cifar10_split": "train",
            "dataset_index": selected_indices,
            "class_id": labels,
            "class_name": [dataset.classes[label] for label in labels],
        }
    )
    return images, labels, sample_table, dataset.classes


def extract_raw_features(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def extract_hsv_histogram_features(
    images: np.ndarray,
    bins: int = 32,
) -> np.ndarray:
    images_float = images.astype(np.float32) / 255.0
    hsv_images = color.rgb2hsv(images_float)
    features = np.empty((len(images), bins * 3), dtype=np.float32)

    for i, image in enumerate(hsv_images):
        parts = []
        for channel in range(3):
            hist, _ = np.histogram(
                image[..., channel],
                bins=bins,
                range=(0.0, 1.0),
            )
            hist = hist.astype(np.float32)
            hist /= hist.sum()
            parts.append(hist)
        features[i] = np.concatenate(parts)
    return features


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    images_float = images.astype(np.float32) / 255.0
    gray_images = color.rgb2gray(images_float)
    features = [
        hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        for image in gray_images
    ]
    return np.asarray(features, dtype=np.float32)


def build_representations(images: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "raw": extract_raw_features(images),
        "hsv": extract_hsv_histogram_features(images),
        "hog": extract_hog_features(images),
    }


def fit_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    pca50 = PCA(n_components=50, whiten=False, random_state=RANDOM_STATE)
    x50 = pca50.fit_transform(features)

    pca2 = PCA(n_components=2, whiten=False, random_state=RANDOM_STATE)
    x2 = pca2.fit_transform(features)

    explained_50 = float(pca50.explained_variance_ratio_.sum())
    return x50, x2, explained_50


def make_kmeans(k: int) -> KMeans:
    return KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )


def scan_k_values(x50: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        model = make_kmeans(k)
        cluster_labels = model.fit_predict(x50)
        rows.append(
            {
                "k": k,
                "sse": float(model.inertia_),
                "silhouette": float(silhouette_score(x50, cluster_labels)),
            }
        )
    return pd.DataFrame(rows)


def plot_k_scan(scan_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(scan_df["k"], scan_df["sse"], marker="o", label="SSE")
    ax1.set_xlabel("k")
    ax1.set_ylabel("SSE")
    ax1.set_xticks(list(K_VALUES))

    ax2 = ax1.twinx()
    ax2.plot(
        scan_df["k"],
        scan_df["silhouette"],
        marker="s",
        linestyle="--",
        label="Silhouette",
    )
    ax2.set_ylabel("Silhouette")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fit_with_resource_measurement(
    x50: np.ndarray,
    k: int,
) -> tuple[KMeans, float, float]:
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    stop_event = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop_event.is_set():
            peak_rss = max(peak_rss, process.memory_info().rss)
            time.sleep(0.01)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()

    start = time.perf_counter()
    model = make_kmeans(k)
    model.fit(x50)
    elapsed = time.perf_counter() - start

    stop_event.set()
    monitor.join()
    peak_rss = max(peak_rss, process.memory_info().rss)

    peak_delta_mb = max(0, peak_rss - baseline_rss) / (1024**2)
    return model, elapsed, peak_delta_mb


def plot_pca2_clusters(
    x2: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    output_path: Path,
    title: str,
) -> None:
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8")
    fig, ax = plt.subplots(figsize=(7, 6))

    for cluster_id in range(k):
        mask = cluster_labels == cluster_id
        ax.scatter(
            x2[mask, 0],
            x2[mask, 1],
            s=14,
            alpha=0.65,
            marker=markers[cluster_id % len(markers)],
            label=f"cluster {cluster_id}",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def representative_indices(
    x50: np.ndarray,
    cluster_labels: np.ndarray,
    centers: np.ndarray,
    top_n: int = 3,
) -> dict[int, np.ndarray]:
    representatives: dict[int, np.ndarray] = {}
    for cluster_id, center in enumerate(centers):
        members = np.flatnonzero(cluster_labels == cluster_id)
        distances = np.linalg.norm(x50[members] - center, axis=1)
        representatives[cluster_id] = members[np.argsort(distances)[:top_n]]
    return representatives


def plot_representatives(
    images: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str],
    representatives: dict[int, np.ndarray],
    output_path: Path,
    title: str,
) -> pd.DataFrame:
    k = len(representatives)
    fig, axes = plt.subplots(k, 3, figsize=(7.5, 2.2 * k), squeeze=False)
    rows = []

    for cluster_id, indices in representatives.items():
        for col, sample_index in enumerate(indices):
            ax = axes[cluster_id, col]
            ax.imshow(images[sample_index])
            true_name = class_names[int(true_labels[sample_index])]
            ax.set_title(f"C{cluster_id} | {true_name}", fontsize=9)
            ax.axis("off")
            rows.append(
                {
                    "cluster": cluster_id,
                    "rank": col + 1,
                    "sample_id": int(sample_index),
                    "true_class_id": int(true_labels[sample_index]),
                    "true_class_name": true_name,
                }
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def save_cluster_composition(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    true_names = [class_names[int(label)] for label in true_labels]
    table = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series(true_names, name="true_class"),
    )
    table.to_csv(output_path, encoding="utf-8-sig")


def run_representation(
    key: str,
    features: np.ndarray,
    images: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str],
    results_dir: Path,
    final_k: int | None,
) -> dict[str, float | int | str] | None:
    name = REPRESENTATIONS[key]
    rep_dir = results_dir / name
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] 原始特征形状: {features.shape}")
    x50, x2, explained_50 = fit_pca(features)
    print(f"[{name}] PCA-50 累计解释方差: {explained_50:.4f}")

    scan_df = scan_k_values(x50)
    scan_df.to_csv(rep_dir / "k_scan.csv", index=False, encoding="utf-8-sig")
    plot_k_scan(
        scan_df,
        rep_dir / "k_scan.png",
        title=f"{name}: SSE and Silhouette vs k",
    )

    suggested_k = int(scan_df.loc[scan_df["silhouette"].idxmax(), "k"])
    print(
        f"[{name}] 扫描完成。Silhouette 单指标参考 k={suggested_k}；"
        "正式最终 k 请结合 SSE 肘部曲线共同确定。"
    )

    if final_k is None:
        return None

    model, elapsed, peak_memory_delta_mb = fit_with_resource_measurement(x50, final_k)
    cluster_labels = model.labels_

    silhouette = float(silhouette_score(x50, cluster_labels))
    ch = float(calinski_harabasz_score(x50, cluster_labels))
    db = float(davies_bouldin_score(x50, cluster_labels))

    plot_pca2_clusters(
        x2,
        cluster_labels,
        final_k,
        rep_dir / "pca2_clusters.png",
        title=f"{name}: PCA-2D visualization (k={final_k})",
    )

    representatives = representative_indices(
        x50,
        cluster_labels,
        model.cluster_centers_,
        top_n=3,
    )
    representative_df = plot_representatives(
        images,
        true_labels,
        class_names,
        representatives,
        rep_dir / "representatives.png",
        title=f"{name}: 3 nearest samples per cluster",
    )
    representative_df.to_csv(
        rep_dir / "representatives.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_cluster_composition(
        cluster_labels,
        true_labels,
        class_names,
        rep_dir / "cluster_composition.csv",
    )

    return {
        "representation": name,
        "feature_dim": int(features.shape[1]),
        "pca50_explained_variance": explained_50,
        "final_k": final_k,
        "silhouette": silhouette,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "kmeans_time_sec": elapsed,
        "kmeans_peak_memory_delta_mb": peak_memory_delta_mb,
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    images, true_labels, sample_table, class_names = load_subset(
        args.data_dir,
        args.samples_per_class,
    )
    sample_table.to_csv(
        args.results_dir / "sample_list.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"已加载 {len(images)} 张图片：{', '.join(SELECTED_CLASSES)}，"
        f"每类 {args.samples_per_class} 张。"
    )

    representations = build_representations(images)
    final_k_map = {
        "raw": args.final_k_raw,
        "hsv": args.final_k_hsv,
        "hog": args.final_k_hog,
    }

    final_rows = []
    for key in ("raw", "hsv", "hog"):
        row = run_representation(
            key=key,
            features=representations[key],
            images=images,
            true_labels=true_labels,
            class_names=class_names,
            results_dir=args.results_dir,
            final_k=final_k_map[key],
        )
        if row is not None:
            final_rows.append(row)

    if not final_rows:
        print(
            "\n首次扫描阶段完成。请查看 results/*/k_scan.png 和 k_scan.csv，"
            "结合 SSE 肘部与 Silhouette 曲线确定三种表示的最终 k，"
            "再同时传入 --final-k-raw / --final-k-hsv / --final-k-hog 重新运行。"
        )
        return

    pd.DataFrame(final_rows).to_csv(
        args.results_dir / "final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n最终聚类阶段完成，汇总结果已写入 results/final_metrics.csv。")


if __name__ == "__main__":
    main()
