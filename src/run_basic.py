from __future__ import annotations          # 延后解析类型注解，便于描述函数输入输出。

import argparse                             # 读取终端传入的数据目录、样本量和最终 k 等参数。
import os                                   # 获取当前进程编号，供内存测量使用。
import threading                            # 在后台线程中定时采样内存。
import time                                 # 精确计时，并控制内存采样间隔。
from pathlib import Path                    # 用路径对象组织数据和结果文件的位置。

import matplotlib                           # 配置绘图后端。

matplotlib.use("Agg")                       # 直接保存图片，不弹出绘图窗口。
import matplotlib.pyplot as plt             # 绘制曲线、散点图和代表图片。
import numpy as np                          # 处理图像数组、特征矩阵和距离计算。
import pandas as pd                         # 组织样本清单、统计表并读写 CSV。
import psutil                               # 读取当前进程的物理内存占用（RSS）。
from skimage import color                   # 完成 RGB→HSV、RGB→灰度转换。
from skimage.feature import hog             # 提取局部梯度方向直方图特征。
from sklearn.cluster import KMeans          # 按特征距离将样本划分为 k 个簇。
from sklearn.decomposition import PCA       # 主成分分析，压缩特征维度。
from sklearn.metrics import (
    calinski_harabasz_score,                # CH：比较簇间与簇内离散程度，通常越大越好。
    davies_bouldin_score,                   # DB：比较簇内松散程度与簇间距离，通常越小越好。
    silhouette_score,                       # 轮廓系数：衡量簇内接近、簇间分离程度，通常越大越好。
)
from torchvision.datasets import CIFAR10    # 下载或读取 CIFAR-10 图像与类别标签。


RANDOM_STATE = 42                                                      # int：固定随机种子，用于抽样、PCA 和聚类。
SELECTED_CLASSES = ("airplane", "automobile", "frog", "cat", "dog")    # tuple[str, ...]：选用的五类。
K_VALUES = tuple(range(2, 13))                                         # tuple[int, ...]：扫描 2～12 个簇，包含两端。
N_INIT = 20                                                            # int：每次 fit 尝试 20 次初始化，保留 SSE 最小的解。

REPRESENTATIONS = {                                                    # dict[str, str]：程序中的表示简称 → 结果子目录名称。
    "raw": "raw_pixel",
    "hsv": "hsv_histogram",
    "hog": "hog",
}


def parse_args() -> argparse.Namespace:
    """读取并检查命令行参数，区分首次扫描与指定最终 k 的运行方式。

    输入：无函数参数；从终端读取以下命令行选项。
        --data-dir / --results-dir（Path）：数据目录 / 输出目录。
        --samples-per-class（int）：每类样本数，允许 10～600，默认 600。
        --final-k-raw/hsv/hog（int | None）：各表示的最终簇数，2～12；
            默认均为 None，仅扫描；若指定，三个必须同时提供。
    输出：argparse.Namespace，包含上述选项对应的属性。
    """
    parser = argparse.ArgumentParser(
        description="CIFAR-10 子集表示构建与 k-means 聚类 Basic 实验"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))  # CIFAR-10 存放目录。
    parser.add_argument("--results-dir", type=Path, default=Path("results"))  # 本轮结果目录。
    parser.add_argument("--samples-per-class", type=int, default=600)  # 五类使用相同样本数。
    parser.add_argument("--final-k-raw", type=int, choices=K_VALUES, default=None)  # Raw 最终 k。
    parser.add_argument("--final-k-hsv", type=int, choices=K_VALUES, default=None)  # HSV 最终 k。
    parser.add_argument("--final-k-hog", type=int, choices=K_VALUES, default=None)  # HoG 最终 k。
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
    """读取训练集，固定种子按类别均衡抽样，再打乱样本顺序。

    输入：
        data_dir（Path）：CIFAR-10 本地目录，缺少数据时允许下载。
        samples_per_class（int）：每个指定类别抽取的图片数。
    输出：四元组，N=5×samples_per_class。
        images（np.ndarray，uint8，N×32×32×3）：RGB 图像，值域 0～255。
        labels（np.ndarray，int64，N）：与图像逐行对应的原始类别编号。
        sample_table（pd.DataFrame）：样本编号、训练集索引、类别等清单。
        class_names（list[str]）：CIFAR-10 全部十类名称，按原始类别编号索引。
    """
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
    """将每张 RGB 图像展平并缩放，构建原始像素表示。

    输入：images（np.ndarray，uint8，N×32×32×3），值域 0～255 的图像。
    输出：np.ndarray（float32，N×3072），像素值缩放至 [0,1] 的特征矩阵。
    """
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def extract_hsv_histogram_features(
    images: np.ndarray,
    bins: int = 32,
) -> np.ndarray:
    """统计每张图的 H/S/V 颜色分布，分别归一化后拼接。

    输入：
        images（np.ndarray，uint8，N×32×32×3）：RGB 图像，值域 0～255。
        bins（int）：每个通道的直方图区间数，默认 32。
    输出：np.ndarray（float32，N×(3*bins)），默认 96 维；
        每个通道的直方图之和为 1，拼接后的整行之和为 3。
    """
    images_float = images.astype(np.float32) / 255.0
    hsv_images = color.rgb2hsv(images_float)
    features = np.empty((len(images), bins * 3), dtype=np.float32)

    for i, image in enumerate(hsv_images):
        parts = []
        for channel in range(3):
            hist, _ = np.histogram(
                image[..., channel],
                bins=bins,  # 将该通道等宽分成 bins 个区间。
                range=(0.0, 1.0),  # skimage 转换后的 H、S、V 均在此范围。
            )
            hist = hist.astype(np.float32)
            hist /= hist.sum()
            parts.append(hist)
        features[i] = np.concatenate(parts)
    return features


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """将图像转灰度，提取描述局部边缘方向的 HoG 特征。

    输入：images（np.ndarray，uint8，N×32×32×3），RGB 图像。
    输出：np.ndarray（float32，N×D），每张图的 HoG 向量；
        D 由库根据图像和参数计算，当前设置实际为 1764，不硬编码维度。
    """
    images_float = images.astype(np.float32) / 255.0
    gray_images = color.rgb2gray(images_float)
    features = [
        hog(
            image,
            orientations=9,  # 将梯度方向划分为 9 个区间。
            pixels_per_cell=(4, 4),  # 每个局部单元覆盖 4×4 像素。
            cells_per_block=(2, 2),  # 每 2×2 个单元组成一个归一化块。
            block_norm="L2-Hys",  # 对块特征做 L2 归一化、截断后再次归一化。
            feature_vector=True,  # 将各块特征展平为一维向量。
        )
        for image in gray_images
    ]
    return np.asarray(features, dtype=np.float32)


def build_representations(images: np.ndarray) -> dict[str, np.ndarray]:
    """为同一批图片分别构建三种特征，保持样本行顺序一致。

    输入：images（np.ndarray，uint8，N×32×32×3），RGB 图像。
    输出：dict[str, np.ndarray]，键 raw/hsv/hog 分别对应 N×D 特征矩阵；
        三者独立使用，不在这里拼接。
    """
    return {
        "raw": extract_raw_features(images),
        "hsv": extract_hsv_histogram_features(images),
        "hog": extract_hog_features(images),
    }


def fit_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """从同一原特征分别拟合 50 维与 2 维 PCA，不做串联降维。

    输入：features（np.ndarray，N×D），每行一个样本，N 和 D 均需至少 50。
    输出：三元组。
        x50（np.ndarray，N×50）：用于正式聚类的特征。
        x2（np.ndarray，N×2）：仅用于绘图的二维坐标。
        explained_50（float）：前 50 个主成分累计解释方差比例，不是准确率。
    """
    # n_components 指保留维数；whiten=False 保留各主成分的相对方差尺度。
    pca50 = PCA(n_components=50, whiten=False, random_state=RANDOM_STATE)
    x50 = pca50.fit_transform(features)

    pca2 = PCA(n_components=2, whiten=False, random_state=RANDOM_STATE)
    x2 = pca2.fit_transform(features)

    explained_50 = float(pca50.explained_variance_ratio_.sum())
    return x50, x2, explained_50


def make_kmeans(k: int) -> KMeans:
    """按项目统一参数构造 k-means 模型，此处尚不执行拟合。

    输入：k（int），要求划分的簇数，主实验扫描范围为 2～12。
    输出：KMeans，尚未拟合的模型；后续调用 fit 或 fit_predict 进行聚类。
    """
    return KMeans(
        n_clusters=k,  # 指定簇数，不直接使用真实类别数。
        init="k-means++",  # 用距离加权方式选择初始中心。
        n_init=N_INIT,  # 多次初始化后选择 SSE 最小的结果。
        random_state=RANDOM_STATE,  # 固定初始化的随机种子。
    )


def scan_k_values(x50: np.ndarray) -> pd.DataFrame:
    """依次尝试 k=2～12，记录选取最终 k 所需的两项指标。

    输入：x50（np.ndarray，N×50），PCA 降维后的聚类特征。
    输出：pd.DataFrame，共 11 行，列为 k（int）、sse（float，簇内平方
        距离总和）和 silhouette（float，全体样本平均轮廓系数）。
    """
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
    """绘制选 k 曲线：左轴为 SSE，右轴为平均轮廓系数。

    输入：scan_df（pd.DataFrame），含 k/sse/silhouette 列的扫描表；
        output_path（Path），图片保存路径；title（str），图标题。
    输出：None；将图像写入 output_path，并关闭绘图对象。
    """
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
    """执行一次 fit，记录耗时和采样得到的进程 RSS 峰值增量。

    输入：x50（np.ndarray，N×50），聚类特征；k（int），簇数。
    输出：三元组：model（KMeans，已拟合模型）；elapsed（float，fit
        耗时，秒）；peak_delta_mb（float，RSS 峰值增量，实际单位 MiB）。
    说明：每 10 ms 采样，受内存复用和漏采短暂峰值影响，不代表总内存。
    """
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss  # 拟合前进程驻留物理内存，单位字节。
    peak_rss = baseline_rss
    stop_event = threading.Event()

    def sample_memory() -> None:
        """后台采样当前进程内存，直到收到停止信号。

        输入：无显式参数；使用外层 process（psutil.Process）、
            stop_event（threading.Event）及 peak_rss（int，字节）。
        输出：None；更新外层 peak_rss，保存观测到的最大值。
        """
        nonlocal peak_rss
        while not stop_event.is_set():
            peak_rss = max(peak_rss, process.memory_info().rss)
            time.sleep(0.01)  # 采样间隔为 0.01 秒，即 10 ms。

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()

    model = make_kmeans(k)
    start = time.perf_counter()
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
    """在二维坐标上展示 50 维聚类标签，不在二维空间重新聚类。

    输入：x2（np.ndarray，N×2），绘图坐标；cluster_labels（np.ndarray，
        N 个整数），与坐标逐行对应的簇编号；k（int），簇数；
        output_path（Path），输出图片路径；title（str），图标题。
    输出：None；保存带有簇图例的二维散点图。
    """
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
    """在各簇内部选择距离其 50 维中心最近的真实样本。

    输入：x50（np.ndarray，N×50），聚类特征；cluster_labels（np.ndarray，
        N 个整数），簇编号；centers（np.ndarray，k×50），簇中心；
        top_n（int），每簇最多选几张，默认 3。
    输出：dict[int, np.ndarray]，簇编号 → 按距离从近到远排列的样本行索引；
        若簇内不足 top_n 张，则只返回实际样本，不补齐。
    """
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
    """按簇展示代表图片，并整理可追溯的代表样本清单。

    输入：images（np.ndarray，N×32×32×3），RGB 图像；
        true_labels（np.ndarray，N 个整数），真实类别，仅用于标题与清单；
        class_names（list[str]），原始类别编号对应的名称；
        representatives（dict[int, np.ndarray]），簇编号对应的样本行索引，
            当前布局每簇最多展示 3 张；
        output_path（Path），输出图路径；title（str），总标题。
    输出：pd.DataFrame，包含簇编号、簇内排名、样本行索引及真实类别；
        同时保存代表图。此流程的样本行索引与 sample_list 的 sample_id 一致。
    """
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
    """在聚类结束后统计各簇包含的五类样本数量，辅助解释结果。

    输入：cluster_labels（np.ndarray，N 个整数），聚类簇编号；
        true_labels（np.ndarray，N 个整数），对应的真实类别编号；
        class_names（list[str]），按类别编号索引的名称；
        output_path（Path），CSV 保存路径。
    输出：None；保存“行=簇、列=真实类别、值=样本数”的交叉统计表。
    """
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
    """完成一种表示的降维、k 扫描，以及可选的最终聚类与展示。

    输入：key（str），raw/hsv/hog 之一；features（np.ndarray，N×D），
        对应表示的特征；images（np.ndarray，N×32×32×3），原图；
        true_labels（np.ndarray，N 个整数），仅供事后解释的真实类别；
        class_names（list[str]），类别编号到名称的对应；
        results_dir（Path），结果根目录；final_k（int | None），最终簇数，
            为 None 时仅输出扫描结果。
    输出：dict[str, float | int | str] | None；有 final_k 时返回包含表示名、
        维数、解释方差、最终 k、三项指标、秒级耗时和 MiB 内存增量的字典，
        否则返回 None。扫描表/曲线与最终图表分别保存到该表示子目录。
    """
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
    """串联 Basic 主流程，并汇总三种表示的最终结果。

    输入：无函数参数；命令行参数由 parse_args 读取，类型和含义见其说明。
    输出：None；保存样本清单和各表示的扫描结果。若同时提供三个最终 k，
        额外保存最终指标、散点图、代表图片及类别组成，并打印运行信息。
    """
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
