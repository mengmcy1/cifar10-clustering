"""Plus 阶段：k-means 与 GMM 在同一 PCA-50D 空间的全 K 对照实验。"""

import argparse                             # 读取样本清单、数据目录与结果目录等命令行参数。
import time                                 # 对单次 fit 精确计时。
from pathlib import Path                    # 组织数据、样本清单与结果文件的位置。

import matplotlib                           # 配置绘图后端。

matplotlib.use("Agg")                       # 直接保存图片，不弹出绘图窗口。
import matplotlib.pyplot as plt             # 绘制扫描曲线、指标对比图和二维总览。
import numpy as np                          # 处理特征矩阵、硬分组标签与分量计数。
import pandas as pd                         # 组织扫描记录、类别统计并读写 CSV。
from sklearn.cluster import KMeans          # k-means 模型类型，仅用于函数类型注解。
from sklearn.metrics import (
    calinski_harabasz_score,                # CH：比较簇间与簇内离散程度，通常越大越好。
    davies_bouldin_score,                   # DB：比较簇内松散程度与簇间距离，通常越小越好。
    silhouette_score,                       # 轮廓系数：衡量簇内接近、簇间分离程度，通常越大越好。
)
from sklearn.mixture import GaussianMixture  # 高斯混合模型，用 EM 拟合多个高斯分量。

from run_basic import (                     # 复用主流程，保持样本、表示、PCA 与 k-means 口径一致。
    K_VALUES,                               # tuple[int, ...]：扫描 K=2～12，包含两端。
    RANDOM_STATE,                           # int：固定随机种子 42。
    REPRESENTATIONS,                        # dict[str, str]：表示简称 → 结果子目录名称。
    SELECTED_CLASSES,                       # tuple[str, ...]：选用的五个类别。
    build_representations,                  # 为同一批图片构建三种特征矩阵。
    fit_pca,                                # 从原特征分别拟合 PCA-50 与 PCA-2。
    load_subset,                            # 固定种子抽样并返回图像、标签与样本清单。
    make_kmeans,                            # 按项目统一参数构造 k-means 模型。
)

GMM_COVARIANCE_TYPE = "diag"    # str：每个分量在各维上有独立方差，不估计非对角协方差。
GMM_INIT_PARAMS = "kmeans"      # str：用一次 k-means 结果初始化各分量，不表示复用 Basic 模型。
GMM_N_INIT = 10                 # int：每次 fit 尝试 10 次初始化，保留下界最高的解。
GMM_MAX_ITER = 300              # int：单次初始化最多 300 轮 EM 迭代。
GMM_TOL = 1e-3                  # float：EM 收敛阈值，下界增益低于该值即停止。
GMM_REG_COVAR = 1e-6            # float：加到方差上的正则项，避免方差退化。


def parse_args() -> argparse.Namespace:
    """读取命令行参数，定位样本清单、数据目录与结果目录。

    输入：无函数参数；从终端读取以下命令行选项。
        --data-dir（Path）：已下载的 CIFAR-10 目录，默认 data。
        --sample-list（Path）：n3000 样本清单，默认
            results/n3000/sample_list.csv，用于保证与 Basic 完全相同样本。
        --results-dir（Path）：本轮结果根目录，默认 results/plus/gmm。
    输出：argparse.Namespace，包含上述选项对应的属性。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))  # 已有数据目录。
    parser.add_argument("--sample-list", type=Path, default=Path("results/n3000/sample_list.csv"))  # 固定样本清单。
    parser.add_argument("--results-dir", type=Path, default=Path("results/plus/gmm"))  # Plus 结果根目录。
    return parser.parse_args()


def load_mother_samples(
    sample_list: Path,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """按已有清单重新抽样并校验，保证输入与 Basic 主实验一致。

    输入：
        sample_list（Path）：五类等量的样本清单 CSV，含类别名列。
        data_dir（Path）：CIFAR-10 本地目录。
    输出：二元组，N=清单行数（当前为 3000）。
        images（np.ndarray，uint8，N×32×32×3）：RGB 图像，值域 0～255。
        true_names（np.ndarray，字符串，N）：真实类别名称，仅用于事后
            类别统计，不进入特征、PCA 或任何模型拟合。
    """
    saved = pd.read_csv(sample_list)
    counts = saved["class_name"].value_counts()
    if set(counts.index) != set(SELECTED_CLASSES) or counts.nunique() != 1:
        raise ValueError("样本清单必须包含指定五类且每类数量相同。")
    images, labels, samples, class_names = load_subset(data_dir, int(counts.iloc[0]))
    # 清单不一致会直接报错，避免在错误样本上产生看似合理的对照结果。
    pd.testing.assert_frame_equal(samples, saved)
    true_names = np.array([class_names[label] for label in labels])
    return images, true_names


def make_gmm(k: int) -> GaussianMixture:
    """按已确认参数构造 GMM 模型，此处尚不执行拟合。

    输入：k（int），高斯分量数，扫描范围为 2～12。
    输出：GaussianMixture，尚未拟合的模型；diag 协方差是控制复杂度的
        简化假设，不表示 PCA 后各分量内部必然去相关。
    """
    return GaussianMixture(
        n_components=k,
        covariance_type=GMM_COVARIANCE_TYPE,
        init_params=GMM_INIT_PARAMS,
        n_init=GMM_N_INIT,
        random_state=RANDOM_STATE,
        max_iter=GMM_MAX_ITER,
        tol=GMM_TOL,
        reg_covar=GMM_REG_COVAR,
    )


def time_fit(model: KMeans | GaussianMixture, x50: np.ndarray) -> float:
    """对已构造的模型仅计时一次 fit，两算法使用同一口径。

    输入：model（KMeans | GaussianMixture），已构造、未拟合的模型；
        x50（np.ndarray，N×50），PCA 降维后的聚类特征。
    输出：float，fit 耗时（秒），不含模型构造、predict 或指标计算。
    """
    start = time.perf_counter()
    model.fit(x50)
    return time.perf_counter() - start


def hard_grouping_metrics(
    x50: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[float, float, float, str]:
    """计算硬分组的三项指标，非空分组不足 2 时保留原因不填 0。

    输入：x50（np.ndarray，N×50），聚类特征；labels（np.ndarray，
        N 个整数，0～k-1），硬分组标签；k（int），设定的簇数或分量数。
    输出：四元组：silhouette、calinski_harabasz、davies_bouldin（均为
        float，无定义时为 NaN）与 invalid_reason（str，正常时为空串；
        非空硬分组不足 2 时为具体原因，指标随之无定义）。
    """
    n_nonempty = int((np.bincount(labels, minlength=k) > 0).sum())
    if n_nonempty < 2:
        return np.nan, np.nan, np.nan, "fewer than 2 non-empty hard components"
    return (
        float(silhouette_score(x50, labels)),
        float(calinski_harabasz_score(x50, labels)),
        float(davies_bouldin_score(x50, labels)),
        "",
    )


def grouping_sizes(labels: np.ndarray, k: int) -> np.ndarray:
    """统计全部 k 个编号的硬分组大小，空分组保留为 0。

    输入：labels（np.ndarray，N 个整数，0～k-1），硬分组标签；
        k（int），设定的簇数或分量数。
    输出：np.ndarray（int64，k），各编号对应的样本数，总和为 N。
    """
    return np.bincount(labels, minlength=k)


def run_kmeans_scan(x50: np.ndarray, name: str) -> list[dict]:
    """在固定特征上重扫 k-means 的 K=2～12，补齐三项指标与同口径耗时。

    输入：x50（np.ndarray，N×50），本表示的 PCA-50 特征；
        name（str），表示名称，仅写入记录。
    输出：list[dict]，11 行记录；不适用 GMM 的 BIC 等字段不出现，
        不伪造占位值。
    """
    rows = []
    for k in K_VALUES:
        model = make_kmeans(k)
        elapsed = time_fit(model, x50)
        labels = model.labels_
        sizes = grouping_sizes(labels, k)
        silhouette, ch, db, reason = hard_grouping_metrics(x50, labels, k)
        rows.append(
            {
                "representation": name,
                "algorithm": "kmeans",
                "k": k,
                "silhouette": silhouette,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "fit_time_sec": elapsed,
                "n_iter": int(model.n_iter_),  # 最终保留解的迭代数，不是 20 次初始化的总和。
                "n_nonempty_components": int((sizes > 0).sum()),
                "min_component_size": int(sizes.min()),
                "max_component_size": int(sizes.max()),
                "invalid_reason": reason,
            }
        )
        print(f"[{name}] kmeans K={k}: fit {elapsed:.3f}s, Silhouette {silhouette:.4f}", flush=True)
    return rows


def run_gmm_scan(
    x50: np.ndarray,
    name: str,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """在固定特征上扫描 GMM 的 K=2～12，记录拟合状态与硬分组指标。

    输入：x50（np.ndarray，N×50），本表示的 PCA-50 特征；
        name（str），表示名称，仅写入记录。
    输出：二元组。
        rows（list[dict]）：11 行记录，含 BIC、对数似然下界、收敛标记等
            GMM 特有字段；空硬分量时在 invalid_reason 中说明。
        labels_by_k（dict[int, np.ndarray]）：K → predict 得到的硬标签
            （N 个整数），供类别统计与二维总览使用。
    """
    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in K_VALUES:
        model = make_gmm(k)
        elapsed = time_fit(model, x50)
        labels = model.predict(x50)
        labels_by_k[k] = labels
        sizes = grouping_sizes(labels, k)
        n_nonempty = int((sizes > 0).sum())
        silhouette, ch, db, reason = hard_grouping_metrics(x50, labels, k)
        if not reason and n_nonempty < k:
            reason = f"{k - n_nonempty} empty hard component(s)"
        rows.append(
            {
                "representation": name,
                "algorithm": "gmm",
                "k": k,
                "silhouette": silhouette,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "fit_time_sec": elapsed,
                "bic": float(model.bic(x50)),  # 贝叶斯信息准则，只在同表示同空间内比较，越低越好。
                "lower_bound": float(model.lower_bound_),  # 最终保留解的对数似然下界。
                "converged": bool(model.converged_),  # 与硬分组是否有效分别判断。
                "n_iter": int(model.n_iter_),  # 最终保留解的 EM 迭代数，不是所有初始化的总和。
                "n_nonempty_components": n_nonempty,
                "min_component_size": int(sizes.min()),
                "max_component_size": int(sizes.max()),
                "invalid_reason": reason,
            }
        )
        print(
            f"[{name}] gmm K={k}: fit {elapsed:.3f}s, Silhouette {silhouette:.4f}, "
            f"converged={model.converged_}, non-empty {n_nonempty}/{k}, sizes={sizes.tolist()}",
            flush=True,
        )
    return rows, labels_by_k


def composition_rows(
    labels_by_k: dict[int, np.ndarray],
    true_names: np.ndarray,
) -> list[dict]:
    """按 K 逐个统计各硬分量的样本数与真实类别构成，空分量保留零行。

    输入：labels_by_k（dict[int, np.ndarray]），K → 硬标签（N 个整数）；
        true_names（np.ndarray，字符串，N），真实类别名称，仅用于统计。
    输出：list[dict]，共 77 行（2+3+…+12），列为 k、component、size、
        fraction（分量大小/N，分母为全部样本数）及五个类别数量。
    """
    rows = []
    n_samples = len(true_names)
    for k, labels in labels_by_k.items():
        for component in range(k):
            mask = labels == component
            size = int(mask.sum())
            row = {
                "k": k,
                "component": component,
                "size": size,
                "fraction": size / n_samples,
            }
            row.update({c: int(np.sum(true_names[mask] == c)) for c in SELECTED_CLASSES})
            rows.append(row)
    return rows


def formal_candidate_mask(gmm_df: pd.DataFrame) -> pd.Series:
    """标记正式候选：已收敛、三项指标有效且非空硬分组数等于设定 K。

    输入：gmm_df（pd.DataFrame），单一表示的 GMM 扫描记录。
    输出：pd.Series（bool，11 行），True 行为正式候选；其余配置保留
        诊断数据，不进入候选。这是本项目的筛选约定，不表示概率模型无效。
    """
    # 三项指标同时为有限值才算有效；NaN 表示无定义，无穷值同样排除。
    metrics_valid = np.isfinite(
        gmm_df[["silhouette", "calinski_harabasz", "davies_bouldin"]]
    ).all(axis=1)
    return (
        gmm_df["converged"]
        & metrics_valid
        & (gmm_df["n_nonempty_components"] == gmm_df["k"])
    )


def plot_gmm_scan(gmm_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """绘制 GMM 扫描图：上方 BIC-K，下方 Silhouette-K，异常配置单独标记。

    输入：gmm_df（pd.DataFrame），单一表示的 GMM 扫描记录；
        output_path（Path），图片保存路径；title（str），图标题。
    输出：None；保存两个独立纵轴的子图，非候选配置用红色叉号标出，
        不隐藏后当作正常结果。
    """
    candidate = formal_candidate_mask(gmm_df)
    abnormal = gmm_df[~candidate]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax1.plot(gmm_df["k"], gmm_df["bic"], marker="o", label="BIC (all configs)")
    ax1.set_ylabel("BIC (lower is better)")
    ax2.plot(gmm_df["k"], gmm_df["silhouette"], marker="s", label="Silhouette (all configs)")
    ax2.set_ylabel("Silhouette")
    ax2.set_xlabel("K")
    ax2.set_xticks(list(K_VALUES))

    for ax, column in ((ax1, "bic"), (ax2, "silhouette")):
        ax.scatter(
            abnormal["k"],
            abnormal[column],
            marker="x",
            s=80,
            color="red",
            zorder=3,
            label="not a formal candidate",
        )
        ax.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(
    kmeans_df: pd.DataFrame,
    gmm_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """绘制同表示、同 K 下 k-means 与 GMM 的三项指标对比图。

    输入：kmeans_df / gmm_df（pd.DataFrame），同一表示两种算法的扫描
        记录；output_path（Path），图片保存路径；title（str），图标题。
    输出：None；保存 Silhouette、CH、DB 三个子图，各含两条算法曲线；
        非正式候选的 GMM 配置标红叉，指标无定义的位置在对应 K 处文字
        注明，不把异常配置当作正常同 K 比较；不在两算法之间比较 BIC。
    """
    metrics = (
        ("silhouette", "Silhouette"),
        ("calinski_harabasz", "Calinski-Harabasz"),
        ("davies_bouldin", "Davies-Bouldin"),
    )
    abnormal = gmm_df[~formal_candidate_mask(gmm_df)]
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for ax, (column, label) in zip(axes, metrics):
        ax.plot(kmeans_df["k"], kmeans_df[column], marker="o", label="k-means")
        ax.plot(gmm_df["k"], gmm_df[column], marker="s", linestyle="--", label="GMM (diag)")
        ax.scatter(
            abnormal["k"],
            abnormal[column],
            marker="x",
            s=80,
            color="red",
            zorder=3,
            label="GMM not a formal candidate",
        )
        # NaN 或无穷不会画出可见的点，需要在对应 K 处显式注明。
        y0, y1 = ax.get_ylim()
        y_text = y0 + 0.02 * (y1 - y0)
        for df, algo in ((kmeans_df, "k-means"), (gmm_df, "GMM")):
            for k in df.loc[~np.isfinite(df[column]), "k"]:
                ax.text(k, y_text, f"{algo} n/a", ha="center", va="bottom", fontsize=7, color="red")
        ax.set_ylabel(label)
        ax.legend(loc="best")
    axes[-1].set_xlabel("K")
    axes[-1].set_xticks(list(K_VALUES))

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gmm_pca2_overview(
    x2: np.ndarray,
    labels_by_k: dict[int, np.ndarray],
    gmm_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """在固定二维坐标上总览 GMM 各 K 的硬分组，异常配置在标题中注明。

    输入：x2（np.ndarray，N×2），仅用于展示的 PCA-2D 坐标；
        labels_by_k（dict[int, np.ndarray]），K → 50 维空间得到的硬标签；
        gmm_df（pd.DataFrame），单一表示的 GMM 扫描记录，用于标题标注；
        output_path（Path），图片保存路径；title（str），总标题。
    输出：None；保存 4×3 子图网格。各 K 独立拟合，分量编号和颜色
        不代表跨 K 一一对应。
    """
    scans = gmm_df.set_index("k")
    colors = plt.get_cmap("tab20").colors[:12]
    fig, axes = plt.subplots(4, 3, figsize=(16, 18), sharex=True, sharey=True)

    for ax, k in zip(axes.flat, K_VALUES):
        labels = labels_by_k[k]
        for component in range(k):
            mask = labels == component
            ax.scatter(
                x2[mask, 0],
                x2[mask, 1],
                s=5,
                alpha=0.55,
                color=colors[component],
                linewidths=0,
            )
        record = scans.loc[k]
        sil_text = f"{record['silhouette']:.4f}" if pd.notna(record["silhouette"]) else "n/a"
        subtitle = f"K={k} | Silhouette={sil_text}"
        if not record["converged"]:
            subtitle += " | not converged"
        if record["n_nonempty_components"] < k:
            subtitle += f" | non-empty {record['n_nonempty_components']}/{k}"
        ax.set_title(subtitle)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.tick_params(labelbottom=True, labelleft=True)

    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[c], label=f"component {c}")
        for c in range(12)
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=3, title="Component IDs (within each K)")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_comparison_table(
    kmeans_df: pd.DataFrame,
    gmm_df: pd.DataFrame,
) -> pd.DataFrame:
    """按表示与 K 合并两种算法的共同字段，并保留 GMM 的异常状态。

    输入：kmeans_df / gmm_df（pd.DataFrame），两种算法的全部扫描记录。
    输出：pd.DataFrame，每行一个表示×K，列为两算法各自的三项指标与
        fit 耗时（秒），以及 GMM 的 converged、n_nonempty_components、
        invalid_reason 和 formal_candidate 标记；单独查看本表也能识别
        异常配置。BIC 等无法共同比较的字段不进入本表。
    """
    common = (
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "fit_time_sec",
    )
    left = kmeans_df[["representation", "k", *common]]
    right = gmm_df[["representation", "k", *common]]
    merged = left.merge(right, on=("representation", "k"), suffixes=("_kmeans", "_gmm"))
    status = gmm_df[["representation", "k", "converged", "n_nonempty_components", "invalid_reason"]].copy()
    status = status.rename(columns={
        "converged": "gmm_converged",
        "n_nonempty_components": "gmm_n_nonempty_components",
        "invalid_reason": "gmm_invalid_reason",
    })
    status["gmm_formal_candidate"] = formal_candidate_mask(gmm_df).to_numpy()
    return merged.merge(status, on=("representation", "k"))


def print_candidate_summary(gmm_df: pd.DataFrame, name: str) -> None:
    """在控制台汇总单一表示的正式候选与两个候选 K，不做最终决定。

    输入：gmm_df（pd.DataFrame），单一表示的 GMM 扫描记录；
        name（str），表示名称，仅用于打印。
    输出：None；打印候选数量、BIC 最低 K 与 Silhouette 最高 K；
        没有合格候选时如实说明，不强选 K。
    """
    candidate = gmm_df[formal_candidate_mask(gmm_df)]
    if candidate.empty:
        print(f"[{name}] 无合格 GMM 候选（要求收敛、指标有效、非空硬分组数等于 K）。", flush=True)
        return
    bic_best = int(candidate.loc[candidate["bic"].idxmin(), "k"])
    sil_best = int(candidate.loc[candidate["silhouette"].idxmax(), "k"])
    print(
        f"[{name}] 合格候选 {len(candidate)}/11 个；"
        f"候选内 BIC 最低 K={bic_best}，Silhouette 最高 K={sil_best}；"
        "最终配置待审阅结果后确定。",
        flush=True,
    )


def main() -> None:
    """串联三种表示的 k-means 与 GMM 全 K 对照流程。

    输入：无函数参数；命令行参数由 parse_args 读取，类型和含义见其说明。
    输出：None；在结果根目录保存 kmeans_scan_metrics.csv、
        gmm_scan_metrics.csv、kmeans_vs_gmm_metrics.csv；在各表示子目录
        保存 all_k_component_composition.csv、gmm_k_scan.png、
        kmeans_vs_gmm.png、pca2_all_k.png，并打印运行与候选信息。
    """
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    images, true_names = load_mother_samples(args.sample_list, args.data_dir)
    print(f"已按清单载入 {len(images)} 张图片，与 Basic 样本一致。", flush=True)

    kmeans_rows: list[dict] = []
    gmm_rows: list[dict] = []
    for key, features in build_representations(images).items():
        name = REPRESENTATIONS[key]
        rep_dir = args.results_dir / name
        rep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{name}] 原始特征形状: {features.shape}", flush=True)
        x50, x2, explained_50 = fit_pca(features)
        print(f"[{name}] PCA-50 累计解释方差: {explained_50:.4f}", flush=True)

        kmeans_rows.extend(run_kmeans_scan(x50, name))
        gmm_scan, labels_by_k = run_gmm_scan(x50, name)
        gmm_rows.extend(gmm_scan)

        gmm_df = pd.DataFrame(gmm_scan)
        kmeans_df = pd.DataFrame(kmeans_rows[-len(K_VALUES):])
        composition = composition_rows(labels_by_k, true_names)
        pd.DataFrame(composition).to_csv(
            rep_dir / "all_k_component_composition.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_gmm_scan(gmm_df, rep_dir / "gmm_k_scan.png", title=f"{name}: GMM BIC and Silhouette vs K")
        plot_metric_comparison(
            kmeans_df,
            gmm_df,
            rep_dir / "kmeans_vs_gmm.png",
            title=f"{name}: k-means vs GMM metrics at same K",
        )
        plot_gmm_pca2_overview(
            x2,
            labels_by_k,
            gmm_df,
            rep_dir / "pca2_all_k.png",
            title=f"{name}: GMM in PCA-50D, displayed in PCA-2D\n"
            "Same samples and axes; colors do not track components across K",
        )
        print_candidate_summary(gmm_df, name)

    kmeans_all = pd.DataFrame(kmeans_rows)
    gmm_all = pd.DataFrame(gmm_rows)
    kmeans_all.to_csv(args.results_dir / "kmeans_scan_metrics.csv", index=False, encoding="utf-8-sig")
    gmm_all.to_csv(args.results_dir / "gmm_scan_metrics.csv", index=False, encoding="utf-8-sig")
    build_comparison_table(kmeans_all, gmm_all).to_csv(
        args.results_dir / "kmeans_vs_gmm_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n对照扫描完成，结果已写入 results/plus/gmm/。", flush=True)


if __name__ == "__main__":
    main()
