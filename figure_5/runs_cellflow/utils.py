import os
import gc
import yaml
from functools import partial
import inspect
from typing import Iterable, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sp
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, wasserstein_distance, entropy
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import scanpy as sc
import anndata as ad
import umap

import jax.tree as jt
import jax.numpy as jnp
import cfp
from cfp.external._scvi import CFJaxSCVI
from cfp.metrics import compute_metrics, compute_metrics_fast


def load_vae(
    h5ad: str,
    latent_space: str,
    h5ad_vae_train: str,
    vae_train_path: str,
    h5ad_vae_eval: str | None = None,
    vae_eval_path: str | None = None,
) -> tuple[CFJaxSCVI | None, CFJaxSCVI | None]:
    if vae_train_path is not None:
        if vae_eval_path is not None:
            if latent_space == "pca":
                raise ValueError("Latent space is PCA but VAE paths are provided.")
        else:
            raise ValueError("Both VAE paths must be provided.")
        h5ad_vae_eval = h5ad_vae_eval or h5ad
        adata_vae_train = sc.read_h5ad(h5ad_vae_train)
        adata_vae_eval = sc.read_h5ad(h5ad_vae_eval)
        CFJaxSCVI.setup_anndata(adata_vae_train, layer="counts")
        vae_train = CFJaxSCVI.load(vae_train_path, adata_vae_train)
        CFJaxSCVI.setup_anndata(adata_vae_eval, layer="counts")
        vae_eval = CFJaxSCVI.load(vae_eval_path, adata_vae_eval)
    else:
        vae_train, vae_eval = None, None
    return vae_train, vae_eval


def get_onehot_dict(categories: list[str] | np.ndarray) -> dict[str, np.ndarray]:
    dataset_enc = OneHotEncoder()
    dataset_enc.fit(np.array(categories).reshape(-1, 1))
    onehot_dict = {}
    for cat in categories:
        dataset_onehot = (
            dataset_enc.transform(np.array([cat]).reshape(-1, 1)).toarray().flatten()
        )
        onehot_dict[cat] = dataset_onehot
    return onehot_dict


def get_covariate_reps(
    adata: ad.AnnData,
    molecules: list[str],
    train_datasets: list[str],
) -> None:
    # condition representations
    mol_onehot_dict = get_onehot_dict(molecules)
    adata.uns["conditions"] = {}
    condition_keys = []
    for mol in molecules:
        concs = adata.obs[mol + "_conc"].unique()
        mol_onehot = mol_onehot_dict[mol]
        for conc in concs:
            cond_cells = adata.obs[mol + "_conc"] == conc
            cond_id = mol + "_" + str(conc)
            if cond_id in condition_keys:
                continue
            condition_keys.append(cond_id)
            adata.obs[cond_id] = cond_cells
            adata.uns["conditions"][cond_id] = mol_onehot * np.log1p(float(conc))

    # dataset/media representations (sample covariates)
    if len(train_datasets) > 1:
        adata.uns["dataset"] = get_onehot_dict(train_datasets)
    if "glut_pre" in train_datasets or "gaba_pre" in train_datasets:
        adata.uns["media"] = get_onehot_dict(adata.obs["media"].unique().tolist())

    return adata, condition_keys


def get_combs_idx(adata: ad.AnnData, combs: list[str], conds: list[str]) -> np.ndarray:
    combs_idx = np.zeros(adata.n_obs, dtype=bool)
    for comb in combs:
        mols = comb.split("+") if comb != "ctrl" else [comb]
        combs_idx = combs_idx | adata.obs["comb"].apply(
            lambda x: all([mol in x for mol in mols])
            and all([mol in mols for mol in x.split("+")])
        )
    if len(conds) != 0:
        combs_idx = combs_idx | adata.obs["condition"].isin(conds)
    return combs_idx


def split_data(
    adata: ad.AnnData,
    test_dataset: str,
    train_combs: list[str],
    exclude_combs: list[str],
    eval_combs: list[str],
    sweep_combs: list[str],
    train_conds: list[str],
    eval_conds: list[str],
    sweep_conds: list[str],
) -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    # train adata
    if exclude_combs:
        train_combs = [
            comb for comb in adata.obs["comb"].unique() if comb not in exclude_combs
        ]
    train_filt = get_combs_idx(adata, train_combs, train_conds)
    train_filt = train_filt | (adata.obs["dataset"] != test_dataset)
    adata_train = adata[train_filt]

    # eval adata
    eval_filt = get_combs_idx(adata, eval_combs, eval_conds)
    eval_filt = eval_filt & (adata.obs["dataset"] == test_dataset)
    adata_eval = adata[eval_filt]

    # sweep conditions
    sweep_filt = get_combs_idx(adata, sweep_combs, sweep_conds)
    sweep_filt = sweep_filt & (adata.obs["dataset"] == test_dataset)
    if not np.all(~sweep_filt | eval_filt):
        raise ValueError("Sweep conditions must be a subset of eval conditions.")
    sweep_conds = adata[sweep_filt].obs["condition"].unique()

    return adata_train, adata_eval, sweep_conds


def run_pca(
    adata: ad.AnnData,
    adata_train: ad.AnnData,
    adata_eval: ad.AnnData,
    latent_space: str,
    n_dims_train: int | None,
    n_dims_eval: int,
) -> None:
    # run PCA
    cfp.pp.centered_pca(adata, n_comps=n_dims_eval, method="rapids")
    adata_eval.obsm["X_pca_all"] = adata[
        adata.obs_names.isin(adata_eval.obs_names)
    ].obsm["X_pca"]
    if latent_space == "pca":
        cfp.pp.centered_pca(adata_train, n_comps=n_dims_train, method="rapids")
        cfp.pp.project_pca(adata_eval, adata_train)


def generate_adata_with_source(
    adata_train: ad.AnnData,
    molecules: list[str],
    sample_rep: str,
    n_src_cells: int = 10000,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Add control cells to the training dataset."""
    np.random.seed(seed)
    samples = []
    for i in range(n_src_cells):
        sample = adata_train.obsm[sample_rep][
            np.random.choice(adata_train.n_obs, n_samples), :
        ].mean(axis=0)
        samples.append(sample)
    samples = np.array(samples)
    samples_obs = pd.DataFrame(
        {col: 0.0 for col in [mol + "_conc" for mol in molecules]},
        index=range(samples.shape[0]),
    )
    samples_obs["dataset"] = "CTRL"
    samples_obs["media"] = "CTRL"
    samples_obs["condition"] = "CTRL"
    adata_ctrl = sc.AnnData(
        X=csr_matrix(np.zeros((samples.shape[0], adata_train.n_vars))), obs=samples_obs
    )
    adata_ctrl.obsm[sample_rep] = samples
    adata_ctrl.var_names = adata_train.var_names
    adata_train_full = ad.concat([adata_train, adata_ctrl], join="outer")
    adata_train_full.obs["CTRL"] = adata_train_full.obs["dataset"] == "CTRL"
    adata_ctrl.obs["CTRL"] = True
    return adata_ctrl, adata_train_full


def get_adata_rep(
    adata: ad.AnnData,
    rep: str,
) -> np.ndarray:
    if rep == "X":
        if sp.sparse.issparse(adata.X):
            return adata.X.toarray()
        else:
            return adata.X
    else:
        return adata.obsm[rep]


def compute_cfp_metrics(
    adata_gt: ad.AnnData,
    adata_pred: ad.AnnData,
    rep_gt: str,
    rep_pred: str,
    pred_name: str = "",
    fast: bool = True,
) -> dict[str, dict[str, float]]:
    conditions = adata_gt.obs["condition"].unique()
    gt_data = {
        cond: get_adata_rep(adata_gt[adata_gt.obs["condition"] == cond], rep_gt)
        for cond in conditions
    }
    pred_data = {
        cond: get_adata_rep(adata_pred[adata_pred.obs["condition"] == cond], rep_pred)
        for cond in conditions
    }
    metrics_func = compute_metrics_fast if fast else compute_metrics
    metrics = jt.map(metrics_func, gt_data, pred_data)
    metrics_df = pd.DataFrame(metrics).T
    if pred_name:
        metrics_df = metrics_df.add_suffix(f"_{pred_name}")
    metrics_df.insert(0, "condition", metrics_df.index)
    return metrics_df.reset_index(drop=True)


def weighted_mae(true_props: np.ndarray, pred_props: np.ndarray) -> float:
    """Mean absolute error in proportions weighted by abundance."""
    absolute_errors = np.abs(true_props - pred_props)
    weighted_errors = absolute_errors * true_props
    return np.sum(weighted_errors) / np.sum(true_props)


def compute_cluster_dist_metrics(
    true_props: np.ndarray, pred_props: np.ndarray
) -> dict[str, float]:
    metrics = {
        "cosine": cosine(true_props, pred_props),
        "pearson": pearsonr(true_props, pred_props)[0],
        "wasserstein": wasserstein_distance(true_props, pred_props),
        "kd_truefirst": entropy(true_props, pred_props),
        "kd_predfirst": entropy(pred_props, true_props),
        "mae": np.mean(np.abs(true_props - pred_props)),
        "wmae": weighted_mae(true_props, pred_props),
    }
    return metrics


def get_detected_clusters(x: jnp.ndarray, threshold: float = 0.05) -> jnp.ndarray:
    value_counts = jnp.unique_counts(x)
    value_frac = value_counts[1] / x.shape[0]
    select_vals = value_counts[0][value_frac > threshold]
    return select_vals


def compute_precision_recall(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_threshold: int = 0.05,
    y_thresholds: Iterable = [0.05],
) -> dict[str, float]:
    precrec_dict = {}
    for threshold in y_thresholds:
        x_detect = get_detected_clusters(x, threshold=x_threshold)
        y_detect = get_detected_clusters(y, threshold=threshold)
        # Get true positives, false positives, and false negatives
        tp = jnp.intersect1d(x_detect, y_detect).shape[0]
        fp = jnp.setdiff1d(y_detect, x_detect).shape[0]
        fn = jnp.setdiff1d(x_detect, y_detect).shape[0]
        # Compute precision and recall
        # But catch division by zero
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precrec_dict[f"precision_{threshold}"] = precision
        precrec_dict[f"recall_{threshold}"] = recall
    return precrec_dict


def compute_cluster_metrics(
    adata_gt: ad.AnnData,
    adata_pred: ad.AnnData,
    rep_gt: str,
    rep_pred: str,
    n_neighbors: list[int] | int,
    label_keys: list[str],
    thresholds: list[float] | np.ndarray = np.arange(1, 20, 0.5) / 100,
    pred_name: str = "",
) -> pd.DataFrame:
    conditions = adata_pred.obs["condition"].unique()
    precrec_func = partial(compute_precision_recall, y_thresholds=thresholds)
    n_neighbors_list = n_neighbors if isinstance(n_neighbors, list) else [n_neighbors]
    metrics_dict = {}
    for n_neighbors in n_neighbors_list:
        cfp.pp.compute_wknn(
            adata_gt,
            adata_pred,
            n_neighbors=n_neighbors,
            ref_rep_key=rep_gt,
            query_rep_key=rep_pred,
        )
        for label_key in label_keys:
            cfp.pp.transfer_labels(adata_pred, adata_gt, label_key=label_key)

            # cluster distribution metrics
            true_props = {
                cond: adata_gt[adata_gt.obs["condition"] == cond]
                .obs[label_key]
                .value_counts(normalize=True)
                for cond in conditions
            }
            pred_props = {
                cond: adata_pred[adata_pred.obs["condition"] == cond]
                .obs[f"{label_key}_transfer"]
                .value_counts(normalize=True)
                .reindex(true_props[cond].index)
                .fillna(0)
                for cond in conditions
            }
            cluster_dist_metrics = jt.map(
                compute_cluster_dist_metrics, true_props, pred_props
            )
            cluster_dist_metrics_df = pd.DataFrame(cluster_dist_metrics).T

            # precision-recall metrics
            cluster_data_gt = {
                cond: jnp.array(
                    adata_gt.obs[label_key][
                        adata_gt.obs["condition"] == cond
                    ].values.astype(int)
                )
                for cond in conditions
            }
            cluster_data_pred = {
                cond: jnp.array(
                    adata_pred.obs[f"{label_key}_transfer"][
                        adata_pred.obs["condition"] == cond
                    ].values.astype(int)
                )
                for cond in conditions
            }
            precrec = jt.map(precrec_func, cluster_data_gt, cluster_data_pred)
            precrec_df = pd.DataFrame(precrec).T

            # combine metrics
            metrics_df = pd.concat([cluster_dist_metrics_df, precrec_df], axis=1)
            if pred_name:
                metrics_df = metrics_df.add_suffix(f"_{pred_name}")
            metrics_df.insert(0, "n_neighbors", n_neighbors)
            metrics_df.insert(0, "label_key", label_key)
            metrics_df.insert(0, "condition", metrics_df.index)
            metrics_dict[(n_neighbors, label_key)] = metrics_df

    metrics_df = pd.concat(metrics_dict.values(), axis=0)

    return metrics_df


def umap_fit_transform(
    adata_gt: ad.AnnData,
    adata_pred: ad.AnnData,
    rep_gt: str,
    rep_pred: str,
    key_added: str = "X_umap",
    method: Literal["umap", "cuml"] = "umap",
) -> None:
    if method == "umap":
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=2,
            n_epochs=500,
            learning_rate=1.0,
            init="spectral",
            min_dist=0.4,
            spread=1.0,
            negative_sample_rate=5,
            a=None,
            b=None,
            random_state=212,
            n_jobs=-1,
        )
    elif method == "cuml":
        raise NotImplementedError("cuML UMAP is not supported right now.")
        # umap_model = cuml.UMAP(
        #     n_neighbors=15,
        #     n_components=2,
        #     n_epochs=500,
        #     learning_rate=1.0,
        #     init="spectral",
        #     min_dist=0.4,
        #     spread=1.0,
        #     negative_sample_rate=5,
        #     a=None,
        #     b=None,
        #     random_state=212,
        # )
    adata_gt.obsm[key_added] = umap_model.fit_transform(adata_gt.obsm[rep_gt])
    adata_pred.obsm[key_added] = umap_model.transform(adata_pred.obsm[rep_pred])


def plot_predictions_umap(
    adata_gt: ad.AnnData,
    adata_pred: ad.AnnData,
    umap_rep: str,
    plot_combs: list[str],
    plot_conds: list[str],
    out_dir: str | None = None,
    space_name: str = "",
) -> None:
    deepcolors = sns.color_palette("deep")
    suffix = "_" + space_name if space_name else ""
    for cond_col, cond in zip(
        ["comb"] * len(plot_combs) + ["condition"] * len(plot_conds),
        plot_combs + plot_conds,
    ):
        single1, single2 = cond.rsplit("+", 1)

        adatas_plot = [
            adata_gt[adata_gt.obs[cond_col] == single1],
            adata_gt[adata_gt.obs[cond_col] == single2],
            adata_gt[adata_gt.obs[cond_col] == cond],
            adata_pred[adata_pred.obs[cond_col] == cond],
        ]
        titles = [
            single1,
            single2,
            f"{cond}, GT",
            f"{cond}, predicted",
        ]

        fig, axs = plt.subplots(1, 4, figsize=(20, 4))
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            sns.scatterplot(
                x=adata_gt.obsm[umap_rep][:, 0],
                y=adata_gt.obsm[umap_rep][:, 1],
                c="lightgrey",
                s=3,
                alpha=0.8,
                linewidth=0,
                rasterized=True,
                ax=ax,
            )
            sns.scatterplot(
                x=adatas_plot[i].obsm[umap_rep][:, 0],
                y=adatas_plot[i].obsm[umap_rep][:, 1],
                s=6,
                linewidth=0,
                color=deepcolors[0] if i < 3 else deepcolors[3],
                rasterized=True,
                ax=ax,
            )
            ax.set_title(titles[i])
            ax.axis("off")
        plt.tight_layout()
        if out_dir is not None:
            plt.savefig(
                os.path.join(out_dir, f"umap_{cond}{suffix}.pdf"),
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()


def detect_clusters(
    clusters: pd.Series,
    threshold: float,
) -> list:
    cluster_counts = clusters.value_counts()
    detected_clusters = cluster_counts[
        cluster_counts > threshold * cluster_counts.sum()
    ].index.tolist()
    return detected_clusters


def plot_predictions_heatmap(
    adata_gt: ad.AnnData,
    adata_train: ad.AnnData,
    adata_pred: ad.AnnData,
    plot_combs: list[str],
    plot_conds: list[str],
    out_dir: str,
    label_keys: list[str] | str,
    thresholds: list[float] | np.ndarray = np.arange(1, 6) / 100,
) -> None:
    label_keys = [label_keys] if isinstance(label_keys, str) else label_keys
    for label_key in label_keys:

        # assign each cluster to division based on the majority of cells
        clusters_divisions = {}
        clusters_regions = {}
        clusters_neurons = {}
        all_clusters = sorted(set(adata_gt.obs[label_key].unique().astype(int)))
        all_clusters = list(map(str, all_clusters))
        for cluster in all_clusters:
            cluster_cells = adata_gt.obs[label_key] == cluster
            cluster_division = (
                adata_gt.obs[cluster_cells]["Division"].value_counts().index[0]
            )
            cluster_region = (
                adata_gt.obs[cluster_cells]["Region"].value_counts().index[0]
            )
            cluster_neuron = (
                adata_gt.obs[cluster_cells]["Neuron_type"].value_counts().index[0]
            )
            clusters_divisions[cluster] = cluster_division
            clusters_regions[cluster] = cluster_region
            clusters_neurons[cluster] = cluster_neuron
        clusters_annot = pd.DataFrame(
            {
                "Division": pd.Series(clusters_divisions),
                "Region": pd.Series(clusters_regions),
                "Neuron_type": pd.Series(clusters_neurons),
            }
        )
        clusters_annot["ones"] = 1
        clusters_annot["Neuron"] = (
            clusters_annot["Division"]
            + "_"
            + clusters_annot["Region"]
            + "_"
            + clusters_annot["Neuron_type"]
        )

        # plot heatmap
        for threshold in thresholds:
            for cond_col, cond in zip(
                ["comb"] * len(plot_combs) + ["condition"] * len(plot_conds),
                plot_combs + plot_conds,
            ):
                single1, single2 = cond.rsplit("+", 1)

                # plot the heatmaps
                conc_filter_gt = np.ones(adata_gt.n_obs, dtype=bool)
                conc_filter_pred = np.ones(adata_pred.n_obs, dtype=bool)
                if cond_col == "comb":
                    if "RA" in cond:
                        conc_filter_gt &= adata_gt.obs["condition"].str.contains(
                            "RA_3|RA_4"
                        )
                        conc_filter_pred &= adata_pred.obs["condition"].str.contains(
                            "RA_3|RA_4"
                        )
                    if "CHIR" in cond:
                        conc_filter_gt &= adata_gt.obs["condition"].str.contains(
                            "CHIR_3|CHIR_4"
                        )
                        conc_filter_pred &= adata_pred.obs["condition"].str.contains(
                            "CHIR_3|CHIR_4"
                        )

                adatas_plot = [
                    adata_gt[(adata_gt.obs[cond_col] == single1) & conc_filter_gt],
                    adata_gt[adata_gt.obs[cond_col] == single2],
                    adata_gt[(adata_gt.obs[cond_col] == cond) & conc_filter_gt],
                    adata_pred[(adata_pred.obs[cond_col] == cond) & conc_filter_pred],
                    adata_train,
                ]

                clusters_present = set()
                for i, adata in enumerate(adatas_plot):
                    if i == 4:
                        continue
                    clusters = adata.obs[
                        label_key if i < 3 else f"{label_key}_transfer"
                    ]
                    clusters = detect_clusters(clusters, threshold)
                    clusters_present.update(clusters)

                adata_gt_plot = adata_gt[adata_gt.obs[label_key].isin(clusters_present)]
                sc.tl.dendrogram(adata_gt_plot, groupby=label_key)
                linkage_matrix = adata_gt_plot.uns["dendrogram_" + label_key]["linkage"]
                clusters_ordered = adata_gt_plot.uns["dendrogram_" + label_key][
                    "categories_ordered"
                ]
                clusters_annot_cond = clusters_annot.loc[clusters_ordered]

                fig, axes = plt.subplots(
                    9,
                    1,
                    figsize=(10, 3.3),
                    height_ratios=[0.2, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1],
                )
                sns.set_style("white")

                # plot the dendrogram
                with plt.rc_context({"lines.linewidth": 0.5}):
                    ax = axes[0]
                    dendrogram(
                        linkage_matrix,
                        ax=ax,
                        orientation="top",
                        no_labels=True,
                        link_color_func=lambda x: "black",
                    )
                    ax.yaxis.set_ticks([])
                    ax.yaxis.set_ticklabels([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                # plot color coded divisions
                handles, labels = [], []
                for i, annot_type in enumerate(["Division", "Region", "Neuron_type"]):
                    i += 1
                    ax = axes[i]
                    if annot_type == "Division":
                        palette = sns.color_palette("Accent")
                        palette = {
                            "CNS": palette[0],
                            "PNS": palette[1],
                            "Other": palette[2],
                        }
                    elif annot_type == "Region":
                        palette = sns.color_palette("tab10")
                        palette = {
                            "TG": palette[0],
                            "Spinal cord": palette[1],
                            "DRG": palette[2],
                            "ENS": palette[3],
                            "Forebrain": palette[4],
                            "SYM": palette[5],
                            "Hindbrain": palette[6],
                            "Midbrain": palette[7],
                        }
                    else:
                        palette = sns.color_palette("Dark2")
                        palette = {
                            "NOR": palette[0],
                            "GLUT": palette[1],
                            "CHO": palette[2],
                            "GABA": palette[3],
                            "GLY": palette[4],
                        }

                    sns.barplot(
                        x=clusters_annot_cond.index,
                        y="ones",
                        hue=annot_type,
                        order=clusters_ordered,
                        data=clusters_annot_cond,
                        ax=ax,
                        width=1,
                        palette=palette,
                    )
                    cur_handles, cur_labels = ax.get_legend_handles_labels()
                    handles.append(cur_handles)
                    labels.append(cur_labels)
                    ax.get_legend().remove()
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    ax.set_ylim(0, 1)
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                titles = [
                    single1,
                    single2,
                    cond + ", GT",
                    cond + ", predicted",
                    "Training set",
                ]
                for i, ax in enumerate(axes):
                    if i < 4:
                        continue
                    _i = i - 4
                    label_key_cur = f"{label_key}_transfer" if _i == 3 else label_key
                    adata_cluster_counts = (
                        adatas_plot[_i].obs[label_key_cur].value_counts().sort_index()
                    )
                    adata_cluster_counts = np.log2(adata_cluster_counts)
                    cluster_counts = pd.DataFrame(
                        {
                            "adata": adata_cluster_counts.reindex(clusters_ordered)
                            .fillna(0)
                            .astype(int)
                        }
                    )
                    sns.heatmap(
                        cluster_counts[["adata"]].T,
                        annot=False,
                        cmap="Blues",
                        ax=ax,
                        cbar=False,
                    )

                    ax.set_ylabel(
                        titles[_i], rotation=0, va="center", ha="right", fontsize=11
                    )
                    ax.yaxis.set_ticks([])
                    ax.yaxis.set_ticklabels([])
                    ax.xaxis.set_ticks([])
                    ax.xaxis.set_ticklabels([])
                    ax.set_xlabel("Cluster", fontsize=14)

                for i, annot_type in enumerate(["Division", "Region", "Neuron_type"]):
                    legend_offset = 0 if i == 0 else 2.4 if i == 1 else 8
                    legend = plt.legend(
                        handles[i],
                        labels[i],
                        title=annot_type,
                        bbox_to_anchor=(1.3, 10 - legend_offset),
                    )
                    fig.add_artist(legend)

                plt.subplots_adjust(hspace=0.05)
                plt.savefig(
                    os.path.join(
                        out_dir, f"heatmap_{cond}_{label_key}_{threshold}.pdf"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
                gc.collect()


def _load_cf_results(
    h5ad: str,
    train_datasets: list[Literal["glut_post", "glut_pre", "gaba_post", "gaba_pre"]],
    test_dataset: Literal["glut_post", "gaba_post"],
    train_combs: list[str],
    exclude_combs: list[str],
    eval_combs: list[str],
    sweep_combs: list[str],
    train_conds: list[str],
    eval_conds: list[str],
    sweep_conds: list[str],
    latent_space: Literal["pca"] | str,
    n_dims_train: int | None,
    n_dims_eval: int,
    h5ad_vae_train: str | None,
    vae_train_path: str | None,
    h5ad_vae_eval: str | None,
    vae_eval_path: str | None,
    out_dir: str,
    out_prefix: str,
    save_adata: bool,
    save_model: bool,
) -> dict:
    # load data
    molecules = np.array(["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"])
    adata = sc.read_h5ad(h5ad)
    adata = adata[adata.obs["dataset"].isin(train_datasets)]

    # load VAEs
    vae_train, vae_eval = load_vae(
        h5ad, latent_space, h5ad_vae_train, vae_train_path, h5ad_vae_eval, vae_eval_path
    )

    # create covariate representations
    adata, condition_keys = get_covariate_reps(adata, molecules, train_datasets)

    # split data
    adata_train, adata_eval, sweep_conds = split_data(
        adata,
        test_dataset,
        train_combs,
        exclude_combs,
        eval_combs,
        sweep_combs,
        train_conds,
        eval_conds,
        sweep_conds,
    )

    # run PCA
    run_pca(adata, adata_train, adata_eval, latent_space, n_dims_train, n_dims_eval)

    # load results
    if save_model:
        cf = cfp.model.CellFlow.load(
            os.path.join(out_dir, f"{out_prefix}_CellFlow.pkl")
        )
    else:
        cf = None
    if save_adata:
        adata_pred = sc.read_h5ad(os.path.join(out_dir, f"{out_prefix}_pred.h5ad"))
    else:
        adata_pred = None

    return {
        "adata": adata,
        "adata_train": adata_train,
        "adata_eval": adata_eval,
        "cf": cf,
        "adata_pred": adata_pred,
        "vae_train": vae_train,
        "vae_eval": vae_eval,
    }


def plot_condition_embedding(
    cf: cfp.model.CellFlow,
    adata: ad.AnnData,
    out_dir: str | None = None,
):
    molecules = ["FGF8", "XAV", "RA", "CHIR", "SHH", "BMP4"]
    deepcolors = sns.color_palette("deep")

    conditions = adata.obs.drop_duplicates("condition")
    cond_embed = cf.get_condition_embedding(
        covariate_data=conditions,
        rep_dict=adata.uns,
        condition_id_key="condition",
    )
    pca = PCA(n_components=min(20, cond_embed.shape[1]))
    cond_embed_pca = pca.fit_transform(cond_embed)

    # plot PCA explained variance
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(range(1, 21), np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    if out_dir is not None:
        plt.savefig(
            os.path.join(out_dir, "cond_embed_pca_variance.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()

    # plot PCs 1 and 2
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    for i, mol in enumerate(molecules):
        ax = axs[i]
        mol_filt = np.array([mol in cond for cond in conditions["condition"]])
        sns.scatterplot(
            x=cond_embed_pca[~mol_filt, 0],
            y=cond_embed_pca[~mol_filt, 1],
            s=50,
            linewidth=0.5,
            alpha=0.8,
            ax=ax,
            color=deepcolors[7],
        )
        sns.scatterplot(
            x=cond_embed_pca[mol_filt, 0],
            y=cond_embed_pca[mol_filt, 1],
            hue=conditions[mol + "_conc"][mol_filt],
            palette="viridis",
            s=50,
            linewidth=0.5,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(mol)
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(
            os.path.join(out_dir, "cond_embed_pca_12.pdf"), dpi=300, bbox_inches="tight"
        )
    else:
        plt.show()

    # plot PCs 1 and 3
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    for i, mol in enumerate(molecules):
        ax = axs[i]
        mol_filt = np.array([mol in cond for cond in conditions["condition"]])
        sns.scatterplot(
            x=cond_embed_pca[~mol_filt, 0],
            y=cond_embed_pca[~mol_filt, 2],
            s=50,
            linewidth=0.5,
            alpha=0.8,
            ax=ax,
            color=deepcolors[7],
        )
        sns.scatterplot(
            x=cond_embed_pca[mol_filt, 0],
            y=cond_embed_pca[mol_filt, 2],
            hue=conditions[mol + "_conc"][mol_filt],
            palette="viridis",
            s=50,
            linewidth=0.5,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(mol)
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(
            os.path.join(out_dir, "cond_embed_pca_13.pdf"), dpi=300, bbox_inches="tight"
        )
    else:
        plt.show()


def load_cf_results(
    config_filename: str,
) -> dict:
    with open(config_filename, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    func_args = inspect.signature(_load_cf_results).parameters.keys()
    config_filtered = {k: v for k, v in config.items() if k in func_args}
    return _load_cf_results(**config_filtered)
