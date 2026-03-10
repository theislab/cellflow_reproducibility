"""Compute systema metrics (Pearson-delta, centroid accuracy, etc.) for the new_donor use case.

Uses the same data splits, predictions, and methods as the cell_eval pipeline,
but computes the metrics from the systema benchmark (https://github.com/mlbio-epfl/systema).

Metrics computed per (method, donor, split/wandb):
  - pearson_delta_all:  PearsonΔ on all genes (delta = centroid - control_mean)
  - pearson_delta_top20: PearsonΔ on top-20 DE genes
  - mse_all:            MSE on all genes (centroid-level)
  - rmse_all:           RMSE on all genes
  - jaccard_top20:      Jaccard similarity of top-20 DE gene sets (true vs pred)
  - centroid_accuracy:  Fraction of other perturbations farther than the correct one

Usage:
  For baselines (identity, mean_model_1, mean_model_2, closest_embedding):
    python compute_systema_metrics.py --method <method> --donor <DonorN> --split-idx <idx>

  For cellflow:
    python compute_systema_metrics.py --method cellflow --donor <DonorN> --wandb-name <name> --num-train-cytos <k>
"""

import argparse
import logging
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths (same as cell_eval)
DATA_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc"
ADATA_FULL_PATH = os.path.join(DATA_DIR, "pbmc_with_pca.h5ad")
SPLIT_DIR = os.path.join(DATA_DIR, "new_donor")
CELLFLOW_PRED_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor"

# The 10 held-out test cytokines for k=65
TEST_CYTOKINES = [
    "ADSF", "BAFF", "CD27L", "FasL", "IFN-gamma",
    "IFN-omega", "IL-1Ra", "IL-32-beta", "M-CSF", "OX40L",
]


def densify(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def load_common_data():
    logger.info("Loading adata_full...")
    adata_full = sc.read_h5ad(ADATA_FULL_PATH)
    return adata_full


def get_control_cells(adata_full, donor, max_cells=10000):
    adata_ctrl = adata_full[
        (adata_full.obs["cytokine"] == "PBS") & (adata_full.obs["donor"] == donor)
    ].copy()
    if adata_ctrl.n_obs > max_cells:
        sc.pp.subsample(adata_ctrl, n_obs=max_cells)
    adata_ctrl.X = densify(adata_ctrl.X)
    return adata_ctrl


def get_ground_truth(adata_full, donor, cytokine):
    adata_gt = adata_full[
        (adata_full.obs["donor"] == donor) & (adata_full.obs["cytokine"] == cytokine)
    ].copy()
    adata_gt.X = densify(adata_gt.X)
    return adata_gt


# --- Prediction methods (identical to cell_eval) ---


def predict_identity(adata_ctrl, cytokine):
    adata_pred = adata_ctrl.copy()
    adata_pred.obs["cytokine"] = cytokine
    return adata_pred


def predict_mean_model_1(adata_train, adata_ctrl_donor, donor):
    """Mean displacement of same donor across training cytokines."""
    control_mean = np.asarray(adata_ctrl_donor.X.mean(axis=0)).squeeze()
    adata_pert = adata_train[
        (adata_train.obs["donor"] == donor) & (adata_train.obs["cytokine"] != "PBS")
    ]
    displacement_vecs = []
    for cyto in adata_pert.obs["cytokine"].unique():
        pert_mean = np.asarray(
            adata_pert[adata_pert.obs["cytokine"] == cyto].X.mean(axis=0)
        ).squeeze()
        displacement_vecs.append(pert_mean - control_mean)
    displacement = np.array(displacement_vecs).mean(axis=0)
    pred_X = densify(adata_ctrl_donor.X) + displacement
    return pred_X, displacement


def predict_mean_model_2(adata_train, adata_ctrl_donor, cytokine):
    """Mean displacement of same cytokine across training donors."""
    control_mean = np.asarray(adata_ctrl_donor.X.mean(axis=0)).squeeze()
    adata_pert = adata_train[adata_train.obs["cytokine"] == cytokine]
    displacement_vecs = []
    for d in adata_pert.obs["donor"].unique():
        pert_mean = np.asarray(
            adata_pert[adata_pert.obs["donor"] == d].X.mean(axis=0)
        ).squeeze()
        displacement_vecs.append(pert_mean - control_mean)
    if len(displacement_vecs) == 0:
        return densify(adata_ctrl_donor.X)
    displacement = np.array(displacement_vecs).mean(axis=0)
    return densify(adata_ctrl_donor.X) + displacement


def find_closest_donor(donor_held_out, donor_embeddings):
    emb_0 = donor_embeddings[donor_held_out]
    closest_donor = None
    closest_dist = np.inf
    for ref, ref_emb in donor_embeddings.items():
        if ref == donor_held_out:
            continue
        dist = np.sum((emb_0 - ref_emb) ** 2)
        if dist < closest_dist:
            closest_dist = dist
            closest_donor = ref
    return closest_donor


def predict_closest_embedding(adata_train, donor_held_out, closest_donor, cytokine):
    """Use closest donor's response for the specific cytokine."""
    adata_pred = adata_train[
        (adata_train.obs["cytokine"] == cytokine)
        & (adata_train.obs["donor"] == closest_donor)
    ].copy()
    adata_pred.X = densify(adata_pred.X)
    return adata_pred


def load_cellflow_prediction(donor, cytokine, wandb_name, num_train_cytos, var_names):
    pred_file = f"{wandb_name}_{donor}_{cytokine}_{num_train_cytos}_preds.h5ad"
    pred_path = os.path.join(CELLFLOW_PRED_DIR, pred_file)
    if not os.path.exists(pred_path):
        logger.warning(f"CellFlow prediction not found: {pred_path}")
        return None
    adata_pred = sc.read_h5ad(pred_path)
    adata_pred.X = adata_pred.layers["X_recon"]
    adata_pred.X = densify(adata_pred.X)
    adata_pred.var_names = var_names
    return adata_pred


# --- Systema metric functions ---


def get_topk_de_gene_ids(ctrl_mean, post_mean, k=20):
    """Get indices of top-k differentially expressed genes by |delta|."""
    delta = np.abs(post_mean - ctrl_mean)
    return np.argsort(delta)[-k:]


def jaccard_similarity(list1, list2):
    """Jaccard similarity between two index sets."""
    s1, s2 = set(list1), set(list2)
    if len(s1 | s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def compute_per_perturbation_metrics(ctrl_mean, gt_centroids, pred_centroids, cytokines):
    """Compute per-perturbation systema metrics.

    Parameters
    ----------
    ctrl_mean : np.ndarray, shape (n_genes,)
    gt_centroids : dict[str, np.ndarray], cytokine -> ground truth centroid
    pred_centroids : dict[str, np.ndarray], cytokine -> predicted centroid
    cytokines : list[str]

    Returns
    -------
    pd.DataFrame with per-perturbation metrics
    """
    rows = []
    for cyto in cytokines:
        gt_cent = gt_centroids[cyto]
        pred_cent = pred_centroids[cyto]

        delta_true = gt_cent - ctrl_mean
        delta_pred = pred_cent - ctrl_mean

        # PearsonΔ (all genes)
        corr_all, _ = pearsonr(delta_true, delta_pred)

        # PearsonΔ (top 20 DE genes)
        top20_true = get_topk_de_gene_ids(ctrl_mean, gt_cent, k=20)
        if len(delta_true[top20_true]) >= 2:
            corr_top20, _ = pearsonr(delta_true[top20_true], delta_pred[top20_true])
        else:
            corr_top20 = np.nan

        # MSE & RMSE (on deltas)
        mse = np.mean((delta_true - delta_pred) ** 2)
        rmse = np.sqrt(mse)

        # Jaccard of top-20 DE genes
        top20_pred = get_topk_de_gene_ids(ctrl_mean, pred_cent, k=20)
        jacc = jaccard_similarity(top20_true, top20_pred)

        rows.append({
            "perturbation": cyto,
            "pearson_delta_all": corr_all,
            "pearson_delta_top20": corr_top20,
            "mse_delta": mse,
            "rmse_delta": rmse,
            "jaccard_top20_de": jacc,
        })

    return pd.DataFrame(rows)


def compute_centroid_accuracy(gt_centroids, pred_centroids, cytokines):
    """Compute centroid accuracy: for each perturbation, fraction of other
    perturbations whose GT centroid is farther from the prediction than the
    correct GT centroid.

    Parameters
    ----------
    gt_centroids : dict[str, np.ndarray]
    pred_centroids : dict[str, np.ndarray]
    cytokines : list[str]

    Returns
    -------
    dict[str, float] mapping cytokine -> centroid accuracy
    """
    n = len(cytokines)
    if n < 2:
        return {c: np.nan for c in cytokines}

    gt_matrix = np.array([gt_centroids[c] for c in cytokines])
    pred_matrix = np.array([pred_centroids[c] for c in cytokines])

    # Distance from each prediction to each GT centroid
    dists = cdist(pred_matrix, gt_matrix, metric="euclidean")

    accuracies = {}
    for i, cyto in enumerate(cytokines):
        self_dist = dists[i, i]
        # Fraction of other perturbations that are farther away
        n_farther = np.sum(dists[i, :] > self_dist)
        accuracies[cyto] = n_farther / (n - 1)

    return accuracies


def main():
    parser = argparse.ArgumentParser(description="Compute systema metrics for new_donor")
    parser.add_argument(
        "--method",
        required=True,
        choices=["identity", "mean_model_1", "mean_model_2", "closest_embedding", "cellflow"],
    )
    parser.add_argument("--donor", required=True)
    parser.add_argument("--split-idx", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--num-train-cytos", type=int, default=65)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    method = args.method
    donor = args.donor
    outdir = args.outdir

    # Load common data
    adata_full = load_common_data()
    var_names = adata_full.var_names.copy()

    # Get control cells
    adata_ctrl = get_control_cells(adata_full, donor)
    ctrl_mean = np.asarray(adata_ctrl.X.mean(axis=0)).squeeze()
    logger.info(f"Control cells for {donor}: {adata_ctrl.n_obs}")

    # Determine test cytokines and load split-specific data if needed
    if method in ("mean_model_1", "mean_model_2", "closest_embedding"):
        if args.split_idx is None:
            raise ValueError(f"--split-idx required for method {method}")
        split_idx = args.split_idx
        logger.info(f"Loading training data for {donor}, split {split_idx}...")
        adata_train = sc.read_h5ad(
            os.path.join(SPLIT_DIR, donor, str(split_idx), f"adata_train_{donor}.h5ad")
        )
        split_info = adata_train.uns["split_info"][split_idx]
        cytokines_to_impute = split_info["cytokines_to_impute"]
        num_train = len(split_info["cytokines_to_train_data"])
        logger.info(f"Split {split_idx}: {num_train} train cytokines, {len(cytokines_to_impute)} to impute")

        # Get control cells from training data for the held-out donor
        adata_ctrl_train = adata_train[
            adata_train.obs["is_control"].to_numpy() & (adata_train.obs["donor"] == donor)
        ].copy()
        if adata_ctrl_train.n_obs > 10000:
            sc.pp.subsample(adata_ctrl_train, n_obs=10000)
        adata_ctrl_train.X = densify(adata_ctrl_train.X)

        if method == "closest_embedding":
            donor_embeddings = adata_train.uns["donor_embeddings"]
            closest_donor = find_closest_donor(donor, donor_embeddings)
            logger.info(f"Closest donor to {donor}: {closest_donor}")

        prefix = f"{method}_{donor}_{split_idx}_{num_train}"
    elif method == "identity":
        cytokines_to_impute = TEST_CYTOKINES
        prefix = f"identity_{donor}"
    elif method == "cellflow":
        if args.wandb_name is None:
            raise ValueError("--wandb-name required for cellflow")
        cytokines_to_impute = TEST_CYTOKINES
        prefix = f"cellflow_{donor}_{args.wandb_name}_{args.num_train_cytos}"

    logger.info(f"Cytokines to evaluate: {cytokines_to_impute}")

    # Generate predictions and compute centroids for each test cytokine
    gt_centroids = {}
    pred_centroids = {}
    valid_cytokines = []

    for cytokine in cytokines_to_impute:
        # Ground truth
        adata_gt = get_ground_truth(adata_full, donor, cytokine)
        if adata_gt.n_obs == 0:
            logger.warning(f"No ground truth cells for {donor}_{cytokine}, skipping")
            continue

        # Generate prediction
        if method == "identity":
            adata_pred_cyto = predict_identity(adata_ctrl, cytokine)

        elif method == "mean_model_1":
            pred_X, _ = predict_mean_model_1(adata_train, adata_ctrl_train, donor)
            adata_pred_cyto = ad.AnnData(
                X=pred_X,
                var=adata_ctrl_train.var.copy(),
            )

        elif method == "mean_model_2":
            pred_X = predict_mean_model_2(adata_train, adata_ctrl_train, cytokine)
            adata_pred_cyto = ad.AnnData(
                X=pred_X,
                var=adata_ctrl_train.var.copy(),
            )

        elif method == "closest_embedding":
            adata_pred_cyto = predict_closest_embedding(
                adata_train, donor, closest_donor, cytokine
            )
            if adata_pred_cyto.n_obs == 0:
                logger.warning(f"No prediction for closest_embedding {donor}_{cytokine}, skipping")
                continue

        elif method == "cellflow":
            adata_pred_cyto = load_cellflow_prediction(
                donor, cytokine, args.wandb_name, args.num_train_cytos, var_names
            )
            if adata_pred_cyto is None:
                logger.warning(f"CellFlow prediction not found for {donor}_{cytokine}, skipping")
                continue

        # Compute centroids
        gt_cent = np.asarray(adata_gt.X.mean(axis=0)).squeeze()
        pred_cent = np.asarray(densify(adata_pred_cyto.X).mean(axis=0)).squeeze()

        gt_centroids[cytokine] = gt_cent
        pred_centroids[cytokine] = pred_cent
        valid_cytokines.append(cytokine)

    if len(valid_cytokines) == 0:
        logger.error("No valid cytokines to evaluate")
        sys.exit(1)

    # Free memory
    del adata_full
    if method in ("mean_model_1", "mean_model_2", "closest_embedding"):
        del adata_train

    logger.info(f"Computing systema metrics for {len(valid_cytokines)} cytokines...")

    # Compute per-perturbation metrics
    results_df = compute_per_perturbation_metrics(
        ctrl_mean, gt_centroids, pred_centroids, valid_cytokines
    )

    # Compute centroid accuracy
    cent_acc = compute_centroid_accuracy(gt_centroids, pred_centroids, valid_cytokines)
    results_df["centroid_accuracy"] = results_df["perturbation"].map(cent_acc)

    # Aggregated results (mean across perturbations)
    metric_cols = [c for c in results_df.columns if c != "perturbation"]
    agg_results = results_df[metric_cols].mean().to_frame().T

    # Save results
    os.makedirs(outdir, exist_ok=True)
    results_path = os.path.join(outdir, f"{prefix}_results.csv")
    agg_results_path = os.path.join(outdir, f"{prefix}_agg_results.csv")
    results_df.to_csv(results_path, index=False)
    agg_results.to_csv(agg_results_path, index=False)

    logger.info(f"Per-perturbation results ({results_df.shape}): {results_path}")
    logger.info(f"Aggregated results ({agg_results.shape}): {agg_results_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
