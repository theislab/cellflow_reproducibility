"""Compute cell-eval metrics per cell type for the new_donor use case.

First transfers cell type labels to predictions via PCA projection + 1-NN
(matching the cfp.preprocessing.project_pca / transfer_labels approach),
then computes cell-eval metrics for each cell type separately.

Usage:
  For baselines (identity, mean_model_1, mean_model_2, closest_embedding):
    python compute_cell_eval_metrics_per_ct.py --method <method> --donor <DonorN> --split-idx <idx>

  For cellflow:
    python compute_cell_eval_metrics_per_ct.py --method cellflow --donor <DonorN> --wandb-name <name> --num-train-cytos <k>
"""

import argparse
import logging
import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors

from cell_eval import MetricsEvaluator

# Monkey-patch pdex to handle NaN p-values from Wilcoxon test
import pdex._single_cell as _pdex_sc

_orig_fdc = _pdex_sc.false_discovery_control


def _safe_fdc(ps, *a, **kw):
    mask = np.isnan(ps)
    if mask.any():
        ps = ps.copy()
        ps[mask] = 1.0
    return _orig_fdc(ps, *a, **kw)


_pdex_sc.false_discovery_control = _safe_fdc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc"
ADATA_FULL_PATH = os.path.join(DATA_DIR, "pbmc_with_pca.h5ad")
SPLIT_DIR = os.path.join(DATA_DIR, "new_donor")
CELLFLOW_PRED_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_donor"
IDCS_TO_KEEP_PATH = os.path.join(DATA_DIR, "idcs_to_keep.pkl")

CELL_TYPE_COL = "cell_type_new"
MIN_CELLS_PER_CT = 30

# The 10 held-out test cytokines for k=65
TEST_CYTOKINES = [
    "ADSF", "BAFF", "CD27L", "FasL", "IFN-gamma",
    "IFN-omega", "IL-1Ra", "IL-32-beta", "M-CSF", "OX40L",
]


def densify(X):
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


# --- PCA projection and label transfer (matching cfp.preprocessing) ---


def project_pca(adata_query, ref_adata, obsm_key_added="X_pca"):
    """Project query data into reference PCA space.

    Replicates cfp.preprocessing.project_pca:
        X_pca = (X - X_mean.T) @ PCs
    """
    ref_means = np.asarray(ref_adata.varm["X_mean"])   # (n_genes, 1)
    ref_pcs = np.asarray(ref_adata.varm["PCs"])         # (n_genes, n_pcs)
    X = densify(adata_query.X)
    adata_query.obsm[obsm_key_added] = (X - ref_means.T) @ ref_pcs


def transfer_cell_types(adata_query, adata_ref, query_rep="X_pca", ref_rep="X_pca",
                        n_neighbors=1):
    """Transfer cell type labels from reference to query using KNN.

    With n_neighbors=1 this matches cfp's compute_wknn + transfer_labels.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(adata_ref.obsm[ref_rep])
    _, indices = nn.kneighbors(adata_query.obsm[query_rep])
    ref_labels = adata_ref.obs[CELL_TYPE_COL].values
    adata_query.obs[CELL_TYPE_COL] = ref_labels[indices.flatten()]


# --- Data loading ---


def load_common_data():
    logger.info("Loading adata_full...")
    adata_full = sc.read_h5ad(ADATA_FULL_PATH)
    return adata_full


def load_reference(adata_full):
    logger.info("Loading reference (idcs_to_keep) for cell type transfer...")
    with open(IDCS_TO_KEEP_PATH, "rb") as f:
        idcs_to_keep = pickle.load(f)
    adata_ref = adata_full[adata_full.obs_names.isin(idcs_to_keep)].copy()
    return adata_ref


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


# --- Prediction methods ---


def predict_identity(adata_ctrl, cytokine):
    adata_pred = adata_ctrl.copy()
    adata_pred.obs["cytokine"] = cytokine
    return adata_pred


def predict_mean_model_1(adata_train, adata_ctrl_donor, donor):
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
    return pred_X


def predict_mean_model_2(adata_train, adata_ctrl_donor, cytokine):
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
    adata_pred = adata_train[
        (adata_train.obs["cytokine"] == cytokine)
        & (adata_train.obs["donor"] == closest_donor)
    ].copy()
    adata_pred.X = densify(adata_pred.X)
    return adata_pred


def load_cellflow_prediction(donor, cytokine, wandb_name, num_train_cytos):
    pred_file = f"{wandb_name}_{donor}_{cytokine}_{num_train_cytos}_preds.h5ad"
    pred_path = os.path.join(CELLFLOW_PRED_DIR, pred_file)
    if not os.path.exists(pred_path):
        logger.warning(f"CellFlow prediction not found: {pred_path}")
        return None
    adata_pred = sc.read_h5ad(pred_path)
    adata_pred.X = adata_pred.layers["X_recon"]
    adata_pred.X = densify(adata_pred.X)
    return adata_pred


# --- Build combined AnnData with cell type ---


def build_combined_adata(adata_ctrl, cytokine_adatas, cytokines, var_names):
    """Build combined AnnData with control + perturbation cells, including cell_type."""
    parts = []

    # Control part — has cell_type from adata_full
    ctrl = adata_ctrl.copy()
    ct_labels_ctrl = ctrl.obs[CELL_TYPE_COL].values.copy()
    ctrl.obs = ctrl.obs[[]].copy()
    ctrl.obs["cytokine"] = "PBS"
    ctrl.obs[CELL_TYPE_COL] = ct_labels_ctrl
    ctrl = ad.AnnData(
        X=densify(ctrl.X),
        obs=ctrl.obs,
        var=pd.DataFrame(index=var_names),
    )
    parts.append(ctrl)

    # Perturbation parts — cell_type from adata_full (gt) or from transfer (pred)
    for cyto, adata in zip(cytokines, cytokine_adatas):
        ct_labels = (
            adata.obs[CELL_TYPE_COL].values.copy()
            if CELL_TYPE_COL in adata.obs.columns
            else np.array(["unknown"] * adata.shape[0])
        )
        obs = pd.DataFrame({
            "cytokine": [cyto] * adata.shape[0],
            CELL_TYPE_COL: ct_labels,
        })
        part = ad.AnnData(
            X=densify(adata.X),
            obs=obs,
            var=pd.DataFrame(index=var_names),
        )
        parts.append(part)

    combined = ad.concat(parts, join="outer")
    combined.obs_names_make_unique()
    combined.X = np.clip(combined.X, 0, None)
    return combined


# --- Run cell-eval ---


def run_cell_eval(adata_real, adata_pred, outdir, prefix, num_threads=8, write_csv=True):
    os.makedirs(outdir, exist_ok=True)
    try:
        evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert="PBS",
            pert_col="cytokine",
            num_threads=num_threads,
            outdir=outdir,
            prefix=prefix,
        )
        results, agg_results = evaluator.compute(
            profile="full",
            write_csv=write_csv,
        )
        return results, agg_results
    except Exception as e:
        logger.warning(f"Full profile failed ({e}), retrying with skip_de=True...")
        evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert="PBS",
            pert_col="cytokine",
            num_threads=num_threads,
            outdir=outdir,
            prefix=prefix,
            skip_de=True,
        )
        results, agg_results = evaluator.compute(
            profile="anndata",
            write_csv=write_csv,
        )
        return results, agg_results


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Compute cell-eval metrics per cell type for new_donor"
    )
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
    parser.add_argument("--min-cells", type=int, default=MIN_CELLS_PER_CT)
    args = parser.parse_args()

    method = args.method
    donor = args.donor
    outdir = args.outdir
    min_cells = args.min_cells

    # Load common data
    adata_full = load_common_data()
    var_names = adata_full.var_names.copy()

    # Load reference for cell type transfer
    adata_ref = load_reference(adata_full)
    logger.info(f"Reference cells: {adata_ref.n_obs}")

    # Get control cells (with cell type from adata_full)
    adata_ctrl = get_control_cells(adata_full, donor)
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
        logger.info(
            f"Split {split_idx}: {num_train} train cytokines, "
            f"{len(cytokines_to_impute)} to impute"
        )

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

    # Generate predictions for each test cytokine
    gt_adatas = []
    pred_adatas = []
    valid_cytokines = []

    for cytokine in cytokines_to_impute:
        adata_gt = get_ground_truth(adata_full, donor, cytokine)
        if adata_gt.n_obs == 0:
            logger.warning(f"No ground truth cells for {donor}_{cytokine}, skipping")
            continue

        if method == "identity":
            adata_pred_cyto = predict_identity(adata_ctrl, cytokine)

        elif method == "mean_model_1":
            pred_X = predict_mean_model_1(adata_train, adata_ctrl_train, donor)
            adata_pred_cyto = ad.AnnData(
                X=pred_X, var=adata_ctrl_train.var.copy(),
            )
            adata_pred_cyto.obs["cytokine"] = cytokine
            adata_pred_cyto.obs["donor"] = donor

        elif method == "mean_model_2":
            pred_X = predict_mean_model_2(adata_train, adata_ctrl_train, cytokine)
            adata_pred_cyto = ad.AnnData(
                X=pred_X, var=adata_ctrl_train.var.copy(),
            )
            adata_pred_cyto.obs["cytokine"] = cytokine
            adata_pred_cyto.obs["donor"] = donor

        elif method == "closest_embedding":
            adata_pred_cyto = predict_closest_embedding(
                adata_train, donor, closest_donor, cytokine
            )
            if adata_pred_cyto.n_obs == 0:
                logger.warning(
                    f"No prediction for closest_embedding {donor}_{cytokine}, skipping"
                )
                continue

        elif method == "cellflow":
            adata_pred_cyto = load_cellflow_prediction(
                donor, cytokine, args.wandb_name, args.num_train_cytos
            )
            if adata_pred_cyto is None:
                logger.warning(
                    f"CellFlow prediction not found for {donor}_{cytokine}, skipping"
                )
                continue
            adata_pred_cyto.var_names = var_names

        # Transfer cell type labels to prediction
        project_pca(adata_pred_cyto, adata_ref)
        transfer_cell_types(adata_pred_cyto, adata_ref)
        logger.info(
            f"  {cytokine}: transferred cell types to {adata_pred_cyto.n_obs} pred cells"
        )

        gt_adatas.append(adata_gt)
        pred_adatas.append(adata_pred_cyto)
        valid_cytokines.append(cytokine)

    if len(valid_cytokines) == 0:
        logger.error("No valid cytokines to evaluate")
        sys.exit(1)

    logger.info(f"Building combined AnnData for {len(valid_cytokines)} cytokines...")

    # Build combined adatas with cell_type column
    adata_real = build_combined_adata(adata_ctrl, gt_adatas, valid_cytokines, var_names)
    adata_pred_combined = build_combined_adata(
        adata_ctrl, pred_adatas, valid_cytokines, var_names
    )

    logger.info(f"adata_real: {adata_real.shape}, adata_pred: {adata_pred_combined.shape}")

    # Free memory
    del adata_full, adata_ref
    if method in ("mean_model_1", "mean_model_2", "closest_embedding"):
        del adata_train

    # --- Overall cell-eval ---
    logger.info("Running overall cell-eval...")
    results, agg_results = run_cell_eval(
        adata_real, adata_pred_combined,
        outdir=outdir, prefix=prefix, num_threads=args.num_threads,
    )
    logger.info(f"Overall: results {results.shape}, agg {agg_results.shape}")

    # --- Per-cell-type cell-eval ---
    logger.info("Running per-cell-type cell-eval...")
    cell_types = sorted(
        set(adata_real.obs[CELL_TYPE_COL].dropna().unique())
        & set(adata_pred_combined.obs[CELL_TYPE_COL].dropna().unique())
    )
    logger.info(f"Cell types to evaluate: {cell_types}")

    ct_results_list = []
    ct_agg_list = []

    for ct in cell_types:
        real_ct = adata_real[adata_real.obs[CELL_TYPE_COL] == ct].copy()
        pred_ct = adata_pred_combined[
            adata_pred_combined.obs[CELL_TYPE_COL] == ct
        ].copy()

        # Need enough cells and both PBS + at least one perturbation
        real_perts = set(real_ct.obs["cytokine"].unique())
        pred_perts = set(pred_ct.obs["cytokine"].unique())
        if "PBS" not in real_perts or "PBS" not in pred_perts:
            logger.info(f"  {ct}: skipping (no PBS control cells)")
            continue
        if len(real_perts) < 2 or len(pred_perts) < 2:
            logger.info(f"  {ct}: skipping (fewer than 2 conditions)")
            continue

        # Only keep cytokines present in both
        common_perts = real_perts & pred_perts
        real_ct = real_ct[real_ct.obs["cytokine"].isin(common_perts)].copy()
        pred_ct = pred_ct[pred_ct.obs["cytokine"].isin(common_perts)].copy()

        if real_ct.n_obs < min_cells or pred_ct.n_obs < min_cells:
            logger.info(
                f"  {ct}: skipping (real={real_ct.n_obs}, pred={pred_ct.n_obs} < {min_cells})"
            )
            continue

        logger.info(
            f"  {ct}: real={real_ct.n_obs}, pred={pred_ct.n_obs}, "
            f"perts={sorted(common_perts - {'PBS'})}"
        )

        ct_safe = ct.replace(" ", "_").replace("/", "_")
        ct_prefix = f"{prefix}_ct_{ct_safe}"
        try:
            ct_res, ct_agg = run_cell_eval(
                real_ct, pred_ct,
                outdir=outdir, prefix=ct_prefix,
                num_threads=args.num_threads, write_csv=False,
            )
            # Add cell type column — handle polars and pandas DataFrames
            if isinstance(ct_res, pl.DataFrame):
                ct_res = ct_res.with_columns(pl.lit(ct).alias(CELL_TYPE_COL)).to_pandas()
            else:
                ct_res[CELL_TYPE_COL] = ct
            if isinstance(ct_agg, pl.DataFrame):
                ct_agg = ct_agg.with_columns(pl.lit(ct).alias(CELL_TYPE_COL)).to_pandas()
            else:
                ct_agg[CELL_TYPE_COL] = ct
            ct_results_list.append(ct_res)
            ct_agg_list.append(ct_agg)
        except Exception as e:
            logger.warning(f"  {ct}: cell-eval failed ({e})")

    # Save combined per-cell-type results
    if ct_results_list:
        ct_results_all = pd.concat(ct_results_list, ignore_index=True)
        ct_results_all.to_csv(
            os.path.join(outdir, f"{prefix}_per_ct_results.csv"), index=False
        )
        logger.info(f"Per-cell-type results: {ct_results_all.shape}")
    else:
        logger.warning("No per-cell-type results produced")

    if ct_agg_list:
        ct_agg_all = pd.concat(ct_agg_list, ignore_index=True)
        ct_agg_all.to_csv(
            os.path.join(outdir, f"{prefix}_per_ct_agg_results.csv"), index=False
        )
        logger.info(f"Per-cell-type aggregated results: {ct_agg_all.shape}")
    else:
        logger.warning("No per-cell-type aggregated results produced")

    logger.info("Done!")


if __name__ == "__main__":
    main()
