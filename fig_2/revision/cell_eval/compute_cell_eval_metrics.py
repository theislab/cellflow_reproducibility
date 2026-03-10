"""Compute cell-eval metrics for the new_donor use case.

Usage:
  For baselines (identity, mean_model_1, mean_model_2, closest_embedding):
    python compute_cell_eval_metrics.py --method <method> --donor <DonorN> --split-idx <idx>

  For cellflow:
    python compute_cell_eval_metrics.py --method cellflow --donor <DonorN> --wandb-name <name> --num-train-cytos <k>
"""

import argparse
import glob
import logging
import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from cell_eval import MetricsEvaluator

# Monkey-patch pdex to handle NaN p-values from Wilcoxon test
# (some genes have constant expression, producing NaN p-values)
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


# --- Prediction methods ---


def predict_identity(adata_ctrl, cytokine):
    adata_pred = adata_ctrl.copy()
    adata_pred.obs["cytokine"] = cytokine
    return adata_pred


def predict_mean_model_1(adata_train, adata_ctrl_donor, donor):
    """Mean displacement of same donor across training cytokines (constant prediction)."""
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


def build_combined_adata(adata_ctrl, cytokine_adatas, cytokines, var_names):
    """Build combined AnnData with control + perturbation cells.

    Ensures all parts have the same genes and a 'cytokine' column.
    """
    parts = []

    # Control part
    ctrl = adata_ctrl.copy()
    ctrl.obs = ctrl.obs[[]].copy()
    ctrl.obs["cytokine"] = "PBS"
    ctrl = ad.AnnData(
        X=densify(ctrl.X),
        obs=ctrl.obs,
        var=pd.DataFrame(index=var_names),
    )
    parts.append(ctrl)

    # Perturbation parts
    for cyto, adata in zip(cytokines, cytokine_adatas):
        obs = pd.DataFrame({"cytokine": [cyto] * adata.shape[0]})
        part = ad.AnnData(
            X=densify(adata.X) if adata.X is not None else np.zeros((adata.shape[0], len(var_names))),
            obs=obs,
            var=pd.DataFrame(index=var_names),
        )
        parts.append(part)

    combined = ad.concat(parts, join="outer")
    combined.obs_names_make_unique()
    # Clip negative values to 0 — cell-eval requires non-negative (log-normalized) data,
    # but displacement-based models (mean_model, cellflow) can produce negatives.
    combined.X = np.clip(combined.X, 0, None)
    return combined


def run_cell_eval(adata_real, adata_pred, outdir, prefix, num_threads=8):
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
            write_csv=True,
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
            write_csv=True,
        )
        return results, agg_results


def main():
    parser = argparse.ArgumentParser(description="Compute cell-eval metrics for new_donor")
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

    # Generate predictions for each test cytokine
    gt_adatas = []
    pred_adatas = []
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
            adata_pred_cyto.obs["cytokine"] = cytokine
            adata_pred_cyto.obs["donor"] = donor

        elif method == "mean_model_2":
            pred_X = predict_mean_model_2(adata_train, adata_ctrl_train, cytokine)
            adata_pred_cyto = ad.AnnData(
                X=pred_X,
                var=adata_ctrl_train.var.copy(),
            )
            adata_pred_cyto.obs["cytokine"] = cytokine
            adata_pred_cyto.obs["donor"] = donor

        elif method == "closest_embedding":
            adata_pred_cyto = predict_closest_embedding(
                adata_train, donor, closest_donor, cytokine
            )
            if adata_pred_cyto.n_obs == 0:
                logger.warning(f"No prediction for closest_embedding {donor}_{cytokine}, skipping")
                continue

        elif method == "cellflow":
            adata_pred_cyto = load_cellflow_prediction(
                donor, cytokine, args.wandb_name, args.num_train_cytos
            )
            if adata_pred_cyto is None:
                logger.warning(f"CellFlow prediction not found for {donor}_{cytokine}, skipping")
                continue
            adata_pred_cyto.var_names = var_names

        gt_adatas.append(adata_gt)
        pred_adatas.append(adata_pred_cyto)
        valid_cytokines.append(cytokine)

    if len(valid_cytokines) == 0:
        logger.error("No valid cytokines to evaluate")
        sys.exit(1)

    logger.info(f"Building combined AnnData for {len(valid_cytokines)} cytokines...")

    # Build combined adata_real and adata_pred
    adata_real = build_combined_adata(adata_ctrl, gt_adatas, valid_cytokines, var_names)
    adata_pred_combined = build_combined_adata(
        adata_ctrl, pred_adatas, valid_cytokines, var_names
    )

    logger.info(
        f"adata_real: {adata_real.shape}, adata_pred: {adata_pred_combined.shape}"
    )
    logger.info(
        f"Perturbations in real: {sorted(adata_real.obs['cytokine'].unique())}"
    )
    logger.info(
        f"Perturbations in pred: {sorted(adata_pred_combined.obs['cytokine'].unique())}"
    )

    # Free memory
    del adata_full
    if method in ("mean_model_1", "mean_model_2", "closest_embedding"):
        del adata_train

    # Run cell-eval
    logger.info("Running cell-eval...")
    results, agg_results = run_cell_eval(
        adata_real,
        adata_pred_combined,
        outdir=outdir,
        prefix=prefix,
        num_threads=args.num_threads,
    )

    logger.info(f"Done! Results saved to {outdir}")
    logger.info(f"Per-perturbation results shape: {results.shape}")
    logger.info(f"Aggregated results shape: {agg_results.shape}")


if __name__ == "__main__":
    main()
