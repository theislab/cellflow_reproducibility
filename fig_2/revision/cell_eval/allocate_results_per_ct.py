"""Aggregate per-cell-type cell-eval results into a single DataFrame.

Usage:
    python allocate_results_per_ct.py [--indir <dir>] [--outfile <file>]
"""

import argparse
import os

import polars as pl

# Columns that should always be Float64 (may be inferred as String when all null)
FLOAT_COLS = [
    "overlap_at_N", "overlap_at_50", "overlap_at_100", "overlap_at_200", "overlap_at_500",
    "precision_at_N", "precision_at_50", "precision_at_100", "precision_at_200", "precision_at_500",
    "de_spearman_sig", "de_direction_match", "de_spearman_lfc_sig",
    "de_sig_genes_recall", "de_nsig_counts_real", "de_nsig_counts_pred",
    "pr_auc", "roc_auc",
    "pearson_delta", "mse", "mae", "mse_delta", "mae_delta",
    "discrimination_score_l1", "discrimination_score_l2", "discrimination_score_cosine",
    "pearson_edistance", "clustering_agreement",
]


def _cast_float_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Cast known numeric columns to Float64 (handles null-only String columns)."""
    for col in FLOAT_COLS:
        if col in df.columns and df[col].dtype != pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    return df


def parse_prefix(prefix):
    """Parse method, donor, split/wandb, num_train from prefix."""
    parts = prefix.split("_")
    if parts[0] == "identity":
        return "identity", parts[1], "none", 65
    elif parts[0] == "cellflow":
        num_train = int(parts[-1])
        donor = parts[1]
        split_or_wandb = "_".join(parts[2:-1])
        return "cellflow", donor, split_or_wandb, num_train
    elif parts[0] in ("mean", "closest"):
        if parts[0] == "mean":
            method = f"mean_model_{parts[2]}"
            donor = parts[3]
            split_or_wandb = parts[4]
            num_train = int(parts[5])
        else:
            method = "closest_embedding"
            donor = parts[2]
            split_or_wandb = parts[3]
            num_train = int(parts[4])
        return method, donor, split_or_wandb, num_train
    return None, None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        default="/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_cell_eval_per_ct",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_cell_eval_per_ct/cell_eval_per_ct_aggregated.csv",
    )
    args = parser.parse_args()

    indir = args.indir

    # Collect per-ct results and per-ct agg results
    ct_result_dfs = []
    ct_agg_dfs = []
    # Also collect overall results (same as allocate_results.py)
    agg_dfs = []
    result_dfs = []

    for fname in sorted(os.listdir(indir)):
        # Per-cell-type results
        if fname.endswith("_per_ct_results.csv"):
            prefix = fname.replace("_per_ct_results.csv", "")
            method, donor, split_or_wandb, num_train = parse_prefix(prefix)
            if method is None:
                print(f"Skipping unrecognized: {fname}")
                continue
            try:
                df = _cast_float_cols(pl.read_csv(os.path.join(indir, fname)))
                df = df.with_columns(
                    pl.lit(method).alias("method"),
                    pl.lit(donor).alias("donor"),
                    pl.lit(split_or_wandb).alias("split_or_wandb"),
                    pl.lit(num_train).alias("num_train_cytos"),
                    pl.lit(prefix).alias("prefix"),
                )
                ct_result_dfs.append(df)
            except Exception as e:
                print(f"Error reading {fname}: {e}")

        elif fname.endswith("_per_ct_agg_results.csv"):
            prefix = fname.replace("_per_ct_agg_results.csv", "")
            method, donor, split_or_wandb, num_train = parse_prefix(prefix)
            if method is None:
                print(f"Skipping unrecognized: {fname}")
                continue
            try:
                df = _cast_float_cols(pl.read_csv(os.path.join(indir, fname)))
                df = df.with_columns(
                    pl.lit(method).alias("method"),
                    pl.lit(donor).alias("donor"),
                    pl.lit(split_or_wandb).alias("split_or_wandb"),
                    pl.lit(num_train).alias("num_train_cytos"),
                    pl.lit(prefix).alias("prefix"),
                )
                ct_agg_dfs.append(df)
            except Exception as e:
                print(f"Error reading {fname}: {e}")

        # Overall results (same logic as allocate_results.py)
        elif fname.endswith("_agg_results.csv") and "_per_ct_" not in fname:
            prefix = fname.replace("_agg_results.csv", "")
            method, donor, split_or_wandb, num_train = parse_prefix(prefix)
            if method is None:
                continue
            try:
                df = _cast_float_cols(pl.read_csv(os.path.join(indir, fname)))
                df = df.with_columns(
                    pl.lit(method).alias("method"),
                    pl.lit(donor).alias("donor"),
                    pl.lit(split_or_wandb).alias("split_or_wandb"),
                    pl.lit(num_train).alias("num_train_cytos"),
                    pl.lit(prefix).alias("prefix"),
                )
                agg_dfs.append(df)
            except Exception as e:
                print(f"Error reading {fname}: {e}")

        elif fname.endswith("_results.csv") and "_per_ct_" not in fname:
            prefix = fname.replace("_results.csv", "")
            method, donor, split_or_wandb, num_train = parse_prefix(prefix)
            if method is None:
                continue
            try:
                df = _cast_float_cols(pl.read_csv(os.path.join(indir, fname)))
                df = df.with_columns(
                    pl.lit(method).alias("method"),
                    pl.lit(donor).alias("donor"),
                    pl.lit(split_or_wandb).alias("split_or_wandb"),
                    pl.lit(num_train).alias("num_train_cytos"),
                    pl.lit(prefix).alias("prefix"),
                )
                result_dfs.append(df)
            except Exception as e:
                print(f"Error reading {fname}: {e}")

    # Write per-cell-type results
    if ct_result_dfs:
        combined = pl.concat(ct_result_dfs, how="diagonal")
        outpath = args.outfile.replace(".csv", "_per_perturbation.csv")
        combined.write_csv(outpath)
        print(f"Per-CT per-perturbation results ({combined.shape}): {outpath}")
    else:
        print("No per-CT per-perturbation results found.")

    if ct_agg_dfs:
        combined = pl.concat(ct_agg_dfs, how="diagonal")
        combined.write_csv(args.outfile)
        print(f"Per-CT aggregated results ({combined.shape}): {args.outfile}")
    else:
        print("No per-CT aggregated results found.")

    # Write overall results
    if agg_dfs:
        combined = pl.concat(agg_dfs, how="diagonal")
        outpath = args.outfile.replace("per_ct_aggregated", "overall_aggregated")
        combined.write_csv(outpath)
        print(f"Overall aggregated results ({combined.shape}): {outpath}")

    if result_dfs:
        combined = pl.concat(result_dfs, how="diagonal")
        outpath = args.outfile.replace("per_ct_aggregated", "overall_per_perturbation")
        combined.write_csv(outpath)
        print(f"Overall per-perturbation results ({combined.shape}): {outpath}")


if __name__ == "__main__":
    main()
