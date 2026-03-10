"""Aggregate systema metric results from all methods into a single DataFrame.

Usage:
    python allocate_results.py [--indir <metrics_dir>] [--outfile <output.csv>]
"""

import argparse
import os

import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        default="/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_systema",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics_new_donor_systema/systema_metrics_aggregated.csv",
    )
    args = parser.parse_args()

    indir = args.indir

    agg_dfs = []
    result_dfs = []

    for fname in sorted(os.listdir(indir)):
        if not fname.endswith("_agg_results.csv"):
            continue

        prefix = fname.replace("_agg_results.csv", "")

        # Parse method and metadata from prefix
        parts = prefix.split("_")
        if parts[0] == "identity":
            method = "identity"
            donor = parts[1]
            split_or_wandb = "none"
            num_train = 65
        elif parts[0] == "cellflow":
            method = "cellflow"
            donor = parts[1]
            num_train = int(parts[-1])
            split_or_wandb = "_".join(parts[2:-1])
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
        else:
            print(f"Skipping unrecognized file: {fname}")
            continue

        filepath = os.path.join(indir, fname)
        try:
            df = pl.read_csv(
                filepath,
                schema_overrides={
                    "pearson_delta_all": pl.Float64,
                    "pearson_delta_top20": pl.Float64,
                    "mse_delta": pl.Float64,
                    "rmse_delta": pl.Float64,
                    "jaccard_top20_de": pl.Float64,
                    "centroid_accuracy": pl.Float64,
                },
            )
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

        # Also read per-perturbation results
        results_fname = fname.replace("_agg_results.csv", "_results.csv")
        results_path = os.path.join(indir, results_fname)
        if os.path.exists(results_path):
            try:
                df_r = pl.read_csv(
                    results_path,
                    schema_overrides={
                        "pearson_delta_all": pl.Float64,
                        "pearson_delta_top20": pl.Float64,
                        "mse_delta": pl.Float64,
                        "rmse_delta": pl.Float64,
                        "jaccard_top20_de": pl.Float64,
                        "centroid_accuracy": pl.Float64,
                    },
                )
                df_r = df_r.with_columns(
                    pl.lit(method).alias("method"),
                    pl.lit(donor).alias("donor"),
                    pl.lit(split_or_wandb).alias("split_or_wandb"),
                    pl.lit(num_train).alias("num_train_cytos"),
                    pl.lit(prefix).alias("prefix"),
                )
                result_dfs.append(df_r)
            except Exception as e:
                print(f"Error reading {results_fname}: {e}")

    if agg_dfs:
        combined_agg = pl.concat(agg_dfs, how="diagonal")
        combined_agg.write_csv(args.outfile)
        print(f"Aggregated results ({combined_agg.shape}): {args.outfile}")
    else:
        print("No aggregated results found.")

    if result_dfs:
        per_pert_outfile = args.outfile.replace("_aggregated.csv", "_per_perturbation.csv")
        combined_results = pl.concat(result_dfs, how="diagonal")
        combined_results.write_csv(per_pert_outfile)
        print(f"Per-perturbation results ({combined_results.shape}): {per_pert_outfile}")
    else:
        print("No per-perturbation results found.")


if __name__ == "__main__":
    main()
