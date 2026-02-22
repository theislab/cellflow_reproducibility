"""Build a unified paired DataFrame across all models for the new_cytokine experiment.

For each (donor_cytokine, idx_given_cytokine) combination, this script produces one row
with metrics from all models: CellFlow (cf), CellFlow stochastic (cf_stoch),
mean_same_donor (m1), mean_same_cytokine (m2), closest_embedding (cl), and identity (id).
This enables paired statistical tests (e.g., paired t-test) across models.

The key insight is that idx_given_cytokine determines the exact train/test split
(which donors saw the held-out cytokine during training), and models must be compared
on the same split for a valid paired test.

Output: a CSV with columns:
  - donor, cytokine, donor_cytokine, idx_given_cytokine, num_donors_in_train,
    donors_in_train (as sorted comma-separated string)
  - {metric}_cf, {metric}_cf_stoch, {metric}_m1, {metric}_m2, {metric}_cl, {metric}_id for each metric
"""

import os
import re
import pandas as pd
import scanpy as sc


# --- Config ---
DATA_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/new_cytokine"
CF_METRICS = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics/metrics_new_cytokine.csv"
CF_STOCH_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/pbmc_new_cytokine_stochastic_metrics"
CF_STOCH_REG = "0.1"  # regularization value to use for stochastic model
MEAN_DONOR_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/models/additive_model/pbmc_new_cytokine/mean_model_same_donor"
MEAN_CYTO_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/models/additive_model/pbmc_new_cytokine/mean_model_same_cytokine"
CLOSEST_EMB_DIR = "/lustre/groups/ml01/workspace/ot_perturbation/models/additive_model/pbmc_new_cytokine/closest_embedding"
ID_METRICS = "/lustre/groups/ml01/workspace/ot_perturbation/models/identity/pbmc_new_cytokine/metrics_new_cytokine.csv"
OUT_PATH = "/lustre/groups/ml01/workspace/ot_perturbation/data/pbmc/metrics/paired_metrics_new_cytokine.csv"

CYTOKINES_HELD_OUT = [
    "4-1BBL", "ADSF", "APRIL", "BAFF", "C5a", "IFN-beta",
    "IFN-epsilon", "IL-13", "IL-15", "Noggin", "OSM", "OX40L",
]
METRICS = [
    "ood_e_distance",
    "decoded_ood_r_squared",
    "mean_e_distance_per_cell_type",
    "mean_decoded_r_sq_per_cell_type",
    "mean_deg_r_sq_per_cell_type",
]

ALL_DONORS = {
    "Donor1", "Donor2", "Donor3", "Donor4", "Donor5", "Donor6",
    "Donor7", "Donor8", "Donor9", "Donor10", "Donor11", "Donor12",
}

# Stochastic model uses different column names for some metrics
STOCH_METRIC_MAP = {
    "ood_e_distance": "mean_ood_e_distance",
    "decoded_ood_r_squared": "mean_decoded_ood_r_squared",
    "mean_e_distance_per_cell_type": "mean_e_distance_per_cell_type",
    "mean_decoded_r_sq_per_cell_type": "mean_decoded_r_sq_per_cell_type",
    "mean_deg_r_sq_per_cell_type": "mean_deg_r_sq_per_cell_type",
}


def parse_donors(s):
    """Parse donors_in_train string to a sorted tuple."""
    return tuple(sorted(re.findall(r"Donor\d+", str(s))))


def donors_to_str(donors_tuple):
    """Convert donors tuple to a consistent string key."""
    return ",".join(donors_tuple)


# --- Step 1: Build split_info mapping ---
# For each (cytokine_held_out, idx_given_cytokine), get the donors_to_train_data
print("Loading split_info from h5ad files...")
split_mapping = {}  # (cytokine, idx) -> donors_tuple
for cyto in CYTOKINES_HELD_OUT:
    path = os.path.join(DATA_DIR, f"adata_rest_{cyto}.h5ad")
    adata = sc.read_h5ad(path, backed="r")
    split_info = adata.uns["split_info"]
    for idx_str, info in split_info.items():
        donors = tuple(sorted(info["donors_to_train_data"]))
        split_mapping[(cyto, int(idx_str))] = donors
    del adata
print(f"  Built {len(split_mapping)} (cytokine, idx) -> donors mappings")

# Reverse mapping: (cytokine, donors_tuple) -> idx_given_cytokine
reverse_mapping = {}
for (cyto, idx), donors in split_mapping.items():
    key = (cyto, donors)
    if key not in reverse_mapping:
        reverse_mapping[key] = idx
    # If multiple idx map to same (cyto, donors), keep the first (shouldn't happen)
print(f"  Built {len(reverse_mapping)} reverse mappings")

# --- Step 2: Load CellFlow metrics and add idx_given_cytokine ---
print("Loading CellFlow metrics...")
df_cf = pd.read_csv(CF_METRICS)
df_cf["donors_tuple"] = df_cf["donors_in_train"].apply(parse_donors)
df_cf["idx_given_cytokine"] = df_cf.apply(
    lambda row: reverse_mapping.get((row["cytokine"], row["donors_tuple"])), axis=1
)

n_matched = df_cf["idx_given_cytokine"].notna().sum()
print(f"  {n_matched}/{len(df_cf)} rows matched to idx_given_cytokine")

# Drop rows without a match and deduplicate (keep first per unique key)
df_cf = df_cf.dropna(subset=["idx_given_cytokine"])
df_cf["idx_given_cytokine"] = df_cf["idx_given_cytokine"].astype(int)
df_cf = df_cf.drop_duplicates(subset=["donor_cytokine", "idx_given_cytokine"], keep="first")
print(f"  {len(df_cf)} rows after dedup")

# --- Step 2b: Load CellFlow stochastic metrics ---
# Each CSV contains metrics for one wandb run (one held-out cytokine, one split).
# Donors_in_train is inferred from which donors are missing from the index.
print("Loading CellFlow stochastic metrics...")
stoch_files = [
    f for f in os.listdir(CF_STOCH_DIR)
    if f.endswith(f"_{CF_STOCH_REG}.csv") and f != "df_calibration.csv"
]

stoch_rows = []
for f in stoch_files:
    df_s = pd.read_csv(os.path.join(CF_STOCH_DIR, f), index_col=0)
    # Infer cytokine and donors from the index (e.g., "Donor10_OX40L")
    cytokines_in_file = set(idx.split("_", 1)[1] for idx in df_s.index)
    assert len(cytokines_in_file) == 1
    cyto = cytokines_in_file.pop()
    donors_imputed = set(idx.split("_")[0] for idx in df_s.index)
    donors_in_train = tuple(sorted(ALL_DONORS - donors_imputed))

    # Look up idx_given_cytokine
    idx_key = (cyto, donors_in_train)
    idx_given = reverse_mapping.get(idx_key)
    if idx_given is None:
        continue

    for donor_cyto, row_data in df_s.iterrows():
        row = {}
        row["donor_cytokine"] = donor_cyto
        row["idx_given_cytokine"] = idx_given
        for metric, stoch_col in STOCH_METRIC_MAP.items():
            if stoch_col in row_data.index:
                row[metric] = row_data[stoch_col]
        stoch_rows.append(row)

df_cf_stoch = pd.DataFrame(stoch_rows)
df_cf_stoch = df_cf_stoch.drop_duplicates(
    subset=["donor_cytokine", "idx_given_cytokine"], keep="first"
)
print(f"  {len(df_cf_stoch)} rows from {len(stoch_files)} files (reg={CF_STOCH_REG})")

# --- Step 3: Load mean_model_same_donor (m1) from individual files ---
# Metrics are constant per condition (split-independent), so just take one per donor_cytokine
print("Loading mean_same_donor metrics...")
df_m1_combined = pd.read_csv(os.path.join(MEAN_DONOR_DIR, "metrics_new_cytokine.csv"), index_col=0)
# Deduplicate: one row per (donor, cytokine)
df_m1 = df_m1_combined.drop_duplicates(subset=["donor", "cytokine"], keep="first").copy()
df_m1["donor_cytokine"] = df_m1["donor"] + "_" + df_m1["cytokine"]
print(f"  {len(df_m1)} unique conditions")

# --- Step 4: Load mean_model_same_cytokine (m2) from individual files ---
# Metrics depend on idx_given_cytokine, so we need to recover it from filenames
print("Loading mean_same_cytokine metrics from individual files...")
m2_dir = MEAN_CYTO_DIR
m2_files = [f for f in os.listdir(m2_dir) if f.endswith(".csv") and f != "metrics_new_cytokine.csv"]

m2_rows = []
for f in m2_files:
    parts = f.replace(".csv", "").split("_")
    idx_given_cytokine = int(parts[0])
    split_idx = int(parts[-1])
    donor = parts[1]
    cytokine = "_".join(parts[2:-1])

    df_single = pd.read_csv(os.path.join(m2_dir, f), index_col=0)
    row = df_single.iloc[:, 0].to_dict()
    row["idx_given_cytokine"] = idx_given_cytokine
    row["donor"] = donor
    row["cytokine"] = cytokine
    row["donor_cytokine"] = f"{donor}_{cytokine}"
    m2_rows.append(row)

df_m2 = pd.DataFrame(m2_rows)
# Deduplicate (shouldn't be needed but just in case)
df_m2 = df_m2.drop_duplicates(subset=["donor_cytokine", "idx_given_cytokine"], keep="first")
print(f"  {len(df_m2)} rows from {len(m2_files)} files")

# --- Step 5: Load closest embedding (cl) ---
# Metrics are constant per condition (split-independent), so just take one per donor_cytokine
print("Loading closest_embedding metrics...")
df_cl_combined = pd.read_csv(os.path.join(CLOSEST_EMB_DIR, "metrics_new_cytokine.csv"), index_col=0)
df_cl = df_cl_combined.drop_duplicates(subset=["donor", "cytokine"], keep="first").copy()
df_cl["donor_cytokine"] = df_cl["donor"] + "_" + df_cl["cytokine"]
print(f"  {len(df_cl)} unique conditions")

# --- Step 6: Load identity model (id) ---
print("Loading identity metrics...")
df_id = pd.read_csv(ID_METRICS, index_col=0)
df_id["donor_cytokine"] = df_id.index
if "donor" not in df_id.columns:
    df_id["donor"] = df_id["donor_cytokine"].str.split("_").str[0]
    df_id["cytokine"] = df_id["donor_cytokine"].str.split("_", n=1).str[1]
print(f"  {len(df_id)} rows")

# --- Step 6: Merge everything ---
print("Merging...")

# Start with CellFlow as the base (it has the most granular split info)
merge_keys = ["donor_cytokine", "idx_given_cytokine"]

# Add num_donors_in_train and donors_in_train string
df_cf["num_donors_in_train"] = df_cf["num_donors_in_train"].astype(int)
df_cf["donors_in_train_str"] = df_cf["donors_tuple"].apply(donors_to_str)

result = df_cf[["donor", "cytokine", "donor_cytokine", "idx_given_cytokine",
                "num_donors_in_train", "donors_in_train_str"]].copy()

# Add CellFlow metrics
for m in METRICS:
    if m in df_cf.columns:
        result[f"{m}_cf"] = df_cf[m].values

# Add CellFlow stochastic metrics - merge on (donor_cytokine, idx_given_cytokine)
stoch_metrics = df_cf_stoch[
    ["donor_cytokine", "idx_given_cytokine"] + [m for m in METRICS if m in df_cf_stoch.columns]
].copy()
stoch_metrics = stoch_metrics.rename(
    columns={m: f"{m}_cf_stoch" for m in METRICS if m in df_cf_stoch.columns}
)
result = result.merge(stoch_metrics, on=["donor_cytokine", "idx_given_cytokine"], how="left")

# Add mean_same_donor (m1) - merge on donor_cytokine only (split-independent)
m1_metrics = df_m1[["donor_cytokine"] + [m for m in METRICS if m in df_m1.columns]].copy()
m1_metrics = m1_metrics.rename(columns={m: f"{m}_m1" for m in METRICS if m in df_m1.columns})
result = result.merge(m1_metrics, on="donor_cytokine", how="left")

# Add mean_same_cytokine (m2) - merge on (donor_cytokine, idx_given_cytokine)
m2_metrics = df_m2[["donor_cytokine", "idx_given_cytokine"] + [m for m in METRICS if m in df_m2.columns]].copy()
m2_metrics = m2_metrics.rename(columns={m: f"{m}_m2" for m in METRICS if m in df_m2.columns})
result = result.merge(m2_metrics, on=["donor_cytokine", "idx_given_cytokine"], how="left")

# Add closest_embedding (cl) - merge on donor_cytokine only (split-independent)
cl_metrics = df_cl[["donor_cytokine"] + [m for m in METRICS if m in df_cl.columns]].copy()
cl_metrics = cl_metrics.rename(columns={m: f"{m}_cl" for m in METRICS if m in df_cl.columns})
result = result.merge(cl_metrics, on="donor_cytokine", how="left")

# Add identity (id) - merge on donor_cytokine only (split-independent)
id_metrics = df_id[["donor_cytokine"] + [m for m in METRICS if m in df_id.columns]].copy()
id_metrics = id_metrics.rename(columns={m: f"{m}_id" for m in METRICS if m in df_id.columns})
result = result.merge(id_metrics, on="donor_cytokine", how="left")

print(f"\nFinal DataFrame: {result.shape}")
print(f"Columns: {list(result.columns)}")
print(f"\nNon-null counts per model:")
for model in ["cf", "cf_stoch", "m1", "m2", "cl", "id"]:
    col = f"{METRICS[0]}_{model}"
    if col in result.columns:
        print(f"  {model}: {result[col].notna().sum()}/{len(result)}")

print(f"\nRows per num_donors_in_train:")
print(result["num_donors_in_train"].value_counts().sort_index())

# Save
result.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
