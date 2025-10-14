import sys
import os
import warnings
import yaml
from typing import Literal
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad
import pandas as pd
import cpa
import inspect
from sklearn.metrics import r2_score
import hydra
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import json
import time
import traceback
import random



@hydra.main(config_path="conf", config_name="train")
def run(config):
    split = config.generic_params.split
    adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_train_{split}.h5ad"
    adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_test_{split}.h5ad"
    adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_ood_{split}.h5ad"

    adata_train = sc.read(adata_train_path)
    adata_test = sc.read(adata_test_path)
    adata_ood = sc.read(adata_ood_path)
    

    adata = sc.read("/lustre/groups/ml01/workspace/ot_perturbation/hparam/cpa_combosciplex/combo_sciplex_prep_hvg_filtered.h5ad")
    
    adata.obs["index"] = adata.obs_names.values
    adata.obs["index_adapted"] = adata.obs.apply(lambda x: x["index"][:-2], axis=1)
    adata.obs.set_index("index_adapted", inplace=True)

    adata_ood.obs["cov_drug_dose"] = adata.obs["cov_drug_dose"]
    adata_ood.obs["condition_ID"] = adata.obs["condition_ID"]
    adata_ood.obs["log_dose"] = adata.obs["log_dose"]
    adata_ood.obs["smiles_rdkit"] = adata.obs["smiles_rdkit"]
    adata_ood.obs["cell_type"] = adata.obs["cell_type"]

    adata_test.obs["cov_drug_dose"] = adata.obs["cov_drug_dose"]
    adata_test.obs["condition_ID"] = adata.obs["condition_ID"]
    adata_test.obs["log_dose"] = adata.obs["log_dose"]
    adata_test.obs["smiles_rdkit"] = adata.obs["smiles_rdkit"]
    adata_test.obs["cell_type"] = adata.obs["cell_type"]

    adata_train.obs["cov_drug_dose"] = adata.obs["cov_drug_dose"]
    adata_train.obs["condition_ID"] = adata.obs["condition_ID"]
    adata_train.obs["log_dose"] = adata.obs["log_dose"]
    adata_train.obs["smiles_rdkit"] = adata.obs["smiles_rdkit"]
    adata_train.obs["cell_type"] = adata.obs["cell_type"]
    if config.generic_params.use_counts:
        adata_train.X = adata_train.layers["counts"].copy()
        adata_test.X = adata_test.layers["counts"].copy()
        adata_ood.X = adata_ood.layers["counts"].copy()
    control_cells_gex = adata_ood[adata_ood.obs["condition"]=="control"].X.A
    
    all_data = []
    obs_df = pd.DataFrame(columns=["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"])
    
    for condition in adata_ood.obs["condition"].cat.categories:
        if condition=="control":
            continue
        all_data.append(control_cells_gex)
        filtered_row = pd.Series(adata_ood[adata_ood.obs["condition"] == condition].obs[["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"]].iloc[0].values)
        filtered_row_df = pd.DataFrame([filtered_row.values], columns=["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"])
        duplicated_rows = pd.concat([filtered_row_df] * 500, ignore_index=True)
        obs_df = pd.concat([obs_df, duplicated_rows], ignore_index=True)

    # Stack all data vertically to create a single array
    all_data_array = np.vstack(all_data)


    # Create the Anndata object
    adata_ood_new = ad.AnnData(X=all_data_array, obs=obs_df)
    adata_ood_new.var = adata_ood.var


    control_cells_gex = adata_test[adata_test.obs["condition"]=="control"].X.A
    
    all_data = []
    obs_df = pd.DataFrame(columns=["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"])
    
    for condition in adata_test.obs["condition"].cat.categories:
        if condition=="control":
            continue
        all_data.append(control_cells_gex)
        filtered_row = pd.Series(adata_test[adata_test.obs["condition"] == condition].obs[["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"]].iloc[0].values)
        filtered_row_df = pd.DataFrame([filtered_row.values], columns=["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"])
        duplicated_rows = pd.concat([filtered_row_df] * 500, ignore_index=True)
        obs_df = pd.concat([obs_df, duplicated_rows], ignore_index=True)

    # Stack all data vertically to create a single array
    all_data_array = np.vstack(all_data)


    # Create the Anndata object
    adata_test_new = ad.AnnData(X=all_data_array, obs=obs_df)
    adata_test_new.var = adata_test.var

    adata_combined = ad.concat((adata_train, adata_test_new, adata_ood_new), label="split", keys=["train", "test", "ood"], join="inner")

    frac_valid = adata[adata.obs["split_1ct_MEC"] == "valid"].n_obs / (
        adata[adata.obs["split_1ct_MEC"] == "valid"].n_obs + adata[adata.obs["split_1ct_MEC"] == "train"].n_obs
    )

    def create_split(x):
        if x["split"] != "train":
            return x["split"]
        is_train = np.random.choice(2, p=[frac_valid, 1 - frac_valid])
        if is_train:
            return "train_train"
        return "train_valid"


    adata_combined.obs["split_1ct_MEC"] = adata_combined.obs.apply(create_split, axis=1)
    adata_combined.obs['cov_drug_dose'] = adata.obs['cov_drug_dose']
    adata_combined.uns = adata_train.uns

    recon_loss="nb" if config.generic_params.use_counts else "gauss"


     
    cpa.CPA.setup_anndata(adata_combined,
                      perturbation_key='condition_ID',
                      dosage_key='log_dose',
                      control_group='CHEMBL504',
                      batch_key=None,
                      smiles_key='smiles_rdkit',
                      is_count_data=config.generic_params.use_counts,
                      categorical_covariate_keys=['cell_type'],
                      deg_uns_key='rank_genes_groups_cov_all' if config.generic_params.use_deg_stopping_criterion else None,
                      deg_uns_cat_key='cov_drug_dose',
                      max_comb_len=2,
                     )
    
    ae_hparams = {k:None if v=="to_none" else v for k,v in config.ae_hparams.items()}
    
    model = cpa.CPA(
        adata=adata_combined,
        split_key="split_1ct_MEC",
        train_split="train_train",
        valid_split="train_valid",
        recon_loss=recon_loss,
        **ae_hparams,
    )
    
    trainer_hparams = {k:None if v=="to_none" else v for k,v in config.trainer_hparams.items()}
    model.train(
        max_epochs=2000,
        use_gpu=True,
        batch_size=128,
        plan_kwargs=trainer_hparams,
        early_stopping_patience=10,
        check_val_every_n_epoch=5,
        save_path=f"/lustre/groups/ml01/workspace/ot_perturbation/models/chemcpa/combosciplex_count/{split}",
    )


    model.predict(adata_combined)

    adata_combined.write(
        f"/lustre/groups/ml01/workspace/ot_perturbation/models/chemcpa/combosciplex/adata_with_predictions_{split}.h5ad"
    )
    return 1.0



if __name__ == "__main__":
    import os
    try:
        os.environ["HYDRA_FULL_ERROR"] = "1"
        run()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)