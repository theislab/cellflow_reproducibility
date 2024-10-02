import cpa
import scanpy as sc
import pandas as pd
import numpy as np

import anndata
from sklearn.metrics import r2_score
import anndata as ad

import itertools

adata_train_path = "/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_train_1.h5ad"
adata_test_path = "/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_test_1.h5ad"
adata_ood_path = "/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_ood_1.h5ad"


adata_train = sc.read(adata_train_path)
adata_test = sc.read(adata_test_path)
adata_ood = sc.read(adata_ood_path)


try:
    data_path = 'combo_sciplex_prep_hvg_filtered.h5ad'
    adata = sc.read(data_path)
except:
    import gdown
    gdown.download('https://drive.google.com/uc?export=download&id=1RRV0_qYKGTvD3oCklKfoZQFYqKJy4l6t')
    data_path = 'combo_sciplex_prep_hvg_filtered.h5ad'
    adata = sc.read(data_path)

adata

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

control_cells_gex = adata_ood[adata_ood.obs["condition"]=="control"].X.A
obs_names = list(adata_ood[adata_ood.obs["condition"]=="control"].obs_names)

all_data = []
obs_df = pd.DataFrame(columns=["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"])
indices = []

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
obs_names = list(adata_test[adata_test.obs["condition"]=="control"].obs_names)

all_data = []
obs_df = pd.DataFrame(columns=["condition_ID", "log_dose", "smiles_rdkit", "cell_type", "condition"])
indices = []

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

adata_combined = anndata.concat((adata_train, adata_test_new, adata_ood_new), label="split", keys=["train", "test", "ood"], join="inner")

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

cpa.CPA.setup_anndata(adata_combined,
                      perturbation_key='condition_ID',
                      dosage_key='log_dose',
                      control_group='CHEMBL504',
                      batch_key=None,
                      is_count_data=False,
                      categorical_covariate_keys=['cell_type'],
                      deg_uns_cat_key='rank_genes_groups_cov',
                      max_comb_len=2,
                     )

ae_hparams = {
    "n_latent": 128,
    "recon_loss": "gauss",
    "doser_type": "logsigm",
    "n_hidden_encoder": 512,
    "n_layers_encoder": 3,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 3,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": True,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.1,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 434,
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 30,
    "n_epochs_adv_warmup": 50,
    "n_epochs_mixup_warmup": 3,
    "mixup_alpha": 0.1,
    "adv_steps": 2,
    "n_hidden_adv": 64,
    "n_layers_adv": 2,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "reg_adv": 20.0,
    "pen_adv": 20.0,
    "lr": 0.0003,
    "wd": 4e-07,
    "adv_lr": 0.0003,
    "adv_wd": 4e-07,
    "adv_loss": "cce",
    "doser_lr": 0.0003,
    "doser_wd": 4e-07,
    "do_clip_grad": False,
    "gradient_clip_value": 1.0,
    "step_size_lr": 45,
}


params = list(trainer_params.keys())
params.append("r_squared")
df_res = pd.DataFrame(index=params)

n_epochs_pretrain_ae = [30, 50]
n_epochs_adv_warmup = [50, 100]
n_epochs_mixup_warmup = [3, 10]
mixup_alpha = [0.1, 0.5]
adv_steps = [None, 2]
step_size_lr = [10, 45]

combinations_dicts = [
    {
        'n_epochs_pretrain_ae': ae,
        'n_epochs_adv_warmup': adv_warmup,
        'n_epochs_mixup_warmup': mixup_warmup,
        'mixup_alpha': alpha,
        'adv_steps': adv,
        'step_size_lr': lr
    }
    for ae, adv_warmup, mixup_warmup, alpha, adv, lr in itertools.product(
        n_epochs_pretrain_ae,
        n_epochs_adv_warmup,
        n_epochs_mixup_warmup,
        mixup_alpha,
        adv_steps,
        step_size_lr
    )
]


for i,comb in enumerate(combinations_dicts):

    trainer_params.update(comb)


    model = cpa.CPA(
        adata=adata_combined,
        split_key="split_1ct_MEC",
        train_split="train_train",
        valid_split="train_valid",
        **ae_hparams,
    )


    model.train(
        max_epochs=2000,
        use_gpu=True,
        batch_size=128,
        plan_kwargs=trainer_params,
        early_stopping_patience=10,
        check_val_every_n_epoch=5,
        save_path="/lustre/groups/ml01/workspace/ot_perturbation/models/cpa/combosciplex/1",
    )

    model.predict(adata_combined)

    adata_pred_ood = adata_combined[adata_combined.obs["split"]=="ood"]


    r_squared = {}
    for cond in adata_ood.obs["condition"].cat.categories:
        if cond == "control":
            continue
        true = adata_ood[adata_ood.obs["condition"] == cond].X.toarray()
        pred = adata_pred_ood[adata_pred_ood.obs["condition"] == cond].obsm["CPA_pred"]
        r_squared[cond] = r2_score(true.mean(axis=0), pred.mean(axis=0))

    df = pd.DataFrame(trainer_params, index=[f"params_{i}"]).T
    df.loc["r_squared"] = np.mean(list(r_squared.values()))
    df_res[f"params_{i}"] = df[f"params_{i}"]
    df_res.to_csv("result_hsearch.csv")




