import jax 
jax.config.update("jax_enable_x64", True)
import cellflow
import scanpy as sc
import anndata as ad
import numpy as np
import cellflow.preprocessing as cfpp
import pandas as pd
import sys
import os
import pickle

args = sys.argv
split, model_name, emb_type = int(args[1]), args[2], args[3]


adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{split}.h5ad"
adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{split}.h5ad"
adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{split}.h5ad"
adata_train = sc.read_h5ad(adata_train_path)
adata_test = sc.read_h5ad(adata_test_path)
adata_ood = sc.read_h5ad(adata_ood_path)



embedding_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/cellflow/embeddings_from_tahoe"

def get_embeddings(type: str):
    if type == "closest_cell_line":
        with open(os.path.join(embedding_dir, "sci_cl_cell_line.pkl"), "rb") as f:
            embedding_dict = pickle.load(f)
            
    elif type == "direct_embedding":
        with open(os.path.join(embedding_dir, "sci_emb.pkl"), "rb") as f:
            embedding_dict = pickle.load(f)
    elif type == "concatenated_drug_embeddings":
        with open(os.path.join(embedding_dir, "sci_drug_concatenated_from_tahoe.pkl"), "rb") as f:
            embedding_dict = pickle.load(f)
    else:
        raise ValueError
    return embedding_dict

adata_train.obs["cell_line_drug"] = adata_train.obs["cell_line"].astype("str") + "_" + adata_train.obs["drug"].astype("str")
adata_ood.obs["cell_line_drug"] = adata_ood.obs["cell_line"].astype("str") + "_" + adata_ood.obs["drug"].astype("str")

drug_embeddings = get_embeddings(emb_type)

if emb_type == "direct_embedding":
    adata_train.uns["cell_line_drug_embedding"] = drug_embeddings
    adata_ood.uns["cell_line_drug_embedding"] = drug_embeddings
    perturbation_covariates = {"cell_line_drug": ["cell_line_drug"], "dose": ["dose"]}
    perturbation_covariate_reps = {"cell_line_drug": "cell_line_drug_embedding"}
    sample_covariates = []
    sample_covariate_reps = {}
    split_covariates = ["cell_line"]
    layers_before_pool = {"cell_line_drug": 
                            {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
                        "dose":
                            {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                        }

elif emb_type == "closest_cell_line":
    adata_train.uns["cell_line_drug_embedding"] = drug_embeddings
    adata_ood.uns["cell_line_drug_embedding"] = drug_embeddings
    perturbation_covariates = {"cell_line_drug": ["cell_line_drug"], "dose": ["dose"]}
    perturbation_covariate_reps = {"cell_line_drug": "cell_line_drug_embedding"}
    sample_covariate_reps = {}
    sample_covariates = []
    split_covariates = ["cell_line"]
    layers_before_pool = {"cell_line_drug": 
                            {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
                        "dose":
                            {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                        }

elif emb_type == "concatenated_drug_embeddings":
    adata_train.uns["drug_embedding"] = drug_embeddings
    adata_ood.uns["drug_embedding"] = drug_embeddings
    
    df_cl_emb = pd.read_csv("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex_tahoe/cell_line_embedding_full_ccle_300_scaled.csv")
    with open("/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex_tahoe/id_to_cell_line.pkl", "rb") as f:
        id_to_cell_line = pickle.load(f)
    ccle_embs_sciplex = {}
    for cl in ["A549", "K562", "MCF7"]:
        ccle_embs_sciplex[cl] = df_cl_emb[df_cl_emb["stripped_cell_line_name"]==cl][[str(el) for el in np.arange(300)]].values.squeeze()
    adata_train.uns["cell_line_embedding"] = ccle_embs_sciplex
    adata_ood.uns["cell_line_embedding"] = ccle_embs_sciplex
    perturbation_covariates = {"drug": ["drug"], "dose": ["dose"]}
    perturbation_covariate_reps = {"drug": "drug_embedding"}
    sample_covariates = ["cell_line"]
    sample_covariate_reps = {"cell_line": "cell_line_embedding"}
    split_covariates = ["cell_line"]
    layers_before_pool = {"drug": 
                            {"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.5},
                        "dose":
                            {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                        "cell_line":
                            {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.2},
                        }




    

adata_ood_ctrl = adata_ood[adata_ood.obs["condition"].str.contains("Vehicle")]
adata_test_ctrl = adata_test[adata_test.obs["condition"].str.contains("Vehicle")]
adata_ood_ctrl.obs["control"] = True
adata_test_ctrl.obs["control"] = True
covariate_data_ood = adata_ood[~adata_ood.obs["condition"].str.contains("Vehicle")].obs.drop_duplicates(subset=["condition"])
covariate_data_test = adata_test[~adata_test.obs["condition"].str.contains("Vehicle")].obs.drop_duplicates(subset=["condition"])

cf = cellflow.model.CellFlow.load(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/sciplex_new/{model_name}_CellFlow.pkl")

preds_ood = cf.predict(adata=adata_ood_ctrl, sample_rep="X_pca", condition_id_key="condition", covariate_data=covariate_data_ood)


all_data = []
conditions = []

for condition, array in preds_ood.items():
    all_data.append(array)
    conditions.extend([condition] * array.shape[0])

# Stack all data vertically to create a single array
all_data_array = np.vstack(all_data)

# Create a DataFrame for the .obs attribute
obs_data = pd.DataFrame({
    'condition': conditions
})

# Create the Anndata object
adata_ood_result = ad.AnnData(X=np.empty((len(all_data_array),2001)), obs=obs_data)
adata_ood_result.obsm["X_pca_pred"] = all_data_array


cfpp.reconstruct_pca(query_adata=adata_ood_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")

adata_ood_result.write(f"/lustre/groups/ml01/workspace/ot_perturbation/models/cellflow/sciplex/predictions/adata_ood_with_predictions_{split}_{emb_type}.h5ad")
