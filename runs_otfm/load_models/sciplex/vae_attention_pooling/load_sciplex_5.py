import jax 
jax.config.update("jax_enable_x64", True)
import cfp
import scanpy as sc
import anndata as ad
import numpy as np
import cfp.preprocessing as cfpp
import pandas as pd
from cfp.external import CFJaxSCVI


split = 5
model_name = "young-cherry-451"

adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{split}.h5ad"
adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{split}.h5ad"
adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{split}.h5ad"
adata_train = sc.read_h5ad(adata_train_path)
adata_test = sc.read_h5ad(adata_test_path)
adata_ood = sc.read_h5ad(adata_ood_path)

vae = CFJaxSCVI.load(f"/lustre/groups/ml01/workspace/ot_perturbation/models/vaes/sciplex/32_1024_{split}", adata=adata_train)
adata_train.obsm["X_scVI"] = vae.get_latent_representation(adata_train)
adata_test.obsm["X_scVI"] = vae.get_latent_representation(adata_test)
adata_ood.obsm["X_scVI"] = vae.get_latent_representation(adata_ood)

    

adata_ood_ctrl = adata_ood[adata_ood.obs["condition"].str.contains("Vehicle")]
adata_test_ctrl = adata_test[adata_test.obs["condition"].str.contains("Vehicle")]
adata_ood_ctrl.obs["control"] = True
adata_test_ctrl.obs["control"] = True
covariate_data_ood = adata_ood[~adata_ood.obs["condition"].str.contains("Vehicle")].obs.drop_duplicates(subset=["condition"])
covariate_data_test = adata_test[~adata_test.obs["condition"].str.contains("Vehicle")].obs.drop_duplicates(subset=["condition"])

cf = cfp.model.CellFlow.load(f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/out/{model_name}_CellFlow.pkl")

preds_ood = cf.predict(adata=adata_ood_ctrl, sample_rep="X_scVI", condition_id_key="condition", covariate_data=covariate_data_ood)


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
adata_ood_result.obsm["X_scVI_pred"] = all_data_array
adata_ood_result.layers["pred_reconstruction"] = vae.get_reconstructed_expression(adata_ood_result, use_rep="X_scVI_pred")
adata_ood_result.write(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/sciplex/vae_attention_pooling/adata_ood_with_predictions_{split}.h5ad")



preds_test = cf.predict(adata=adata_test_ctrl, sample_rep="X_scVI", condition_id_key="condition", covariate_data=covariate_data_test)
all_data = []
conditions = []

for condition, array in preds_test.items():
    all_data.append(array)
    conditions.extend([condition] * array.shape[0])

# Stack all data vertically to create a single array
all_data_array = np.vstack(all_data)

# Create a DataFrame for the .obs attribute
obs_data = pd.DataFrame({
    'condition': conditions
})

# Create the Anndata object
adata_test_result = ad.AnnData(X=np.empty((len(all_data_array),2001)), obs=obs_data)
adata_test_result.obsm["X_scVI_pred"] = all_data_array

adata_test_result.layers["pred_reconstruction"] = vae.get_reconstructed_expression(adata_test_result, use_rep="X_scVI_pred")

adata_test_result.write(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/sciplex/vae_attention_pooling/adata_test_with_predictions_{split}.h5ad")
