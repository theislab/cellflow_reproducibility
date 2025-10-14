import cfp
import scanpy as sc
import anndata as ad
import numpy as np
import sys

import cfp.preprocessing as cfpp

if __name__ == "__main__":

    split = sys.argv[1]
    model_name = sys.argv[2]

    adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_train_{split}.h5ad"
    adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_test_{split}.h5ad"
    adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/combosciplex/adata_ood_{split}.h5ad"
    adata_train = sc.read_h5ad(adata_train_path)
    adata_test = sc.read_h5ad(adata_test_path)
    adata_ood = sc.read_h5ad(adata_ood_path)
        
    def prepare_data(adata_train, adata_test, adata_ood):
        
        adata_tmp =  adata_train[adata_train.obs["Drug1"].drop_duplicates().index]
        ecfp_dict = {drug: adata_tmp[adata_tmp.obs["Drug1"]==drug].obsm["ecfp_drug_1"] for drug in adata_tmp.obs["Drug1"]}

        adata_tmp =  adata_train[adata_train.obs["Drug2"].drop_duplicates().index]
        ecfp_dict.update({drug: adata_tmp[adata_tmp.obs["Drug2"]==drug].obsm["ecfp_drug_2"] for drug in adata_tmp.obs["Drug2"]})

        adata_tmp =  adata_ood[adata_ood.obs["Drug1"].drop_duplicates().index]
        ecfp_dict.update({drug: adata_tmp[adata_tmp.obs["Drug1"]==drug].obsm["ecfp_drug_1"] for drug in adata_tmp.obs["Drug1"]})

        adata_tmp =  adata_ood[adata_ood.obs["Drug2"].drop_duplicates().index]
        ecfp_dict.update({drug: adata_tmp[adata_tmp.obs["Drug2"]==drug].obsm["ecfp_drug_2"] for drug in adata_tmp.obs["Drug2"]})

            
        adata_train.uns['ecfp_rep'] = ecfp_dict
        adata_test.uns['ecfp_rep'] = ecfp_dict
        adata_ood.uns['ecfp_rep'] = ecfp_dict
        return adata_train, adata_test, adata_ood

    adata_train, adata_test, adata_ood = prepare_data(adata_train, adata_test, adata_ood)

    adata_ood_ctrl = adata_ood[adata_ood.obs["condition"]=="control"]
    adata_test_ctrl = adata_test[adata_test.obs["condition"]=="control"]
    adata_ood_ctrl.obs["control"] = True
    adata_test_ctrl.obs["control"] = True
    covariate_data_ood = adata_ood[adata_ood.obs["condition"]!="control"].obs.drop_duplicates(subset=["condition"])
    covariate_data_test = adata_test[adata_test.obs["condition"]!="control"].obs.drop_duplicates(subset=["condition"])

    cf = cfp.model.CellFlow.load(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/combosciplex/{model_name}_CellFlow.pkl")

    preds_ood = cf.predict(adata=adata_ood_ctrl, sample_rep="X_pca", condition_id_key="condition", covariate_data=covariate_data_ood)

    preds_test = cf.predict(adata=adata_test_ctrl, sample_rep="X_pca", condition_id_key="condition", covariate_data=covariate_data_test)

    import pandas as pd
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
    adata_ood_result = ad.AnnData(X=np.empty((len(all_data_array),2000)), obs=obs_data)
    adata_ood_result.obsm["X_pca_pred"] = all_data_array

    cfpp.reconstruct_pca(query_adata=adata_ood_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")

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
    adata_test_result = ad.AnnData(X=np.empty((len(all_data_array),2000)), obs=obs_data)
    adata_test_result.obsm["X_pca_pred"] = all_data_array                       

    cfpp.reconstruct_pca(query_adata=adata_test_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")

    adata_ood_result.write(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/combosciplex/adata_ood_with_predictions_{split}.h5ad")
    adata_test_result.write(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/combosciplex/adata_test_with_predictions_{split}.h5ad")