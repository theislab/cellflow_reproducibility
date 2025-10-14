import pandas as pd
import numpy as np
import scanpy as sc
import os
from cfp.preprocessing import transfer_labels, compute_wknn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import cfp
import anndata as ad



data_dir = "/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/zebrafish/first_t_cns"
output_dir = "/lustre/groups/ml01/workspace/cell_flow_zebrafish/results/single_condition/interpolate_time_cns"

adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")
adata = adata[adata.obs["tissue"] == "Central Nervous System"]
    
model_name = "fresh-field-2"
ood_cond = "control_control_24"

covariate_data = adata[adata.obs["condition"]==ood_cond].obs.drop_duplicates(subset=["condition"])
def duplicate_and_interpolate(df, column, start, end, steps):
    result = pd.DataFrame()
    
    for _, row in df.iterrows():
        new_rows = pd.DataFrame([row] * steps)  # Duplicate the row
        new_rows[column] = np.log(np.linspace(start, end, steps))
        result = pd.concat([result, new_rows], ignore_index=True)
    
    return result

covariate_data_interpolated = duplicate_and_interpolate(covariate_data, 'logtime', 18, 36, 10)
covariate_data_interpolated.loc[:, "condition"] = covariate_data_interpolated.apply(lambda x: "_".join([x.gene_target, str(int(np.round(np.exp(x.logtime))))]), axis=1)

adata_ctrl = adata[(adata.obs["gene_target"]=="control_control") & (adata.obs["timepoint"]==18)]
adata_ctrl.uns = adata.uns.copy()


cf = cfp.model.CellFlow.load(f"/lustre/groups/ml01/workspace/ot_perturbation/models/otfm/zebrafish/first_t_cns/{model_name}_CellFlow.pkl")

preds_ood = cf.predict(adata=adata_ctrl, sample_rep="X_aligned", condition_id_key="condition", covariate_data=covariate_data_interpolated)


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
adata_ood_result.obsm["X_aligned"] = all_data_array

adata_ood_result.write_h5ad(os.path.join(output_dir, f"interpolated_{ood_cond}_no_annotation.h5ad"))

compute_wknn(ref_adata=adata, query_adata=adata_ood_result, n_neighbors=1, ref_rep_key="X_aligned", query_rep_key="X_aligned")
transfer_labels(query_adata=adata_ood_result, ref_adata=adata, label_key="cell_type_broad")
adata_ood_result.obs["cell_type_broad"] = adata_ood_result.obs["cell_type_broad_transfer"]
adata_ood_result.write_h5ad(os.path.join(output_dir, f"interpolated_{ood_cond}_{model_name}_with_annotation.h5ad"))

adata_ood_result.obs.to_csv(os.path.join(output_dir, f"interpolated_{ood_cond}_{model_name}_obs.csv"))

