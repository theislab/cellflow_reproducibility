import jax
jax.config.update("jax_enable_x64", True)

import scvi
import scanpy as sc
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from cfp.external import CFJaxSCVI

output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/vae"
adata = sc.read_h5ad("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/zebrafish_processed.h5ad")

n_latents = [32, 64, 128]
n_hiddens = [1024, 2048]


df_result = pd.DataFrame(columns=['r2_ood'])
adata_train = adata[adata.obs["is_control"]].copy()
adata_other = adata[~adata.obs["is_control"]].copy()
sc.pp.subsample(adata_other, fraction=0.01)
CFJaxSCVI.setup_anndata(adata_train)

for n_latent in n_latents:
    for n_hidden in n_hiddens:
        vae = CFJaxSCVI(adata_train, gene_likelihood="normal", n_latent = n_latent, n_hidden = n_hidden)

        vae.train(
            max_epochs=3000,
            batch_size=1024,
            plan_kwargs=dict(
                lr=1e-4,
            ),
            early_stopping=True,
            early_stopping_patience=20,
        )
        config = str(n_latent)+ "_"+str(n_hidden) 
        res = [vae.history[k].iloc[-1].values[0] for k in vae.history.keys()]
        
        adata_other.obsm["X_scVI"] = vae.get_latent_representation(adata_other)
        adata_other.obsm["reconstruction"] = vae.get_reconstructed_expression(adata_other, give_mean="True")
        r2_ood = np.mean([r2_score(np.array(np.mean(adata_other[adata_other.obs["condition"]==cond].X, axis=0))[0,:], np.mean(adata_other[adata_other.obs["condition"]==cond].obsm["reconstruction"], axis=0)) for cond in adata_other.obs["condition"].cat.categories])
        
        df_result.loc[config] = r2_ood
        df_result.to_csv('zebrafish_vae_hyper.csv')
        print(res)
        vae.save(os.path.join("/lustre/groups/ml01/workspace/ot_perturbation/data/zebrafish_new/vae", f"vae_{config}"))
