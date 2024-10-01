import jax
jax.config.update("jax_enable_x64", True)

import scvi
import scanpy as sc
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from cfp.external import CFJaxSCVI

split = 5
output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex"
adata_train = sc.read(os.path.join(output_dir, f"adata_train_{split}.h5ad"))
adata_test = sc.read(os.path.join(output_dir, f"adata_test_{split}.h5ad"))
adata_ood = sc.read(os.path.join(output_dir, f"adata_ood_{split}.h5ad")) 


n_latents = [10, 32, 64, 128, 256]
n_hiddens = [512, 1024, 2048]


df_result = pd.DataFrame(columns=['r2_ood'])

CFJaxSCVI.setup_anndata(adata_train)

for n_latent in n_latents:
    for n_hidden in n_hiddens:
        vae = CFJaxSCVI(adata_train, gene_likelihood="normal", n_latent = n_latent, n_hidden = n_hidden)

        vae.train(
            max_epochs=2000,
            batch_size=1024,
            plan_kwargs=dict(
                lr=1e-4,
            ),
            early_stopping=True,
            early_stopping_patience=20,
        )
        config = str(n_latent)+ "_"+str(n_hidden) 
        res = [vae.history[k].iloc[-1].values[0] for k in vae.history.keys()]
        #adata_train.obsm["X_scVI"] = vae.get_latent_representation(adata_train)
        #adata_train.obsm["reconstruction"] = vae.get_reconstructed_expression(adata_train, give_mean="True")
        #try:
        #    r2_train = r2_score(np.array(np.mean(adata_train.X, axis=0))[0,:], np.mean(adata_train.obsm["reconstruction"], axis=0))
        #except:
        #    r2_train = 0.0
        adata_ood.obsm["X_scVI"] = vae.get_latent_representation(adata_ood)
        adata_ood.obsm["reconstruction"] = vae.get_reconstructed_expression(adata_ood, give_mean="True")
        try:
            r2_ood = r2_score(np.mean(adata_ood.obsm["reconstruction"], axis=0), np.array(np.mean(adata_ood.X, axis=0))[0,:])
        except:
            r2_ood = 0.0
        df_result.loc[config] = r2_ood
        df_result.to_csv('new_result_hyper_sciplex_split_5.csv')
        print(res)
