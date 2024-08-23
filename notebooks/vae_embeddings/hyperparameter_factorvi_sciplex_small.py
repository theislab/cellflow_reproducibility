import jax
jax.config.update("jax_enable_x64", True)

import juniper
import scanpy as sc
import os
import scvi
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, r2_score
import numpy as np
import pkgutil
import importlib



output_dir = "/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex"
adata_train = sc.read(os.path.join(output_dir, "adata_train_biolord_split_30.h5ad"))
adata_test = sc.read(os.path.join(output_dir, "adata_test_biolord_split_30.h5ad"))
adata_ood = sc.read(os.path.join(output_dir, "adata_ood_biolord_split_30.h5ad")) 


layers = [[1024, 1024], [1024, 1024, 1024], [2048, 2048]]
n_latents = [10, 16, 20]
disentangling_weights = [0.0, 0.5, 1.0, 2.0]
kld_weights = [0.1, 1.0] 

df_result = pd.DataFrame(columns=['validation_loss', 'elbo_validation', 'reconstruction_loss_validation', 'kl_local_validation', 'kl_global_validation', 'disentangling_loss_validation', 'train_loss', 'rec_loss', 'kld_loss', 'elbo_train', 'reconstruction_loss_train', 'kl_local_train', 'kl_global_train', 'disentangling_loss_train', 'r2_train', 'r2_ood'])

juniper.latent.model.FactorVI.setup_anndata(adata_train)

for n_latent in n_latents:
    for dis_weight in disentangling_weights:
        for kld_weight in kld_weights:
            for layer in layers:
                vae = juniper.latent.model.FactorVI(adata_train, count_input=False, n_latent = n_latent, encoder_layers= layer, decoder_layers = layer, disentangling_weight=dis_weight, kld_weight=kld_weight)

                vae.train(
                    max_epochs=5000,
                    batch_size=1024,
                    plan_kwargs=dict(
                        lr=1e-4,
                    ),
                    early_stopping=True,
                    early_stopping_patience=20,
                )
                config = str(n_latent)+ "_"+str(dis_weight) + "_" + str(kld_weight) + "_" + str(layer)
                res = [vae.history[k].iloc[-1].values[0] for k in vae.history.keys()]
                adata_train.obsm["X_scVI"] = vae.get_latent_representation(adata_train)
                adata_train.obsm["reconstruction"] = vae.get_reconstructed_expression(adata_train, give_mean="True")
                try:
                    r2_train = r2_score(np.mean(adata_train.obsm["reconstruction"], axis=0), np.array(np.mean(adata_train.X, axis=0))[0,:])
                except:
                    r2_train = 0.0
                adata_ood.obsm["X_scVI"] = vae.get_latent_representation(adata_ood)
                adata_ood.obsm["reconstruction"] = vae.get_reconstructed_expression(adata_ood, give_mean="True")
                try:
                    r2_ood = r2_score(np.mean(adata_ood.obsm["reconstruction"], axis=0), np.array(np.mean(adata_ood.X, axis=0))[0,:])
                except:
                    r2_train = 0.0
                res.extend([r2_train, r2_ood])
                df_result.loc[config] =res
                df_result.to_csv('result_hyper_factorvi_sciplex_small.csv')
                print(res)
