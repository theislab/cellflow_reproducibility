import jax
jax.config.update("jax_enable_x64", True)

import argparse

import scvi
import scanpy as sc
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from cfp.external import CFJaxSCVI


def run_vae(
    output_dir: str,
    filename_adata_train: str,
    filename_adata_val: str,
    filename_adata_test: str,
    n_latents: list[int],
    n_hiddens: list[int],
    gene_likelihood: str,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping: bool,
    early_stopping_patience: int,
    return_results: bool = True,
) -> None | pd.DataFrame:
    
    all_args = locals()

    def evaluate(adata, vae) -> list:
        adata.obsm["X_scVI"] = vae.get_latent_representation(adata)
        adata.obsm["reconstruction"] = vae.get_reconstructed_expression(adata, give_mean="True")
        
        r2_scores = []
        for cond in adata.obs["condition"].cat.categories:
            truth = np.mean(adata[adata.obs["condition"]==cond].X, axis=0)
            reconstruction = np.mean(adata[adata.obs["condition"]==cond].obsm["reconstruction"], axis=0)
            r2_scores.append(r2_score(truth, reconstruction))
        
        return r2_scores
        

    df_result = pd.DataFrame(columns=["n_latent", "n_hidden", "r2_ood"])
    df_result.attrs["all_args"] = all_args

    for n_latent in n_latents:
        for n_hidden in n_hiddens:
    
            # start with clean data to avoid potential leakage across iterations        
            adata_train = sc.read(os.path.join(output_dir, filename_adata_train))
            adata_test = sc.read(os.path.join(output_dir, filename_adata_val))
            adata_ood = sc.read(os.path.join(output_dir, filename_adata_test))
            
            CFJaxSCVI.setup_anndata(adata_train)
            
            vae = CFJaxSCVI(
                adata_train, 
                gene_likelihood=gene_likelihood, 
                n_latent = n_latent, 
                n_hidden = n_hidden
            )

            vae.train(
                max_epochs=max_epochs,
                batch_size=batch_size,
                plan_kwargs=dict(
                    lr=learning_rate,
                ),
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
            )
            
            res = [vae.history[k].iloc[-1].values[0] for k in vae.history.keys()]
            print(res)
            
            r2_scores_test = evaluate(adata_test, vae)
            r2_scores_ood = evaluate(adata_ood, vae)
            row_idx = df_result.shape[0]
            df_result.loc[row_idx, "n_latent"] = n_latent
            df_result.loc[row_idx, "n_hidden"] = n_hidden
            df_result.loc[row_idx, "r2_test"] = np.mean(r2_scores_test)
            df_result.loc[row_idx, "r2_ood"] = np.mean(r2_scores_ood)
            df_result.to_csv(os.path.join(output_dir, f"{filename_adata_train}.result_hyper.csv"))
            
    
    if return_results:
        return df_result


def str2bool(arg: str | bool) -> bool:
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, str):
        if arg.lower() == "true":
            return True
        else:
            return False
    raise TypeError(f"arg should have type str or bool but has type {type(arg)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # "/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex"
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--filename_adata_train", type=str)
    parser.add_argument("--filename_adata_val", type=str)
    parser.add_argument("--filename_adata_test", type=str)
    parser.add_argument("--n_latents", nargs="+", default=[10, 32, 64, 128, 256])
    parser.add_argument("--n_hiddens", nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--gene_likelihood", type=str, default="normal")
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--early_stopping", type=str2bool, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    args = parser.parse_args()
    
    if not isinstance(args.n_latents, list):
        args.n_latents = [args.n_latents]
    if not isinstance(args.n_hiddens, list):
        args.n_hiddens = [args.n_hiddens]
    args.n_latents = [int(dim) for dim in args.n_latents]
    args.n_hiddens = [int(dim) for dim in args.n_hiddens]
    
    print(args)
    
    df_result = run_vae(
        output_dir=args.output_dir,
        filename_adata_train=args.filename_adata_train,
        filename_adata_val=args.filename_adata_val,
        filename_adata_test=args.filename_adata_test,
        n_latents=args.n_latents,
        n_hiddens=args.n_hiddens,
        gene_likelihood=args.gene_likelihood,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )



