import scanpy as sc
import numpy as np
import pandas as pd
import os, sys
import anndata as ad
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

directory = "/home/icb/lea.zimmermann/projects/pertot/data/satija_h5ad_counts"
anndata_list = {}
for filename in os.listdir(directory):
    if filename.endswith("_seq.h5ad"):
        filename_parts = filename.split("_")
        anndata_list[filename_parts[0]] = os.path.join(directory,filename)
        
ad.experimental.concat_on_disk(anndata_list, out_file='merged.h5ad', join='outer')