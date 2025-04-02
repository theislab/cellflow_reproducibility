source('~/scripts/R/master.R')
source('~/scripts/R/colors.R')

library(ggbeeswarm)
library(yaml)
library(tidygraph)
library(ggraph)

select <- dplyr::select

setwd('~/projects/cellflow/')

#### Colors ####
morph_pathways = c(
    "SAG"="SHH",
    "SHH"="SHH",
    "PM"="SHH",
    "CycA"="SHH",
    "FGF2"="FGF",
    "FGF4"="FGF",
    "FGF8"="FGF",
    "FGF17"="FGF",
    "FGF19"="FGF",
    "BMP4"="TGFb",
    "BMP7"="TGFb",
    "LDN"="TGFb",
    "Activin"="TGFb",
    "CHIR"="WNT",
    "XAV"="WNT",
    "IWP2"="WNT",
    "Rspondin2"="WNT",
    "Rspondin3"="WNT",
    "RA"="RA",
    "SR11237"="RA",
    "DAPT"="Notch",
    "EGF"="EGF",
    "Insulin"="Insulin"
)

morph_pathways_mode = c(
    "SAG"= "SHH_1",
    "SHH"= "SHH_1",
    "PM"= "SHH_1",
    "CycA"= "SHH_-1",
    "FGF2"= "FGF_1",
    "FGF4"= "FGF_1",
    "FGF8"= "FGF_1",
    "FGF17"= "FGF_1",
    "FGF19"= "FGF_1",
    "BMP4"= "TGFb_1",
    "BMP7"= "TGFb_1",
    "LDN"= "TGFb_-1",
    "Activin"= "TGFb_1",
    "CHIR"= "WNT_1",
    "XAV"= "WNT_-1",
    "IWP2"= "WNT_-1",
    "Rspondin2"= "WNT_1",
    "Rspondin3"= "WNT_1",
    "RA"= "RA_1",
    "SR11237"= "RA_1",
    "DAPT"= "Notch_-1",
    "EGF"= "EGF_1",
    "Insulin"= "Insulin_1"
)


pathway_colors <- c(
    "FGF"="#E54106",
    "EGF"="#D2B2D6",
    "TGFb"="#FE9900",
    "SHH"="#AD649B",
    "WNT"="#60A8E1",
    "RA"="#00C681",
    "Notch"="#E8D642",
    "Insulin"="#A7D8FF"
)

region_colors <- c(
    "Cortex"="#F06292",
    "Telencephalon"="#BA68C8",
    "Forebrain"="#BA68C8",
    "Hippocampus"="#FFCDD2",
    "Subcortex"="#B39DDB",
    "Striatum"="#AB47BC",
    "Hypothalamus"="#673AB7",
    "Thalamus"="#0288D1",
    "Diencephalon"="#7986CB",
    "Midbrain"="#00BCD4", 
    "Midbrain ventral"="#4DB6AC", 
    "Midbrain dorsal"="#00897B", 
    "Hindbrain"="#A5D6A7", 
    "Cerebellum"="#558B2F",
    "Pons"="#8BC34A",
    "Medulla"="#DCE775",
    "Spinal cord"="#FFC01E", 
    "Other"="grey",
    "ENS"="#a1887f", 
    "SYM"="#D2927D" 
)
type_colors <- c("GLUT"="#EB984E", "CHO"="#F4D03F", "NOR"="#AF7AC5")
class_colors <- c(
    "Neuron"="#4682E2",
    "Neurolast"="#6500CC",
    "Neuronal IPC"="#B80CCB",
    "Radial glia"="#00BA51",
    "Gliolast"="#318068",
    "Oligo"="#96BF00",
    "Firolast"="#CE7F48",
    "Neural crest"="#F5C599",
    "Placodes"="#755000",
    "Immune"="#EDC300",
    "Vascular"="#F20000",
    "Erythrocyte"="#FF477E"
)

dataset_colors <- c("nadya"="#FFD274", "neal"="#898BFF", "fatima"="#BA377E", "observed"="#566573")

model_colors <- c(
    "mean" = "#BDBDBD",
    "barycenter" = "#BDBDBD",
    "union" = "#8f97a8",
    "background" = "#8f97a8",
    "train_dataset" = "#8f97a8",
    "individual" = "#566573",
    "single" = "#566573",
    "closest" = "#566573",
    "cpa" = "#26B0B6",
    "biolord" = "#BD6DED",
    "cellflow" = "#B12F8C"
)

model_order <- c(
    "closest"="Closest train\ncondition",
    "single"="Closest single\nmorphogen",
    "cellflow"="CellFlow",
    "biolord"="biolord",
    "cpa"="CPA",
    "union"="Union",
    "barycenter"="Barycenter",
    "train_dataset"="Train dataset"
)

timing_colors <- c(
    "early"="#82C5FF",
    "mid"="#3F84BF",
    "late"="#105795",
    "early-mid"="#6C99C0",
    "mid-late"="#005AA8",
    "early-late"="#00172D",
    "different"="white"
)

dataset_names <- c(
    "Azbukina et al."="nadya",
    "Amin et al."="neal",
    "Sanchis-Calleja et al."="fatima"
)

PLOT_DIR = "/home/fleckj/projects/cellflow/plots/paper/figure_6/"
DATA_DIR = "/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/cellflow_0a37dcb9/v1+2/"


#### Data ####
adata <- anndata::read_h5ad("/home/fleckj/projects/cellflow/results/organoid_annots/organoids_combined_full_v6_annot.h5ad")
ref_adata <- anndata::read_h5ad("/projects/site/pred/organoid-atlas/data/public_datasets/scg/human_brain/BraunLinnarsson2022/braun_2022_fetal_brain_v3.1umap_common_hv2k_wknn.h5ad")

ref_cluster_meta <- ref_adata$obs %>% 
    as_tibble() %>% 
    group_by(Clusters) %>%
    summarize(
        Region=mode(Region),
        Subregion=mode(Subregion),
        CellClass=mode(CellClass),
        cluster=first(Clusters)
    )

condition_meta <- adata$obs %>% 
    as_tibble() %>% 
    dplyr::select(condition, mol_comb, dataset) %>% 
    distinct() %>% 
    mutate(comb_length = str_split(mol_comb, "\\+"))
condition_meta$comb_length <- map_int(condition_meta$comb_length, length)

condition_meta %>% View

meta <- adata$obs %>% 
    as_tibble(rownames="cell") 

condition_meta_annot <- read_tsv(str_c(DATA_DIR, "condition_meta_annot.tsv"))

pred_condition_length <- condition_meta_annot %>% 
    select(condition, n_mols) %>% 
    distinct()

morph_usage_mat <- condition_meta_annot %>% 
    select(condition, contains("_conc")) %>% 
    distinct() %>% 
    {colnames(.) <- str_remove(colnames(.), "_conc"); .} %>%
    column_to_rownames("condition") %>%
    as.matrix() %>% Matrix(sparse=TRUE)
morph_usage_mat[is.na(morph_usage_mat)] <- 0
morph_usage_mat <- (morph_usage_mat>0) * 1

morph_usage_mat %>% colnames()

morph_pathways_use <- morph_pathways_mode[colnames(morph_usage_mat)]
pw_usage_mat <- Pando::aggregate_matrix(t(morph_usage_mat), as.character(morph_pathways_use)) %>% 
    {(.>0)*1}

pw_usage_df <- t(pw_usage_mat) %>% 
    as_tibble(rownames="condition") %>% 
    pivot_longer(!condition, names_to="pathway_mode", values_to="use") %>% 
    mutate(
        pathway=str_replace(pathway_mode, "_.*", ""),
        mode=str_replace(pathway_mode, ".*_", "")
    ) %>% 
    filter(use>0) 

morph_pw_df <- morph_pathways %>% 
    enframe(name="mol_comb", value="pathway") 

cond_meta_pw <- condition_meta_annot %>% 
    separate_rows(mol_comb, sep="\\+") %>%
    inner_join(morph_pw_df)  %>% 
    select(condition, mol_comb, pathway) %>% 
    mutate(pathway=factor(pathway, levels=names(pathway_colors))) %>%
    arrange(pathway) %>% 
    mutate(mol_comb=factor(mol_comb, levels=rev(unique(.$mol_comb))))


#### Plot ####
condition_meta_subregion <- read_tsv(str_c(DATA_DIR, "organoid_cond_preds_subregion_transfer.tsv"))
condition_meta_region <- read_tsv(str_c(DATA_DIR, "organoid_cond_preds_region_transfer.tsv"))
condition_meta_cluster <- read_tsv(str_c(DATA_DIR, "organoid_cond_preds_cluster_transfer.tsv"))

3*(condition_meta_subregion$condition %>% unique() %>% length())

condition_meta_region$Region_transfer %>% unique()
condition_meta_subregion$Subregion_transfer %>% unique()

condition_annot_meta <- condition_meta_cluster %>% 
    select(-1) %>% 
    rename(Clusters=Clusters_transfer) %>%
    inner_join(ref_cluster_meta)

condition_meta_subregion$Subregion_transfer %>% unique()
condition_meta_region$Region_transfer %>% unique()

condition_meta_region_split <- condition_meta_subregion %>% 
    select(-1) %>% 
    filter(dataset!="observed", condition%in%condition_meta_annot$condition) %>%
    mutate(
        region=case_when(
            str_detect(Subregion_transfer, "Midbrain") ~ "Midbrain",
            T ~ Subregion_transfer
        )
    ) %>% 
    mutate(ds_cond = str_c(dataset, "_", condition)) %>% 
    group_by(dataset) %>% 
    group_split()

condition_meta_region_mat <- condition_meta_region_split %>% 
    map(~{
        .x %>% 
        ungroup() %>%
        select(condition, region, n_cells) %>% 
        group_by(condition, region) %>%
        summarize(n_cells=sum(n_cells)) %>%
        pivot_wider(names_from = region, values_from = n_cells, values_fill=0) %>% 
        column_to_rownames("condition") %>%
        as.matrix() %>% Matrix(sparse=TRUE)
    }) 

cond_orders <- condition_meta_region_mat %>% 
    map(~{
        .x %>% dist() %>% hclust(method="ward.D2") %>% {.$labels[.$order]}
    })

plots <- map(seq_along(condition_meta_region_split), ~{
    plot_df <- condition_meta_region_split[[.x]] %>% 
        group_by(condition, region) %>%
        summarize(n_cells=sum(n_cells)) %>%
        mutate(region=factor(region, levels=names(region_colors))) %>% 
        arrange(region) %>% 
        mutate(condition=factor(condition, levels=cond_orders[[3]]))

    p <- ggplot(plot_df, aes(n_cells, condition, fill=region)) +
        geom_bar(stat="identity", position="fill") +
        scale_fill_manual(values=region_colors) 
    ggsave(p, filename = str_c(PLOT_DIR, "/predictions/bar_region_composition_split_sorted_", .x, ".png"), bg="white", width = 10, height = 40)
    
    return(p)
})

p <- wrap_plots(plots, ncol=1) & 
    theme_dr() +
    no_legend() +
    no_y_text() +
    no_x_text() 
ggsave(p, filename = str_c(PLOT_DIR, "/predictions/bar_region_composition_split_sorted.png"), bg="white", width = 10, height = 20)


plot_df$n_cells %>% hist()

plot_df <- condition_annot_meta %>% 
    group_by(dataset, condition, CellClass) %>%
    summarize(n_cells=sum(n_cells)) %>%
    mutate(class=factor(CellClass, levels=names(class_colors))) %>% 
    arrange(class) %>% 
    mutate(condition=factor(condition, levels=cond_orders[[3]]))

plot_df %>% filter(is.na(class)) %>% pull(CellClass) %>% unique()

p <- ggplot(plot_df, aes(condition, n_cells, fill=class)) +
    geom_bar(stat="identity", position="fill") +
    facet_grid(dataset~.) +
    scale_fill_manual(values=class_colors) +
    theme_dr() +
    no_legend() +
    no_y_text() +
    no_x_text() 
ggsave(p, filename = str_c(PLOT_DIR, "/predictions/bar_class_composition_split_sorted.png"), bg="white", width = 40, height = 12)


dendro_graph <- condition_meta_region_mat[[3]] %>% dist() %>% hclust() %>% as_tbl_graph() 
ggraph(dendro_graph, layout = 'dendrogram') +
    geom_edge_elbow(linewidth=1) 
ggsave(filename = str_c(PLOT_DIR, "/predictions/dendrogram_elbow_sorted.pdf"), bg="white", width = 40, height = 12)

ggraph(dendro_graph, layout = 'dendrogram') +
    geom_edge_diagonal() 
ggsave(filename = str_c(PLOT_DIR, "/predictions/dendrogram_diag_sorted.png"), bg="white", width = 40, height = 12)


plot_df <- cond_meta_pw %>% 
    mutate(condition=factor(condition, levels=cond_orders[[3]])) 

p <- ggplot(plot_df, aes(condition, mol_comb, fill=pathway)) +
    geom_tile() +
    no_x_text() +
    scale_fill_manual(values=pathway_colors, na.value="white") +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    ) +
    theme_dr() +
    no_legend() +
    no_y_text() +
    no_x_text() 
p
ggsave(p, filename = str_c(PLOT_DIR, "/predictions/tile_pathway_composition_sorted.png"), bg="white", width = 30, height = 8)


plot_df <- condition_annot_meta %>% 
    filter(dataset!="observed") %>%
    select(condition) %>% 
    distinct() %>%
    mutate(
        time_comb=str_extract_all(condition, "early-late|mid-late|early-mid|early|mid|late"),
        condition=factor(condition, levels=cond_orders[[3]])
    )
plot_df$time_comb <- plot_df$time_comb  %>% map_chr(~{str_c(., collapse="+")}) 
plot_df$time_comb_simple=case_when(
    plot_df$time_comb %in% c("early-late+early-late+early-late+early-late", "early-late+early-late+early-late", "early-late+early-late", "early-late") ~ "early-late",
    plot_df$time_comb %in% c("mid-late+mid-late+mid-late+mid-late", "mid-late+mid-late+mid-late", "mid-late+mid-late", "mid-late") ~ "mid-late",
    plot_df$time_comb %in% c("early-mid+early-mid+early-mid+early-mid", "early-mid+early-mid+early-mid", "early-mid+early-mid", "early-mid") ~ "early-mid",
    plot_df$time_comb %in% c("early+early+early+early", "early+early+early", "early+early", "early") ~ "early",
    plot_df$time_comb %in% c("mid+mid+mid+mid", "mid+mid+mid", "mid+mid", "mid") ~ "mid",
    plot_df$time_comb %in% c("late+late+late+late", "late+late+late", "late+late", "late") ~ "late",
    T ~ "different"
)
plot_df$time_comb_simple %>% table()

plot_df %>% View()

p <- ggplot(plot_df, aes(condition, fill=time_comb_simple)) +
    geom_bar(position="fill") +
    scale_fill_manual(values=timing_colors) +
    theme_dr() +
    no_legend() +
    no_y_text() +
    no_x_text()
p
ggsave(p, filename = str_c(PLOT_DIR, "/predictions/bar_timing_composition_sorted.png"), bg="white", width = 40, height = 3)


#### Linear model to analyze subregion composition ####
library(glmnetUtils)

cond_model_meta <- condition_meta_annot %>% 
    ungroup() %>%
    mutate(ds_cond=str_c(dataset, "_", condition)) %>%
    select(ds_cond, FGF2_conc, FGF8_conc, FGF19_conc, FGF17_conc, BMP4_conc, BMP7_conc, RA_conc, SAG_conc, SHH_conc, CHIR_conc, LDN_conc, XAV_conc, CycA_conc, Insulin_conc, dataset) %>%
    distinct() %>%
    mutate(
        FGF2_conc = as.numeric(!is.na(FGF2_conc)),
        FGF8_conc = as.numeric(!is.na(FGF8_conc)),
        FGF19_conc = as.numeric(!is.na(FGF19_conc)),
        FGF17_conc = as.numeric(!is.na(FGF17_conc)),
        RA_conc = as.numeric(!is.na(RA_conc)),
        SAG_conc = as.numeric(!is.na(SAG_conc)),
        SHH_conc = as.numeric(!is.na(SHH_conc)),
        BMP4_conc = as.numeric(!is.na(BMP4_conc)),
        BMP7_conc = as.numeric(!is.na(BMP7_conc)),
        CHIR_conc = as.numeric(!is.na(CHIR_conc)),
        LDN_conc = as.numeric(!is.na(LDN_conc)),
        XAV_conc = as.numeric(!is.na(XAV_conc)),
        CycA_conc = as.numeric(!is.na(CycA_conc)),
        Insulin_conc = as.numeric(!is.na(Insulin_conc))
    )

condition_region_meta <- condition_meta_subregion %>%
    mutate(region=Subregion_transfer) %>% 
    mutate(
        region=case_when(
            region %in% c("Midbrain dorsal", "Midbrain ventral") ~ "Midbrain",
            TRUE ~ region
        )
    ) %>% 
    filter(!region%in%c("Hindbrain")) %>%
    inner_join(pred_condition_length) %>%
    mutate(ds_cond = str_c(dataset, "_", condition)) 

condition_comp_mat <- condition_region_meta %>% 
    filter(dataset!="observed", n_cells>0) %>%
    filter(n_mols<=2) %>%
    mutate(ds_cond=str_c(dataset, "_", condition)) %>%
    ungroup() %>%
    select(ds_cond, region, n_cells) %>% 
    pivot_wider(names_from=region, values_from=n_cells, values_fill=0) %>% 
    column_to_rownames("ds_cond") %>% 
    as.matrix() %>% Matrix(sparse=T)

condition_comp_mat <- condition_comp_mat / rowSums(condition_comp_mat)

rownames(condition_comp_mat) %>% length()
rownames(condition_comp_mat) %>% unique() %>% length()

model_features <- condition_comp_mat
colnames(model_features) <- str_replace_all(colnames(model_features), " ", "_")

model_all <- map_dfr(1:ncol(model_features), function(i){

    y <- model_features[,i,drop=FALSE]
    x <- column_to_rownames(cond_model_meta, 'ds_cond')
    mol_names <- colnames(x)[str_detect(colnames(x), "_conc")] %>% 
        str_remove("_conc") %>% 
        unique()
    colnames(x)[str_detect(colnames(x), "_conc")] <- mol_names

    cond_intersect <- intersect(rownames(x), rownames(y))

    mol_int <- combn(mol_names, 2)  %>% 
        t() %>% apply(1, function(x) str_c(x, collapse=':'))
    formula_str <- reformulate(
        str_c(
            # "dataset + ",
            paste(mol_names, collapse=' + '),
            " + ",
            paste(mol_int, collapse=' + ')
        ),
        response = colnames(model_features)[i],
        intercept = FALSE
    )

    model_mat <- as.data.frame(cbind(y[cond_intersect, ], x[cond_intersect, ]))
    colnames(model_mat)[1] <- colnames(model_features)[i]

    mframe <- model.frame(formula=formula_str, model_mat)

    model_fit <- cv.glmnet(formula=formula_str, data=mframe, alpha=1)
    coefs <- model_fit$glmnet.fit$beta[,model_fit$lambda==model_fit$lambda.min]
    coefs %>%
        enframe('term', 'coef') %>%
        mutate(region_coarse=colnames(model_features)[i]) %>%
        return()
})

model_all_mat <- model_all %>% 
    mutate(term=str_replace(term, ":", "\\+")) %>%
    pivot_wider(names_from = term, values_from = coef) %>%
    column_to_rownames("region_coarse") %>%
    as.matrix()

coef_order <- model_all_mat %>% t() %>% dist() %>% hclust(method="ward.D2") %>% {.$labels[.$order]}
coef_dendro <- model_all_mat %>% t() %>% dist() %>% hclust(method="ward.D2")

plot_df <- model_all %>% 
    mutate(term=str_replace(term, ":", "\\+")) %>%
    filter(region_coarse%in%names(region_colors)) %>% 
    mutate(
        term=factor(term, levels=coef_order), region_coarse=factor(region_coarse, levels=rev(names(region_colors))),
        single=factor(ifelse(str_detect(term, "\\+"), "Combination", "Single"), levels=c("Single", "Combination")),
        coef_clip=pmin(pmax(coef, -0.1), 0.1)
        # coef_clip=coef
    ) 

colgrad <- bigrad(pals::brewer.rdbu, bias=2)
ggplot(plot_df, aes(term, region_coarse, fill=coef_clip)) +
    geom_tile() +
    geom_vline(xintercept=0) +
    facet_grid(~single, scales="free", space="free") +
    scale_fill_gradientn(colors=rev(colgrad), limits=c(-0.1, 0.1)) +
    rotate_x_text(90) 


ggplot(plot_df, aes(term, region_coarse, fill=coef_clip)) +
    geom_tile() +
    geom_vline(xintercept=0) +
    facet_grid(~single, scales="free", space="free") +
    scale_fill_gradientn(colors=rev(colgrad), limits=c(-0.1, 0.1)) +
    article_text() +
    rotate_x_text(90) +
    no_legend() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color="black", fill=NA, linewidth=0.5),
        axis.line = element_blank(),
        strip.text = element_text(size=6),
        panel.spacing = unit(0.1, "lines")
    ) +
    labs(fill="Coefficient", x="Morphogen pathway modulators", y="Region")
ggsave(filename = str_c(PLOT_DIR, "predictions/lm_subregion_composition_coefs_heatmap.pdf"), bg="white", width=18, height=4.5, dpi=300, units="cm")




#### Distance to reference vs distance to observed ####
ref_cluster_dist <- read_tsv("/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/cellflow_0a37dcb9/v1+2/organoid_cond_preds_ref_cluster_dists.tsv")
ref_cluster_counts <- read_tsv("/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/cellflow_0a37dcb9/v1+2/organoid_cond_preds_cluster_transfer.tsv") %>% select(-1) 
obs_condition_dist <- read_tsv("/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/cellflow_0a37dcb9/v1+2/organoid_cond_preds_obs_condition_dists.tsv")
obs_obs_dist <- read_tsv("/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/cellflow_0a37dcb9/v1+2/organoid_obs_obs_dists.tsv")

ref_cluster_counts_all <- ref_cluster_counts %>% 
    rename(cluster=Clusters_transfer)

ref_cluster_counts_use <- ref_cluster_counts %>% 
    filter(dataset!="observed", n_cells>30) %>% 
    rename(cluster=Clusters_transfer)

ref_cluster_counts_obs <- ref_cluster_counts %>% 
    filter(dataset=="observed") %>% 
    rename(cluster=Clusters_transfer)

closest_obs <- ref_cluster_dist %>% 
    filter(dataset=="observed") %>% 
    inner_join(ref_cluster_counts_obs) %>%
    group_by(condition) %>%
    mutate(frac_cells=n_cells/sum(n_cells)) %>% 
    filter(frac_cells>0.02) %>% 
    group_by(cluster) %>% 
    filter(edist==min(edist)) %>%
    rename(closest_obs_edist=edist, obs_condition=condition, obs_n_cells=n_cells) %>% 
    select(-dataset)

pred_min_logfc <- ref_cluster_dist %>% 
    filter(dataset!="observed") %>% 
    inner_join(ref_cluster_counts_use) %>%
    filter(n_cells>60) %>% 
    inner_join(closest_obs) %>% 
    group_by(condition, dataset) %>% 
    mutate(min_edist_logfc=log2(edist/closest_obs_edist)) %>% 
    summarize(min_edist_logfc=min(min_edist_logfc))

pred_closest_ref <- ref_cluster_dist %>% 
    filter(dataset!="observed") %>% 
    inner_join(closest_obs)  %>% 
    inner_join(ref_cluster_counts_use) %>%
    mutate(edist_logfc=log2(edist/closest_obs_edist)) %>% 
    group_by(condition, dataset) %>% 
    filter(edist==min(edist)) %>% 
    rename(closest_ref_edist=edist) 

pred_closest_obs <- obs_condition_dist %>% 
    group_by(dataset, condition) %>% 
    summarize(min_obs_edist=min(edist))

obs_closest_obs <- obs_obs_dist %>% 
    filter(condition!=obs_condition) %>% 
    group_by(condition) %>% 
    summarize(
        min_obs_edist=min(edist),
        closest_cond=obs_condition[which.min(edist)]
    )

pred_mean_ref_dist <- ref_cluster_dist %>% 
    filter(dataset!="observed") %>% 
    inner_join(closest_obs)  %>% 
    inner_join(ref_cluster_counts_use) %>%
    group_by(condition, dataset) %>% 
    mutate(
        edist_weighted=edist*(n_cells/mean(n_cells))
    ) %>% 
    summarize(
        weighted_ref_edist=mean(edist_weighted),
    ) %>% 
    inner_join(pred_min_logfc)



#### Scatter plot OOD vs realistic ####
plot_df <- pred_mean_ref_dist %>% 
    inner_join(pred_closest_obs) %>% 
    ungroup() %>% 
    mutate(gates=case_when(
        min_obs_edist > 1 & weighted_ref_edist < 2 ~ "OOD & close to reference",
        min_obs_edist > 1 ~ "OOD",
        weighted_ref_edist < 2 ~ "Close to reference",
        T ~ "Far"
    )) %>% 
    mutate(ds_cond=str_c(dataset, "_", condition)) 

gradient <- rev(pals::ocean.curl(10))
logfc_colors <- colorRampPalette(c(gradient[1:3], "lightgrey", gradient[8:10]))(100)

density_theme <- function(){
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.line = element_blank(),
        axis.title = element_blank(),
    )
}

p <- ggplot(arrange(plot_df, abs(min_edist_logfc)), aes(weighted_ref_edist, min_obs_edist)) +
    geom_hex(bins=30) +
    geom_hline(yintercept=1, color="black", linetype="dashed", size=0.25) +
    geom_vline(xintercept=2, color="black", linetype="dashed", size=0.25) +
    scale_fill_gradientn(colors=pals::brewer.rdpu(100), trans="log10") +
    scale_x_continuous(breaks=seq(0,10,2)) +
    scale_y_continuous(breaks=seq(0,10,1)) +
    no_legend() +
    article_text() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    ) +
    labs(x="Weighted E-distance\n(mean over clusters)", y="E-distance\n(closest measured condition)")
p
ggsave(p, filename=str_c(PLOT_DIR, "ref_cluster_dist/edist_closest_obs_vs_mean_weighted_ref_hex.pdf"), bg="white", width=5, height=4.3, dpi=300, units="cm")

cond_plot <- c(
    # "neal_XAV_3.7_late_FGF8_5.4_late_BMP4_5.59_late"
    # "neal_SAG_6.0_mid_FGF2_5.4_mid_FGF8_5.4_early-late"
    "nadya_SAG_6.0_early_BMP7_4.85_early_CHIR_3.48_late"
)
ggplot(arrange(plot_df, abs(min_edist_logfc)), aes(weighted_ref_edist, min_obs_edist)) +
    geom_hex(bins=30) +
    geom_point(data=filter(plot_df, ds_cond%in%cond_plot), aes(weighted_ref_edist, min_obs_edist), color="black", size=4) +
    geom_hline(yintercept=1, color="black", linetype="dashed", size=0.25) +
    geom_vline(xintercept=2, color="black", linetype="dashed", size=0.25) +
    scale_fill_gradientn(colors=pals::brewer.rdpu(100), trans="log10") +
    scale_x_continuous(breaks=seq(0,10,2)) +
    scale_y_continuous(breaks=seq(0,10,1)) +
    no_legend() +
    article_text() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    ) +
    labs(x="Weighted E-distance\n(mean over clusters)", y="E-distance\n(closest measured condition)")


p <- ggplot(arrange(plot_df, abs(min_edist_logfc)), aes(weighted_ref_edist, min_obs_edist, color=min_edist_logfc)) +
    geom_point(size=0.005, shape=16) +
    geom_hline(yintercept=1, color="black", linetype="dashed", size=0.25) +
    geom_vline(xintercept=2, color="black", linetype="dashed", size=0.25) +
    scale_color_gradientn(colors=logfc_colors, limits=c(-3,3)) +
    scale_x_continuous(breaks=seq(0,10,2)) +
    scale_y_continuous(breaks=seq(0,10,1)) +
    # coord_equal() +
    article_text() +
    no_legend() +
    no_label() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    ) +
    labs(x="Weighted E-distance\n(mean over clusters)", y="E-distance\n(closest measured condition)", fill="Min e-distance LogFC\nvs closest train condition")
p
# ggsave(p, filename=str_c(PLOT_DIR, "ref_cluster_dist/edist_closest_obs_vs_mean_weighted_ref.pdf"), bg="white", width=6, height=4.5, dpi=300, units="cm")

px <- ggplot(plot_df, aes(weighted_ref_edist)) +
    geom_density(fill="darkgrey", color=NA) +
    density_theme() +
    coord_flip() +
    scale_y_reverse()

py <- ggplot(plot_df, aes(min_obs_edist)) +
    geom_density(fill="darkgrey", color=NA) +
    density_theme() +
    scale_y_reverse()

design <- "
211111
211111
211111
211111
#33333
"

pall <- p + px + py + plot_layout(design = design)
ggsave(pall, filename=str_c(PLOT_DIR, "ref_cluster_dist/edist_closest_obs_vs_mean_weighted_ref_density.pdf"), bg="white", width=7.5, height=5, dpi=300, units="cm")



#### Heatmap with selected conditions ####
condition_meta_subregion <- read_tsv(str_c(DATA_DIR, "organoid_cond_preds_subregion_transfer.tsv")) %>% 
    select(dataset, condition, Subregion_transfer, n_cells_subregion=n_cells) %>% 
    group_by(dataset, condition)

pred_select_conditions <- pred_mean_ref_dist %>% 
    inner_join(pred_closest_obs) %>% 
    ungroup() %>% 
    mutate(gates=case_when(
        min_obs_edist > 1 & weighted_ref_edist < 2 ~ "OOD & close to reference",
        min_obs_edist > 1 ~ "OOD",
        weighted_ref_edist < 2 ~ "Close to reference",
        T ~ "Far"
    ))

selected_conds <- pred_select_conditions %>% 
    filter(gates=="OOD & close to reference") %>% 
    select(condition, dataset, min_obs_edist, weighted_ref_edist)

selected_conds %>% View()

selected_conds %>% write_tsv(str_c(DATA_DIR, "selected_conditions_closest_obs_vs_mean_ref_edist.tsv"))

pred_mean_ref_dist <- ref_cluster_dist %>% 
    inner_join(ref_cluster_counts_all) %>% 
    inner_join(selected_conds) %>% 
    mutate(ds_cond=str_c(dataset, "_", condition))

pred_mean_ref_dist_mat <- pred_mean_ref_dist %>% 
    select(ds_cond, cluster, edist) %>% 
    pivot_wider(names_from=cluster, values_from=edist, values_fill=999) %>%
    column_to_rownames("ds_cond") %>%
    as.matrix() %>% Matrix(sparse=TRUE)

cond_order <- pred_mean_ref_dist_mat %>% dist() %>% hclust() %>% {.$labels[.$order]}
clust_order <- pred_mean_ref_dist_mat %>% t() %>% dist() %>% hclust() %>% {.$labels[.$order]}

plot_df <- pred_mean_ref_dist %>% 
    group_by(condition, dataset) %>%
    inner_join(pred_min_logfc) %>% 
    mutate(median_edist=median(edist)) %>%
    arrange(median_edist) %>% 
    ungroup() %>%
    inner_join(ref_cluster_meta) %>% 
    mutate(
        condition=factor(condition, levels=unique(.$condition)),
        dataset=factor(dataset, levels=unique(.$dataset)),
    ) %>% 
    mutate(
        ds_cond=factor(ds_cond, levels=cond_order),
        cluster=factor(cluster, levels=clust_order)
    )

plot_df$ds_cond

p <- ggplot(plot_df, aes(ds_cond, cluster, color=edist, size=n_cells)) +
    geom_point(shape=16) +
    scale_color_gradientn(colors=pals::magma(100), transform="log10", limits=c(0.1, 10)) +
    scale_size_continuous(range=c(0.0001, 0.5), transform="log10") +
    no_x_text() +
    no_y_text() +
    no_legend() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
    ) +
    labs(x="Annotated cluster", y="Condition")
p
ggsave(p, filename=str_c(PLOT_DIR, "ref_cluster_dist/edist_per_cluster_heatmap.pdf"), bg="white", width=20, height=12, dpi=300, units="cm")

p_region_ref <- ggplot(plot_df, aes(cluster, fill=Subregion)) +
    geom_bar(position="fill") +
    scale_fill_manual(values=region_colors) +
    theme_dr() +
    no_legend() 
ggsave(p_region_ref, filename=str_c(PLOT_DIR, "ref_cluster_dist/region_per_cluster_barplot.pdf"), bg="white", width=10, height=3, dpi=300, units="cm")


p_class_ref <- ggplot(plot_df, aes(cluster, fill=CellClass)) +
    geom_bar(position="fill") +
    scale_fill_manual(values=class_colors) +
    theme_dr() +
    no_legend() 
ggsave(p_class_ref, filename=str_c(PLOT_DIR, "ref_cluster_dist/class_per_cluster_barplot.pdf"), bg="white", width=10, height=3, dpi=300, units="cm")

plot_df_region <- condition_meta_subregion %>% 
    inner_join(selected_conds) %>%
    mutate(ds_cond = str_c(dataset, "_", condition)) %>%
    mutate(ds_cond = factor(ds_cond, levels=cond_order))

p_region_ref <- ggplot(plot_df_region, aes(ds_cond, n_cells_subregion, fill=Subregion_transfer)) +
    geom_bar(position="fill", stat="identity") +
    scale_fill_manual(values=region_colors) +
    theme_dr() +
    no_legend() 
ggsave(p_region_ref, filename=str_c(PLOT_DIR, "ref_cluster_dist/region_per_cond_barplot.pdf"), bg="white", width=10, height=3, dpi=300, units="cm")


plot_df_region_edist <- condition_meta_subregion %>% 
    inner_join(selected_conds) %>%
    mutate(ds_cond = str_c(dataset, "_", condition)) %>%
    arrange(min_obs_edist) %>%
    mutate(ds_cond = factor(ds_cond, levels=unique(.$ds_cond))) 

plot_df_region_edist$weighted_ref_edist

p_region_edist <- ggplot(plot_df_region_edist, aes(ds_cond, n_cells_subregion, fill=Subregion_transfer)) +
    geom_bar(position="fill", stat="identity") +
    scale_fill_manual(values=region_colors) +
    theme_dr() +
    no_legend() 
ggsave(p_region_ref_edist, filename=str_c(PLOT_DIR, "ref_cluster_dist/region_per_cond_barplot_edist_sort.pdf"), bg="white", width=10, height=3, dpi=300, units="cm")


plot_df_region_ref_edist <- condition_meta_subregion %>% 
    inner_join(selected_conds) %>%
    mutate(ds_cond = str_c(dataset, "_", condition)) %>%
    arrange(weighted_ref_edist) %>%
    mutate(ds_cond = factor(ds_cond, levels=unique(.$ds_cond))) 

plot_df_region_edist$weighted_ref_edist

p_region_ref_edist <- ggplot(plot_df_region_edist, aes(ds_cond, n_cells_subregion, fill=Subregion_transfer)) +
    geom_bar(position="fill", stat="identity") +
    scale_fill_manual(values=region_colors) +
    theme_dr() +
    no_legend() 
ggsave(p_region_ref_edist, filename=str_c(PLOT_DIR, "ref_cluster_dist/region_per_cond_barplot_edist_ref_sort.pdf"), bg="white", width=10, height=3, dpi=300, units="cm")


#### Check examples of generated cell type distributions ####
cond_plot <- c(
    "neal_XAV_3.7_late_FGF8_5.4_late_BMP4_5.59_late",
    "neal_CHIR_4.5_mid-late_XAV_3.7_mid-late_CycA_5.18_mid-late",
    "nadya_SAG_6.0_early_BMP7_4.85_early_CHIR_3.48_late",
    "neal_SAG_6.0_mid_FGF2_5.4_mid_FGF8_5.4_early-late",
    "neal_SAG_6.0_mid_FGF8_5.4_early-late_BMP7_4.85_mid",
    "neal_SAG_6.0_mid_FGF2_5.4_early-late_LDN_5.0_mid",
    "neal_SAG_6.0_mid_FGF2_5.4_early-late_LDN_5.0_early"
)

map(set_names(cond_plot), function(cond_check){
    print(cond_check)

    cond_path <- str_c(DATA_DIR, "/selected/", cond_check, ".h5ad")
    cond_adata <- anndata::read_h5ad(cond_path)

    cond_umap <- cond_adata$obsm$X_umap %>% 
        as_tibble() %>% 
        set_names(c("UMAP1", "UMAP2"))

    cond_meta <- cond_adata$obs %>% 
        as_tibble() %>% 
        bind_cols(cond_umap) %>% 
        group_by(Clusters_transfer) %>%
        mutate(n_cells_cluster=n()) %>%
        mutate(
            cluster=Clusters_transfer,
            CellClass=CellClass_transfer,
            major_clusters=case_when(
                n_cells_cluster>60 ~ as.character(Clusters_transfer),
                T ~ NA
            )
        ) %>% 
        inner_join(ref_cluster_meta) %>% 
        mutate(
            Subregion=factor(Subregion, levels=names(region_colors))
        )

    p1 <- ggplot(cond_meta, aes(UMAP1, UMAP2, fill=Subregion)) +
        geom_point(size=3, shape=21, color="black") +
        geom_point(size=3, shape=21, stroke=0) +
        scale_fill_manual(values=region_colors) +
        theme_dr() +
        guides_dr() +
        ggtitle(cond_check)

    p2 <- ggplot(cond_meta, aes(UMAP1, UMAP2, fill=CellClass)) +
        geom_point(size=3, shape=21, color="black") +
        geom_point(size=3, shape=21, stroke=0) +
        scale_fill_manual(values=class_colors) +
        theme_dr() +
        guides_dr()

    p3 <- ggplot(arrange(cond_meta, !is.na(major_clusters)), aes(UMAP1, UMAP2, fill=major_clusters)) +
        geom_point(size=3, shape=21, color="black") +
        geom_point(size=3, shape=21, stroke=0) +
        theme_dr() +
        guides_dr()

    p <- p1 + p2 + p3
    ggsave(p, filename=str_c(PLOT_DIR, "prediction_umaps/", cond_check, ".png"), bg="white", width=16, height=6)
})




CHECK_PLOT_DIR = "/home/fleckj/projects/cellflow/plots/organoid_cond_search/predictions/cellflow_0a37dcb9/v1+2/"

all_plots <- map(set_names(levels(plot_df_region_edist$ds_cond)), function(cond_check){
    print(cond_check)

    cond_path <- str_c(DATA_DIR, "/selected/", cond_check, ".h5ad")
    cond_adata <- anndata::read_h5ad(cond_path)

    cond_umap <- cond_adata$obsm$X_umap %>% 
        as_tibble() %>% 
        set_names(c("UMAP1", "UMAP2"))

    cond_meta <- cond_adata$obs %>% 
        as_tibble() %>% 
        bind_cols(cond_umap) %>% 
        group_by(Clusters_transfer) %>%
        mutate(n_cells_cluster=n()) %>%
        mutate(
            cluster=Clusters_transfer,
            CellClass=CellClass_transfer,
            major_clusters=case_when(
                n_cells_cluster>60 ~ as.character(Clusters_transfer),
                T ~ NA
            )
        ) %>% 
        inner_join(ref_cluster_meta) %>% 
        mutate(
            Subregion=factor(Subregion, levels=names(region_colors))
        )

    p1 <- ggplot(cond_meta, aes(UMAP1, UMAP2, color=Subregion)) +
        geom_point() +
        scale_color_manual(values=region_colors) +
        theme_dr() +
        guides_dr() +
        ggtitle(cond_check)

    p2 <- ggplot(cond_meta, aes(UMAP1, UMAP2, color=CellClass)) +
        geom_point() +
        scale_color_manual(values=class_colors) +
        theme_dr() +
        guides_dr()

    p3 <- ggplot(arrange(cond_meta, !is.na(major_clusters)), aes(UMAP1, UMAP2, color=major_clusters)) +
        geom_point() +
        theme_dr() +
        guides_dr()

    p <- p1 + p2 + p3
    ggsave(p, filename=str_c(CHECK_PLOT_DIR, "selected/umaps/", cond_check, ".png"), bg="white", width=16, height=6)

    return(p)
})


all_plots_edist_ref <- all_plots[levels(plot_df_region_edist$ds_cond)]
p_all <- wrap_plots(all_plots_edist_ref, ncol=1)
ggsave(p_all, filename=str_c(CHECK_PLOT_DIR, "selected/umaps/selected_conditions_edist_ref.png"), bg="white", width=10, height=4000, limitsize=FALSE, dpi=3)


all_plots_edist_obs <- all_plots[levels(plot_df_region_ref_edist$ds_cond)]

selected_conds %>% mutate(ds_cond = str_c(dataset, "_", condition)) %>% arrange(desc(min_obs_edist)) %>% write_csv(str_c(DATA_DIR, "selected_conds_ref_edist_sorted.csv"))



#### Check if any of these are close to missing clusters ####
cluster_presence <- read_tsv("/projects/site/pred/organoid-atlas/projects/cellflow/data/Braun_cl_avg_presence_score.tsv")
missing_clusters <- cluster_presence %>% filter(underrepresented) %>% pull(1)

pred_missing_ref_dist <- pred_mean_ref_dist %>% 
    filter(cluster %in% missing_clusters) %>%
    group_by(cluster) %>%
    filter(edist==min(edist))

pred_dist_missing %>% View()




