source('~/scripts/R/master.R')
source('~/scripts/R/colors.R')

library(Pando)
library(ggbeeswarm)
library(yaml)

select <- dplyr::select

setwd('~/projects/cellflow/')

#### Colors ####
region_colors <- c(
    "Cortex"="#F06292",
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

dataset_colors <- c("nadya"="#FFD274", "neal"="#898BFF", "fatima"="#BA377E")

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

dataset_names <- c(
    "Azbukina et al."="nadya",
    "Amin et al."="neal",
    "Sanchis-Calleja et al."="fatima"
)


PLOT_DIR = "/home/fleckj/projects/cellflow/plots/paper/figure_6/"
DATA_DIR = "/home/fleckj/projects/cellflow/results/train_eval_organoids_common/cellflow_0a37dcb9/"


#### Data ####
adata <- anndata::read_h5ad("/home/fleckj/projects/cellflow/results/organoid_annots/organoids_combined_full_v6_annot.h5ad")

adata$obs %>% colnames()

condition_meta <- adata$obs %>% 
    as_tibble() %>% 
    dplyr::select(condition, mol_comb, dataset) %>% 
    distinct() %>% 
    mutate(comb_length = str_split(mol_comb, "\\+"))
condition_meta$comb_length <- map_int(condition_meta$comb_length, length)


#### Metrics ####
metrics_files <- list.files(DATA_DIR, pattern = "metrics.tsv", recursive = T, full.names = T)
names(metrics_files) <- metrics_files

dist_metrics_files <- metrics_files[str_detect(metrics_files, "/metrics.tsv")]
cluster_metrics_files <- metrics_files[str_detect(metrics_files, "/cluster_metrics.tsv")]
precrec_metrics_files <- metrics_files[str_detect(metrics_files, "/precrec_metrics.tsv")]

dist_metrics_df <- map_dfr(dist_metrics_files, read_tsv, .id="file") %>% 
    rename("condition"=2) %>% 
    mutate(
        trial_name=str_split(file, "/", simplify=T)[, 8],
        split_type=str_split(file, "/", simplify=T)[, 10],
        split_task=ifelse(str_detect(file, "transfer"), "transfer", "combination"),
        model="cellflow"
    ) %>% 
    inner_join(condition_meta)

metrics_df$trial_name %>% unique()
metrics_df$split_type
metrics_df$file
metrics_df$split_task %>% unique()

precrec_metrics_df <- map_dfr(precrec_metrics_files, read_tsv, .id="file") %>%
    rename("condition"=2) %>% 
    mutate(
        trial_name=str_split(file, "/", simplify=T)[, 8],
        split_type=str_split(file, "/", simplify=T)[, 10],
        split_task=ifelse(str_detect(file, "transfer"), "transfer", "combination"),
        model="cellflow"
    ) %>% 
    inner_join(condition_meta)

cluster_metrics_df <- map_dfr(cluster_metrics_files, read_tsv, .id="file") %>%
    rename("condition"=2) %>% 
    mutate(
        trial_name=str_split(file, "/", simplify=T)[, 8],
        split_type=str_split(file, "/", simplify=T)[, 10],
        split_task=ifelse(str_detect(file, "transfer"), "transfer", "combination"),
        model="cellflow"
    ) %>% 
    inner_join(condition_meta)


dist_metrics_df %>% filter(split_task=="combination") %>% pull(split_type) %>% unique() %>% length()
dist_metrics_df %>% filter(split_task=="transfer") %>% pull(split_type) %>% unique() %>% length()
dist_metrics_df %>% View()


#### Baselines ####
comb_baseline_cluster_metrics_df <- read_tsv("/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/combination_baseline_cluster_metrics.tsv") %>% 
    rename(condition=holdout_condition)

comb_baseline_dist_metrics_df <- read_tsv("/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/combination_baseline_dist_metrics.tsv") %>% 
    rename(condition=holdout_condition)

trans_baseline_cluster_metrics_df <- read_tsv("/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/transfer_baseline_cluster_metrics.tsv") %>% 
    rename(condition=holdout_condition)

trans_baseline_dist_metrics_df <- read_tsv("/pmount/projects/site/pred/organoid-atlas/USERS/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/transfer_baseline_dist_metrics.tsv") %>% 
    rename(condition=holdout_condition)


#### Embeddings and stats of datasets ####
umap_df <- adata$obsm$X_umap
rownames(umap_df) <- rownames(adata$obs)
colnames(umap_df) <- c("UMAP1", "UMAP2")
umap_df <- as_tibble(umap_df, rownames="cell")

meta <- adata$obs %>% 
    as_tibble(rownames="cell") %>%
    inner_join(umap_df, by="cell")

# Plot each molecule usage
all_mol_cols <- colnames(meta)[str_detect(colnames(meta), "_conc")]
plots <- map(all_mol_cols, ~{
    meta$has_mol <- meta[[.x]] > 0
    meta <- arrange(meta, has_mol)
    mol_names <- str_split(.x, "_", simplify=T)[, 1]
    p <- ggplot(meta, aes(UMAP1, UMAP2, color=has_mol, fill=has_mol)) +
        geom_point(size=1, shape=21, color="black") +
        geom_point(size=1, shape=21, stroke=0) +
        scale_fill_manual(values=c("grey", "black")) +
        scale_color_manual(values=c("grey", "black")) +
        theme_dr() +
        no_legend() +
        ggtitle(mol_names) +
        theme(plot.title = element_text(size=6))
    return(p)
})
p <- wrap_plots(plots, ncol=12, nrow=2)
ggsave(str_c(PLOT_DIR, "annotations/umap_single_mols.png"), plot=p,  bg="white", width=17, height=3, units="cm")

p <- ggplot(meta, aes(UMAP1, UMAP2, color=subregion_pred_wknn, fill=subregion_pred_wknn)) +
    geom_point(size=0.8, shape=21, color="black") +
    geom_point(size=0.8, shape=21, stroke=0) +
    scale_color_manual(values=region_colors) +
    scale_fill_manual(values=region_colors) +
    theme_dr() +
    guides_dr()
ggsave(str_c(PLOT_DIR, "annotations/umap_subregion_pred_wknn.png"), plot=p,  bg="white", width=12, height=8)


p <- ggplot(meta, aes(UMAP1, UMAP2, color=subregion_pred_wknn, fill=subregion_pred_wknn)) +
    geom_point(size=0.8, shape=21, color="black") +
    geom_point(size=0.8, shape=21, stroke=0) +
    scale_color_manual(values=region_colors) +
    scale_fill_manual(values=region_colors) +
    theme_dr() +
    guides_dr()
ggsave(str_c(PLOT_DIR, "annotations/umap_subregion_pred_wknn.png"), plot=p,  bg="white", width=12, height=8)

p <- ggplot(meta, aes(UMAP1, UMAP2, color=class_pred_wknn, fill=class_pred_wknn)) +
    geom_point(size=0.8, shape=21, color="black") +
    geom_point(size=0.8, shape=21, stroke=0) +
    scale_color_manual(values=many) +
    scale_fill_manual(values=many) +
    theme_dr() +
    guides_dr()
ggsave(str_c(PLOT_DIR, "annotations/umap_class_pred_wknn.png"), plot=p,  bg="white", width=12, height=8)

comb_meta <- meta %>% 
    mutate(class_pred_wknn=factor(class_pred_wknn)) %>% 
    filter(condition=="bmp7 chir")

ggplot(comb_meta, aes(leiden_4, fill=class_pred_wknn)) +
    geom_bar() +
    scale_fill_manual(values=many, breaks=levels(comb_meta$class_pred_wknn))

meta <- adata$obs %>% 
    as_tibble(rownames="cell") %>%
    inner_join(umap_df, by="cell")  %>% 
    sample_n(1000)

p <- ggplot(meta, aes(UMAP1, UMAP2, color=dataset, fill=dataset)) +
    geom_point(size=1.5, shape=21, color="black") +
    geom_point(size=1.5, shape=21, stroke=0) +
    scale_color_manual(values=dataset_colors) +
    scale_fill_manual(values=dataset_colors) +
    theme_dr() +
    guides_dr() 
ggsave(str_c(PLOT_DIR, "annotations/umap_dataset.png"), plot=p,  bg="white", width=12, height=8)


#### Distribution of start/end times ####
dataset_meta <- adata$obs %>%
    as_tibble() %>% 
    select(1:69, "dataset", "condition") %>% 
    distinct() 

timing_meta <- dataset_meta[,1:69] %>% 
    mutate_all(as.character) %>%
    mutate_all(as.numeric) %>% 
    bind_cols(select(dataset_meta, "dataset", "condition"))

timing_meta <- timing_meta[!is.na(rowSums(timing_meta_num)), ]

start_time_cols <- colnames(timing_meta)[str_detect(colnames(timing_meta), "start_time")]
mol_cols <- start_time_cols %>% str_replace("_start_time", "")

condition_timing <- map_dfr(set_names(mol_cols), ~{
    start_col <- paste0(.x, "_start_time")
    end_col <- paste0(.x, "_end_time")
    conc_col <- paste0(.x, "_conc")
    p_df <- timing_meta %>% 
        select(condition, dataset, !!sym(start_col), !!sym(end_col), !!sym(conc_col)) %>% 
        mutate(mol=.x) %>% 
        rename("start_time"=!!sym(start_col), "end_time"=!!sym(end_col), "conc"=!!sym(conc_col)) %>% 
        return()
}, .id="mol") %>% 
    mutate(dataset=factor(dataset, levels=names(dataset_colors))) %>%
    arrange(dataset) %>% 
    mutate(condition=factor(condition, levels=unique(.$condition))) %>% 
    filter(conc!=0)

plot_df <- condition_timing
p <- ggplot(plot_df, aes(x = start_time, xend = end_time, y=condition, color=dataset)) +
    geom_segment(linewidth=0.1) +
    scale_color_manual(values=dataset_colors) +
    facet_wrap(~mol) +
    scale_x_continuous(limits=c(0, 36), breaks=seq(0, 36, 10)) +
    article_text() +
    no_y_text() +
    no_legend() +
    no_label() +
    no_margin() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(),
        panel.border = element_rect(color = "darkgrey", fill = NA, linewidth = 0.5),
        panel.spacing = unit(0.05, "lines"),
        axis.line=element_blank(),
    ) +
    labs(x = "Time", y = "Condition")
p
ggsave(p, filename = paste0(PLOT_DIR, "time_per_condition.pdf"), bg="white", unit="cm", width = 6, height = 7)
 
p <- ggplot(plot_df, aes(y = start_time, yend = end_time, x=condition, color=dataset)) +
    geom_segment(linewidth=0.05) +
    scale_color_manual(values=dataset_colors) +
    facet_grid(mol~.) +
    scale_y_continuous(limits=c(0, 36), breaks=seq(0, 36, 10)) +
    article_text() +
    no_y_text() +
    no_legend() +
    no_label() +
    no_margin() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(),
        panel.border = element_rect(color = "darkgrey", fill = NA, linewidth = 0.25),
        panel.spacing = unit(0.05, "lines"),
        axis.line=element_blank(),
    ) +
    labs(x = "Time", y = "Condition")
p
ggsave(p, filename = paste0(PLOT_DIR, "time_per_condition_long.pdf"), bg="white", unit="cm", width = 6, height = 7)
 

#### Plot all single morphogens ####
meta <- adata$obs %>% 
    as_tibble(rownames="cell") %>%
    inner_join(umap_df, by="cell")

all_mols <- meta$mol_comb %>% str_split("\\+") %>% unlist() %>% unique()

plots <- map(all_mols, ~{
    meta$has_mol <- str_detect(meta$mol_comb, .x)
    meta <- arrange(meta, has_mol)
    p <- ggplot(meta, aes(UMAP1, UMAP2, color=has_mol, fill=has_mol)) +
        geom_point(size=1, shape=21, color="black") +
        geom_point(size=1, shape=21, stroke=0) +
        scale_fill_manual(values=c("grey", "#BA377E")) +
        scale_color_manual(values=c("grey", "#BA377E")) +
        theme_dr() +
        no_legend() +
        ggtitle(.x) +
        theme(plot.title = element_text(size=60))
    # ggsave(str_c(PLOT_DIR, "annotations/umap_single_", .x, ".png"), plot=p,  bg="white", width=12, height=8)
})

p <- wrap_plots(plots, ncol=5, nrow=5)
ggsave(str_c(PLOT_DIR, "annotations/umap_single_combs.png"), plot=p,  bg="white", width=35, height=35)


#### Metrics of combination task ####
dist_metrics_df$model <- "cellflow"

plot_df <- dist_metrics_df %>% 
    filter(split_task=="combination") %>%
    bind_rows(comb_baseline_dist_metrics_df) %>% 
    select(-c(dataset, mol_comb, comb_length)) %>% 
    inner_join(condition_meta) %>% 
    filter(mol_comb!="BMP4+CHIR") %>% 
    mutate(
        model=factor(model, levels=names(model_order)),
        dataset_name=fct_recode(dataset, !!!dataset_names)
    )

plot_df %>% filter(model=="cellflow") %>% pull(e_distance) %>% median()

ggplot(plot_df, aes(e_distance, model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    scale_x_log10() +
    facet_grid(dataset_name~.) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2),
        panel.spacing=unit(0.1, "lines")
    ) +
    labs(x="Energy distance", y="Model") 
ggsave(paste0(PLOT_DIR, "metrics/comb_metrics_bg_edist.pdf"), unit="cm", width=4, height=3.2)


plot_df <- cluster_metrics_df %>% 
    filter(split_task=="combination") %>%
    bind_rows(comb_baseline_cluster_metrics_df) %>% 
    select(-c(dataset, mol_comb, comb_length)) %>% 
    inner_join(condition_meta) %>% 
    filter(mol_comb!="BMP4+CHIR") %>% 
    mutate(
        model=factor(model, levels=names(model_order)),
        dataset_name=fct_recode(dataset, !!!dataset_names)
    )

plot_df %>% filter(model=="cellflow") %>% pull(cosine) %>% {1-.} %>% median()

ggplot(plot_df, aes(1-cosine, model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    facet_grid(dataset_name~.) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2),
        panel.spacing=unit(0.1, "lines")
    ) +
    labs(x="Cosine similarity", y="Model") 
ggsave(str_c(PLOT_DIR, "metrics/comb_metrics_bg_cosine.pdf"), bg="white", width=4, height=3.2, units="cm")



#### Split by mol comb and plot improvement over union baseline ####
comb_union_baseline_metrics_df <- comb_baseline_dist_metrics_df %>% 
    filter(model=="union")

comb_fc_df <- dist_metrics_df %>% 
    filter(split_task=="combination") %>%
    mutate(split_name=str_split(split_type, "_", simplify=T)[, 3]) %>%
    inner_join(comb_union_baseline_metrics_df, by=c("condition", "split_task", "dataset", "split_name"), suffix=c("", "_bg")) %>% 
    mutate(
        e_distance_logfc=log2(e_distance / e_distance_bg),
        mmd_logfc=log2(mmd / mmd_bg),
        sinkhorn_div_logfc=log2(sinkhorn_div_1 / sinkhorn_div_1_bg),
        dataset_name=fct_recode(dataset, !!!dataset_names)
    ) 

2**(-comb_fc_df$e_distance_logfc %>% mean())
2**(-comb_fc_df$mmd_logfc %>% mean())
2**(-comb_fc_df$sinkhorn_div_logfc %>% mean())

plot_df <- comb_fc_df
ggplot(plot_df, aes(condition, -e_distance_logfc, fill=dataset)) +
    geom_bar(stat="identity") +
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey") +
    scale_fill_manual(values=dataset_colors) +
    facet_grid(~dataset_name + split_name, scales="free", space="free") +
    article_text() +
    no_x_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2),
        panel.spacing=unit(0.1, "lines"),
        panel.border = element_rect(color = "darkgrey", fill = NA, linewidth = 0.25),
    ) +
    labs(x="Condition", y="E-distance log2FC\n(Baseline vs CellFlow)")
ggsave(str_c(PLOT_DIR, "metrics/comb_metrics_bg_edist_logfc.pdf"), bg="white", width=12, height=3.2, units="cm")


ggplot(plot_df, aes(condition, -e_distance_logfc, fill=comb_length)) +
    geom_bar(stat="identity", color='black', linewidth=0.2) +
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey") +
    facet_grid(~dataset_name + split_name, scales="free", space="free") +
    scale_fill_gradientn(colors=pals::brewer.blues(100)) +
    article_text() +
    no_x_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2),
        panel.spacing=unit(0.1, "lines"),
        panel.border = element_rect(color = "darkgrey", fill = NA, linewidth = 0.25),
    ) +
    labs(x="Condition", y="E-distance log2FC\n(Baseline vs CellFlow)")
ggsave(str_c(PLOT_DIR, "metrics/comb_metrics_bg_edist_logfc_comb_length.pdf"), bg="white", width=2.5, height=3.5, units="cm")


#### Relationship between performance and combination length and timing ####
plot_df <- comb_fc_df %>% 
    inner_join(condition_timing)

ggplot(plot_df, aes(x = start_time, xend = end_time, y=-e_distance_logfc, color=dataset)) +
    geom_segment(linewidth=0.2) +
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey") +
    scale_color_manual(values=dataset_colors) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
    ) +
    labs(x = "Time", y = "E-distance log2FC\n(Baseline vs CellFlow)") 
ggsave(str_c(PLOT_DIR, "metrics/comb_metrics_bg_edist_logfc_timing.pdf"), bg="white", width=3, height=5, units="cm")



#### Metrics of transfer task ####
plot_df <- dist_metrics_df %>% 
    filter(split_task=="transfer") %>%
    bind_rows(trans_baseline_dist_metrics_df) %>%
    select(-c(dataset, mol_comb, comb_length)) %>%
    inner_join(condition_meta) %>%
    mutate(model=factor(model, levels=names(model_order)))

ggplot(plot_df, aes(e_distance, model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    facet_grid(dataset~.) +
    article_text() +
    no_legend() +
    theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), axis.line=element_line(size=0.2)) +
    labs(x="Energy distance", y="Model")
ggsave(str_c(PLOT_DIR, "metrics/trans_metrics_bg_edist.pdf"), bg="white", width=4, height=3.5, units="cm")


ggplot(plot_df, aes(e_distance, model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    scale_y_discrete(labels=model_order) +
    facet_grid(dataset~comb_length) +
    article_text() +
    no_legend() +
    theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), axis.line=element_line(size=0.2)) +
    labs(x="Energy distance", y="Model")


plot_df <- cluster_metrics_df %>% 
    filter(split_task=="transfer") %>%
    bind_rows(trans_baseline_cluster_metrics_df) %>% 
    select(-c(dataset, mol_comb, comb_length)) %>%
    inner_join(condition_meta) %>%
    mutate(model=factor(model, levels=names(model_order)))

ggplot(plot_df, aes(1-cosine, model, fill=model)) +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=model_colors) +
    facet_grid(dataset~.) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="Cosine similarity", y="Model")
ggsave(str_c(PLOT_DIR, "metrics/trans_metrics_bg_cosine.pdf"), bg="white", width=4, height=3.5, units="cm")


#### Compute logfc vs baseline ####
trans_fc_df <- dist_metrics_df %>% 
    filter(split_task=="transfer") %>%
    inner_join(trans_baseline_dist_metrics_df, by=c("condition", "split_task", "dataset"), suffix=c("", "_bg")) %>% 
    mutate(
        e_distance_logfc=log2(e_distance / e_distance_bg),
        mmd_logfc=log2(mmd / mmd_bg),
        sinkhorn_div_logfc=log2(sinkhorn_div_1 / sinkhorn_div_1_bg),
        dataset_name=fct_recode(dataset, !!!dataset_names)
    ) 

2**(-1*(trans_fc_df %>% filter(dataset!="nadya") %>% pull(e_distance_logfc) %>% mean()))

plot_df <- trans_fc_df
ggplot(plot_df, aes(-e_distance_logfc, dataset_name, fill=dataset)) +
    geom_vline(xintercept=0, linetype="dashed", color="darkgrey") +
    geom_boxplot(size=0.2,  width=0.8, outlier.size=0.2, outlier.shape=16) +
    scale_fill_manual(values=dataset_colors) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="E-distance log2FC\n(Baseline vs CellFlow)", y="Dataset")
ggsave(str_c(PLOT_DIR, "metrics/trans_metrics_bg_edist_logfc.pdf"), bg="white", width=5, height=3, units="cm")



#### Transfer performance for different comb lengths ####
plot_df <- trans_fc_df
ggplot(plot_df, aes(factor(comb_length), -e_distance_logfc, fill=dataset)) +
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey", linewidth=0.25) +
    geom_jitter(shape=21, size=0.5, width=0.3, stroke=0.2) +
    geom_boxplot(fill="lightgrey", size=0.2,  width=0.6, outlier.shape=NA, alpha=0.5) +
    scale_fill_manual(values=dataset_colors) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="Combination length", y="E-distance log2FC\n(Baseline vs CellFlow)")
ggsave(str_c(PLOT_DIR, "metrics/trans_metrics_bg_edist_logfc_comb_length.pdf"), bg="white", width=3, height=4, units="cm")


ggplot(plot_df, aes(factor(comb_length), -e_distance_logfc, fill=dataset)) +
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey", linewidth=0.25) +
    geom_jitter(shape=21, size=2, width=0.3, stroke=0.2) +
    geom_boxplot(fill="lightgrey", size=0.2,  width=0.6, outlier.shape=NA, alpha=0.5) +
    scale_fill_manual(values=dataset_colors) +
    facet_wrap(~split_type) +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="Combination length", y="E-distance log2FC\n(Baseline vs CellFlow)")


plot_df <- trans_fc_df %>% 
    inner_join(condition_timing)

ggplot(plot_df, aes(x = start_time, xend = end_time, y=-e_distance_logfc, color=dataset)) +
    geom_segment(linewidth=0.2) +
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey") +
    scale_color_manual(values=dataset_colors) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
    ) +
    labs(x = "Time", y = "E-distance log2FC\n(Baseline vs CellFlow)")
ggsave(str_c(PLOT_DIR, "metrics/comb_metrics_bg_edist_logfc_timing.pdf"), bg="white", width=3, height=5, units="cm")


#### Distribution of comb lengths across datasets ####
plot_df <- condition_meta %>% 
    mutate(dataset_name=fct_recode(dataset, !!!dataset_names))
ggplot(plot_df, aes(comb_length, fill=dataset)) +
    geom_histogram(binwidth=1) +
    facet_grid(dataset_name~., scales="free") +
    scale_fill_manual(values=dataset_colors) +
    scale_x_continuous(breaks=seq(1, 7, 1)) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="Combination length", y="# conditions")
ggsave(str_c(PLOT_DIR, "metrics/comb_length_dist.pdf"), bg="white", width=4, height=4, units="cm")


#### Check performance vs distance to closest train condition ####
obs_obs_dist <- read_tsv("/home/fleckj/projects/cellflow/results/organoid_cond_search/predictions/cellflow_0a37dcb9/v1+2/organoid_obs_obs_dists.tsv")
split_meta <- dist_metrics_df %>% 
    select(condition, split_type, dataset) %>% 
    distinct()

closest_train_cond <- obs_obs_dist %>% 
    inner_join(split_meta, by=c("condition"="condition")) %>% 
    inner_join(split_meta, by=c("obs_condition"="condition"), suffix=c("", "_2")) %>% 
    group_by(condition, obs_condition) %>% 
    filter(!any(split_type==split_type_2)) %>% 
    filter(dataset==dataset_2) %>%
    group_by(condition) %>%
    summarize(
        closest_train_dist=min(edist),
        closest_train_cond=obs_condition[which.min(edist)]
    )

closest_train_cond %>% View()

plot_df <- bind_rows("transfer"=trans_fc_df, "combination"=comb_fc_df, .id="task") %>%
    inner_join(closest_train_cond, by="condition")

ggplot(plot_df, aes(closest_train_dist, -e_distance_logfc, fill=task)) +
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey", linewidth=0.25) +
    geom_point(shape=21, size=0.8, stroke=0.1) +
    scale_fill_manual(values=many[3:4]) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="E-distance\nto closest train condition", y="E-distance log2FC\n(Baseline vs CellFlow)")
ggsave(str_c(PLOT_DIR, "metrics/edist_vs_closest_train_dist.pdf"), bg="white", width=4, height=4, units="cm")


ggplot(plot_df, aes(closest_train_dist, e_distance, fill=task)) +  
    geom_abline(intercept=0, slope=1, linetype="dashed", color="darkgrey", linewidth=0.25) +
    geom_point(shape=21, size=0.8, stroke=0.1) +
    scale_fill_manual(values=many[3:4]) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="E-distance\nto closest train condition", y="E-distance\nto ground truth")
ggsave(str_c(PLOT_DIR, "metrics/edist_vs_closest_train_dist_raw.pdf"), bg="white", width=4, height=4, units="cm")

ggplot(plot_df, aes(e_distance, -e_distance_logfc, fill=task)) +  
    geom_hline(yintercept=0, linetype="dashed", color="darkgrey", linewidth=0.25) +
    geom_point(shape=21, size=0.8, stroke=0.1) +
    scale_fill_manual(values=many[3:4]) +
    article_text() +
    no_legend() +
    theme(
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(), 
        axis.line=element_line(size=0.2)
    ) +
    labs(x="E-distance\nto ground truth", y="E-distance log2FC\n(Baseline vs CellFlow)")
ggsave(str_c(PLOT_DIR, "metrics/edist_vs_edist_logfc.pdf"), bg="white", width=4, height=4, units="cm")




#### UMAPs with predictions and ground truth for each test condition ####
adata_pred_umap <- anndata::read_h5ad(str_c(DATA_DIR, "/combination_neal_BMP7+CHIR/test_predictions.h5ad"))
adata_pred <- anndata::read_h5ad(str_c(DATA_DIR, "/combination_neal_BMP7+CHIR/gt_test_predictions.h5ad"))
adata_pred <- adata_pred[adata_pred$obs$split=="pred", ]
adata_cond <- anndata::read_h5ad("/home/fleckj/projects/cellflow/data/datasets/organoids_combined/v8/combination/neal/BMP7+CHIR/adata_full.h5ad")

umap_df <- adata$obsm$X_umap
rownames(umap_df) <- rownames(adata$obs)
colnames(umap_df) <- c("UMAP1", "UMAP2")
umap_df <- as_tibble(umap_df, rownames="cell")

gt_meta <- adata_cond$obs %>% 
    as_tibble(rownames="cell") %>%
    inner_join(umap_df, by="cell")

pred_umap_df <- adata_pred_umap$obsm$X_umap
rownames(pred_umap_df) <- rownames(adata_pred_umap$obs)
colnames(pred_umap_df) <- c("UMAP1", "UMAP2")
pred_umap_df <- as_tibble(pred_umap_df, rownames="cell")

pred_meta <- adata_pred$obs %>% 
    as_tibble(rownames="cell") %>%
    inner_join(pred_umap_df, by="cell")

meta <- bind_rows("gt"=gt_meta, "pred"=pred_meta, .id="type") 

plot_df_gt <- meta  %>% 
    filter(type=="gt") %>% 
    sample_n(200000)
plot_df2 <- meta  %>% 
    filter(type=="pred" & condition=="bmp7 chir") %>% 
    sample_n(1000)
plot_df <- bind_rows(plot_df_gt, plot_df2)
plot_df$this <- (plot_df$type=="pred") & (plot_df$condition=="bmp7 chir")
p1 <- ggplot(arrange(plot_df, this), aes(UMAP1, UMAP2, color=this, fill=this)) +
    geom_point(size=1.5, shape=21, stroke=0) +
    scale_color_manual(values=c("gray", "#B12F8C")) +
    scale_fill_manual(values=c("gray", "#B12F8C")) +
    theme_dr() +
    guides_dr() 
ggsave(str_c(PLOT_DIR, "eval_predictions/umap_pred_bmp7_chir.png"), plot=p1,  bg="white", width=12, height=8)

plot_df <- plot_df_gt
plot_df$this <- plot_df$condition=="bmp7 chir"
p1 <- ggplot(arrange(plot_df, this), aes(UMAP1, UMAP2, color=this, fill=this)) +
    geom_point(size=1.5, shape=21, stroke=0) +
    scale_color_manual(values=c("gray", "#455A64")) +
    scale_fill_manual(values=c("gray", "#455A64")) +
    theme_dr() +
    guides_dr() 
ggsave(str_c(PLOT_DIR, "eval_predictions/umap_gt_bmp7_chir.png"), plot=p1,  bg="white", width=12, height=8)

plot_df$this <- plot_df$condition=="chir"
p1 <- ggplot(arrange(plot_df, this), aes(UMAP1, UMAP2, color=this, fill=this)) +
    geom_point(size=1.5, shape=21, stroke=0) +
    scale_color_manual(values=c("gray", "#455A64")) +
    scale_fill_manual(values=c("gray", "#455A64")) +
    theme_dr() +
    guides_dr() 
ggsave(str_c(PLOT_DIR, "eval_predictions/umap_gt_chir.png"), plot=p1,  bg="white", width=12, height=8)

plot_df$this <- plot_df$condition=="bmp7"
p1 <- ggplot(arrange(plot_df, this), aes(UMAP1, UMAP2, color=this, fill=this)) +
    geom_point(size=1.5, shape=21, stroke=0) +
    scale_color_manual(values=c("gray", "#455A64")) +
    scale_fill_manual(values=c("gray", "#455A64")) +
    theme_dr() +
    guides_dr() 
ggsave(str_c(PLOT_DIR, "eval_predictions/umap_gt_bmp7.png"), plot=p1,  bg="white", width=12, height=8)



#### Barplot with predicted clusters ####
cluster_meta <- meta %>% 
    filter(type=="gt") %>%
    group_by(leiden_2) %>% 
    mutate(leiden_2_region=mode(subregion_pred_wknn))  %>% 
    group_by(leiden_3) %>%
    mutate(leiden_3_region=mode(subregion_pred_wknn)) %>%
    group_by(leiden_4) %>%
    mutate(leiden_4_region=mode(subregion_pred_wknn))  %>% 
    select(leiden_2, leiden_2_region, leiden_3, leiden_3_region, leiden_4, leiden_4_region) %>%
    distinct()

meta %>% 
    filter(type=="pred") %>% 
    select(contains("leiden"))

pred_meta <- meta %>% 
    filter(type=="pred") %>% 
    mutate(
        leiden_2=leiden_2_transfer,
        leiden_3=leiden_3_transfer,
        leiden_4=leiden_4_transfer
    )

leiden_4_fracs <- meta %>% 
    filter(type=="gt") %>%  
    bind_rows(pred_meta) %>%
    inner_join(cluster_meta) %>% 
    filter(condition %in% c("bmp7 chir", "chir", "bmp7")) %>%
    group_by(type, condition) %>% 
    mutate(cond_ncells=n()) %>%
    group_by(type, condition, leiden_4, leiden_4_region) %>% 
    summarize(cluster_frac=n()/cond_ncells[1]) %>% 
    group_by(leiden_4) %>% 
    mutate(max_frac=max(cluster_frac)) %>%
    filter(max_frac>0.005) %>%
    distinct() %>% 
    ungroup() %>%
    mutate(leiden_4_region=factor(leiden_4_region, levels=names(region_colors))) %>% 
    arrange(leiden_4_region) %>% 
    mutate(leiden_4=factor(leiden_4, levels=unique(.$leiden_4)))

leiden_4_fracs$max_frac

plot_df <- leiden_4_fracs %>% 
    filter(type=="gt" & condition=="bmp7 chir")
p1 <- ggplot(plot_df, aes(leiden_4, cluster_frac, fill=leiden_4_region)) +
    geom_bar(stat="identity") +
    scale_fill_manual(values=region_colors) +
    scale_x_discrete(limits=unique(leiden_4_fracs$leiden_4)) +
    scale_y_continuous(expand=c(0,0)) +
    article_text() +
    no_legend() +
    no_x_text() +
    no_label() +
    no_y_text() +
    no_margin() +
    theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), axis.line=element_line(size=0.2)) +
    labs(y="Fraction of cells", x="Cluster")
ggsave(str_c(PLOT_DIR, "eval_predictions/leiden_4_bar_gt_bmp7_chir.pdf"), plot=p1,  bg="white", width=2, height=1, units="cm")

plot_df <- leiden_4_fracs %>% 
    filter(type=="pred" & condition=="bmp7 chir")
p2 <- ggplot(plot_df, aes(leiden_4, cluster_frac, fill=leiden_4_region)) +
    geom_bar(stat="identity") +
    scale_fill_manual(values=region_colors) +
    scale_x_discrete(limits=unique(leiden_4_fracs$leiden_4)) +
    scale_y_continuous(expand=c(0,0)) +
    article_text() +
    no_legend() +
    no_x_text() +
    no_label() +
    no_y_text() +
    no_margin() +
    theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), axis.line=element_line(size=0.2)) +
    labs(y="Fraction of cells", x="Cluster")
ggsave(str_c(PLOT_DIR, "eval_predictions/leiden_4_bar_pred_bmp7_chir.pdf"), plot=p2,  bg="white", width=2, height=1, units="cm")

plot_df <- leiden_4_fracs %>% 
    filter(type=="gt" & condition=="chir")
p3 <- ggplot(plot_df, aes(leiden_4, cluster_frac, fill=leiden_4_region)) +
    geom_bar(stat="identity") +
    scale_fill_manual(values=region_colors) +
    scale_x_discrete(limits=unique(leiden_4_fracs$leiden_4)) +
    scale_y_continuous(expand=c(0,0)) +
    article_text() +
    no_legend() +
    no_x_text() +
    no_label() +
    no_y_text() +
    no_margin() +
    theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), axis.line=element_line(size=0.2)) +
    labs(y="Fraction of cells", x="Cluster")
ggsave(str_c(PLOT_DIR, "eval_predictions/leiden_4_bar_gt_chir.pdf"), plot=p3,  bg="white", width=2, height=1, units="cm")

plot_df <- leiden_4_fracs %>% 
    filter(type=="gt" & condition=="bmp7")
p4 <- ggplot(plot_df, aes(leiden_4, cluster_frac, fill=leiden_4_region)) +
    geom_bar(stat="identity") +
    scale_fill_manual(values=region_colors) +
    scale_x_discrete(limits=unique(leiden_4_fracs$leiden_4)) +
    scale_y_continuous(expand=c(0,0)) +
    article_text() +
    no_legend() +
    no_x_text() +
    no_label() +
    no_y_text() +
    no_margin() +
    theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(), axis.line=element_line(size=0.2)) +
    labs(y="Fraction of cells", x="Cluster")
ggsave(str_c(PLOT_DIR, "eval_predictions/leiden_4_bar_gt_bmp7.pdf"), plot=p4,  bg="white", width=2, height=1, units="cm")

p3 / p4 / p1 / p2
ggsave(str_c(PLOT_DIR, "eval_predictions/leiden_4_bar_all.pdf"),  bg="white", width=6, height=5, units="cm")


