library('purrr')
library('tidyverse')
library('jsonlite')
library('data.table')
library(ComplexHeatmap)
library('magrittr')

nice_cols_1   = c(
    "#DC050C", "#FB8072", "#1965B0", "#7BAFDE", "#882E72", "#B17BA6", 
    "#FF7F00", "#FDB462", "#E7298A", "#E78AC3", "#33A02C", "#B2DF8A", 
    "#55A1B1", "#8DD3C7", "#A6761D", "#E6AB02", "#7570B3", "#BEAED4")
nice_cols_2 = c("#666666", "#999999", "#aa8282", "#d4b7b7", "#8600bf", "#ba5ce3", 
    "#808000", "#aeae5c", "#1e90ff", "#00bfff", "#56ff0d", "#ffff00")

layer_vals    = seq(2,40,length.out=3)
layer_cols    = circlize::colorRamp2(layer_vals, RColorBrewer::brewer.pal(3, 'Purples'))

K_vals    = seq(0,10,length.out=3)
K_cols    = circlize::colorRamp2(K_vals, RColorBrewer::brewer.pal(3, 'Blues'))

model_ord     = c('ER', 'BA', 'SBM')
model_cols    = RColorBrewer::brewer.pal(3, 'Greys') %>% setNames(model_ord)

features_vals  = seq(0, 250,length.out=9)
features_cols  = circlize::colorRamp2(features_vals, RColorBrewer::brewer.pal(9, 'Greens'))

ratio_vals  = seq(0, 1,length.out=3)
ratio_cols  = circlize::colorRamp2(ratio_vals, RColorBrewer::brewer.pal(3, 'Greens'))

signal_vals  = seq(0, 10,length.out=3)
signal_cols  = circlize::colorRamp2(signal_vals, RColorBrewer::brewer.pal(3, 'Reds'))


res2hm <- function(res_mat){
    cols_dt = limma::strsplit2(colnames(res_mat), '_') %>%
        as.data.table %>%
        setnames(c('classifier', 'n_hidden_GNN', 'n_hidden_FC', 'K')) %>%
        .[, .(classifier=factor(classifier), 
            n_hidden_GNN=as.numeric(n_hidden_GNN), 
            n_hidden_FC=as.numeric(n_hidden_FC), 
            K=as.numeric(K))] %>%
        .[, id := colnames(..res)] %>%
        .[order(n_hidden_FC, n_hidden_GNN, K)]


    rows_dt = limma::strsplit2(rownames(res_mat), '_') %>%
        as.data.table %>%
        setnames(c('method', 'n_features', 'n_char_features', 'signal_test', 'model')) %>%
        .[, .(method=factor(method), 
            n_features=as.numeric(n_features), 
            n_char_features=as.numeric(n_char_features), 
            signal_test=as.numeric(signal_test), 
            model=as.factor(model))] %>%
        .[, id := rownames(..res_mat)] %>%  
        .[, char_ratio := n_char_features / n_features] %>%
        .[order(signal_test, n_features, char_ratio, model)]

    res_mat %<>% .[rows_dt$id,]
    res_mat %<>% .[, cols_dt$id]

    split_cols = cols_dt$classifier
    split_rows = rows_dt$method

    col_annots  = HeatmapAnnotation(
        n_hidden_GNN=cols_dt$n_hidden_GNN,
        n_hidden_FC=cols_dt$n_hidden_FC,
        K=cols_dt$K,
        col=list(
            n_hidden_GNN=layer_cols,
            n_hidden_FC=layer_cols,
            K=K_cols),
        annotation_legend_param = list(
            n_hidden_FC=list(
                title='# layers',
                at=layer_vals,
                labels=layer_vals
            ),
            K=list(
                title='K (Chebnet)',
                at=K_vals,
                labels=K_vals
            )
        ), show_legend = c(FALSE, TRUE, TRUE)
    )
        # col=list(
        #     matter  = matter_cols,
        #     lesion  = lesion_cols,
        #     # outlier = outlier_cols
        #     disease = disease_cols,
        #     source  = source_cols,
        #     age = age_cols,
        #     sex = sex_cols
        #     ),
        # annotation_legend_param = list(
        #     matter  = list(
        #         title   = 'matter',
        #         at      = matter_ord,
        #         labels  = matter_ord
        #         ),
        #     lesion = list(
        #         title   = 'lesion\ntype',
        #         at      = lesion_ord,
        #         labels  = lesion_ord
        #         ),
        #     disease = list(
        #         title   = 'disease\nstatus',
        #         at      = disease_ord,
        #         labels  = disease_ord
        #         ),
        #     source  = list(
        #         title   = 'sample\nsource',
        #         at      = source_ord,
        #         labels  = source_ord
        #         ),
        #     age = list(title='age'),
        #     sex = list(
        #         title   = 'sex',
        #         at      = sex_ord,
        #         labels  = sex_ord
        #         )

        #     )
        # )
    row_annots  = rowAnnotation(
        model=rows_dt$model,
        # n_features=rows_dt$n_features,
        # n_char_features=rows_dt$n_char_features,
        char_ratio=rows_dt$char_ratio,
        signal_test=rows_dt$signal_test,
        col=list(
            model=model_cols,
            n_features=features_cols,
            n_char_features=features_cols,
            signal_test=signal_cols),
        annotation_legend_param = list(
            model=list(
                title='model',
                at=model_ord,
                labels=model_ord
                ),
            # n_features=list(
            #     title='# features',
            #     at=features_vals,
            #     labels=features_vals
            #     ),
            # n_char_features=list(
            #     title='# char. features',
            #     at=features_vals,
            #     labels=features_vals
            #     ),
            char_ratio=list(
                title='prop. char. feat.',
                at=ratio_vals,
                labels=ratio_vals
            ),
            signal_test=list(
                title='signal (test)',
                at=signal_vals,
                labels=signal_vals
            )
        )
    )
    res_range   = c(0,max(res_mat))
    res_vals    = seq(res_range[1],res_range[2],length.out=9)
    res_cols    = circlize::colorRamp2(res_vals, viridis::viridis(9))

    # define how to split 
    # res_mat[]
    rownames(res_mat) = NULL
    colnames(res_mat) = NULL

    hm_obj      = Heatmap(
        matrix=res_mat, 
        col=res_cols, name='Accuracy', 
        cluster_rows=FALSE, cluster_columns=FALSE, 
        # row_labels=NULL,
        # row_title='conos celltype', column_title='library'
        # column_names_side="none", 
        # row_names_gp=gpar(fontsize = 5), column_names_gp=gpar(fontsize = 5),
        right_annotation=row_annots, top_annotation=col_annots,
        row_split=split_rows, 
        column_split=split_cols,
        cluster_row_slices=FALSE, cluster_column_slices=FALSE
    )

    hm_obj
}