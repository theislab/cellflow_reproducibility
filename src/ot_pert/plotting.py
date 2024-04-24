import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from adjustText import adjust_text

# This is taken from CPA github

def get_palette(
        n_colors,
        palette_name='Set1'
):
    try:
        palette = sns.color_palette(palette_name)
    except:
        print('Palette not found. Using default palette tab10')
        palette = sns.color_palette()
    while len(palette) < n_colors:
        palette += palette

    return palette
    
def get_colors(
        labels,
        palette=None,
        palette_name=None
):
    n_colors = len(labels)
    if palette is None:
        palette = get_palette(n_colors, palette_name)
    col_dict = dict(zip(labels, palette[:n_colors]))
    return col_dict


def plot_embedding(
        emb,
        labels=None,
        col_dict=None,
        title=None,
        show_lines=False,
        show_text=False,
        show_legend=True,
        axis_equal=False,
        circle_size=40,
        circe_transparency=1.0,
        line_transparency=0.8,
        line_width=1.0,
        fontsize=9,
        fig_width=4,
        fig_height=4,
        file_name=None,
        file_format=None,
        labels_name=None,
        width_ratios=[7, 1],
        bbox=(1.3, 0.7)
):
    sns.set_style("white")

    # create data structure suitable for embedding
    df = pd.DataFrame(emb, columns=['dim1', 'dim2'])
    if not (labels is None):
        if labels_name is None:
            labels_name = 'labels'
        df[labels_name] = labels

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    sns.despine(left=False, bottom=False, right=True)

    if (col_dict is None) and not (labels is None):
        col_dict = get_colors(labels)

    sns.scatterplot(
        x="dim1",
        y="dim2",
        hue=labels_name,
        palette=col_dict,
        alpha=circe_transparency,
        edgecolor="none",
        s=circle_size,
        data=df,
        ax=ax)

    try:
        ax.legend_.remove()
    except:
        pass

    if show_lines:
        for i in range(len(emb)):
            if col_dict is None:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=None
                )
            else:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=col_dict[labels[i]]
                )

    if show_text and not (labels is None):
        texts = []
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx_label = np.where(labels == label)[0]
            texts.append(
                ax.text(
                    np.mean(emb[idx_label, 0]),
                    np.mean(emb[idx_label, 1]),
                    label,
                    fontsize=fontsize
                )
            )

        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='black', lw=0.1),
            ax=ax
        )

    if axis_equal:
        ax.axis('equal')
        ax.axis('square')

    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold")

    ax.set_xlabel('dim1', fontsize=fontsize)
    ax.set_ylabel('dim2', fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    plt.tight_layout()
    