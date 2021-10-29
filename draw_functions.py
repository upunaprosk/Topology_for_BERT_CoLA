import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_heatmaps_trained_nontrained(
        sf,
        thresholds=[],
        vmin=None,
        vmax=None,
        transpose=False,
        stats_name="",
        title=None,
        annot=None,
        annot_color='w',
        color_scheme='coolwarm',
        figsize=(14, 5),
        pad=0.5,
        ticks=[i for i in range(12)],
        ticks_font_size=9,
        precision=2,
        labels_font_size=10,
        annot_font_size=10,
        subtitles_font_size=16,
        subtitles_pad=12,
        enable_grid=False,
        zorder=0,
        pdf_file=None,
        norm=None,
        topological=True):
    # function for drawing 2 plots for each model
    # sf - list of dicts (received from make_comparison_data)
    if vmax is None or vmin is None:
        vmin = 1e10
        vmax = -1e10
        for j in range(len(sf)):
            surfaces = sf[j]
            for array in surfaces.values():
                vmax = max(vmax, max(array.flatten()))
                vmin = min(vmin, min(array.flatten()))
            if not norm:
                vmax = 1
                vmin = -1

    plots_dim = len(sf)
    if not transpose:
        fig, axs = plt.subplots(len(sf[0].keys()), plots_dim, squeeze=False)
    else:
        fig, axs = plt.subplots(plots_dim, len(sf[0].keys()), squeeze=False)
    fig.set_size_inches(figsize)
    fig.tight_layout(pad=pad)
    for j in range(len(sf)):
        surfaces = sf[j]
        k = 0
        for key in sorted(surfaces.keys()):
            Z = surfaces[key]
            cell_num = (k, j) if not transpose else (j, k)
            if norm:
                cp = axs[cell_num].imshow(Z, interpolation='none', cmap=color_scheme, vmin=vmin, vmax=vmax, norm=norm)
            else:
                cp = axs[cell_num].imshow(Z, interpolation='none', cmap=color_scheme, vmin=vmin, vmax=vmax)
            if labels_font_size > 0:
                axs[cell_num].set_xlabel('head', fontsize=subtitles_font_size)
                axs[cell_num].set_ylabel(key + '\nlayer', fontsize=subtitles_font_size)
            if k == 0:
                value = ''
                try:
                    value = thresholds[j]
                except:
                    value = ''
                if subtitles_font_size > 0:
                    axs[cell_num].set_title(value, fontsize=subtitles_font_size, pad=subtitles_pad)
                else:
                    axs[cell_num].set_title(value)

            axs[cell_num].set_xticks(ticks)
            axs[cell_num].set_yticks(ticks)
            for tick in axs[cell_num].xaxis.get_major_ticks():
                tick.label.set_fontsize(ticks_font_size)
            for tick in axs[cell_num].yaxis.get_major_ticks():
                tick.label.set_fontsize(ticks_font_size)
            if enable_grid:
                axs[cell_num].xaxis.grid(True, color='black', zorder=zorder)
                axs[cell_num].yaxis.grid(True, color='black', zorder=zorder)

            if annot is not None:
                for y in range(Z.shape[0]):
                    for x in range(Z.shape[1]):
                        formatstring = "{:." + str(precision) + "f}"
                        axs[cell_num].text(x,
                                           y,
                                           formatstring.format(annot[j][key][y, x]),
                                           fontsize=annot_font_size,
                                           color=annot_color,
                                           horizontalalignment='center',
                                           verticalalignment='center')
            k += 1
    common_cb = fig.colorbar(cp, ax=axs[:, :])
    common_cb.ax.tick_params(labelsize=subtitles_font_size)
    for ax in axs.flat:
        ax.label_outer()
    args = {}
    if thresholds:
        args = {'x': 0.4}
        plt.subplots_adjust(top=0.75, bottom=0.1, left=0.07, right=0.81, hspace=0.15, wspace=0.15)
    if topological:
        title = title + '\n\nthresholds' if thresholds else title
    else:
        title = 'features'
    fig.suptitle(title, fontsize=subtitles_font_size, **args)
    if pdf_file is not None:
        pdf_file.savefig()
    plt.rcParams["font.family"] = "serif"
    plt.show()


def middle_tick(value1, value2):
    return np.mean([value1, value2])


def plot_histogram(layer, head, feat, features, func, feature_names, y,ax=None, title=None, color=None, legend=False,
                   force_bars=False, pdf_file=None, bins=21):
    X = func(features, layer, head, feature_names)
    X['y'] = y
    datum = X[[feat, 'y']]
    df = datum.copy()
    x_axis = ax.xaxis
    x_axis.label.set_visible(False)
    df['y'] = df['y'].map({1: 'correct', 0: 'incorrect'})
    if color is not None:
        ax.set_facecolor(color)
    groupped = df.groupby([feat])['y'].value_counts().unstack()
    # groupped.index = groupped.index.astype(int)
    flag = X[[feat]].nunique().values[0]
    locs = []
    intervals = False
    three_main_labs = []
    if force_bars or flag > 3:
        data_to_viz = groupped.copy()
        if flag > 25:
            intervals = True
            trial = groupped.copy()
            data_to_viz = trial.groupby(pd.cut(groupped.index, bins=bins))[['incorrect', 'correct']].count()
            middle_tick_ind = bins // 2
            middle = middle_tick(data_to_viz.index[middle_tick_ind].left,
                                 data_to_viz.index[middle_tick_ind].right) if bins % 2 \
                else middle_tick(data_to_viz.index[middle_tick_ind].right, data_to_viz.index[middle_tick_ind + 1].left)
            three_main_labs = np.round([data_to_viz.index[0].left,
                                        middle,
                                        data_to_viz.index[-1].right], 3)

        if data_to_viz.index.dtype == float:
            data_to_viz.index = np.round(data_to_viz.index, 3);
        data_to_viz.loc[:, ['incorrect']]['incorrect'].plot.bar(width=1, ax=ax, edgecolor='black', legend=legend,
                                                                linewidth=1, grid=False, alpha=0.5, label='incorrect',
                                                                fc=(0, 0, 1, 0.5))
        data_to_viz.loc[:, ['correct']]['correct'].plot.bar(width=1, ax=ax, edgecolor='black', legend=legend,
                                                            linewidth=1, grid=False, alpha=0.5, label='correct',
                                                            fc=(1, 0, 0, 0.5))
    else:
        groupped.plot.line(stacked=True, sharex=False, ax=ax, legend=legend, title=title, marker='o',
                           ms=3, color=[(0, 0, 1, 0.5), (1, 0, 0, 0.5)])
    locs = ax.get_xticks()

    rotation = 45
    if intervals:
        ax.set_xticks([locs[0], locs[bins // 2], locs[-1]])
        ax.set_xticklabels(three_main_labs)
        rotation = 0;
    elif len(locs) > 7:
        map(ax.set_xticks(np.round(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 6), 2)), [])
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_title(title, fontsize=10)
    if pdf_file:
        pdf_file.savefig()
    return

def plot_bt_distributions(cur_matrices, feature,params, y,colormap, NUM_layers = 12, NUM_heads = 12,title = None, start_layer = 9, pdf_file = None):
    if not title: title = feature;
    fig, axes = plt.subplots(nrows=3, ncols=12, figsize=(30, 10))
    fig.suptitle(f'{title}', size='xx-large')
    plt.subplots_adjust(hspace=0.30, wspace=0.25)
    for layer in range(start_layer, NUM_layers):
        for head in range(NUM_heads):
            feat = f'{feature}'
            corr_coeff = cur_matrices[0][feat][layer, head]
            p_val = cur_matrices[1][feat][layer, head]
            plot_histogram(layer, head, feat,
                    *params, y,
                    ax=axes[layer - start_layer, head],
                    title='%d, %d \n (%0.2f, %.2E)'  % (layer, head, corr_coeff, p_val),
                    color=colormap.to_rgba(corr_coeff))
    if pdf_file:
        pdf_file.savefig(fig)
    plt.show()
    plt.close()  # for disabling 20 plots warnings
    return

