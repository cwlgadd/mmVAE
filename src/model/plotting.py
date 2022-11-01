import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import helpers


def cluster_grid(grid,
                 title=None, figsize=[2, 1], save_path=None, 
                 xlabel=None, hide_x=False, remove_columns=False, colnames=None,
                 ylabel=None, y_ticks=[]):       
    """ 
    """
    sns.set()
    
    # Create fig
    gridspec_kw={'width_ratios': [4, .05], 
                 "wspace": .05
                } 
    fig, (ax, ax2) = plt.subplots(1, 2, sharey=False, figsize=(figsize[0]*11.7, figsize[1]*8.27), gridspec_kw=gridspec_kw)

    # Option to not plot for conditions with no grid value above remove_columns
    if remove_columns is not False:
        mask = np.max(grid, axis=0) > remove_columns
    else:
        mask = [True for _ in range(grid.shape[1])]

    # Plot
    sns.heatmap(grid[:, mask], cbar_ax=ax2, cmap="YlGnBu", ax=ax, linewidths=20/grid.shape[0])
 
    # Labels
    if hide_x is True or colnames is None:
        ax.set_xticklabels([])
    else:
        xticks = [b for a, b in zip(mask, colnames) if a]
        ax.set_xticklabels(xticks, rotation=90)    
    ax.set_yticklabels(y_ticks, rotation=0)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel,  fontsize=18)
    if ylabel is not None:
        ax.set_ylabel(ylabel,  fontsize=18)
        
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=16)

    if title is not None:
        ax.set_title(title)   
    
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = ax.get_ylim()                    # discover the values for bottom and top
    b += 0.5                                # Add 0.5 to the bottom
    t -= 0.5                                # Subtract 0.5 from the top
    ax.set_ylim(b, t)                       # update the ylim(bottom, top) values
    
    ax.grid(True)

    if save_path is not None:
        plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight', format='png')
    plt.show()
    
    
def odds_ratio(grid, counts, 
               figsize=[2, 1], save_path=None,
               xlabel=None, hide_x=False, remove_columns=False, colnames=None,
               ylabel=None, y_ticks=[]):

    sns.set()
    
    # Create fig
    gridspec_kw={'width_ratios': [4, 0.5, .05], 
                 "wspace": .025
                } 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(figsize[0]*11.7, figsize[1]*8.27), gridspec_kw=gridspec_kw)
    
    # Option to not plot for conditions with no grid value above remove_columns 
    if remove_columns is not False:
        mask = np.max(grid, axis=0) > remove_columns
    else:
        mask = [True for _ in range(grid.shape[1])]
    
    # Plot
    sns.heatmap(grid[:, mask], cbar_ax=ax3, cbar_kws={"orientation": "vertical"}, cmap="YlGnBu", ax=ax1, linewidths=20/grid.shape[0])

    # Labels
    if hide_x is True or colnames is None:
        ax1.set_xticklabels([])
    else:
        xticks = [b for a, b in zip(mask, colnames) if a]
        ax1.set_xticklabels(xticks, rotation=90)    
    ax1.set_yticklabels(y_ticks, rotation=0)
    
    if xlabel is not None:
        ax1.set_xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        ax1.set_ylabel(ylabel, fontsize=18)
        
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = ax1.get_ylim()          # discover the values for bottom and top
    b += 0.5                              # Add 0.5 to the bottom
    t -= 0.5                              # Subtract 0.5 from the top
    ax1.set_ylim(b, t)             # update the ylim(bottom, top) values
    
    # Histogram plot
    sns.distplot(range(len(counts)), len(counts), hist_kws={'weights': np.flip(counts)}, kde=False, vertical=True, ax=ax2)
    ax2.set_xlabel('% Prevalence', fontsize=18)
    ax2.set_yticklabels([])
    ax2.set_ylim((0, len(counts)-1))
    
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    
    if save_path is not None:
        plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight', format='png')
    plt.show()
    

def cluster_factor_association(grid, 
                               figsize=[1, 1], save_path=None,
                               xlabel=None, x_ticks=[], 
                               ylabel=None, y_ticks=[],):
    sns.set()
    

    # Create fig
    fig, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(figsize[0]*11.7/1.5, figsize[1]*8.27/1.5))

    # Plot
    sns.heatmap(grid, cbar=False, ax=ax1, cmap="Blues", linewidths=20/grid.shape[0]) 

    # Labels
    ax1.set_xticklabels(x_ticks, rotation=0)
    ax1.set_yticklabels(y_ticks, rotation=0)
    if xlabel is not None:
        ax1.set_xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        ax1.set_ylabel(ylabel, fontsize=18)
        
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = ax1.get_ylim()          # discover the values for bottom and top
    b += 0.5                              # Add 0.5 to the bottom
    t -= 0.5                              # Subtract 0.5 from the top
    ax1.set_ylim(b, t)             # update the ylim(bottom, top) values

    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)

    ax1.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path + '.png', dpi=600, bbox_inches='tight', format='png')
    plt.show()

        

def plot_OOD_histogram():
    plt.show()

    
def plot_labelledtSNE(output_dictionary, tSNE, title, col_wrap=5, save_path=None):
    # Note some rare/singleton clusters may not be in the sub-samples.
    # So we drop NaNs in plotting so empty clusters arent passed into FacetGrid
    
    def _annotate(data, **kws):
        cluster = data['cluster'].to_numpy()[0]
        n = len(data)
        ax = plt.gca()
        ax.text(.05, .9, f"Cluster {cluster}", transform=ax.transAxes)
        ax.text(.05, .05, f"{n:.0f}", transform=ax.transAxes)
        
    predicted_cluster = output_dictionary['cluster_allocations'][tSNE.index]
    tSNE['cluster'] = predicted_cluster
    
    sns.scatterplot(data = tSNE, x = "tSNE 1", y = "tSNE 2", hue = "cluster")
    if save_path is not None:
        plt.savefig(save_path + f'allsubcluster.eps', dpi='figure', format='eps')
    plt.show()

    if 'first_acquired' in tSNE.columns:
        # Plot each cluster in a subgrid, but colour by first disease acquired
        grid = sns.FacetGrid(tSNE, col = "cluster", hue = "first_acquired", col_wrap=col_wrap, despine=False, height=2, hue_order = helpers.get_column_order(plot=True))
        grid.map(sns.scatterplot, "tSNE 1", "tSNE 2")
        # grid.map_dataframe(_annotate)
        grid.set_titles(col_template="", row_template="")
        grid.fig.subplots_adjust(wspace=0.2, hspace=0.2)
        grid.add_legend()
        # grid.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=4)
        if save_path is not None:
            plt.savefig(save_path + f'subclusters_split_andlabelled.eps', dpi='figure', format='eps')
        plt.show()
    else:
        # Plot each cluster in a sub-grid
        grid = sns.FacetGrid(tSNE, col = "cluster", hue = "cluster", col_wrap=col_wrap, despine=False, height=2)
        grid.map(sns.scatterplot, "tSNE 1", "tSNE 2")
        grid.map_dataframe(_annotate)
        grid.set_titles(col_template="", row_template="")
        grid.fig.subplots_adjust(wspace=0.2, hspace=0.2)
        grid.add_legend()
        if save_path is not None:
            plt.savefig(save_path + f'subclusters_split.eps', dpi='figure', format='eps')
        plt.show()
