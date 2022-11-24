import numpy as np
import pickle
import matplotlib.pyplot as plt
# Local imports
from model.plotting import *
from experiment import encode
from helpers import post_process


def plot_restarts(diag_frame, all_dicts, losses, labels, 
                  Y_test=None, y_test_params=None, y_test_arch=None,
                  save_path=None, plot_path=None, ):
    
    Y = diag_frame.to_numpy()
    disease_names = diag_frame.columns.tolist()
    
    ###############################
    # =====  Best seed plots ======
    ###############################

    seed = np.argmin(losses)
    print(f"Taking seed {seed} ")
    best_dict = post_process(Y, all_dicts[seed], Y_test, save_path=plot_path)


    # BINARY CLUSTERS
    # ===============
    # Prevalence 
    cluster_grid(best_dict['prevalence_clusters'], 
                 colnames = disease_names,
                 ylabel='Cluster', y_ticks=best_dict['cluster_labels'], 
                 figsize=[2, 2], save_path=f'{plot_path}PrevalenceBinary')  


    # Odds ratio
    odds_ratio(best_dict['OR_clusters'], best_dict['count_clusters'], 
               figsize=[2, 1], save_path=f'{plot_path}OddsRatioBinary',
               x_ticks=disease_names,
               ylabel='Cluster',  y_ticks=best_dict['cluster_labels'], perc_threshold = 1
              )  

    # FACTORS
    # ===============
    # Prevalence 
    cluster_grid(best_dict['prevalence_topics'],
                 colnames=disease_names,
                 ylabel='Latent factors', y_ticks=best_dict['topic_labels'], 
                 save_path=f'{plot_path}PrevalenceTopic')

    # Odds ratio
    odds_ratio(best_dict['OR_topics'], best_dict['count_topics'],
               figsize=[2, 1], save_path=f'{plot_path}OddsRatioTopic',
               x_ticks=disease_names,
               ylabel='Latent factor', y_ticks=best_dict['topic_labels']
              )

    #  CLUSTER-FACTOR ASSOCIATION MATRIX
    # ===============
    cluster_factor_association(best_dict['cluster_factors'].T,
                               xlabel='Cluster', ylabel='Latent factors',
                               x_ticks=best_dict['cluster_labels'], y_ticks=best_dict['topic_labels'], 
                               figsize=[3, 1.5], save_path=f'{plot_path}CTAssociation')
       

    ###############################
    # ==Test out-of-distribution==
    ###############################
        
    if Y_test is not None:
        binary_profiles = encode(Y_test, y_test_params['norm_beta'], 0, f"{save_path}_{seed}.pt",  **y_test_arch)

        # Topics
        prev_factors = (100/ Y_test.shape[0]) * binary_profiles.sum(axis=0)
        prev_factors = prev_factors.tolist()

        
        # Plot histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(1.5*8.27, 8.27))
        sns.distplot(range(len(prev_factors)), len(prev_factors), hist_kws={'weights': prev_factors}, kde=False, vertical=False, ax=ax1)
        ax1.set_xticklabels(range(1, len(prev_factors)+1))
        ax1.set_xticks(np.linspace(0.5, len(prev_factors)-1.5, len(prev_factors)))
        ax1.set_xticklabels(np.arange(len(prev_factors))+1)
        ax1.set_xlabel("Factors", fontsize=18)
        ax1.set_ylabel("% Prevalence", fontsize=18)
        ax1.grid(False)
        ax1.tick_params(labelsize=14)

        # Clusters
        # for l, i in enumerate(range(best_dict["unique_profiles"].shape[0])):
        #     print(f"{l+1}, {best_dict['unique_profiles'][i, :]} {best_dict['counts'][i]}")

        train_profiles, train_counts = best_dict['unique_profiles'], best_dict['counts']
        test_profiles, test_counts = np.unique(binary_profiles, axis=0, return_counts=True)
        indices_sort = np.flip(np.argsort(test_counts))
        test_profiles, test_counts = test_profiles[indices_sort, :], test_counts[indices_sort]

        cluster_labels, cols = [], []
        cluster_count = best_dict["unique_profiles"].shape[0] + 1
        for ind_profile in range(test_profiles.shape[0]):
            #print(f"test profile index {ind_profile}, count {test_counts[ind_profile]}")
            #print(test_profiles[ind_profile])
            test_profile = test_profiles[ind_profile, :]

            # If seen in training
            if np.any(np.all(best_dict['unique_profiles']==test_profile, axis=1)):
                cluster = np.where(np.all(best_dict['unique_profiles']==test_profile, axis=1))[0][0] + 1
                cluster_labels.append(cluster)
                if best_dict['counts'][cluster] > 1000:            
                    cols.append('blue')
                else:
                    cols.append('green')
            # Previously unseen profile
            else:
                cluster_labels.append(cluster_count)
                cols.append('purple')
                cluster_count += 1

        sns.distplot(range(len(test_counts)), len(test_counts), hist_kws={'weights': np.log(test_counts)}, kde=False, ax=ax2)

        for c in ['green', 'blue', 'purple']:
            for patch in [i for i, col in enumerate(cols) if col == c ]:
                ax2.patches[patch].set_facecolor(c)
        ax2.set_xlabel("Clusters", fontsize=18)
        ax2.set_xticks(np.linspace(0.5, len(cluster_labels)-1.5, len(cluster_labels)))
        if len(cluster_labels) < 15:
            ax2.set_xticklabels(cluster_labels)
        else:
            ax2.set_xticklabels([])
        ax2.grid(False)
        ax2.set_ylabel("Log prevalence", fontsize=18)
        ax2.tick_params(labelsize=14)
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='blue', lw=4),
                        Line2D([0], [0], color='green', lw=4),
                        Line2D([0], [0], color='purple', lw=4)]

        ax2.legend(custom_lines, ['Training cluster', 'Rare training cluster', 'Novel cluster'],  prop={'size': 14})
        if plot_path is not None:
            plt.savefig(f'{plot_path}OOD.png', dpi=600, bbox_inches='tight', format='png')
        plt.show()
        
        # Odds ratio plots
        # ############
        # LATENT FACTORS
        prevalence_topics = []
        OR_topics = [] 
        topic_labels = []
        count_topics = []
        eps = 1e-6
        # For each topic (latent dimension)
        for topic in range(test_profiles.shape[1]):
            mask = np.where(np.array(binary_profiles[:, topic]) == 1)[0]
            xmask = np.where(np.array(binary_profiles[:, topic]) != 1)[0]
            if len(mask) > 0:
                prevalence_topics.append(np.sum(Y_test[mask, :], axis=0))
                prob_in = np.mean(Y_test[mask, :], axis=0)
            else:
                prevalence_topics.append(np.zeros_like(Y_test[0, :]))
                prob_in = np.zeros_like(Y_test[0, :])

            if len(xmask) > 0:
                prob_out = (eps + np.mean(Y_test[xmask, :], axis=0))
            else:
                prob_out = eps

            OR_topics.append(prob_in / prob_out)
            topic_labels.append(f'{topic + 1}') #(N={len(mask)})')
            count_topics.append(len(mask))
        OR_topics = np.stack(OR_topics, axis=0)
        prevalence_topics = np.stack(prevalence_topics, axis=0)
        OR_topics[OR_topics < 1] = 0  # np.nan
        OR_topics[OR_topics > 5] = 5
        odds_ratio(OR_topics, count_topics,
                   colnames=disease_names,
                   ylabel='Latent factor', y_ticks=best_dict['topic_labels'], 
                   remove_columns=2, figsize=[2, 1/1,5], save_path=f'{plot_path}OOD_OddsRatioTopic')
        
        # CLUSTERS
        cutoff = 1000
        n_clusters_over = sum(test_counts > cutoff)
        # Memory-alloc
        test_prevalence_clusters = np.zeros((len(test_counts), Y.shape[1]))
        test_OR_clusters = np.zeros((n_clusters_over, Y.shape[1]))
        cluster_factors = np.zeros((n_clusters_over, test_profiles.shape[1]))                  # The latent factors turned on in each cluster
        cluster_labels = []
        count_clusters = []
        # For each cluster above cut-off size
        # for idx, test_cluster in enumerate([i for i in range(len(test_counts)) if test_counts[i] > cutoff]):
        #     # Prevalence: The number of times the condition appears in the cluster
        #     test_prevalence_clusters[idx, :] = np.sum(Y_test[predicted_labels == cluster + 1, :], axis=0)
        #     # Odds ratio: The number of times it appears, divided by the number of times it appears elsewhere (with eps for stability)
        #     test_OR_clusters[idx, :] = np.mean(Y[predicted_labels == cluster + 1, :], axis=0) / (eps + np.mean(Y[predicted_labels != cluster + 1, :], axis=0))
        #     # unique profiles above cut-off size
        #     cluster_factors[idx, :] = unique_profiles[cluster, :]
        #     # Save plotting label for when we report the above metrics
        #     cluster_labels.append(f'{cluster + 1}') # (N={counts[cluster]})') #, topics {np.where(unique_profiles[cluster, :]!=0)[0] + 1}')
        #     count_clusters.append(counts[cluster])

                                 
    return
