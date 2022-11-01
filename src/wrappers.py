import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.relaxed_bernoulli import (LogitRelaxedBernoulli, RelaxedBernoulli)
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_mutual_info_score as AMI
# Local imports
from model.mmvae import RelaxedCategoricalAutoEncoder, encode_mmVAE, decode_mmVAE
from helpers import summarise_binary_profiles, post_process, sim_mat
from model.plotting import *


def fit_mmVAE(Y,
              enc_h,
              dec_h, 
              norm_beta=1.0, 
              epochs=10, 
              batch_size=512, 
              lr=1e-3, 
              verbose=1, 
              constrain=[False, False],
              tmp_schedule=[4.0, 0.4, 0.4], 
              L0_kwargs={},
              l1 = 0.,
              anneal=True,
              save_path=None,
              validate_every=5):
    """
    Procedure for fitting the clustering model.
    :return:
    """
    # Train/val split
    Y = torch.Tensor(Y)
    Y_train, Y_val = torch.utils.data.random_split(Y, [int(Y.shape[0]*0.95), Y.shape[0] - int(Y.shape[0]*0.95)])
    train_loader = DataLoader(Y_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Y_val, batch_size=batch_size, shuffle=True)    

    # Temperature scheduler of the form from the paper:  http://proceedings.mlr.press/v124/wang20b/wang20b.pdf
    scale = np.log(tmp_schedule[1] / tmp_schedule[0]) / epochs   #* len(train_loader)
    temperature_fn = lambda x: np.max((tmp_schedule[2], tmp_schedule[0]*torch.exp(scale * torch.tensor(x))))
    # Annealing scheduler 
    if anneal:
        kl_anneal_schedule = [0 for _ in range(5)] + [x for epoch, x in enumerate(np.linspace(0, 1, epochs-5))]
    else:
        kl_anneal_schedule = [0 for _ in range(5)] + [1 for _ in range(epochs-5)]

    # Create model
    model = RelaxedCategoricalAutoEncoder(norm_beta=norm_beta, enc_h=enc_h, dec_h=dec_h, constrain=constrain, L0_kwargs=L0_kwargs, l1=l1)

    # Create optimiser and learning rate scheduler
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(opt, 'min', patience=2, verbose=True, factor=0.75, min_lr=1e-3)
    
    model.train()
    lrs, train_recons, train_Hs, val_recons, val_Hs = [], [], [], [], []
    for epoch in range(epochs):
        
        # Training 
        # Update concrete distribution temperature
        temp = temperature_fn(epoch)
        kl_anneal = kl_anneal_schedule[epoch]
        
        train_loss, train_recon, train_H = 0, 0, 0
        for batch in train_loader:
            opt.zero_grad()
            disease_prob, mu = model(batch, temp)
            loss, recon, H = model.loss_function(disease_prob, batch, mu, kl_anneal = kl_anneal)
            loss.backward()
            opt.step()
            # Record
            train_loss += loss.item() / (batch.shape[0] * len(train_loader))
            train_recon += recon / (batch.shape[0] * len(train_loader))
            train_H += H / (batch.shape[0] * len(train_loader))
        if verbose > 0:
            print(f'====> Training Epoch: {epoch + 1} Train loss: {train_loss:.3f} (Reconstruction loss:{train_recon:.5f} and entropy:{train_H:.5f}). Temperature {temp:.1f}. kl_anneal: {kl_anneal:.2f}')
            
        # Validation check
        if epoch % validate_every == 0:
            with torch.no_grad():
                val_loss, val_recon, val_H = 0, 0, 0
                for val_batch in val_loader:
                    disease_prob, mu = model(val_batch, tmp_schedule[-1])
                    loss, recon, H = model.loss_function(disease_prob, val_batch, mu, kl_anneal = 1)
                    # Record
                    val_loss += loss.item() / (val_batch.shape[0] * len(val_loader))
                    val_recon += recon / (val_batch.shape[0] * len(val_loader))
                    val_H += H / (val_batch.shape[0] * len(val_loader))
            if verbose > 0:
                print(f'\t====> Validating: Validation loss: {val_loss:.3f} (Reconstruction loss:{val_recon:.5f} and entropy: {val_H:.5f}). Temperature {tmp_schedule[-1]:.1f}. kl_anneal: 1')
                
            scheduler.step(val_loss) 
        
        lrs.append(opt.param_groups[0]["lr"])
        train_recons.append(train_recon)
        train_Hs.append(train_H)
        val_recons.append(val_recon)
        val_Hs.append(val_H)

        # with torch.no_grad():
        #     if verbose > 1:
        #         _, z_mean = model(Y, temp)
        #         z_mean_binary = 1 * (z_mean.numpy() > 0.5)
        #         counts, unique_profiles, cluster_allocations = summarise_binary_profiles(z_mean_binary)
        #         print(f"\t====> {len(counts)} clusters,  of resp. sizes: {counts}.")
        #         if verbose > 2:
        #             print(f"\t\t====> Unique_profiles {unique_profiles}")

    # Report final
    with torch.no_grad():
        _, z_mean = model(Y, temp)
        z_mean_binary = 1 * (z_mean.numpy() > 0.5)
        counts, unique_profiles, cluster_allocations = summarise_binary_profiles(z_mean_binary)
        print(f"Final:\n====> Train loss: {train_loss:.3f} (Reconstruction loss:{train_recon:.5f} and entropy:{train_H:.5f}).\n====> Validation loss: {val_loss:.3f} (Reconstruction loss:{val_recon:.5f} and entropy: {val_H:.5f}).")
        print(f"\t\t====> Counts at epoch {epoch}: {len(counts)} clusters, with {len(counts[counts>1000])} above 1000 patients")
        if verbose > 1:
            plt.plot(range(epochs), train_recons, label='train recon')
            plt.plot(range(epochs), val_recons, label='val recon')
            plt.plot(range(epochs), train_Hs, label='train Entropy')
            plt.plot(range(epochs), val_Hs, label='val Entropy')
            plt.show()
        
        decode = torch.Tensor(np.eye(enc_h[-1]))
        for layer in model.dec_layers:
            decode = layer(decode)
        disentangled_y = decode.detach().numpy()
            
    # Save model parameters
    if save_path is not None:
        torch.save(model.state_dict(), save_path + ".pt")

    return {"z_mean": z_mean.numpy(),
            "z_binary": z_mean_binary,
            "counts": counts,
            "unique_profiles": unique_profiles,
            "cluster_allocations": 1 + cluster_allocations,
            "disentangled_y": disentangled_y,
            "lrs": lrs,
            "val_loss": val_loss
            }, model
    

def fit_restarts(diag_frame, architecture, params, 
                 n_restarts=5, force_retrain=False, sim_samples=None,
                 Y_test=None, 
                 plot=True, yaxis_scale=1, plot_threshold_frac=0.005, save_path=None, plot_path=None, ):
        
    Y = diag_frame.to_numpy()
    disease_names = diag_frame.columns.tolist()

    ###############################
    # == Train multiple restarts ==
    ###############################
    labels, losses, all_dicts = [], [], []
    for seed in range(n_restarts):        
        try:
            assert force_retrain is False
            with open(f'{save_path}_{seed}.pickle', 'rb') as file:
                return_dict = pickle.load(file)
            print(f"Loaded {save_path}.pickle, seed {seed}")
        except:
            print(f"Failed to load {save_path}_{seed}.pickle, training...")
            return_dict, model = fit_mmVAE(Y, **architecture, **params, save_path=f"{save_path}_{seed}")
            
            if save_path is not None:
                print(f"... saving to {save_path}_{seed}.pickle")
                with open(f'{save_path}_{seed}.pickle', 'wb') as file:
                    pickle.dump(return_dict, file, protocol=pickle.HIGHEST_PROTOCOL)  
        
        labels.append(return_dict['cluster_allocations'])
        losses.append(return_dict['val_loss'])
        all_dicts.append(return_dict)
    
    
    ###############################
    # Across seed metrics and plots
    ###############################
    avg_ami = []
    if n_restarts > 1:
        adjusted_mutual_score = np.zeros((n_restarts, n_restarts))
        for i in range(n_restarts):
            for j in range(n_restarts):
                if i < j:
                    ami_ij = AMI(labels[i], labels[j], average_method='arithmetic')
                    adjusted_mutual_score[i, j] = ami_ij
                    avg_ami.append(ami_ij)
        print(f"Adjusted mutual information score\n {adjusted_mutual_score}")
        print(f"Averaged adjusted mutual information score\n {np.mean(avg_ami)}")

        if sim_samples is not None:
            allocations = np.stack(labels, axis=1)
            random_index = random.sample(range(allocations.shape[0]), sim_samples)
            similarity_matrix = sim_mat(allocations[random_index, :])
            if plot_path is not None:
                np.savetxt(f"{plot_path}_sim_mat.txt", similarity_matrix, fmt='%.2f')          # save, used in R script to look at some hierarchical clustering

    
    ###############################
    # =====  Best seed plots ======
    ###############################
    threshold = Y.shape[0] * plot_threshold_frac

    if save_path is not None and plot is True:
        seed = np.argmin(losses)
        print(f"Taking seed {seed} ")
        best_dict = post_process(Y, all_dicts[seed], Y_test, cutoff=threshold)


        # BINARY CLUSTERS
        # ===============
        # Prevalence 
        cluster_grid(best_dict['prevalence_clusters'], 
                     colnames = disease_names,
                     ylabel='Cluster', y_ticks=best_dict['cluster_labels'], 
                     figsize=[2, yaxis_scale], save_path=f'{plot_path}PrevalenceBinary')  
        
        
        # Odds ratio
        OR = best_dict['OR_clusters']
        # OR[OR < 1] = 0
        perc = [(100 * i) / Y.shape[0] for i in best_dict['count_clusters']]
        odds_ratio(OR, perc, 
                   colnames=disease_names,
                   ylabel='Cluster',  y_ticks=best_dict['cluster_labels'],
                   figsize=[2.5, yaxis_scale], save_path=f'{plot_path}OddsRatioBinary')  
        

        # FACTORS
        # ===============
        # Prevalence 
        cluster_grid(best_dict['prevalence_topics'],
                     colnames=disease_names,
                     ylabel='Latent factors', y_ticks=best_dict['topic_labels'], 
                     save_path=f'{plot_path}PrevalenceTopic')
        
        # Odds ratio
        OR = best_dict['OR_topics']
        # OR[OR < 1] = 0
        perc = [(100 * i)  / Y.shape[0] for i in best_dict['count_topics']]
        odds_ratio(OR, perc,
                   colnames=disease_names,
                   ylabel='Latent factor', y_ticks=best_dict['topic_labels'], 
                   figsize=[2.5, yaxis_scale], save_path=f'{plot_path}OddsRatioTopic')
        # plot_cluster_grid(best_dict['disentangled_y'] / best_dict['prevalence_topics'], f"{method} disentangled disease probability", ylabels=best_dict['topic_labels'], save_path=f'{plot_path}Disentanglement')
        
        #  CLUSTER-FACTOR ASSOCIATION MATRIX
        # ===============
        cluster_factor_association(best_dict['cluster_factors'].T,
                                   xlabel='Cluster', ylabel='Latent factors',
                                   x_ticks=best_dict['cluster_labels'], y_ticks=best_dict['topic_labels'], 
                                   figsize=[yaxis_scale * 1.5, 1.5], save_path=f'{plot_path}CTAssociation')
       

    ###############################
    # ==Test out-of-distribution==
    ###############################
        
    if Y_test is not None:
        binary_profiles = encode(Y_test, params['norm_beta'], 0, f"{save_path}_{seed}.pt",  **architecture)

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

                                 
    return best_dict, f"{save_path}_{seed}", np.mean(avg_ami)


def encode(Y, norm_beta, tmp, torch_path,  enc_h, dec_h, l1=0., L0_kwargs={}, constrain=[False, False]):
              
    model = RelaxedCategoricalAutoEncoder(enc_h=enc_h, dec_h=dec_h, norm_beta=norm_beta, constrain=constrain, L0_kwargs=L0_kwargs, l1=l1)
    model.load_state_dict(torch.load(torch_path))
    model.eval()
        
    # # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor])
    
    with torch.no_grad():
        # decoding
        z = torch.Tensor(Y)
        for layer in model.enc_layers:
            z = layer(z)

        q_z = RelaxedBernoulli(tmp, logits=z)
        mu = q_z.probs

        z_mean_binary = 1 * (mu.numpy() > 0.5)

        
    return z_mean_binary