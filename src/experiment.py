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
# Local imports
from model.mmvae import RelaxedCategoricalAutoEncoder, encode_mmVAE, decode_mmVAE
from helpers import summarise_binary_profiles, similarity_matrix, process_clusters, process_factors
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
                 n_restarts=5, force_retrain=False, similarity_samples=None,
                 save_path=None):
    """
    Fit multiple mmVAE models
    """
        
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
    # ======= Compare seeds =======
    ###############################
    avg_ami = []
    if n_restarts > 1:
        from sklearn.metrics.cluster import adjusted_mutual_info_score as AMI

        adjusted_mutual_score = np.zeros((n_restarts, n_restarts))
        for i in range(n_restarts):
            for j in range(n_restarts):
                if i < j:
                    ami_ij = AMI(labels[i], labels[j], average_method='arithmetic')
                    adjusted_mutual_score[i, j] = ami_ij
                    avg_ami.append(ami_ij)
        print(f"Adjusted mutual information score\n {adjusted_mutual_score}")
        print(f"Averaged adjusted mutual information score\n {np.mean(avg_ami)}")

        if similarity_samples is not None:
            allocations = np.stack(labels, axis=1)
            random_index = random.sample(range(allocations.shape[0]), similarity_samples)
            similarity_matrix = similarity_matrix(allocations[random_index, :])
            if save_path is not None:
                np.savetxt(f"{save_path}_seed_similarity_matrix.txt", similarity_matrix, fmt='%.2f')          # save, used in R script to look at some hierarchical clustering

                
    return all_dicts, losses, labels, np.mean(avg_ami)



def encode(Y, norm_beta, tmp, torch_path,  enc_h, dec_h, l1=0., L0_kwargs={}, constrain=[False, False]):
              
    model = RelaxedCategoricalAutoEncoder(enc_h=enc_h, dec_h=dec_h, norm_beta=norm_beta, constrain=constrain, L0_kwargs=L0_kwargs, l1=l1)
    model.load_state_dict(torch.load(torch_path))
    model.eval()
        
    with torch.no_grad():
        # decoding
        z = torch.Tensor(Y)
        for layer in model.enc_layers:
            z = layer(z)

        q_z = RelaxedBernoulli(tmp, logits=z)
        mu = q_z.probs

        z_mean_binary = 1 * (mu.numpy() > 0.5)

        
    return z_mean_binary


