import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_column_order(plot=False):
    # TODO: refactor out of code
    print("Deprecated ")
    return get_column_order_SAIL(plot=plot)


def get_column_order_ML4H(plot=False):
    """
    Used for the ML4H@NeurIPs paper. All 79 conditions.
    
    Get the list of conditions to filter, or return the names to be used to plot the multi-morbidity diseases 
    """

    if plot:
        return [ "Cancer", "Asthma", "Female infertility",
                 "Allergic Rhin Conj",
                 "Migraine", "Anxiety", "Depression", "Substance misuse", "Alcohol problem", "Eating disorder",  "SMHmm",  "Other mental",  "Other headache",   
                 "AdrenalAll",
                 "Pituitary", "PCOS",  "Sarcoid",  "Leiomyoma",  "Endometriosis", "Retinal detachment", "PTH", "Heart failure",  "IHD/MI",  "Stroke", 
                 "Interstitial lung", "Blind", 
                 "COPD",  "Solid organ transplant", "Bronchiectasis", "Neuro development", "Atopic eczema", 
                 "Cardiomyopathy", "Cystic fybrosis", "Sickle cell", "Pulminary Heart", "IBS",   "Turners syndrome", "Marfan syndrome", "HIV", 
                "Diabetes", "Diabetes (retino)", 
                 "Hypertension", "Spina bifida", "Congenital Heart",
                 "Vertebrae",  "Thyroid",
                 "Prithrombocytopenia",
                 "Pernicious anaemia",  "Coeliac", "Auto Skin", "Inflam Bowel", "Inflam Eye", "Spond Arth", "Psoriasis",
                 "Osteoporosis", "Chronic back pain", "Peripheral neuro", "Urolithiasis", "Scoliosis", "Cholelithiasis",
                 "Other skin", 
                 "VTE", "Valve",  "Epilepsy", 
                 "Osteoarthritis", 
                 "Slesystemic", "Ulcer peptic",  "Ehler",  "Chronic Liver", "Chronic Kidney", "Inflam Arth",
                 "Atrial fibrillation", "Haemophilia","IIH", "Multiple sclerosis", "Somatoform", "OSA","Deaf", 
                 "Cataract",]
    else:
        return ["CancerAll", "asthmalonglist2018", "female_infertility",
                "AllergicRhinConj",
                "migraine", "AnxietyPTSDdiag",  "depressionDiag", "substance_misuse", "alcoholproblem", "eatingdisorderuom",  "SMHmm",  "OthMental",  "OthHeadache",   
                "AdrenalAll",
                "Pituitary","pcoskoo",  "sarcoid",  "leiomyoma",  "endometriosis", "retinal_detach", "pth", "hfincidenceprevkoo",  "IHD_MI",  "stroketiaincidprevkoo", 
                "interstitiallungdiseasemm", "blindmm", 
                "copd",  "solidorgantransplant", "bronchiectasisdraftv1", "NeuroDev", "atopiceczema_mm", 
                "Cardiomyopathy", "cf", "sickle_cell", "PulmHtn", "ibs_mm",   "turnerssyndrome_imrd", "marfansyndrome_imrd", "HIVall",   
                "DiabAll", "DiabRetino", 
                "hypertension", "spina_bifida", "CongHeart",
                "Vertebrae",  "Thyroid",
                "prithrombocytopenia_imrd",
                "perniciousanaemia",  "coeliac", "AutoSkin", "InflamBowel", "InflamEye", "SpondArth", "psoriasis_mm",
                "osteoporosis", "chronicbackpain", "periph_neuro", "urolithiasis", "scoliosis", "cholelithiasis",
                "OthSkin", 
                "VTEall", "Valve",  "epilepsy_mm", 
                "oa", 
                "slesystemic2019", "ulcer_peptic",  "Ehler",  "ChrLiverAll", "CKDall", "InflamArth",
                "af", "haemophilia_imrd","iih", "ms", "Somatoform", "osafinal","deaf", 
                "cataract",]
    


def get_column_order_SAIL(plot=False):
    """
    Used for the SAIL top-level comparison.  72 conditions.
    HIV and female infertilty are removed as these are not available in SAIL.
    Get the list of conditions to filter, or return the names to be used to plot
    """

    if plot:
        return [ "Cancer", "Asthma", #"Female infertility",
                 "Allergic Rhin Conj",
                 "Migraine", "Anxiety", "Depression", "Substance misuse", "Alcohol problem", "Eating disorder",  "SMHmm",  "Other mental",  "Other headache",   
                 "AdrenalAll",
                 "Pituitary", "PCOS",  "Sarcoid",  "Leiomyoma",  "Endometriosis", "Retinal detachment", "PTH", "Heart failure",  "IHD/MI",  "Stroke", 
                 "Interstitial lung", "Blind", 
                 "COPD",  "Solid organ transplant", "Bronchiectasis", "Neuro development", "Atopic eczema", 
                 "Cardiomyopathy", "Cystic fybrosis", "Sickle cell", "Pulminary Heart", "IBS",   "Turners syndrome", "Marfan syndrome", #"HIV", 
                "Diabetes", "Diabetes (retino)", 
                 "Hypertension", "Spina bifida", "Congenital Heart",
                 "Vertebrae",  "Thyroid",
                 "Prithrombocytopenia",
                 "Pernicious anaemia",  "Coeliac", "Auto Skin", "Inflam Bowel", "Inflam Eye", "Spond Arth", "Psoriasis",
                 "Osteoporosis", "Chronic back pain", "Peripheral neuro", "Urolithiasis", "Scoliosis", "Cholelithiasis",
                 "Other skin", 
                 "VTE", "Valve",  "Epilepsy", 
                 "Osteoarthritis", 
                 "Slesystemic", "Ulcer peptic",  "Ehler",  "Chronic Liver", "Chronic Kidney", "Inflam Arth",
                 "Atrial fibrillation", "Haemophilia","IIH", "Multiple sclerosis", "Somatoform", "OSA","Deaf", 
                 "Cataract",]
    else:
        return ["CancerAll", "asthmalonglist2018", #"female_infertility",
                "AllergicRhinConj",
                "migraine", "AnxietyPTSDdiag",  "depressionDiag", "substance_misuse", "alcoholproblem", "eatingdisorderuom",  "SMHmm",  "OthMental",  "OthHeadache",   
                "AdrenalAll",
                "Pituitary","pcoskoo",  "sarcoid",  "leiomyoma",  "endometriosis", "retinal_detach", "pth", "hfincidenceprevkoo",  "IHD_MI",  "stroketiaincidprevkoo", 
                "interstitiallungdiseasemm", "blindmm", 
                "copd",  "solidorgantransplant", "bronchiectasisdraftv1", "NeuroDev", "atopiceczema_mm", 
                "Cardiomyopathy", "cf", "sickle_cell", "PulmHtn", "ibs_mm",   "turnerssyndrome_imrd", "marfansyndrome_imrd", #"HIVall",   
                "DiabAll", "DiabRetino", 
                "hypertension", "spina_bifida", "CongHeart",
                "Vertebrae",  "Thyroid",
                "prithrombocytopenia_imrd",
                "perniciousanaemia",  "coeliac", "AutoSkin", "InflamBowel", "InflamEye", "SpondArth", "psoriasis_mm",
                "osteoporosis", "chronicbackpain", "periph_neuro", "urolithiasis", "scoliosis", "cholelithiasis",
                "OthSkin", 
                "VTEall", "Valve",  "epilepsy_mm", 
                "oa", 
                "slesystemic2019", "ulcer_peptic",  "Ehler",  "ChrLiverAll", "CKDall", "InflamArth",
                "af", "haemophilia_imrd","iih", "ms", "Somatoform", "osafinal","deaf", 
                "cataract",]
    
    
def get_column_order_SAIL_uncommon(plot=False):
    """
    Used for the SAIL 2nd-level comparison. HIV and female infertilty are removed as these are not available in SAIL. 
    
    Common excluded conditions:
        - asthmalonglist2018
        - AllergicRhinConj
        - migraine
        - AnxietyPTSDdiag
        - depressionDiag
    
    Get the list of conditions to filter, or return the names to be used to plot the multi-morbidity diseases 
    """

    if plot:
        return [ "Cancer",  #"Female infertility",
                 "Substance misuse", "Alcohol problem", "Eating disorder",  "SMHmm",  "Other headache",   
                 "AdrenalAll",
                 "Pituitary", "PCOS",  "Sarcoid",  "Leiomyoma",  "Endometriosis", "Retinal detachment", "PTH", "Heart failure",  "IHD/MI",  "Stroke", 
                 "Interstitial lung", "Blind", 
                 "COPD",  "Solid organ transplant", "Bronchiectasis", "Neuro development", "Atopic eczema", 
                 "Cardiomyopathy", "Cystic fybrosis", "Sickle cell", "Pulminary Heart",  "Turners syndrome", "Marfan syndrome", #"HIV", 
                "Diabetes", "Diabetes (retino)", 
                 "Hypertension", "Spina bifida", "Congenital Heart",
                 "Vertebrae",  "Thyroid",
                 "Prithrombocytopenia",
                 "Pernicious anaemia",  "Coeliac", "Auto Skin", "Inflam Bowel", "Inflam Eye", "Spond Arth", "Psoriasis",
                 "Osteoporosis", "Chronic back pain", "Peripheral neuro", "Urolithiasis", "Scoliosis", "Cholelithiasis",
                 "Other skin", 
                 "VTE", "Valve",  "Epilepsy", 
                 "Osteoarthritis", 
                 "Slesystemic", "Ulcer peptic",  "Ehler",  "Chronic Liver", "Chronic Kidney", "Inflam Arth",
                 "Atrial fibrillation", "Haemophilia","IIH", "Multiple sclerosis", "Somatoform", "OSA","Deaf", 
                 "Cataract",]
    else:
        return ["CancerAll", #"female_infertility",
                "substance_misuse", "alcoholproblem", "eatingdisorderuom",  "SMHmm",  "OthHeadache",   
                "AdrenalAll",
                "Pituitary","pcoskoo",  "sarcoid",  "leiomyoma",  "endometriosis", "retinal_detach", "pth", "hfincidenceprevkoo",  "IHD_MI",  "stroketiaincidprevkoo", 
                "interstitiallungdiseasemm", "blindmm", 
                "copd",  "solidorgantransplant", "bronchiectasisdraftv1", "NeuroDev", "atopiceczema_mm", 
                "Cardiomyopathy", "cf", "sickle_cell", "PulmHtn",  "turnerssyndrome_imrd", "marfansyndrome_imrd", #"HIVall",   
                "DiabAll", "DiabRetino", 
                "hypertension", "spina_bifida", "CongHeart",
                "Vertebrae",  "Thyroid",
                "prithrombocytopenia_imrd",
                "perniciousanaemia",  "coeliac", "AutoSkin", "InflamBowel", "InflamEye", "SpondArth", "psoriasis_mm",
                "osteoporosis", "chronicbackpain", "periph_neuro", "urolithiasis", "scoliosis", "cholelithiasis",
                "OthSkin", 
                "VTEall", "Valve",  "epilepsy_mm", 
                "oa", 
                "slesystemic2019", "ulcer_peptic",  "Ehler",  "ChrLiverAll", "CKDall", "InflamArth",
                "af", "haemophilia_imrd","iih", "ms", "Somatoform", "osafinal","deaf", 
                "cataract",]

        
def filter_frame(data_frame, date_frame=None, verbose=0,
                 diseases=["asthmalonglist2018","hypertension","female_infertility"], sizes=None, remove_duplicates=False):
    """
    Reduce data (and optionally date frame) to one in which we know there is some expected structure. 
        Used in ML4H case study.
    """
    
    # Pre-defined filters
    #skin = (data_frame["atopiceczema_mm"] == 1) | (data_frame["OthSkin"] == 1)
    c_list = [(data_frame[d] == 1) for d in diseases]
 
    dfs = [data_frame[c] for c in c_list]
    for c, df_c in enumerate(dfs):
        df_c["label"] = (c+1) * np.ones_like(df_c.index.values)
        # df_c.loc["label"] = (c+1) * np.ones_like(df_c.index.values)
    
    # if verbose > 0:
    #     print(f"Before sub-sampling: \n\tCluster 1: {len(df1.index)} samples. \n\tCluster 2: {len(df2.index)} samples. \n\tCluster 3: {len(df3.index)} samples")
    
    if sizes is not None:
        dfs = [df_c.sample(n=sizes[c]) for c, df_c in enumerate(dfs)]
        if verbose > 0:
            print(f"After sub-sampling each sub-population size is: {[len(df.index) for df in dfs]}")

    # Combine
    df = pd.concat(dfs, ignore_index=False, axis=0)
    
    # Note some patients satisfy multiple conditions. In this case we exclude the duplicates by default
    if remove_duplicates is True:
        df = df[~df.index.duplicated(keep=False)]
    elif remove_duplicates is 'first':
        #  Alternatively we  include them, each with the different labels. 
        df = df[~df.index.duplicated(keep='first')]
        # TODO: keep only first and do something with overlapping labels
    else:
        pass
        
    if verbose > 0:
        print("After combining separate frames")
        label = df['label'].to_numpy()
        label_count = zip([sum(label == i) for i in np.unique(label)], np.unique(label))
        print([l for l in label_count])

    if date_frame is not None:
        date_frame = date_frame.loc[df.index]

    return df[get_column_order()], date_frame, df['label'].to_numpy()

    
def summarise_binary_profiles(binary_profiles, sort='counts', verbose=0):
    """
    Summarise the binary profiles obtained by mapping to the corners of the latent hyper-cube. 
        Index clusters according to desired `sort' scheme (for visualisation)
    """

    unique_profiles, cluster_allocations, counts = np.unique(binary_profiles, axis=0, return_inverse=True, return_counts=True)
    
    if sort == 'counts':                           # Return in order of largest cluster first
        reorder_inds = counts.argsort()[::-1]
    elif sort == 'divisive':                                         # Return in order of unique profile diversion. e.g. (0,0), (1, 0) (0, 1), (1, 1)
        reorder_inds = [i for i in range(len(counts))]
    else:
        raise NotImplementedError
        
    cluster_allocations2 = np.zeros_like(cluster_allocations)
    for i, e in enumerate(reorder_inds):
        cluster_allocations2[cluster_allocations == e] = i
    
    return counts[reorder_inds], unique_profiles[reorder_inds, :], cluster_allocations2


def similarity_matrix(allocations):
    """
    Calculate similarity matrix between seeds/ensembles
        note: For use in external R hierarchical clustering code.
    """
    num_samples = allocations.shape[0]
    num_draws = allocations.shape[1]
    
    similarity_matrix = np.eye(num_samples)
    for sample1 in range(num_samples):
        for sample2 in range(num_samples):
            if sample2 > sample1:
                pass
            else:
                p_ij = np.mean(allocations[sample1, :] == allocations[sample2, :])
                similarity_matrix[sample1, sample2] = p_ij
                similarity_matrix[sample2, sample1] = p_ij    
    return similarity_matrix


def process_clusters(Y, allocation, profiles, counts, _cutoff=0):
    """
    """
    assert len(counts) == profiles.shape[0]
    assert sum(counts) == Y.shape[0]
    _eps = 1e-12

    # Memory allocation, filtering out rare clusters for plotting
    n_over = sum(counts > _cutoff)
    prevalence = np.zeros((n_over, Y.shape[1]))
    RR = np.zeros((n_over, Y.shape[1]))
    RR_deviation = np.zeros((n_over, Y.shape[1]))
    OR = np.zeros((n_over, Y.shape[1]))
    OR_deviation = np.zeros((n_over, Y.shape[1]))
    CFA = np.zeros((n_over, profiles.shape[1]))                                                                                        # Cluster factor association matrix: The latent factors turned on in each cluster
    y_labels, counts_over = [], []

    for c_idx, cluster in enumerate([i for i in range(len(counts)) if counts[i] > _cutoff]):                                                    # For each cluster above cut-off size
        # Prevalence: The number of times the condition appears in the cluster
        prevalence[c_idx, :] = np.sum(Y[allocation == cluster + 1, :], axis=0)         
        
        a = np.sum(Y[allocation == cluster + 1, :], axis=0)                             # count with condition | in cluster 
        b = np.sum(1 - Y[allocation == cluster + 1, :], axis=0)                         # count without condition | in cluster
        c = np.sum(Y[allocation != cluster + 1, :], axis=0)                             # count with condition | not in cluster
        d = np.sum(1- Y[allocation != cluster + 1, :], axis=0)                          # count without condition | not in cluster
        
        # Relative risks:   
        RR[c_idx, :] = (a/(a+b))/(c/(c+d))
        RR_deviation[c_idx, :] = np.sqrt( 1/a - 1/(a+b) + 1/c - 1/(c+d) )
        assert np.all(RR[c_idx, :] == np.mean(Y[allocation == cluster + 1, :], axis=0) / np.mean(Y[allocation != cluster + 1, :], axis=0))
        
        # Odds ratio
        OR[c_idx, :] = (a*d)/(b*c)
        OR_deviation[c_idx, :] = np.sqrt(1/a + 1/b + 1/c + 1/d)
        
        # Cluster factor association matrix
        CFA[c_idx, :] = profiles[cluster, :]                                                                                                    # unique profiles above cut-off size  
        
        # Misc
        y_labels.append(f'{cluster + 1}') # (N={counts[cluster]})') #, topics {np.where(unique_profiles[cluster, :]!=0)[0] + 1}')             # Save plotting label for when we report the above metrics
        counts_over.append(counts[cluster])

    return prevalence, RR, RR_deviation, OR, OR_deviation, CFA, counts_over, y_labels


def process_factors(Y, z_binary):
        """
        """
        assert Y.shape[0] == z_binary.shape[0]
        _eps = 1e-12                                               # For numerical stability if a factor is empty or has no conditions associated

        L = z_binary.shape[1]
        
        # Memory allocation, filtering out rare clusters for plotting
        prevalence = np.zeros((L, Y.shape[1]))
        RR = np.zeros((L, Y.shape[1]))
        RR_deviation = np.zeros((L, Y.shape[1]))
        OR = np.zeros((L, Y.shape[1]))
        OR_deviation = np.zeros((L, Y.shape[1]))
        y_labels, counts = [], []
        
        for factor in range(L):                                                                                         
            have_factor = np.where(np.array(z_binary[:, factor]) == 1)[0]
            no_factor = np.where(np.array(z_binary[:, factor]) != 1)[0]
            assert len(have_factor) + len(no_factor) == Y.shape[0]
            
            # Prevalence: The number of times the condition appears in the factor
            prevalence[factor, :] = np.sum(Y[have_factor, :], axis=0)         
            
            a = np.sum(Y[have_factor, :], axis=0)                             # count with condition | in factor 
            b = np.sum(1 - Y[have_factor, :], axis=0)                         # count without condition | in factor
            c = np.sum(Y[no_factor, :], axis=0)                               # count with condition | not in factor
            d = np.sum(1- Y[no_factor, :], axis=0)                            # count without condition | not in factor
            
            # Relative risks:  
            if len(have_factor) > 0:
                # If set of patients with latent factor is non-empty
                RR[factor, :] = (a/(a+b))/(c/(c+d))
                RR_deviation[factor, :] = np.sqrt( 1/a - 1/(a+b) + 1/c - 1/(c+d) )
                assert np.all(np.isclose(RR[factor, :], np.mean(Y[have_factor, :], axis=0) / np.mean(Y[no_factor, :], axis=0)))
            else:
                RR[factor, :] = np.nan
                RR_deviation[factor, :] = np.nan
            
            # Odds ratio
            OR[factor, :] = (a*d)/(b*c)
            OR_deviation[factor, :] = np.sqrt(1/a + 1/b + 1/c + 1/d)
            
            y_labels.append(f'{factor + 1}')                                                                                             # Save plotting label for when we report the above metrics
            counts.append(len(have_factor))

        return prevalence, RR, RR_deviation, OR, OR_deviation, counts, y_labels



def post_process(diag_frame, output_dictionary, save_path=None):
    """
    After training our mmVAE model, process the output ready for plotting
    
    Note - patient identifiers are never inputted to this package, and so downstream analysis will need to cross reference any saved files against pre-saved identifiers.
    """
    Y = diag_frame.to_numpy()
        
    ###############################
    # ========= CLUSTERS ==========
    ###############################
    prevalence, RR, RR_dev, OR, OR_dev, CFA, counts, y_labels = process_clusters(Y,
                                                                                 output_dictionary['cluster_allocations'],
                                                                                 output_dictionary['unique_profiles'],
                                                                                 output_dictionary['counts'],
                                                                                )
    output_dictionary['prevalence_clusters'] = prevalence
    output_dictionary['RR_clusters'] = RR
    output_dictionary['RR_deviation_clusters'] = RR_dev
    output_dictionary['OR_clusters'] = OR
    output_dictionary['OR_deviation_clusters'] = OR_dev
    output_dictionary['cluster_factors'] = CFA
    output_dictionary['count_clusters'] = counts
    output_dictionary['cluster_labels'] = y_labels
                                                                 
        
    ###############################
    # ========== FACTORS ===========
    ###############################   
    prevalence, RR, RR_dev, OR, OR_dev, counts, y_labels = process_factors(Y, output_dictionary['z_binary'])
    
    output_dictionary['prevalence_topics'] = prevalence
    output_dictionary['RR_topics'] = RR
    output_dictionary['RR_deviation_topics'] = RR_dev
    output_dictionary['OR_topics'] = OR
    output_dictionary['OR_deviation_topics'] = OR_dev
    output_dictionary['count_topics'] = counts
    output_dictionary['topic_labels'] = y_labels

    
    ###############################
    # =========== SAVE ============
    ###############################       
    if save_path is not None:
        df = pd.DataFrame(output_dictionary['cluster_allocations'])
        df.columns = ["Cluster"]
        df.index.name = "Sample index"
        df.to_excel(save_path + "cluster_allocations.xlsx")
        
        df = pd.DataFrame(output_dictionary['unique_profiles'])
        df.columns = [f'Latent Factor {i+1}' for i in range(output_dictionary['unique_profiles'].shape[1])]
        df.index.name = "Cluster index"
        df.to_excel(save_path + "cluster_factor_association.xlsx")
        
        for key in ["RR_clusters", "RR_deviation_clusters", "OR_clusters", "OR_deviation_clusters"]:
            df= pd.DataFrame(output_dictionary[key])
            df.columns = diag_frame.columns
            df.index.name = "Cluster index"
            df.to_excel(save_path + f"{key}.xlsx")
        
        for key in ["RR_topics", "RR_deviation_topics", "OR_topics", "OR_deviation_topics"]:
            df= pd.DataFrame(output_dictionary[key])
            df.columns = diag_frame.columns
            df.index.name = "Factor index"
            df.to_excel(save_path + f"{key}.xlsx")
            
        df = pd.DataFrame(output_dictionary['prevalence_clusters'])
        df.columns = diag_frame.columns
        df.index.name = "Cluster index"
        df.to_excel(save_path + "prevalence_clusters.xlsx")
        
        df = pd.DataFrame(output_dictionary['prevalence_topics'])
        df.columns = diag_frame.columns
        df.index.name = "Factor index"
        df.to_excel(save_path + "prevalence_factors.xlsx")
        
        df = pd.DataFrame(output_dictionary['count_clusters'])
        df.columns = ["Patients in cluster"]
        df.index.name = "Cluster index"
        df.to_excel(save_path + "count_clusters.xlsx")
        
        df = pd.DataFrame(output_dictionary['count_topics'])
        df.columns = ["Patients in factor"]
        df.index.name = "Factor index"
        df.to_excel(save_path + "count_factors.xlsx")

    
        
    return output_dictionary


# extra date helper functions
# def get_acquired_order(date_frame):
#     """
#     Apply filter to date frame to extract the order in which diseases were acquired
#     """
#     def apply_order(row):
#         acquired_order = pd.to_datetime(row[['Date_' + d for d in get_column_order()]].dropna(), format='%d/%m/%Y').sort_values().index.tolist()
#         acquired_order = [d[5:] for d in acquired_order]
#         return acquired_order
    
#     acquired_order = date_frame.apply(apply_order, axis=1).tolist()
#     date_frame['acquired_order']  = acquired_order 
#     date_frame['first_acquired']  = [d_s[0] for d_s in acquired_order]

#     return date_frame
