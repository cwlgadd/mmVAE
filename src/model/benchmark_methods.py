from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis, NMF, LatentDirichletAllocation
from model.helpers import summarise_binary_profiles

# Clustering methods
def fit_kmeans(Y, L):
    # K means clustering
    kmeans = KMeans(n_clusters=L, random_state=0).fit(Y)
    cluster_allocations = kmeans.labels_
    if np.min(cluster_allocations) == 0:
        cluster_allocations += 1
    return cluster_allocations


def fit_LCA(Y, L):
    raise NotImplementerError("Run this through separate R script")


# Quantised dimensionality reduction methods
def fit_FA(Y, L=2):
    # Factor Analysis
    Z = FactorAnalysis(n_components=L, random_state=0).fit_transform(Y)
    Z_binary = 1 * (Z > 0.5)
    counts, unique_profiles, cluster_allocations = summarise_binary_profiles(Z_binary)
    if np.min(cluster_allocations) == 0:
        cluster_allocations += 1
    return cluster_allocations

def fit_NMF(Y, L=5):
    # Non-negative matrix factorisation
    model = NMF(n_components=L, init='random', random_state=0)
    W = model.fit_transform(Y)
    H = model.components_
    Z_binary = 1 * (W > 0.5)
    counts, unique_profiles, cluster_allocations = summarise_binary_profiles(Z_binary)
    if np.min(cluster_allocations) == 0:
        cluster_allocations += 1
    return cluster_allocations

def fit_LDA(Y, L=5)
    # Latent Dirichlet Allocation
        # This does not scale to the full dataset
    Z = LatentDirichletAllocation(n_components=L, random_state=0).fit_transform(Y)
    Z_binary = 1 * (Z > 0.5)
    counts, unique_profiles, cluster_allocations = summarise_binary_profiles(Z_binary)
    if np.min(cluster_allocations) == 0:
        cluster_allocations += 1
    return cluster_allocations
