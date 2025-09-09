"""
Module to compute distances between clinical notes
Two main functions :
distance_files_by_lab : compute a distance between two clinical notes on one axes
"""

from timeit import default_timer as timer

import numpy as np
import ot
import sklearn
import torch
from sklearn.metrics import euclidean_distances


def distance_files_by_lab(
    sparse_vector1,
    sparse_vector2,
    embedding_matrix=None,
    distance_matrix=None,
    distance_type="cosine",
    use_cuda=False,
    track_time=False,
    verbose=False,
):
    """
    Computes a distance between two clinical notes, based on the embedding of their extracted terms, based on EMD distance
        Parameters
        ----------
        sparse_vector1: scipy csr_matrix containing terms present in doc1 (coming from DictVectorizer)
        sparse_vector2: scipy csr_matrix containing terms present in doc2 (coming from DictVectorizer)
        embedding_matrix: numpy or torch matrix of all terms embeddings (numpy or torch).
            Can be None if distance_matrix is not None
        distance_type: str
                     only if distance_matrix is None, should be "cosine" ou "euclid"
        distance_matrix: numpy or torch matrix
            If specified, contains the distance between all pairs of terms.
            Otherwise the distances will have to be computed in this function.
            If None, distance_type must be specified
        use_cuda: bool
        verbose : bool
            If True : print detailed informations

        Returns
        -------
        Distance between both patients, based on the specified terms

    """
    # Time tracking (to be removed)
    if track_time:
        time_spent = {"EMD": 0, "dist_matrix": 0, "vectors": 0}
    # Terms to consider are those at least in one vector
    # print('sparse_vector1', type(sparse_vector1), sparse_vector1.shape)

    if track_time:
        start = timer()

    # Get indices of terms that appear in at least one vector (document)
    all_term_indices = list(
        set(sparse_vector1.indices.tolist() + sparse_vector2.indices.tolist())
    )
    # Build a one-hot dense vector with only those terms, for each document
    # See documentation of csr_matrix, but basically the idea is NOT to convert the sparse vector
    # into a dense vector by the to_array() method, which produces a vector of the size of the entire vocabulary
    # and takes time to process. We then collect the indices and data from the csr_matrix object
    # and build the dense vectors manually from these information.
    sparse_vector1_data_dict = {
        i: v for i, v in zip(sparse_vector1.indices, sparse_vector1.data)
    }
    sparse_vector2_data_dict = {
        i: v for i, v in zip(sparse_vector2.indices, sparse_vector2.data)
    }
    dense_vector1 = np.array(
        [sparse_vector1_data_dict.get(i, 0) for i in all_term_indices]
    )
    dense_vector2 = np.array(
        [sparse_vector2_data_dict.get(i, 0) for i in all_term_indices]
    )
    # dense_vector1, dense_vector2: 1-D vector with the occurrence number of each terms, reduced to only
    #   the terms appearing in at least one document
    # Normalize
    dense_vector1 /= dense_vector1.sum()
    dense_vector2 /= dense_vector2.sum()

    if use_cuda:
        dense_vector1 = torch.from_numpy(dense_vector1).cuda()
        dense_vector2 = torch.from_numpy(dense_vector2).cuda()

    if track_time:
        time_spent["vectors"] += timer() - start  # type: ignore

    if track_time:
        start = timer()
    # If the global embedding distance matrix is specified,
    # reduce it to only the terms present in the documents
    if distance_matrix is not None:
        local_dist_matrix = distance_matrix[all_term_indices][:, all_term_indices]
    # Otherwise, build it from the embedding matrix
    else:
        # Get embedding submatrix (only terms from vectors)
        embedding_submatrix = embedding_matrix[all_term_indices]  # type: ignore
        # Build similarity matrix from vocabulary embeddings
        if distance_type == "cosine":
            local_dist_matrix = sklearn.metrics.pairwise.cosine_distances(  # type: ignore
                embedding_submatrix
            )  # cosine_distance : ie 1-cos similarity = 1-cos(theta)
        elif distance_type == "euclid":
            if use_cuda:
                raise NotImplementedError()
            else:
                local_dist_matrix = euclidean_distances(embedding_submatrix)
        else:
            raise ValueError()

    # Normalize embedding distance matrix
    if local_dist_matrix.max() != 0:
        local_dist_matrix /= local_dist_matrix.max()  # just for comparison purposes

    if track_time:
        time_spent["dist_matrix"] += timer() - start  # type: ignore

    if track_time:
        start = timer()

    # Compute EMD/Wasserstein distance between the vectors, based on
    # the computed distance matrix.
    # Not clear whether CUDA can really make it faster, as the matrices
    # are pretty small at this point.
    # Efforts to use an approximation of Wasserstein distance (geomloss, sinkhorn)
    # have been unsuccessful but quickly abandoned, I must admit
    # (errors from the loss function).
    emd_distance = ot.emd2(dense_vector1, dense_vector2, local_dist_matrix)

    if track_time:
        time_spent["EMD"] += timer() - start  # type: ignore

    if verbose:
        print(f"\nDistance {distance_type}", emd_distance)

    if track_time:
        return emd_distance, time_spent  # type: ignore
    else:
        return emd_distance
