import numpy

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Measures the cosine of the angle between two vectors, providing a scale-invariant
    similarity metric. Best suited for L2-normalized vectors.
    
    Parameters
    ----------
    a : numpy.ndarray
        First input vector of shape (n,)
    b : numpy.ndarray
        Second input vector of shape (n,)
    
    Returns
    -------
    float
        Cosine similarity in range [-1, 1], where 1 indicates identical direction,
        0 indicates orthogonality, and -1 indicates opposite direction.
    
    Notes
    -----
    This metric is scale-invariant and works well with embeddings that are already
    normalized or when vector magnitude is not meaningful.
    
    Examples
    --------
    >>> a = numpy.array([1, 2, 3])
    >>> b = numpy.array([4, 5, 6])
    >>> cosine_similarity(a, b)
    0.9746318461970762
    """
    dot_product = numpy.dot(a, b)
    norm_a = numpy.linalg.norm(a)
    norm_b = numpy.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def dot_product_similarity(a, b):
    """
    Compute dot product similarity between two vectors.
    
    Calculates the inner product of two vectors. Particularly efficient for
    transformer embeddings and GPU acceleration.
    
    Parameters
    ----------
    a : numpy.ndarray
        First input vector of shape (n,)
    b : numpy.ndarray
        Second input vector of shape (n,)
    
    Returns
    -------
    float
        Dot product value. Higher values indicate greater similarity.
        Range depends on vector magnitudes.
    
    Notes
    -----
    This is the fastest similarity metric and works exceptionally well when vectors
    are pre-normalized (L2 norm = 1). For normalized vectors, this is equivalent
    to cosine similarity but much faster.
    
    Examples
    --------
    >>> a = numpy.array([1, 2, 3])
    >>> b = numpy.array([4, 5, 6])
    >>> dot_product_similarity(a, b)
    32
    """
    return numpy.dot(a, b)


def euclidean_distance(a, b):
    """
    Compute Euclidean (L2) distance between two vectors.
    
    Calculates the straight-line distance between two points in Euclidean space.
    Best for raw vectors where magnitude carries meaningful information.
    
    Parameters
    ----------
    a : numpy.ndarray
        First input vector of shape (n,)
    b : numpy.ndarray
        Second input vector of shape (n,)
    
    Returns
    -------
    float
        Euclidean distance. Lower values indicate greater similarity.
        Range: [0, inf), where 0 means identical vectors.
    
    Notes
    -----
    This metric is sensitive to vector scale/magnitude. Not recommended for
    normalized embeddings where only direction matters.
    
    Examples
    --------
    >>> a = numpy.array([1, 2, 3])
    >>> b = numpy.array([4, 5, 6])
    >>> euclidean_distance(a, b)
    5.196152422706632
    """
    return numpy.linalg.norm(a - b)


def manhattan_distance(a, b):
    """
    Compute Manhattan (L1) distance between two vectors.
    
    Calculates the sum of absolute differences between vector components.
    Also known as taxicab or city block distance.
    
    Parameters
    ----------
    a : numpy.ndarray
        First input vector of shape (n,)
    b : numpy.ndarray
        Second input vector of shape (n,)
    
    Returns
    -------
    float
        Manhattan distance. Lower values indicate greater similarity.
        Range: [0, inf), where 0 means identical vectors.
    
    Notes
    -----
    Less sensitive to outliers than Euclidean distance. Useful when differences
    in individual dimensions should be weighted equally.
    
    Examples
    --------
    >>> a = numpy.array([1, 2, 3])
    >>> b = numpy.array([4, 5, 6])
    >>> manhattan_distance(a, b)
    9
    """
    return numpy.sum(numpy.abs(a - b))


def jaccard_similarity(a, b):
    """
    Compute Jaccard similarity coefficient between two vectors.
    
    Measures similarity between sets by dividing intersection size by union size.
    Best suited for sparse binary or set-based data.
    
    Parameters
    ----------
    a : numpy.ndarray
        First input vector of shape (n,), typically binary (0s and 1s)
    b : numpy.ndarray
        Second input vector of shape (n,), typically binary (0s and 1s)
    
    Returns
    -------
    float
        Jaccard similarity in range [0, 1], where 1 indicates identical sets
        and 0 indicates no overlap.
    
    Notes
    -----
    Works with binary vectors by treating non-zero values as set membership.
    Handles edge case where both vectors are all zeros (returns 0.0).
    
    Examples
    --------
    >>> a = numpy.array([1, 0, 1, 0, 1])
    >>> b = numpy.array([1, 1, 0, 0, 1])
    >>> jaccard_similarity(a, b)
    0.5
    """
    # Convert to boolean for set operations
    a_bool = a != 0
    b_bool = b != 0
    
    intersection = numpy.sum(a_bool & b_bool)
    union = numpy.sum(a_bool | b_bool)
    
    # Handle edge case where both vectors are all zeros
    if union == 0:
        return 0.0
    
    return intersection / union


def soft_cosine_similarity(a, b, similarity_matrix):
    """
    Compute soft cosine similarity between two sparse vectors.
    
    Extends cosine similarity by incorporating semantic relationships between
    features through a similarity matrix. Better for text with term overlap.
    
    Parameters
    ----------
    a : numpy.ndarray
        First input vector of shape (n,)
    b : numpy.ndarray
        Second input vector of shape (n,)
    similarity_matrix : numpy.ndarray
        Feature similarity matrix of shape (n, n) where element (i, j)
        represents similarity between feature i and feature j.
        Should be symmetric with 1s on diagonal.
    
    Returns
    -------
    float
        Soft cosine similarity in range [0, 1] for typical use cases.
        Higher values indicate greater similarity accounting for feature relationships.
    
    Notes
    -----
    The similarity matrix captures relationships between features (e.g., word embeddings
    for text terms). When similarity_matrix is identity, this reduces to standard
    cosine similarity. Computationally more expensive than standard cosine.
    
    Examples
    --------
    >>> a = numpy.array([1, 2, 0])
    >>> b = numpy.array([0, 1, 3])
    >>> # Identity matrix: standard cosine
    >>> sim_matrix = numpy.eye(3)
    >>> soft_cosine_similarity(a, b, sim_matrix)
    0.25819888974716115
    """
    numerator = numpy.dot(a, numpy.dot(similarity_matrix, b))
    
    denominator_a = numpy.sqrt(numpy.dot(a, numpy.dot(similarity_matrix, a)))
    denominator_b = numpy.sqrt(numpy.dot(b, numpy.dot(similarity_matrix, b)))
    
    # Handle edge case where denominator is zero
    if denominator_a == 0 or denominator_b == 0:
        return 0.0
    
    return numerator / (denominator_a * denominator_b)


def learned_similarity(a, b, model):
    """
    Compute learned similarity using a trained model.
    
    Uses a fine-tuned deep learning model to compute similarity. Provides best
    accuracy but slowest retrieval time.
    
    Parameters
    ----------
    a : numpy.ndarray
        First input vector of shape (n,)
    b : numpy.ndarray
        Second input vector of shape (n,)
    model : callable
        Trained model/function that takes two vectors and returns similarity score.
        Should accept inputs of shape (batch_size, n) and return scalar or
        array of similarity scores.
    
    Returns
    -------
    float
        Similarity score as determined by the model. Range and interpretation
        depend on model training.
    
    Notes
    -----
    This is a wrapper for neural network-based similarity models. The model
    should be pre-trained on domain-specific data for best results. Significantly
    slower than mathematical metrics but can capture complex non-linear relationships.
    
    Examples
    --------
    >>> def simple_model(x, y):
    ...     # Example: simple weighted dot product
    ...     weights = numpy.array([0.5, 1.0, 1.5])
    ...     return numpy.dot(x * weights, y * weights)
    >>> a = numpy.array([1, 2, 3])
    >>> b = numpy.array([4, 5, 6])
    >>> learned_similarity(a, b, simple_model)
    71.0
    """
    return model(a, b)


def distance_to_similarity(distance, method='inverse'):
    """
    Convert distance metric to similarity score.

    Utility function to convert distances to similarities
    Transforms distance values (where lower is better) to similarity scores
    (where higher is better).
    
    Parameters
    ----------
    distance : float or numpy.ndarray
        Distance value(s) to convert
    method : {'inverse', 'negative', 'gaussian'}, optional
        Conversion method:
        - 'inverse': similarity = 1 / (1 + distance)
        - 'negative': similarity = -distance
        - 'gaussian': similarity = exp(-distance^2)
        Default is 'inverse'.
    
    Returns
    -------
    float or numpy.ndarray
        Similarity score(s)
    
    Examples
    --------
    >>> distance_to_similarity(5.0, method='inverse')
    0.16666666666666666
    >>> distance_to_similarity(2.0, method='gaussian')
    0.01831563888873418
    """
    if method == 'inverse':
        return 1.0 / (1.0 + distance)
    elif method == 'negative':
        return -distance
    elif method == 'gaussian':
        return numpy.exp(-distance ** 2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse', 'negative', or 'gaussian'")