import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import plotly.express as px
from tqdm import tqdm
from scipy.linalg import eigh, svd, logm, expm, pinv
from scipy.sparse import eye
from sklearn.cluster import KMeans
import warnings
import networkx as nx
import community.community_louvain as community_louvain
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)


def sym_pos_def_dist(A, B, p=2):
    eig = np.linalg.eigvals(np.linalg.inv(A) @ B)
    if p == 1:
        dist = np.sum(np.abs(np.log(eig)))
    else:
        dist = np.sum(np.abs(np.log(eig)) ** p) ** (1 / p)
    return dist


def sym_pos_semi_def_dist(A, B, r, k=1):
    sym = lambda M: (M + M.T) / 2
    A = sym(A)
    B = sym(B)

    eig_A, vec_A = np.linalg.eig(A)
    eig_B, vec_B = np.linalg.eig(B)

    # keep the eigenvectors of the r largest eigenvalues
    vec_A = vec_A[:, np.argsort(eig_A)[-r:]]
    vec_B = vec_B[:, np.argsort(eig_B)[-r:]]

    # numpy returns V transposed compared to matlab
    try:
        OA, S, OB = np.linalg.svd(vec_A.T @ vec_B)
    except:
        OA, S, OB = svd(vec_A.T @ vec_B, lapack_driver='gesvd')
    if np.any(abs(S) > 1.):
        if not np.allclose(abs(S[abs(S) > 1.]), 1.):
            print(f"SVD yields S {S[abs(S) > 1.]}")
        S[S > 1.] = 1.
        S[S < -1.] = -1.
    vTheta = np.arccos(S)
    UA = vec_A @ OA
    UB = vec_B @ OB.T
    RA = sym(UA.T @ A @ UA)
    RB = sym(UB.T @ B @ UB)
    dU = np.linalg.norm(vTheta)
    dR = sym_pos_def_dist(RA, RB)
    d = np.sqrt(dU ** 2 + k * dR ** 2)
    return d


def generate_symmetric_PD_matrices_with_time_dependency_blocks_cutoff_freq(num_matrices, size, f_0, f_1,
                                                               block1=[0, 3], block2=[6, 6], noise_amplitude=0.1):
    matrices = []
    time_values = np.linspace(0, 1, num_matrices)  # Time values for sine function
    t = 0
    # Generate random phases for each pair in each block once, at the start
    phase_block1 = {}
    phase_block2 = {}

    # Create random phases for each pair in block1
    block1_indices = range(block1[0], block1[1])
    for i in block1_indices:
        for j in range(i + 1, block1[1]):
            phase_block1[(i, j)] = 0

    # Create random phases for each pair in block2
    block2_indices = range(block2[0], block2[1])
    for i in block2_indices:
        for j in range(i + 1, block2[1]):
            phase_block2[(i, j)] = 0 #np.random.uniform(0, 2 * np.pi)

    with tqdm(total=num_matrices, desc="Generating Matrices") as pbar:
        while len(matrices) < num_matrices:
            # Create a random symmetric matrix
            A = np.random.uniform(-1, 1, (size, size)) * noise_amplitude

            # Apply time-dependent modifications with consistent phases for each pair in block1
            for i in block1_indices:
                for j in range(i + 1, block1[1]):
                    phase = phase_block1[(i, j)]  # Use pre-set phase for this pair

                    A[i, j] = A[j, i] = np.sin(2 * np.pi * time_values[t] * f_0 + phase) * 0.5 + A[i, j]
            A = (A + A.T) / 2  # Ensure the matrix is symmetric
            np.fill_diagonal(A, 1)  # Fill diagonal with 1

            # Ensure full rank (Positive Definiteness)
            if np.all(np.linalg.eigvals(A) >= 0):  # All eigenvalues >= 0
                matrices.append(A)
                t += 1
                pbar.update(1)

    return matrices


def _riemannian_dist(corrs, eigval_bound=0.01):
    # r: smallest rank
    r = np.min(np.sum(np.linalg.eigvals(corrs) > eigval_bound, axis=1))

    dR = np.zeros((len(corrs), len(corrs)))
    for i, corr_i in enumerate(corrs):
        for j, corr_j in enumerate(corrs[i + 1:]):
            dR[i + j + 1, i] = sym_pos_semi_def_dist(corr_i, corr_j, r)
            dR[i, i + j + 1] = dR[i + j + 1, i]
    return dR


def _get_kernel_riemannian(all_distances, sigma_cutoff=9999999, eps=2):
    closest_distances = np.sort(all_distances)[:, :sigma_cutoff]
    sigma = eps * np.median(closest_distances)
    kernel = np.exp(- (all_distances / (np.sqrt(2) * sigma)) ** 2)
    kernel = (kernel + kernel.T) / 2
    kernel = _make_row_stochastic(kernel)
    return kernel


def _make_row_stochastic(kernel):
    'This does not strictly mean row-stochastic,but normalizes the kernel'
    column_sum = np.sum(kernel, axis=0)
    row_stochastic_kernel = np.einsum("i, j, ij -> ij",
                                      1 / np.sqrt(column_sum),
                                      1 / np.sqrt(column_sum),
                                      kernel)
    return row_stochastic_kernel


def _regularize_by_median_sv(correlations, signal, window_length=0,
                             subsampling=0):
    if subsampling > 0:
        length = signal.shape[-1]
        midpoints = np.linspace(window_length // 2,
                                length - window_length // 2,
                                subsampling)
        midpoints = list(map(int, midpoints))
        u, s, v = np.linalg.svd(signal[..., midpoints])
    else:
        u, s, v = np.linalg.svd(signal)
    regularizer = np.median(s, axis=1)[:, None, None] * np.eye(
        correlations.shape[-1])[None, :, :]
    return correlations + regularizer[:, None, :, :]


def _regularize_by_smallest_ev(correlations, eps=1e-3):
    eig = np.linalg.eigvals(correlations)
    for idx, e in enumerate(eig):
        if any(e.flatten() < 0):
            eps = eps - np.min(e.flatten())
            regularizer = eps * np.eye(correlations.shape[-1])
            correlations[idx] = regularizer[None, :, :] + correlations[idx]
    return correlations


def _get_kernel_euclidean(X, scale_k):
    neighbors = NearestNeighbors(n_neighbors=scale_k, algorithm='auto').fit(X)
    nearest_distances, indices = neighbors.kneighbors(X)
    sigma = np.median(nearest_distances, axis=1)
    nonvanishing_entries = np.exp(- (nearest_distances / sigma[:, None]) ** 2)
    kernel = np.zeros(shape=(len(X), len(X)))
    for i in range(len(X)):
        kernel[i, indices[i, :]] = nonvanishing_entries[i, :]
    kernel = (kernel + kernel.T) / 2
    kernel = _make_row_stochastic(kernel)
    return kernel, nearest_distances


def get_diffusion_embedding(correlations, window_length, scale_k=20,
                            signal=None, subsampling=0, mode='riemannian'):
    """
    :param
    correlations: (Bx)KxNxN K correlation matrices. Will be carried out
    over all first dimensions
    :param
    scale_k: number of nearest neighbors to use for evaluating the scale
    :param
    tol: tolerance when iteration to get Riemannian mean converged
    :param
    maxiter: when to stop Riemannian mean algorithm
    :param
    vector_input: to use diffusion embedding onto vectors, not correlation
    matrices. Only used to test functionality. Use with care.
    :return:
    """

    if ((correlations.ndim == 3 and mode != 'vector_input') or
            (correlations.ndim == 2 and mode == 'vector_input')):
        correlations = np.array([correlations])
    elif ((correlations.ndim == 4 and mode != 'vector_input') or
          (correlations.ndim == 3 and mode == 'vector_input')):
        pass
    else:
        raise ValueError(f"correlations must be shape (Bx)KxNxN but is "
                         f"{correlations.shape}")

    if window_length < correlations.shape[-1] and mode != 'vector_input':
        warnings.warn("Small window_length. Regularizing correlations.")
        if signal is not None:
            if subsampling > 0:
                correlations = _regularize_by_median_sv(
                    correlations, signal, window_length, subsampling)
            else:
                correlations = _regularize_by_median_sv(correlations, signal)
        else:
            correlations = _regularize_by_smallest_ev(correlations)

    distances = []
    diffusion_representations = []
    for corrs in correlations:
        if mode == 'riemannian':
            dists = _riemannian_dist(corrs)
            distances.append(dists)
            kernel = _get_kernel_riemannian(dists)
        elif mode == 'euclidean':
            kernel, dists = _get_kernel_euclidean(
                corrs.reshape(corrs.shape[:-2] + (-1,)), scale_k)
            distances.append(dists)
        else:
            raise ValueError(f'{mode=}')

        eig, vec = np.linalg.eigh(kernel)
        sort_idx = eig.argsort()[-2::-1]
        vec = vec.T[sort_idx]
        vec = eig[sort_idx, None] * vec

        diffusion_representations.append(vec)

    return np.array(diffusion_representations), np.array(distances)


def safe_corr(x, y):
    # Ensure inputs are numpy arrays
    x, y = np.asarray(x), np.asarray(y)

    # Check for constant vectors (std = 0) and prevent division by zero
    if np.std(x) == 0 or np.std(y) == 0:
        return 0  # Correlation is undefined, return 0 or NaN

    # Compute correlation
    corr_matrix = np.corrcoef(x, y)

    return corr_matrix[0, 1]


def get_corr_matrix(matrix):
    num_traces = len(matrix)
    corr_matrix = np.zeros((num_traces, num_traces))
    for i in range(num_traces):
        for j in range(num_traces):
            if i == j:
                corr_matrix[i, j] = 1
            else:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")  # Catch all warnings
                    corr_value = np.corrcoef(matrix[i], matrix[j])[0, 1]

                    if w:  # If a warning was caught, use safe_corr instead
                        corr_value = safe_corr(matrix[i], matrix[j])

                corr_matrix[i, j] = corr_value
    return corr_matrix


def matrix_power_adj(A, p):
    eigvals, eigvecs = np.linalg.eigh(A)

    # Set a threshold for small negative eigenvalues (e.g., 1e-20)
    threshold = 0
    assert np.min(eigvals) > -1e-20, f"negative eigenvalue found {np.min(eigvals)}"
    eigvals = np.maximum(eigvals, threshold)  # Replace negative eigenvalues with zero
    # Raise the eigenvalues to the power of p
    eigvals_p = eigvals ** p  # Eigenvalues raised to the power of p
    # Reconstruct the matrix A^p
    A_inv_p = eigvecs @ np.diag(eigvals_p) @ eigvecs.T  # A^p

    return A_inv_p


def generate_symmetric_PD_matrices_with_time_dependency(num_matrices, size, f_0, f_1):
    matrices = []
    time_values = np.linspace(0, 1, num_matrices)  # Time values for sine function
    t = 0
    while len(matrices) < num_matrices:
        # Create a random symmetric matrix
        A = np.random.uniform(-1, 1, (size, size)) * 0.05
        A = (A + A.T) / 2  # Make the matrix symmetric
        np.fill_diagonal(A, 1)  # Set the diagonal to ones

        # Add time dependency to one off-diagonal element
        A[0, 1] = A[1, 0] = np.sin((2*np.pi*time_values[t]*f_0))*0.5
        A[2, 5] = A[5, 2] = np.cos((2*np.pi*time_values[t]*f_1))*0.5

        # Ensure full rank
        if np.all(np.linalg.eigvals(A) > 0):  # All eigenvalues > 0
            matrices.append(A)
            t += 1

    return matrices


def generate_symmetric_PD_matrices_with_time_dependency_blocks(num_matrices, size, f_0, f_1,
                                                               block1=[0, 3], block2=[6, 9], noise_amplitude=0.1):
    matrices = []
    time_values = np.linspace(0, 1, num_matrices)  # Time values for sine function
    t = 0
    half_matrices = num_matrices // 2  # Halfway point

    # Generate random phases for each pair in each block once, at the start
    phase_block1 = {}
    phase_block2 = {}

    # Create random phases for each pair in block1
    block1_indices = range(block1[0], block1[1])
    for i in block1_indices:
        for j in range(i + 1, block1[1]):
            phase_block1[(i, j)] = 0 #np.random.uniform(0, 2 * np.pi)

    # Create random phases for each pair in block2
    block2_indices = range(block2[0], block2[1])
    for i in block2_indices:
        for j in range(i + 1, block2[1]):
            phase_block2[(i, j)] = 0 #np.random.uniform(0, 2 * np.pi)

    with tqdm(total=num_matrices, desc="Generating Matrices") as pbar:
        while len(matrices) < num_matrices:
            # Create a random symmetric matrix
            A = np.random.uniform(-1, 1, (size, size)) * noise_amplitude

            # Apply time-dependent modifications with consistent phases for each pair in block1
            if t < half_matrices:
                for i in block1_indices:
                    for j in range(i + 1, block1[1]):
                        phase = phase_block1[(i, j)]  # Use pre-set phase for this pair
                        A[i, j] = A[j, i] = np.sin(2 * np.pi * time_values[t] * f_0 + phase) * 0.5 + A[i, j]
            else:
                # Apply time-dependent modifications with consistent phases for each pair in block2
                for i in block2_indices:
                    for j in range(i + 1, block2[1]):
                        phase = phase_block2[(i, j)]  # Use pre-set phase for this pair
                        A[i, j] = A[j, i] = np.sin(2 * np.pi * time_values[t] * f_1 + phase) * 0.5 + A[i, j]

            A = (A + A.T) / 2  # Ensure the matrix is symmetric
            np.fill_diagonal(A, 1)  # Fill diagonal with 1

            # Ensure full rank (Positive Definiteness)
            if np.all(np.linalg.eigvals(A) >= 0):  # All eigenvalues >= 0
                matrices.append(A)
                t += 1
                pbar.update(1)

    return matrices


def filter(w1, w2, p=0.5):
    # Step 1: Compute fractional powers of w1
    w1_half = fractional_matrix_power(w1, 0.5)
    w1_neg_half = np.linalg.inv(w1_half)
    # Step 2: Compute the matrix inside the power operation
    inner_matrix = w1_neg_half @ w2 @ w1_neg_half
    # Step 3: Raise the inner matrix to the power p
    inner_matrix_p = fractional_matrix_power(inner_matrix, p)
    # Step 4: Compute Sp
    sp = w1_half @ inner_matrix_p @ w1_half
    return sp


def apply_LP(matrices):
    covariance_list_after_LP = [fixed_geodes_eff(matrices[i], matrices[i+1], p=0.5) for i in range(0, len(matrices) - 1, 1)]
    covariance_list_after_LP = np.array(covariance_list_after_LP)
    return covariance_list_after_LP


def apply_HP(matrices):
    covariance_list_after_LP = [fixed_geodes_eff(matrices[i], matrices[i+1], p=2) for i in range(0, len(matrices) - 1, 1)]
    covariance_list_after_LP = np.array(covariance_list_after_LP)
    return covariance_list_after_LP


def wavelet_packet(signal, levels):
    """
    Constructs a wavelet packet decomposition of a signal.

    Parameters:
    - signal: array-like, the input signal.
    - levels: int, the number of decomposition levels.
    - lp_filter: function, low-pass filter function (takes a signal and returns the filtered signal).
    - hp_filter: function, high-pass filter function (takes a signal and returns the filtered signal).
    - downsample: function, performs down-sampling (takes a signal and returns the down-sampled signal).
    - up-sample_reconstruct: function, reconstructs the original signal given the LP and HP components.

    Returns:
    - wavelet_tree: dict, the wavelet packet tree. Keys are tuples representing nodes,
     and values are the decomposed signals.
    """
    wavelet_tree = {}  # To store the decomposition tree
    wavelet_tree[(0,)] = signal  # Root node is the original signal

    def decompose(node, current_level):
        if current_level > levels:  # Stop decomposition at the desired level
            return

        # Retrieve the current signal
        current_signal = wavelet_tree[node]
        try:
            # Perform LP and HP filtering
            lp = apply_LP(current_signal)
            hp = apply_HP(current_signal)

            # Downsample the results
            lp_down = lp[0::2]
            hp_down = hp[0::2]

            # Store in the wavelet tree
            left_child = node + (0,)  # Left child (low-pass)
            right_child = node + (1,)  # Right child (high-pass)
            wavelet_tree[left_child] = lp_down
            wavelet_tree[right_child] = hp_down

            # Recursively decompose children
            decompose(left_child, current_level + 1)
            decompose(right_child, current_level + 1)
        except:
            print(f"found negative eigenvalue stopping decent at {node}")

    # Start decomposition from the root node
    decompose((0,), 1)

    return wavelet_tree


def plot_elements(matrices, element1=[0,1], element2=[0,2], element3=[1,2]):
    time_steps = np.arange(matrices.shape[0])  # Create an array for time steps
    element_01 = matrices[:, element1[0], element1[1]]
    element_02 = matrices[:, element2[0], element2[1]]
    element_12 = matrices[:, element3[0], element3[1]]

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Plot each element in a separate subplot
    axs[0].plot(time_steps, element_01, label=f"Element {element1[0]},{element1[1]}", color='r')
    axs[0].set_title(f"Element ({element1[0]},{element1[1]})")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)

    axs[1].plot(time_steps, element_02, label=f"Element {element2[0]},{element2[1]}", color='g')
    axs[1].set_title(f"Element ({element2[0]},{element2[1]})")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Value")
    axs[1].grid(True)

    axs[2].plot(time_steps, element_12, label=f"Element {element3[0]},{element3[1]}", color='b')
    axs[2].set_title(f"Element ({element3[0]},{element3[1]})")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Value")
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def compare_elements(matrices, matrices2):
    # matrices = matrices[1:]
    time_steps = np.arange(matrices.shape[0])  # Create an array for time steps
    element_01 = matrices[:, 0, 1]
    element_02 = matrices[:, 0, 2]
    element_12 = matrices[:, 1, 2]
    element_01r = matrices2[:, 0, 1]
    element_02r = matrices2[:, 0, 2]
    element_12r = matrices2[:, 1, 2]

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Plot each element in a separate subplot
    axs[0].plot(time_steps, element_01, label="Element (0, 1)", color='r')
    axs[0].plot(time_steps, element_01r, label="Element (0, 1) recon", color='g')
    axs[0].set_title("Element (0, 1)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time_steps, element_02, label="Element (0, 2)", color='g')
    axs[1].plot(time_steps, element_02r, label="Element (0, 2) recon", color='b')
    axs[1].set_title("Element (0, 2)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Value")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(time_steps, element_12, label="Element (1, 2)", color='b')
    axs[2].plot(time_steps, element_12r, label="Element (1, 2) recon", color='r')
    axs[2].set_title("Element (1, 2)")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Value")
    axs[2].grid(True)
    axs[2].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


def normalize_eigenvalues(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    normalized_eigenvalues = eigenvalues / max_eigenvalue
    normalized_matrix = eigenvectors @ np.diag(normalized_eigenvalues) @ np.linalg.inv(eigenvectors)

    return normalized_matrix


def clip_eigenvalues(matrix, threshold1=1e5, threshold2=1e-5):
    """
    Clip the eigenvalues of a matrix to a specified threshold.

    Parameters:
    - matrix: 2D numpy array, the matrix whose eigenvalues are to be clipped.
    - threshold: The maximum value to clip eigenvalues to (default is 1e6).

    Returns:
    - clipped_matrix: 2D numpy array, the matrix with clipped eigenvalues.
    """
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Clip eigenvalues that are above the threshold
    clipped_eigenvalues = np.clip(eigenvalues, threshold2, threshold1)

    # Reconstruct the matrix with clipped eigenvalues
    clipped_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ np.linalg.inv(eigenvectors)

    return clipped_matrix


def fixed_geodes_eff(A, B, p):
    """
    Computes the point t along the geodesic stretching from A to B.

    Parameters:
        A (ndarray): PSD matrix of rank `dim`.
        B (ndarray): PSD matrix of rank `dim`.
        p (float): Desired point along the geodesic (t > 0).
    Returns:
        ndarray: The point t along the geodesic.
    """
    # Compute the ranks of A and B based on non-zero eigenvalues
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    dim = min(rank_A, rank_B)  # Set dim to the minimum rank
    if dim == np.shape(B)[0] and dim == np.shape(A)[0]:
        return clip_eigenvalues(np.real(filter(A, B, p)))
    # Get the largest `dim` eigenvalues and eigenvectors for A and B
    S1, U1 = np.linalg.eig(A)
    S2, U2 = np.linalg.eig(B)
    S1 = np.real(S1)  # Take only real parts of eigenvalues
    S2 = np.real(S2)

    U1 = U1[:, np.argsort(-S1)[:dim]]
    U2 = U2[:, np.argsort(-S2)[:dim]]
    S1 = -np.sort(-S1)[:dim]
    S2 = -np.sort(-S2)[:dim]

    # Extract eigenvector subspaces
    VA = np.real(U1[:, :dim])  # Ensure eigenvectors are real
    VB = np.real(U2[:, :dim])

    # Singular Value Decomposition (SVD)
    OA, SAB, OB = svd(VA.T @ VB)
    SAB = np.real(SAB)  # Ensure singular values are real

    UA = VA @ OA
    UB = VB @ OB.T
    theta = np.arccos(np.clip(SAB, -1, 1))  # Clip to avoid numerical errors

    # Compute intermediate matrices
    tmp = UB @ pinv(np.diag(np.sin(theta)))
    X = (eye(A.shape[0]).toarray() @ tmp - UA @ UA.T @ tmp)
    U = UA @ np.diag(np.cos(theta * p)) + X @ np.diag(np.sin(theta * p))

    # Compute R2
    RB2 = OB @ np.diag(S2) @ OB.T
    assert np.all(S1 > 0), "Not all eigenvalues are positive!"
    RA = OA.T @ np.diag(np.sqrt(S1)) @ OA
    RAm1 = OA.T @ np.diag(1 / np.sqrt(S1)) @ OA
    eigenvalues = np.real(np.linalg.eigvals(RAm1 @ RB2 @ RAm1))  # Ensure eigenvalues are real
    assert np.all(eigenvalues >= 0), "Not all eigenvalues are positive!"
    R2 = RA @ expm(p * logm(RAm1 @ RB2 @ RAm1)) @ RA
    # Compute the result
    S = U @ R2 @ U.T
    return clip_eigenvalues(np.real(S))  # Ensure final result is real


def calculate_entropy(eigenvector):
    """Calculate the entropy of a normalized eigenvector."""
    # Take the absolute value of the eigenvector (to avoid negative probabilities)
    probabilities = np.abs(eigenvector)
    # Normalize to get a probability distribution
    probabilities = probabilities / np.sum(probabilities)
    # Compute entropy, avoiding log(0)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # Add a small epsilon to avoid log(0)
    # Compute the maximum possible entropy (log of the number of elements)
    max_entropy = np.log(len(probabilities))
    # Calculate normalized entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    return normalized_entropy


def process_dict_and_calculate_entropy(matrix_dict):
    """Process a dictionary of matrices, get largest eigenvector and compute entropy."""
    entropy_results = {}

    for key, matrix in matrix_dict.items():
        if len(matrix) == 1:
            matrix = matrix[0]
            # Ensure the matrix is square
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Matrix at key '{key}' is not square.")

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            # Find the index of the largest eigenvalue
            largest_idx = np.argmax(eigenvalues)
            # Get the eigenvector corresponding to the largest eigenvalue
            largest_eigenvector = eigenvectors[:, largest_idx]
            # Compute entropy for the eigenvector
            entropy = calculate_entropy(largest_eigenvector)
        else:
            entropy_list = []
            for mat in matrix:
                if mat.shape[0] != mat.shape[1]:
                    raise ValueError("All matrices must be square.")

                    # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(mat)
                largest_idx = np.argmax(eigenvalues)  # Index of largest eigenvalue
                largest_eigenvector = eigenvectors[:, largest_idx]

                # Calculate entropy
                entropy_tmp = calculate_entropy(largest_eigenvector)
                entropy_list.append(entropy_tmp)
            entropy = entropy_list
        # Store the result
        entropy_results[key] = entropy

    return entropy_results


def plot_dict_values_as_histogram(data_dict, bins=50, highlight_keys=None):
    """
    Plot all the values from a dictionary as a histogram, with an option to highlight specific keys.

    Parameters:
        data_dict (dict): A dictionary with keys and numerical values (lists, arrays, or single numbers).
        bins (int): Number of bins for the histogram.
        depth (int): Length of keys to include.
        highlight_keys (list): A list of keys whose values will be highlighted in the histogram.
    """
    # Combine all values into a single list
    all_values = []
    highlight_values = []

    for key, value in data_dict.items():
        if isinstance(value, (list, np.ndarray)):  # If values are a list/array
            all_values.extend(value)
            if highlight_keys and key in highlight_keys:
                highlight_values.extend(value)
        else:  # If values are single numbers
            all_values.append(value)
            if highlight_keys and key in highlight_keys:
                highlight_values.append(value)

    all_values = np.array(all_values)
    highlight_values = np.array(highlight_values)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_values, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, label='All Values')

    if highlight_keys and highlight_values.size > 0:
        highlight_bins = bins  # Default to `bins` if not specified
        plt.hist(
            highlight_values, bins=highlight_bins,
            color='orange', edgecolor='black', alpha=0.9,
            histtype='stepfilled', linewidth=2.5, label=f'Highlighted Keys: {highlight_keys}'
        )

    plt.title("Histogram of All Values in Dictionary")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def get_bottom_percent_keys(entropy_dict, percent=5, exclude_bottom_percent=1):
    # Sort the dictionary by values in ascending order
    sorted_items = sorted(entropy_dict.items(), key=lambda x: x[1])

    # Calculate the number of items to exclude and include
    total_items = len(sorted_items)
    exclude_items = max(1, int(total_items * (exclude_bottom_percent / 100.0)))
    include_items = max(1, int(total_items * (percent / 100.0)))

    # Determine the range of items to include (exclude the bottom percent)
    bottom_keys = [
        item[0] for item in sorted_items[exclude_items:exclude_items + include_items]
    ]

    return bottom_keys


def plot_entropy_vs_frequency(entropy_results):
    keys = list(entropy_results.keys())
    values = list(entropy_results.values())
    # Create the plot
    plt.plot(keys, values, marker='o', linestyle='-', color='b')
    # Add labels and title
    plt.xlabel('Frequency')
    plt.ylabel('Entropy')
    plt.title('Entropy as function of Frequency')
    # Show the plot
    plt.show()


def calculate_importance_scores(vector_list, n_clusters=2, keep_clusters=False):
    """
    Calculate importance scores for each index across a list of vectors using k-means clustering.

    Parameters:
    - vector_list: List of vectors (numpy arrays) to process.
    - n_clusters: Number of clusters for k-means (default is 2).

    Returns:
    - importance_scores: A dictionary where keys are indices, and values are importance scores.
    """
    # Initialize a dictionary to store importance counts for each index
    importance_counts = {}

    # Get the total number of indices (assume all vectors are the same size)
    total_indices = len(vector_list[0]) if vector_list else 0
    clusters = []
    for vector in vector_list:
        # Reshape the vector for k-means clustering (k-means requires 2D input)
        values = vector.reshape(-1, 1)

        # Run k-means clustering to split values into two clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(values)

        # Identify the cluster with higher mean value (important group)
        cluster_means = kmeans.cluster_centers_.flatten()
        important_cluster = np.argmax(cluster_means)

        # Mark indices in the important cluster
        important_indices = np.where(kmeans.labels_ == important_cluster)[0]
        clusters.append(important_indices)
        # Update importance counts for each index
        for idx in important_indices:
            importance_counts[idx] = importance_counts.get(idx, 0) + 1

    # Normalize the importance counts to get scores
    total_vectors = len(vector_list)
    importance_scores = {idx: importance_counts.get(idx, 0) / total_vectors for idx in range(total_indices)}
    if keep_clusters:
        return importance_scores, clusters

    return importance_scores


def get_spectrum(chosen_wave):
    largest_eigenvectors = []
    # Compute eigenvalues and eigenvectors for each matrix
    for matrix in chosen_wave:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = np.argmax(eigenvalues)  # Index of the largest eigenvalue
        largest_eigenvectors.append(eigenvectors[:, max_idx].real)  # Select eigenvector

    # Combine largest eigenvector components over time
    heatmap_data = np.array(np.abs(largest_eigenvectors)).T  # Transpose for heatmap (components as rows)
    return heatmap_data


def plot_importance_scores(importance_scores, title="Importance Scores"):
    """
    Plot the importance scores as a bar chart.

    Parameters:
    - importance_scores: A dictionary where keys are indices, and values are scores.
    - title: The title of the plot.
    """
    # Sort the dictionary by keys for better visualization
    sorted_keys = sorted(importance_scores.keys())
    sorted_scores = [importance_scores[key] for key in sorted_keys]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_keys, sorted_scores, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Importance Score")
    plt.title(title)
    plt.xticks(sorted_keys, rotation=45)  # Rotate x-axis labels for better readability

    # Show grid and the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def get_keys_above_bottom_quantile(data_dict, quantile=0.1, above_zero=False):
    """
    Get keys with values above the bottom `quantile` of a dictionary's values.

    Parameters:
    - data_dict: The dictionary to process.
    - quantile: The bottom quantile threshold (default is 0.1 for 10%).

    Returns:
    - List of keys with values above the bottom quantile.
    """
    if not data_dict:
        return []
    if above_zero:
        return [key for key, value in data_dict.items() if value > 0]
    # Extract all non-zero values
    nonzero_values = [value for value in data_dict.values() if value != 0]
    if not nonzero_values:
        return []  # No non-zero values in the dictionary

    # Get keys with values above the cutoff
    keys_above_cutoff = [key for key, value in data_dict.items() if value > quantile]

    return keys_above_cutoff


def plot_graph_and_get_clusters(clusters):
    G = nx.Graph()

    # Create a dictionary to store the weight (count of co-occurrences)
    weights = defaultdict(int)

    # Compute the mean frequency for each node
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a, b = cluster[i], cluster[j]
                weights[tuple(sorted((a, b)))] += 1

    # Add nodes and edges to the graph based on the co-occurrence counts
    for (a, b), weight in weights.items():
        G.add_edge(a, b, weight=weight)

    # Community detection using Louvain method
    partition = community_louvain.best_partition(G)
    community_nodes = defaultdict(list)

    # Group nodes by their community
    for node, comm in partition.items():
        community_nodes[comm].append(node)

    # Compute the mean frequency for each community
    # Get the positions of nodes
    pos = nx.spring_layout(G)  # You can use other layouts as well

    # Create a list of node colors based on the community
    node_colors = [partition[node] for node in G.nodes()]

    # Plot the graph with community-based coloring
    plt.figure(figsize=(8, 6))

    # Draw the nodes with colors based on their community
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, cmap=plt.cm.jet)

    # Draw the edges with color representing the weight (as before)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_weights, edge_cmap=plt.cm.Blues)

    plt.title("Graph with Communities")
    plt.show()

    return partition