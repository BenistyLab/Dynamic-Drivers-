# Detection of dynamic drivers


## Agenda
* Project overview
* Key features
* Prerequisites
* Code explanation
* Credits and References


### Project overview
This project explores the temporal evolution of symmetric positive semi-definite (SPSD) matrices using Riemannian wavelet packet analysis. By leveraging tools from Riemannian geometry, we provide a framework to analyze structured time-series data with inherent manifold constraints, offering insights into dynamics that are not easily captured by classical methods.
Analysis and toy example for www.biorxiv.org/content/10.1101/2025.03.26.645425v3.abstract

### Key features
* Wavelet Packet Decomposition on SPSD Manifolds: Extends traditional wavelet analysis to the space of SPSD matrices while respecting their geometric structure.
* Riemannian Framework: Utilizes Riemannian metrics to ensure meaningful comparisons and transformations.
* Temporal Dynamics Exploration: Captures hierarchical patterns and variations across different time scales.
* Applications: Suitable for analyzing SPSD matrices (e.g. correlation and covariance matrices) in finance, neuroscience, and other fields where structured temporal data is prevalent.

### Prerequisites
Make sure you have Python 3.8 or higher.

To install the required dependencies, use the `requirements.txt` file:

### Functions
The file `R_utils.py` contains the following functions:

---

#### `sym_pos_semi_def_dist(A, B, r, k=1)`
Computes the distance between two matrices `A` and `B` on the Symmetric Positive Semi-Definite (SPSD) manifold.

- **Parameters:**
  - `A`, `B`: Input matrices.
  - `r`: The minimum rank of `A` and `B`.
  - `k`: Optional parameter (default is `1`).

---

#### `get_diffusion_embedding(correlations, window_length, scale_k=20, signal=None, subsampling=0, mode='riemannian')`
Performs dimensionality reduction using the Diffusion Maps method. Distances can be computed using a Riemannian or Euclidean kernel via the `mode` parameter.

- **Parameters:**
  - `correlations`: A NumPy array of correlation matrices with shape `(time, features, features)`.
  - `window_length`: Integer equal to the number of features.
  - `scale_k`: Scale parameter for diffusion maps (default is `20`).
  - `signal`: Optional signal input (default is `None`).
  - `subsampling`: Subsampling rate (default is `0`).
  - `mode`: Distance metric (`'riemannian'` or `'euclidean'`).

---

#### `filter(w1, w2, p=0.5)`
Computes an intermediate point along the Riemannian geodesic between two SPD matrices.

- **Parameters:**
  - `w1`, `w2`: Input symmetric positive definite matrices.
  - `p`: Interpolation parameter (default is `0.5`).

---

#### `fixed_geodes_eff(A, B, p)`
Extension of `filter()` to support SPSD matrices.

---

#### `apply_LP(matrices)` and `apply_HP(matrices)`
Apply low-pass or high-pass filters to a series of matrices.

- **Parameters:**
  - `matrices`: A sequence of matrices (e.g., list or array).

---

#### `wavelet_packet(signal, levels)`
Constructs a wavelet packet decomposition from a sequence of matrices.

- **Parameters:**
  - `signal`: The matrix sequence to decompose.
  - `levels`: Depth of the wavelet tree. For full decomposition, use `log2(len(signal))`.

---

#### `process_dict_and_calculate_entropy(matrix_dict)`
Processes a dictionary representing a wavelet tree, computes the largest eigenvector in each node, and calculates its entropy.

---

#### `get_bottom_percent_keys(entropy_dict, percent=5, exclude_bottom_percent=1)`
Extracts keys from an entropy dictionary whose values fall in the bottom specified percentile, excluding the very lowest.

- **Parameters:**
  - `entropy_dict`: Dictionary of entropy values.
  - `percent`: Bottom percentile to extract (default is `5`).
  - `exclude_bottom_percent`: Lower-bound cutoff (default is `1`).

---

#### `calculate_importance_scores(vector_list, n_clusters=2, keep_clusters=False)`
Computes importance scores for each index across a list of vectors using k-means clustering.

- **Parameters:**
  - `vector_list`: List of NumPy vectors to process.
  - `n_clusters`: Number of clusters for k-means (default is `2`).
  - `keep_clusters`: Whether to return cluster labels (default is `False`).

---

#### `get_keys_above_bottom_quantile(data_dict, quantile=0.1, above_zero=False)`
Returns dictionary keys whose values are above the specified bottom quantile.

- **Parameters:**
  - `data_dict`: Dictionary of numerical values.
  - `quantile`: Bottom quantile threshold (default is `0.1`).
  - `above_zero`: If `True`, excludes keys with non-positive values.

---
