import R_utils
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import community.community_louvain as community_louvain

np.random.seed(42)


def plot_spectrum(chosen_wave, wave_num):
    largest_eigenvectors = []
    # Compute eigenvalues and eigenvectors for each matrix
    for matrix in chosen_wave:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = np.argmax(eigenvalues)  # Index of the largest eigenvalue
        largest_eigenvectors.append(eigenvectors[:, max_idx].real)  # Select eigenvector

    # Combine largest eigenvector components over time
    heatmap_data = np.array(np.abs(largest_eigenvectors)).T  # Transpose for heatmap (components as rows)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, aspect="auto", cmap="coolwarm", origin="lower")
    plt.colorbar(label="Eigenvector Component Value")
    plt.xlabel("Time Step")
    plt.ylabel("Component Index")
    plt.title(f"Heatmap of Eigenvector (Largest Eigenvalue) for wavelet at {wave_num} Hz")
    plt.xticks(range(len(chosen_wave)), labels=range(1, len(chosen_wave) + 1))
    plt.tight_layout()
    plt.show()


def plot_diffusion_maps(symmetric_matrices, matrix_len, title=''):
    diffusion_representations, distances = R_utils.get_diffusion_embedding(symmetric_matrices, matrix_len)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xs = diffusion_representations[0, 0, :]
    ys = diffusion_representations[0, 1, :]
    zs = diffusion_representations[0, 2, :]

    # Plot with gradient colors
    ax.plot(xs, ys, zs, color='k', linestyle='-', alpha=0.1)
    sc = ax.scatter(xs, ys, zs, marker='o')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.view_init(elev=50, azim=45)
    ax.set_title(f'3D Diffusion map Plot {title}')
    plt.show()


def sum_and_compare(graph):
    def compute_sums(node):
        if isinstance(graph[node], list):
            graph[node] = sum(graph[node])  # Replace list with sum

        children = [k for k in graph if k[:-1] == node]  # Find direct children
        child_sum = sum(graph[child] for child in children)

        if graph[node] < child_sum:
            print(f"Node {node} has value {graph[node]}, which is smaller than its children's sum {child_sum}")

        return graph[node]

    # Process nodes in reverse order (bottom-up)
    for key in sorted(graph.keys(), key=lambda x: (-len(x), x)):
        compute_sums(key)


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
    edges = nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_weights, edge_cmap=plt.cm.Blues)
    plt.title("Graph with Communities")
    plt.show()

    return partition


def make_masks(mean_corr, partition, important_dict):
    mask = np.zeros_like(mean_corr[0], dtype=bool)

    # Loop through all node pairs
    for node1 in important_dict:
        for node2 in important_dict:
            if partition[node1] == partition[node2]:  # Only retain edges within the same community
                mask[node1, node2] = True

    mask2 = 1 - mask

    zeroed_matrices = []
    for matrix in mean_corr:
        zeroed_matrix = matrix * mask
        zeroed_matrices.append(zeroed_matrix)

    zeroed_matrices2 = []
    for matrix in mean_corr:
        zeroed_matrix = matrix * mask2
        zeroed_matrices2.append(zeroed_matrix)

    return zeroed_matrices, zeroed_matrices2


def plot_spectrum_as_matrix(important_spectrum):
    vecs_list = []
    for vec in important_spectrum:
        vecs_list.append(vec)

    matrix = np.array(vecs_list)
    # Plot the matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect='auto', cmap="coolwarm", interpolation="nearest")
    # Add colorbar for reference
    plt.colorbar(label="Value")
    # Label the axes
    plt.xlabel("Vector Index")
    plt.ylabel("Vector Number")
    # Show the plot
    plt.show()


def plot_elements_seq(data):
    rows, cols = np.triu_indices(matrix_size, k=1)  # k=1 excludes diagonal
    num_traces = len(rows)  # Should be 180

    # Prepare figure
    plt.figure(figsize=(10, 20))

    # Extract and plot each unique time trace
    for i in range(num_traces):
        r, c = rows[i], cols[i]  # Get row & col indices
        time_trace = data[:, r, c]  # Extract time series (512 samples)
        plt.plot(time_trace + i * 2, 'k', alpha=0.7)  # Offset each trace for visibility in black

    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (offset)")
    plt.title("Stacked Time Traces of Upper Triangular Matrix Elements (20x20)")
    # plt.savefig("Stacked Time Traces of Upper Triangular Matrix Elements.svg", format="svg", bbox_inches="tight")
    plt.show()


def plot_elements_heatmap(data):
    rows, cols = np.triu_indices(data.shape[1], k=1)  # Extract upper triangle indices
    num_traces = len(rows)  # Should be 180

    # Create a matrix where rows are different matrix elements and columns are time points
    heatmap_data = np.array([data[:, r, c] for r, c in zip(rows, cols)])  # Shape: (180, 512)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', interpolation='nearest')

    # Labels and formatting
    plt.xlabel("Time (samples)")
    plt.ylabel("Upper Triangular Element Index")
    plt.title("Heatmap of Time Traces for Upper Triangular Matrix Elements")
    plt.colorbar(label="Value")
    # plt.savefig("Heatmap of Time Traces for Upper Triangular Matrix Elements.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':

    num_matrices = 512
    matrix_size = 20
    f_0 = 4
    f_1 = 66
    # Generate matrices
    symmetric_matrices = R_utils.generate_symmetric_PD_matrices_with_time_dependency_blocks(num_matrices,
                                                                                            matrix_size, f_0, f_1,
                                                                                            noise_amplitude=0.25)
    symmetric_matrices = np.array(symmetric_matrices)
    plot_elements_heatmap(symmetric_matrices)
    all_spectrum = []
    for vec in symmetric_matrices:
        all_spectrum.append(R_utils.get_spectrum([vec]))
    plot_spectrum_as_matrix(all_spectrum)
    R_utils.plot_elements(symmetric_matrices, element1=[0, 1], element2=[6, 8], element3=[5, 4])

    wave = R_utils.wavelet_packet(symmetric_matrices, 9)

    binary_dict = {
        int("".join(map(str, k)), 2): v
        for k, v in wave.items()
        if len(k) == 10
    }
    all_spectrum = []
    for key in binary_dict.keys():
        all_spectrum.append(R_utils.get_spectrum(binary_dict[key]))
    plot_spectrum_as_matrix(all_spectrum)
    plot_spectrum(binary_dict[f_0], f_0)
    plot_spectrum(binary_dict[70], 70)
    entropy_results = R_utils.process_dict_and_calculate_entropy(binary_dict)

    R_utils.plot_entropy_vs_frequency(entropy_results)

    important_keys = R_utils.get_bottom_percent_keys(entropy_results, percent=10, exclude_bottom_percent=0)
    print("number of important keys: ", len(important_keys))

    important_dict = {}
    for key in entropy_results:
        important_dict[key] = entropy_results[key]
    important_spectrum = []
    for key in important_keys:
        important_spectrum.append(R_utils.get_spectrum(binary_dict[key]))
    R_utils.plot_dict_values_as_histogram(important_dict, bins=50)
    importance_counts, clusters = R_utils.calculate_importance_scores(important_spectrum,
                                                                      n_clusters=2, keep_clusters=True)

    R_utils.plot_importance_scores(importance_counts, title="Importance Scores")
    important_keys = R_utils.get_keys_above_bottom_quantile(importance_counts, quantile=0.2, above_zero=False)
    clusters = [
        [item for item in sublist if item in important_keys]
        for sublist in clusters
    ]
    partition = plot_graph_and_get_clusters(clusters)
    plot_diffusion_maps(symmetric_matrices, 20, title='original signal')
    matrices_w_dynamics, matrices_wo_dynamics = make_masks(symmetric_matrices, partition, important_keys)
    plot_diffusion_maps(np.array(matrices_w_dynamics), 20, title='important')
    plot_diffusion_maps(np.array(matrices_wo_dynamics), 20, title='unimportant')


