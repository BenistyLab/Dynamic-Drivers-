import numpy as np
from scipy.linalg import fractional_matrix_power, logm
import torch
from utils import Import_Data, check_rows, find_rows_with_common_numbers, extract_rows, sort_matrices_by_A
import matplotlib.pyplot as plt
import pickle
from analysis_tools import run_linear_regression_lever_pull_with_kfold, interpolate_points
from tqdm import tqdm
import copy
from sklearn.cluster import KMeans
from analysis_tools import logistic_regression_expert_prediction
import os
import seaborn as sns
from correlation_distances_main.lib.utils import get_diffusion_embedding
from matplotlib.colors import LinearSegmentedColormap
import glob
import pandas as pd


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

OTF77_directory =\
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\No_PD\OTF77 - Single Cell'
#OTF80_directory = r'C:\Users\yonatan.kle\OneDrive - Technion\Desktop\PD_project\raw_data\No_PD\OTF80 - Single Cell'
OTF85_directory =\
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\No_PD\OTF85 - Single Cell'
OTF94_directory =\
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\No_PD\OTF94 - Single Cell'
excel_directory =\
    r"C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\Experiment Log.xlsx"
animal_names = ['OTF77', 'OTF85', 'OTF94']

data_importer = Import_Data(OTF77_directory, excel_directory, animal_names[0])
OTF77 = data_importer.get_data()
#data_importer = Import_Data(OTF80_directory, excel_directory, animal_names[1])
#OTF80 = data_importer.get_data()
data_importer = Import_Data(OTF85_directory, excel_directory, animal_names[1])
OTF85 = data_importer.get_data()
data_importer = Import_Data(OTF94_directory, excel_directory, animal_names[2])
OTF94 = data_importer.get_data()

animals_data_list = [OTF77, OTF85, OTF94]
dirs = [OTF77_directory, OTF85_directory, OTF94_directory]


def get_diffusion_operators(w1, w2, p=0.5, only_s=False, device='cuda'):
    sym = lambda M: (M + M.T) / 2
    # Move matrices to the GPU
    w1 = torch.tensor(sym(w1), device=device)
    w2 = torch.tensor(sym(w2), device=device)

    # Step 1: Compute fractional powers of w1
    w1_half = torch.from_numpy(fractional_matrix_power(w1.cpu().numpy(), 0.5)).to(device).real
    w1_neg_half = torch.from_numpy(fractional_matrix_power(w1.cpu().numpy(), -0.5)).to(device).real

    # Step 2: Compute the matrix inside the power operation
    inner_matrix = w1_neg_half @ w2 @ w1_neg_half

    # Step 3: Raise the inner matrix to the power p
    inner_matrix_p = torch.from_numpy(fractional_matrix_power(inner_matrix.cpu().numpy(), p)).to(device).real

    # Step 4: Compute Sp
    sp = w1_half @ inner_matrix_p @ w1_half

    # Step 5: Compute S_p^(1/2) and S_p^(-1/2)
    sp_half = torch.from_numpy(fractional_matrix_power(sp.cpu().numpy(), 0.5)).to(device).real
    sp_neg_half = torch.from_numpy(fractional_matrix_power(sp.cpu().numpy(), -0.5)).to(device).real

    # Step 6: Calculate the matrix inside the logarithm
    log_inner_matrix = sp_neg_half @ w1 @ sp_neg_half

    # Step 7: Compute the logarithm of the matrix
    log_matrix = torch.from_numpy(logm(log_inner_matrix.cpu().numpy())).to(device).real

    # Step 8: Compute Fp
    fp = sp_half @ log_matrix @ sp_half

    if only_s:
        return sp.cpu().numpy()
    return sp.cpu().numpy(), fp.cpu().numpy()


def plot_matrix_changes(matrix_list, eigenvector_to_plot=0, title='Rois that change inter trial',
                        sort_by=None, save=False):
    total_evals = []
    total_evecs = []
    for i in range(len(matrix_list)):
        eigenvalues, eigenvectors = np.linalg.eig(matrix_list[i])

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        # Get the indices that would sort the eigenvalues in descending order
        sorted_indices = np.argsort(-eigenvalues)

        # Sort the eigenvalues and eigenvectors
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        total_evals.append(eigenvalues_sorted)
        total_evecs.append(np.abs(eigenvectors_sorted[:, eigenvector_to_plot]))
    total_evecs = np.array(total_evecs).T
    if sort_by is None:
        sums = np.sum(total_evecs, axis=1)
        sorted_indices = np.argsort(sums)[::-1]
        total_evecs = total_evecs[sorted_indices]
    else:
        sorted_indices = sort_by
        total_evecs = total_evecs[sorted_indices]

    plt.figure(figsize=(10, 8))
    plt.imshow(total_evecs.squeeze(), cmap='viridis', aspect='auto')
    plt.colorbar()
    # Add labels if needed
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('ROI')

    # Display the plot
    if save:
        plt.savefig(f'{title}.png')
    plt.show()

    return sorted_indices


def plot_diffusion_operators(diffusion_operators, index):
    s_operators_in_trial = diffusion_operators[0]
    f_operators_in_trial = diffusion_operators[1]
    s_operators_in_days = diffusion_operators[2]
    f_operators_in_days = diffusion_operators[3]
    sort_by = plot_matrix_changes(s_operators_in_trial, eigenvector_to_plot=0,
                                  title=f'Rois that stay the same inter trial for {animal_names[index]}', save=False)
    plot_matrix_changes(f_operators_in_trial, eigenvector_to_plot=0,
                        title=f'Rois that change inter trial for {animal_names[index]}', save=False)
    plot_matrix_changes(s_operators_in_days, eigenvector_to_plot=0,
                        title=f'Rois that stay the same between days for {animal_names[index]}', save=False)
    plot_matrix_changes(f_operators_in_days, eigenvector_to_plot=0,
                        title=f'Rois that differ between days for {animal_names[index]}', save=False)


def get_eigenvectors_matrix(operator, eigenvalue=0):
    total_evals = []
    total_evecs = []
    for i in range(len(operator)):
        eigenvalues, eigenvectors = np.linalg.eig(operator[i])

        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        # Get the indices that would sort the eigenvalues in descending order
        sorted_indices = np.argsort(-eigenvalues)

        # Sort the eigenvalues and eigenvectors
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        total_evals.append(eigenvalues_sorted)
        total_evecs.append(np.abs(eigenvectors_sorted[:, eigenvalue]))
    total_evecs = np.array(total_evecs).T
    return total_evecs


def get_clusters(total_evecs):
    # k-means to find the clusters
    if total_evecs.ndim == 3:
        before_col = np.median(total_evecs, axis=1)
    elif total_evecs.ndim == 2:
        before_col = np.median(total_evecs, axis=1).reshape(-1, 1)
    else:
        raise ValueError("total_evecs must be a 2D or 3D array.")

    # Apply KMeans clustering with 2 clusters (high and low)
    kmeans_b4_PD = KMeans(n_clusters=2, random_state=0, n_init=10).fit(before_col)

    # Get the labels (0 or 1) for each row
    labels_b4 = kmeans_b4_PD.labels_ if np.mean(total_evecs[kmeans_b4_PD.labels_ == 1]) > np.mean(
        total_evecs[kmeans_b4_PD.labels_ == 0]) else 1 - kmeans_b4_PD.labels_

    leftover_labels = 1 - labels_b4

    all_ROI = leftover_labels + labels_b4

    list_b4 = [a for a, b in zip(list(range(len(total_evecs))), labels_b4) if b == 1]
    list_leftover = [a for a, b in zip(list(range(len(total_evecs))), leftover_labels) if b == 1]
    all_ROI_list = [a for a, b in zip(list(range(len(total_evecs))), all_ROI) if b == 1]

    segmentation_lists = [list_b4, list_leftover, all_ROI_list]
    return segmentation_lists, len(labels_b4)


def plot_bars_for_operators(data, ylabel, title):
    # Shape of data should be: (animals, operators, values)
    if np.shape(data)[1] == 2:
        list_labels = ['Cluster of Participating ROI ', 'Leftover Cluster', 'All ROI']
        operator_names = ['Inter trial', 'Between days']
    elif np.shape(data)[1] == 4:
        list_labels = ['Cluster of Participating ROI ', 'Leftover Cluster', 'All ROI']
        operator_names = ['S-operators inter trial', 'F-operators inter trial', 'S-operators between days',
                          'F-operators between days']
    else:
        raise ValueError("Data must be a multiple of 2.")

    # Calculate the mean and standard deviation across animals
    mean_data = np.mean(data, axis=0)  # Shape: (operators, values)
    std_data = np.std(data, axis=0) / np.sqrt(np.shape(data)[0])  # Shape: (operators, values)

    # Number of operators and values
    n_operators = mean_data.shape[0]
    n_values = mean_data.shape[1]

    # Set bar width and x locations
    bar_width = 0.2
    x = np.arange(n_operators)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(15, 6))
    # Plot bars for each value across operators
    for i in range(n_values):
        ax.bar(x + i * bar_width, mean_data[:, i], bar_width, yerr=std_data[:, i], label=list_labels[i], capsize=5)

    # Labeling
    ax.set_xlabel('Operators')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (n_values - 1) / 2)
    ax.set_xticklabels([f'Operator {i}' for i in operator_names])
    ax.legend()
    #plt.savefig(f'{title}.pdf', format='pdf')  # You can adjust the dpi for resolution
    plt.show()


def plot_graphs_for_operators(data, ylabel):
    if np.shape(data)[1] == 2:
        list_labels = ['Cluster of Participating ROI ', 'Leftover Cluster', 'All ROI']
        operator_names = ['Inter trial', 'Between days']
    elif np.shape(data)[1] == 4:
        list_labels = ['Cluster of Participating ROI ', 'Leftover Cluster', 'All ROI']
        operator_names = ['S-operators inter trial', 'F-operators inter trial', 'S-operators between days',
                          'F-operators between days']
    else:
        raise ValueError("Data must be a multiple of 2.")
    data = np.array(data)
    mean_data = np.mean(data, axis=0)  # Shape will be (4, 3, 25)
    std_data = np.std(data, axis=0)  # Shape will be (4, 3, 25)

    # Create the subplots
    fig, axs = plt.subplots(1, data.shape[1], figsize=(20, 5))  # 1 row, 4 subplots (for 4 operators)

    # Define time points for the x-axis
    time_points = np.arange(data.shape[-1])  # Assuming 25 time points

    # Plot each operator in a separate subplot
    for operator_idx in range(data.shape[1]):
        ax = axs[operator_idx]

        # Plot each cluster with error bars
        for cluster_idx in range(data.shape[2]):
            mean_trace = mean_data[operator_idx, cluster_idx, :]
            std_trace = std_data[operator_idx, cluster_idx, :]

            ax.errorbar(time_points, mean_trace, yerr=std_trace / np.sqrt(data.shape[0]),
                        label=f'Cluster {list_labels[cluster_idx]}')
        ax.set_title(f'Operator {operator_names[operator_idx]}')
        ax.set_xlabel('Days')
        ax.set_ylabel(ylabel)
        ax.legend()

    # Adjust layout for better viewing
    plt.tight_layout()
    #plt.savefig(f'{ylabel}.pdf', format='pdf')  # You can adjust the dpi for resolution
    plt.show()


def calculate_diffusion_operator():

    diffusion_operators = {}

    for index, animal in enumerate(animals_data_list):
        cov_matrices = []
        for date in animal.keys():
            cov_matrices_dates = []
            imaging = np.array(animal[date]['imagingData.samples'])
            for trial in range(np.shape(imaging)[2]):
                cov_matrix = np.corrcoef(imaging[:, :, trial], rowvar=True)
                cov_matrices_dates.append(cov_matrix)
            cov_matrices.append(cov_matrices_dates)
            # shape of cov_matrices is days X trials X ROI X ROI
        s_days = []
        f_days = []

        for training_day in tqdm(cov_matrices, desc=f"Processing training days for {animal_names[index]}"):
            covariance_list = training_day
            while len(covariance_list) > 2:
                # Apply func to each consecutive pair of matrices
                covariance_list = [get_diffusion_operators(covariance_list[i], covariance_list[i + 1], only_s=True) for
                                   i in
                                   range(0, len(covariance_list) - 1)]
            s_temp, f_temp = get_diffusion_operators(covariance_list[0], covariance_list[1])
            s_days.append(s_temp)
            f_days.append(f_temp)

        s_all = []
        f_all = []
        for i in range(len(s_days) - 1):
            s_temp, f_temp = get_diffusion_operators(s_days[i], s_days[i + 1])
            s_all.append(s_temp)
            f_all.append(f_temp)

        diffusion_operators[animal_names[index]] = [s_days, f_days, s_all, f_all]
    diffusion_operators['Help'] = \
        ('Each animals list contains [S-operators inter trial , F-operators inter trial,'
         ' S-operators between days, F-operators between days]')

    with open('diffusion_operators_single_cell.pkl', 'wb') as file:
        pickle.dump(diffusion_operators, file)

    return diffusion_operators


def calculate_percent_ROI(diffusion_operators):
    total_percent_of_ROI = []
    for index, animal_operators in enumerate(diffusion_operators.keys()):
        if animal_operators == 'Help':
            continue
        percent_of_ROI_per_animal = []
        operator_matrices = []
        for operator_index, diffusion_operator in enumerate(diffusion_operators[animal_operators]):
            total_evecs = get_eigenvectors_matrix(diffusion_operator, eigenvalue=0)
            operator_matrices.append(total_evecs)
        operator_matrices = [np.concatenate((np.expand_dims(operator_matrices[0], axis=2),
                                             np.expand_dims(operator_matrices[1], axis=2)), axis=2),
                             np.concatenate((np.expand_dims(operator_matrices[2], axis=2),
                                             np.expand_dims(operator_matrices[3], axis=2)), axis=2)]
        for operator_index, operator in enumerate(operator_matrices):
            segmentation_lists, number_of_ROI = get_clusters(operator)
            percent_of_ROI_per_animal.append(np.array([len(lst) / number_of_ROI if number_of_ROI > 0
                                                       else 0 for lst in segmentation_lists]))
        total_percent_of_ROI.append(percent_of_ROI_per_animal)

    plot_bars_for_operators(total_percent_of_ROI, ylabel='% of ROI',
                            title='% of ROI in the different clusters per diffusion operator')


def liner_regression_analysis(diffusion_operators):

    R2_across_animals = []
    Mean_across_animals = []
    for index, animal_operators in tqdm(enumerate(diffusion_operators.keys()), total=len(diffusion_operators)):
        if animal_operators == 'Help':
            continue
        R2_per_operator = []
        Mean_per_operator = []
        operator_matrices = []
        for operator_index, diffusion_operator in enumerate(diffusion_operators[animal_operators]):
            total_evecs = get_eigenvectors_matrix(diffusion_operator, eigenvalue=0)
            operator_matrices.append(total_evecs)
        operator_matrices = [
            np.concatenate((np.expand_dims(operator_matrices[0], axis=2), np.expand_dims(operator_matrices[1], axis=2)),
                           axis=2),
            np.concatenate((np.expand_dims(operator_matrices[2], axis=2), np.expand_dims(operator_matrices[3], axis=2)),
                           axis=2)]
        for operator_index, operator in enumerate(operator_matrices):
            segmentation_lists, number_of_ROI = get_clusters(operator)
            R2_per_segment = []
            Mean_per_segment = []
            for segmentation in segmentation_lists:
                temp_animal = copy.deepcopy(animals_data_list[index])
                for train, date in enumerate(temp_animal.keys()):
                    temp_animal[date]['imagingData.samples'] = extract_rows(temp_animal[date]['imagingData.samples'],
                                                                            segmentation)
                    temp_animal[date]['imagingData.roiNames'] = extract_rows(temp_animal[date]['imagingData.roiNames'],
                                                                             segmentation)

                mse_mean_per_date, r2_mean_per_date, mean_non_zero_weights, success, activity, mean_coefficients, PD_index = (
                    run_linear_regression_lever_pull_with_kfold(temp_animal, animal_names[index], k=5, tone=120,
                                                                regularization_type='L1', interp_points=True,
                                                                plot=False))

                R2_per_segment.append(r2_mean_per_date)
                Mean_per_segment.append(activity)
            R2_per_operator.append(R2_per_segment)
            Mean_per_operator.append(Mean_per_segment)
        R2_across_animals.append(R2_per_operator)
        Mean_across_animals.append(Mean_per_operator)

    plot_graphs_for_operators(R2_across_animals, 'R^2 score')

    for idx, r2score_per_animal in enumerate(R2_across_animals):
        r2score_per_animal = np.expand_dims(np.array(r2score_per_animal), axis=0)
        plot_graphs_for_operators(r2score_per_animal, f'R^2 score {animal_names[idx]}')

    plot_graphs_for_operators(Mean_across_animals, 'Mean Activity')


def logistic_regression_analysis(diffusion_operators):

    logistic_regression_results_dict = {}
    for index, animal_operators in tqdm(enumerate(diffusion_operators.keys()), total=len(diffusion_operators)):
        if animal_operators == 'Help':
            continue
        operator_matrices = []
        for operator_index, diffusion_operator in enumerate(diffusion_operators[animal_operators]):
            total_evecs = get_eigenvectors_matrix(diffusion_operator, eigenvalue=0)
            operator_matrices.append(total_evecs)
        operator_matrices = [
            np.concatenate((np.expand_dims(operator_matrices[0], axis=2), np.expand_dims(operator_matrices[1], axis=2)),
                           axis=2),
            np.concatenate((np.expand_dims(operator_matrices[2], axis=2), np.expand_dims(operator_matrices[3], axis=2)),
                           axis=2)]
        temp_logistics_operators = []
        for operator_index, operator in enumerate(operator_matrices):
            segmentation_lists, number_of_ROI = get_clusters(operator)
            temp_logistics = []
            for segmentation in segmentation_lists:
                temp_animal = copy.deepcopy(animals_data_list[index])
                for train, date in enumerate(temp_animal.keys()):
                    temp_animal[date]['imagingData.samples'] = extract_rows(temp_animal[date]['imagingData.samples'],
                                                                            segmentation)
                    temp_animal[date]['imagingData.roiNames'] = extract_rows(temp_animal[date]['imagingData.roiNames'],
                                                                             segmentation)
                temp_logistic = logistic_regression_expert_prediction(temp_animal, animal_names[index], test=True)
                temp_logistics.append(temp_logistic)
            temp_logistics_operators.append(temp_logistics)
        logistic_regression_results_dict[animal_names[index]] = temp_logistics_operators

        with open('diffusion_operators_logistic_single_cell.pkl', 'wb') as file:
            pickle.dump(logistic_regression_results_dict, file)

    return logistic_regression_results_dict


def plot_novice_expert_results(logistic_regression_results_dict):
    novice_expert_all = []
    novice_expert_all_unimp = []
    novice_expert_all_ROI = []
    for result in logistic_regression_results_dict.keys():
        novice_expert_all.append(logistic_regression_results_dict[result][1][0].predictions_expert_interp)
        novice_expert_all_unimp.append(logistic_regression_results_dict[result][1][1].predictions_expert_interp)
        novice_expert_all_ROI.append(logistic_regression_results_dict[result][1][2].predictions_expert_interp)

    novice_expert_all = np.mean(novice_expert_all, axis=0)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(novice_expert_all, cmap="viridis")
    plt.title("Heatmap of Mean Novice expert on important ROI")
    plt.show()

    novice_expert_all_unimp = np.mean(novice_expert_all_unimp, axis=0)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(novice_expert_all_unimp, cmap="viridis")
    plt.title("Heatmap of Mean Novice expert on unimportant ROI")
    plt.show()

    novice_expert_all = []
    novice_expert_all_unimp = []
    novice_expert_all_ROI = []

    novice_expert_all_days = []
    novice_expert_all_unimp_days = []
    novice_expert_all_ROI_days = []

    for result in logistic_regression_results_dict.keys():
        novice_expert_all.append(logistic_regression_results_dict[result][0][0].predictions_expert_interp[:, 14])#TODO: change back to 8
        novice_expert_all_unimp.append(
            logistic_regression_results_dict[result][0][1].predictions_expert_interp[:, 14])#TODO: change back to 8
        novice_expert_all_ROI.append(logistic_regression_results_dict[result][0][2].predictions_expert_interp[:, 14])

        novice_expert_all_days.append(logistic_regression_results_dict[result][1][0].predictions_expert_interp[:, 14])#TODO: change back to 8

        novice_expert_all_unimp_days.append(
            logistic_regression_results_dict[result][1][1].predictions_expert_interp[:, 14]) #TODO: change back to 8
        novice_expert_all_ROI_days.append(logistic_regression_results_dict[result][1][2].predictions_expert_interp[:, 14])

    novice_expert_analysis_inter = [novice_expert_all, novice_expert_all_unimp,novice_expert_all_ROI]
    novice_expert_analysis_days = [novice_expert_all_days, novice_expert_all_unimp_days,novice_expert_all_ROI_days]
    all_data = [novice_expert_analysis_inter, novice_expert_analysis_days]
    plot_graphs_for_operators(np.transpose(all_data, (2, 0, 1, 3)), 'Naive expert prediction')


def calculate_Riemannian(diffusion_operators):

    distance_across_animals = []
    for index, animal_operators in tqdm(enumerate(diffusion_operators.keys()), total=len(diffusion_operators)):
        if animal_operators == 'Help':
            continue
        distance_per_operator = []
        operator_matrices = []
        for operator_index, diffusion_operator in enumerate(diffusion_operators[animal_operators]):
            total_evecs = get_eigenvectors_matrix(diffusion_operator, eigenvalue=0)
            operator_matrices.append(total_evecs)
        operator_matrices = [
            np.concatenate((np.expand_dims(operator_matrices[0], axis=2), np.expand_dims(operator_matrices[1], axis=2)),
                           axis=2),
            np.concatenate((np.expand_dims(operator_matrices[2], axis=2), np.expand_dims(operator_matrices[3], axis=2)),
                           axis=2)]
        temp_logistics_operators = []
        for operator_index, operator in enumerate(operator_matrices):
            segmentation_lists, number_of_ROI = get_clusters(operator)
            distance_per_segment = []
            for seg_index, segmentation in enumerate(segmentation_lists):
                temp_animal = copy.deepcopy(animals_data_list[index])
                for train, date in enumerate(temp_animal.keys()):
                    temp_animal[date]['imagingData.samples'] = extract_rows(temp_animal[date]['imagingData.samples'],
                                                                            segmentation)
                    temp_animal[date]['imagingData.roiNames'] = extract_rows(temp_animal[date]['imagingData.roiNames'],
                                                                             segmentation)

                cov_matrices = []
                for date in temp_animal.keys():
                    imaging = np.array(temp_animal[date]['imagingData.samples'])
                    concatenated_matrix = imaging.reshape(np.shape(imaging)[0], -1)
                    cov_matrix = np.corrcoef(concatenated_matrix, rowvar=True)
                    cov_matrices.append(cov_matrix)
                    window_length = np.shape(imaging)[0]

                cov_matrices = np.array(cov_matrices)

                diffusion_representations, distances = get_diffusion_embedding(cov_matrices, window_length)
                distances_test = distances.squeeze(0)
                expert_distances = distances_test[-1]
                array_min = expert_distances.min()
                array_max = expert_distances.max()

                # Normalized array to range [0, 1]
                normalized_array = (expert_distances - array_min) / (array_max - array_min)
                distance_per_segment.append(interpolate_points(normalized_array, -1))
            distance_per_operator.append(distance_per_segment)
        distance_across_animals.append(distance_per_operator)
    plot_graphs_for_operators(distance_across_animals, 'Riemannian distances from expert session')


def calculate_corrs(diffusion_operators):
    Mean_cov_across_animals = []
    for index, animal_operators in tqdm(enumerate(diffusion_operators.keys()), total=len(diffusion_operators)):
        if animal_operators == 'Help':
            continue
        Mean_cov_per_operator = []
        operator_matrices = []
        for operator_index, diffusion_operator in enumerate(diffusion_operators[animal_operators]):
            total_evecs = get_eigenvectors_matrix(diffusion_operator, eigenvalue=0)
            operator_matrices.append(total_evecs)
        operator_matrices = [
            np.concatenate((np.expand_dims(operator_matrices[0], axis=2), np.expand_dims(operator_matrices[1], axis=2)),
                           axis=2),
            np.concatenate((np.expand_dims(operator_matrices[2], axis=2), np.expand_dims(operator_matrices[3], axis=2)),
                           axis=2)]
        for operator_index, operator in enumerate(operator_matrices):
            segmentation_lists, number_of_ROI = get_clusters(operator)
            Mean_cov_operator = []
            for segmentation in segmentation_lists:
                temp_animal = copy.deepcopy(animals_data_list[index])
                Mean_cov = []
                for train, date in enumerate(temp_animal.keys()):
                    cov_matrices_dates = []
                    temp_animal[date]['imagingData.samples'] = extract_rows(temp_animal[date]['imagingData.samples'],
                                                                            segmentation)
                    temp_animal[date]['imagingData.roiNames'] = extract_rows(temp_animal[date]['imagingData.roiNames'],
                                                                             segmentation)

                    for trial in range(np.shape(temp_animal[date]['imagingData.samples'])[2]):
                        cov_matrix = np.corrcoef(temp_animal[date]['imagingData.samples'][:, :, trial], rowvar=True)
                        cov_matrices_dates.append(np.abs(cov_matrix))
                    Mean_cov.append(np.mean(np.mean(cov_matrices_dates, axis=(1, 2)), axis=0))
                Mean_cov_operator.append(interpolate_points(Mean_cov, -1))
            Mean_cov_per_operator.append(Mean_cov_operator)
        Mean_cov_across_animals.append(Mean_cov_per_operator)

    plot_graphs_for_operators(Mean_cov_across_animals, 'Mean Covariance Score')


def analyze_swc_rois(file_path):
    # Define column names based on the SWC structure
    col_names = ['ID', 'Type', 'X', 'Y', 'Z', 'Parent', 'Dist', 'Label', 'VID']

    # Read the SWC file into a DataFrame
    data = pd.read_csv(file_path, sep='\s+', names=col_names)
    # Find the first bifurcation point (first node with multiple children)
    parent_ids = data['Parent']
    unique_parents = parent_ids.unique()
    first_bifurcation = -1
    for parent in unique_parents:
        child_nodes = data[data['Parent'] == parent]
        if len(child_nodes) > 1:
            first_bifurcation = parent
            break

    if first_bifurcation == -1:
        raise ValueError("No bifurcation point found")

    # Find the two children of the bifurcation point
    child_nodes = data[data['Parent'] == first_bifurcation]['ID'].tolist()

    # Half 1: Nodes after the first child
    half1 = []
    queue = [child_nodes[0]]  # Start with the first child
    while queue:
        current = queue.pop(0)
        half1.append(current)
        queue.extend(data[data['Parent'] == current]['ID'].tolist())

    # Half 2: Nodes after the second child
    half2 = []
    queue = [child_nodes[1]]  # Start with the second child
    while queue:
        current = queue.pop(0)
        half2.append(current)
        queue.extend(data[data['Parent'] == current]['ID'].tolist())

    # Extract ROIs for both halves
    rois_half1 = data[data['ID'].isin(half1)]['Label']
    rois_half1 = rois_half1[rois_half1.str.contains('roi', case=False)].tolist()
    rois_half1 = [int(roi.replace('roi', '')) for roi in rois_half1]

    rois_half2 = data[data['ID'].isin(half2)]['Label']
    rois_half2 = rois_half2[rois_half2.str.contains('roi', case=False)].tolist()
    rois_half2 = [int(roi.replace('roi', '')) for roi in rois_half2]

    return rois_half1, rois_half2


def get_neuron_split_dictionary():
    operator_names = ['Inter trial', 'Between days']
    total_dictionary = {}
    for index, animal_operators in enumerate(diffusion_operators.keys()):
        if animal_operators == 'Help':
            continue
        operator_matrices = []
        for operator_index, diffusion_operator in enumerate(diffusion_operators[animal_operators]):
            total_evecs = get_eigenvectors_matrix(diffusion_operator, eigenvalue=0)
            operator_matrices.append(total_evecs)
        operator_matrices = [
            np.concatenate((np.expand_dims(operator_matrices[0], axis=2), np.expand_dims(operator_matrices[1], axis=2)),
                           axis=2),
            np.concatenate((np.expand_dims(operator_matrices[2], axis=2), np.expand_dims(operator_matrices[3], axis=2)),
                           axis=2)]
        for operator_index, operator in enumerate(operator_matrices):
            segmentation_lists, number_of_ROI = get_clusters(operator)
            for session in animals_data_list[index]:
                important_roi_before = animals_data_list[index][session]['imagingData.roiNames'][
                    segmentation_lists[0], 0]
                important_roi_after = animals_data_list[index][session]['imagingData.roiNames'][
                    segmentation_lists[1], 0]
                break

            swc_files = glob.glob(os.path.join(dirs[index], '**', '*.swc'), recursive=True)

            # Create a dictionary to store the full path for each unique filename
            unique_swc_files = {}
            for swc_file in swc_files:
                filename = os.path.basename(swc_file)
                if filename not in unique_swc_files:
                    unique_swc_files[filename] = swc_file
            # Print the unique filenames with their full paths
            for filename, full_path in unique_swc_files.items():
                rois_half1, rois_half2 = analyze_swc_rois(full_path)
                total_dictionary[f'{animal_names[index]} {operator_names[operator_index]} {filename[:-4]}'] = {}
                total_dictionary[f'{animal_names[index]} {operator_names[operator_index]} {filename[:-4]}'][
                    'before'] = len(np.intersect1d(important_roi_before, rois_half1 + rois_half2))
                total_dictionary[f'{animal_names[index]} {operator_names[operator_index]} {filename[:-4]}'][
                    'after'] = len(np.intersect1d(important_roi_after, rois_half1 + rois_half2))

    total_dictionary = dict(sorted(total_dictionary.items(), key=lambda item: (item[1]['before'], item[1]['after'])))

    return total_dictionary


def plot_all_animal_neurons(neurons_data):
    num_neurons = int(len(neurons_data) / 2)
    labels = []

    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 6))

    # Prepare data
    before_vals = []
    after_vals = []

    for i, neuron in enumerate(neurons_data):
        if 'Inter' in neuron:
            continue
        before_vals.extend([neurons_data[neuron]['before']])
        after_vals.extend([neurons_data[neuron]['after']])

        labels.extend([neuron])

    # X-axis positions
    ind = np.arange(num_neurons)

    # Plot stacked bars
    ax.bar(ind, before_vals, width=0.4, label='Important', color='blue')
    ax.bar(ind, after_vals, width=0.4, bottom=before_vals, label='Unimportant', color='green')

    # Add labels and title
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Neuron split into important and unimportant")
    ax.set_ylabel("Number of ROI")
    ax.legend()

    # Show the plot
    plt.tight_layout()
    #plt.savefig(f'Neuron split into important and unimportant.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':

    for index, animal in enumerate(animals_data_list):
        common_numbers = None
        for train, date in enumerate(animal.keys()):
            ROI_list = animal[date]['imagingData.roiNames']
            check_rows(ROI_list)
            if train == 0:
                common_numbers = set(np.unique(ROI_list))
            else:
                matrix_numbers = set(np.unique(ROI_list))
                common_numbers = common_numbers.intersection(matrix_numbers)
        print(f'Number of common ROIs for {animal_names[index]}: {len(common_numbers)}')
        for train, date in enumerate(animal.keys()):
            common_ROI_rows = find_rows_with_common_numbers(animal[date]['imagingData.roiNames'], common_numbers)
            animal[date]['imagingData.samples'] = extract_rows(animal[date]['imagingData.samples'], common_ROI_rows)
            animal[date]['imagingData.roiNames'] = extract_rows(animal[date]['imagingData.roiNames'], common_ROI_rows)
            animal[date]['imagingData.roiNames'], animal[date]['imagingData.samples'] = sort_matrices_by_A(
                animal[date]['imagingData.roiNames'], animal[date]['imagingData.samples'])

    if os.path.exists(r'.\diffusion_operators_single_cell.pkl'):
        with open('diffusion_operators_single_cell.pkl', 'rb') as file:
            diffusion_operators = pickle.load(file)
    else:
        diffusion_operators = calculate_diffusion_operator()

    # percent of ROI
    calculate_percent_ROI(diffusion_operators)
    # correlations
    calculate_corrs(diffusion_operators)

    # linear regression
    liner_regression_analysis(diffusion_operators)

    # logistic_regression novice expert
    if os.path.exists(r'.\diffusion_operators_logistic_single_cell.pkl'):
        with open('diffusion_operators_logistic_single_cell.pkl', 'rb') as file:
            novice_expert_dict = pickle.load(file)
    else:
        novice_expert_dict = logistic_regression_analysis(diffusion_operators)

    plot_novice_expert_results(novice_expert_dict)

    calculate_Riemannian(diffusion_operators)

    total_dictionary = get_neuron_split_dictionary()
    plot_all_animal_neurons(total_dictionary)


