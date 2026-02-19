from utils import Import_Data, check_rows, find_rows_with_common_numbers, extract_rows, sort_matrices_by_A
from analysis_tools import logistic_regression_expert_prediction
import torch
import numpy as np
import pickle
import os
from analysis_tools import run_linear_regression_lever_pull_with_kfold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from sklearn.decomposition import PCA
from correlation_distances_main.lib.utils import get_diffusion_embedding


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

OTF77_directory = \
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\OTF77'
OTF80_directory = \
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\OTF80'
OTF85_directory = \
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\OTF85'
OTF94_directory = \
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\OTF94'
OTF95_directory = \
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\OTF95'
OTF104_directory = \
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\OTF104'
OTF107_directory = \
    r'C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\OTF107'
excel_directory = \
    r"C:\Users\yonatan.kle\Benisty Lab Dropbox\Benisty Lab Team Folder\datasets\Ca_Motor_PD_Schiller_Mohamad\Experiment Log.xlsx"
animal_names = ['OTF77', 'OTF80', 'OTF85', 'OTF94', 'OTF95', 'OTF104', 'OTF107']

data_importer = Import_Data(OTF77_directory,excel_directory,animal_names[0])
OTF77 = data_importer.get_data()
data_importer = Import_Data(OTF80_directory,excel_directory,animal_names[1])
OTF80 = data_importer.get_data()
data_importer = Import_Data(OTF85_directory,excel_directory,animal_names[2])
OTF85 = data_importer.get_data()
data_importer = Import_Data(OTF94_directory,excel_directory,animal_names[3])
OTF94 = data_importer.get_data()
data_importer = Import_Data(OTF95_directory,excel_directory,animal_names[4])
OTF95 = data_importer.get_data()
data_importer = Import_Data(OTF104_directory,excel_directory,animal_names[5])
OTF104 = data_importer.get_data()
data_importer = Import_Data(OTF107_directory,excel_directory,animal_names[6])
OTF107 = data_importer.get_data()

animals_data_list = [OTF77, OTF80, OTF85, OTF94, OTF95,OTF104, OTF107]


def calculate_row_correlations(matrices, expert_matrix):
    correlations = []

    # Loop through each matrix in the list
    for matrix in matrices:
        matrix_correlations = []

        # Ensure the matrix and expert_matrix have the same number of rows
        assert matrix.shape[0] == expert_matrix.shape[0], "Matrix and expert_matrix must have the same number of rows"
        assert matrix.shape[1] == expert_matrix.shape[
            1], "Matrix and expert_matrix must have the same number of columns"

        # Loop through each row in the matrix
        for row_index in range(matrix.shape[0]):
            row = matrix[row_index]
            expert_row = expert_matrix[row_index]

            # Calculate the correlation coefficient between the matrix row and expert row

            corr = np.corrcoef(row, expert_row)[0, 1]
            matrix_correlations.append(corr)

        correlations.append(matrix_correlations)

    return correlations


def plot_means_and_corrs(means, corrs, PD_index):
    # Ensure both dictionaries have the same keys
    assert means.keys() == corrs.keys(), "The means and corrs dictionaries must have the same keys"

    keys = list(means.keys())
    num_keys = len(keys)

    # Create a figure with subplots, 2 columns for each key
    fig, axes = plt.subplots(num_keys, 2, figsize=(15, num_keys * 5))

    # If there's only one key, wrap axes in a list for consistent indexing
    if num_keys == 1:
        axes = [axes]

    for i, key in enumerate(keys):
        mean_values = means[key]
        corr_values = np.array(corrs[key])
        # Column right before the column of 1s
        col_to_sort_by = PD_index[i] - 2
        # Sort the matrix rows based on the column before the column of 1s
        sorted_matrix = corr_values[:, np.argsort(corr_values[col_to_sort_by, :])]

        # Plot correlation heatmap in the left column
        sns.heatmap(np.array(sorted_matrix).T, ax=axes[i][0], annot=False, cmap="coolwarm")
        axes[i][0].set_title(f'Correlation Heatmap: {key}')
        axes[i][0].set_xlabel('Session')
        axes[i][0].set_ylabel('ROI')

        # Plot means line plot in the right column
        axes[i][1].plot(mean_values, marker='o', label='Mean')
        axes[i][1].set_title(f'Mean Plot: {key}')
        axes[i][1].set_xlabel('Session')
        axes[i][1].set_ylabel('Mean Value')
        axes[i][1].axvline(x=PD_index[i], color='r', linestyle='--',
                           label='PD')
        axes[i][1].legend()

    plt.tight_layout()
    plt.show()


def mean_and_correlation_analysis():
    mean_dict = {}
    corr_dict = {}
    PD_indexes = []
    session_dates = []

    # Get the list of dates
    for index, animal in enumerate(animals_data_list):
        means = []
        session_types = []
        imaging_list = []
        dates = []
        PD_index = 0
        for i, key in enumerate(animal.keys()):
            imaging = animal[key]['imagingData.samples']
            animal[key]['imagingData.samples'] = animal[key]['imagingData.samples']
            means.append(np.mean(imaging))
            df = animal[key]['metadata']
            session_types.append(df['Session Type'][0])
            imaging_list.append(np.mean(imaging, axis=2))
            if PD_index == 0 and df['Session Type'][0] == 'PD':
                expert_index = i - 1
                PD_index = i
                PD_indexes.append(PD_index)
                expert_imaging = imaging_list[expert_index]
        session_dates.append(dates)
        corr_dict[animal_names[index]] = calculate_row_correlations(imaging_list, expert_imaging)
        mean_dict[animal_names[index]] = means

    plot_means_and_corrs(mean_dict, corr_dict, PD_indexes)
    return PD_indexes


def novice_expert_analysis():
    logistic_regression_results_dict = {}

    for index, animal in enumerate(animals_data_list):
        temp_logistic = logistic_regression_expert_prediction(animal, animal_names[index])
        logistic_regression_results_dict[animal_names[index]] = temp_logistic
        # temp_logistic.plot_heat_map(reference='expert')
        # temp_logistic.plot_heat_map(reference='PD')
        # temp_logistic.plot_heat_map(reference='exvex')
        # temp_logistic.plot_prediction_per_segment(reference='expert')
        # temp_logistic.plot_prediction_per_segment(reference='PD')
        # temp_logistic.plot_prediction_per_segment(reference='exvex')

    with open('logistic_regression_analysis.pkl', 'wb') as f:
        pickle.dump(logistic_regression_results_dict, f)


def plot_novice_expert_results(logistic_regression_results_dict):
    novice_expert_all = []
    novice_expertPD_all = []
    novice_expert_expertPD_all = []
    for result in logistic_regression_results_dict.keys():
        novice_expert_all.append(logistic_regression_results_dict[result].predictions_expert_interp)
        novice_expertPD_all.append(logistic_regression_results_dict[result].predictions_expert_PD_interp)
        novice_expert_expertPD_all.append(logistic_regression_results_dict[result].predictions_expert_v_PD_interp)

    novice_expert_all = np.mean(novice_expert_all, axis=0)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(novice_expert_all, cmap="viridis")
    plt.title("Heatmap of Mean Novice expert")
    plt.show()

    novice_expertPD_all = np.mean(novice_expertPD_all, axis=0)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(novice_expertPD_all, cmap="viridis")
    plt.title("Heatmap of Mean Novice expert PD")
    plt.show()

    novice_expert_expertPD_all = np.mean(novice_expert_expertPD_all, axis=0)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(novice_expert_expertPD_all, cmap="viridis")
    plt.title("Heatmap of Mean expert - expert PD")
    plt.show()


def linear_regression_analysis(plot_distances_matrix=True):
    linear_regression_results_dict = {}

    for index, animal in enumerate(animals_data_list):
        mse_mean_per_date, r2_mean_per_date, mean_non_zero_weights, success, activity, mean_coefficients, PD_index = (
            run_linear_regression_lever_pull_with_kfold(animal, animal_names[index], k=5, tone=120,
                                                        regularization_type='L1', interp_points=True))
        linear_regression_results_dict[animal_names[index]] = [mse_mean_per_date, r2_mean_per_date,
                                                               mean_non_zero_weights, success, activity,
                                                               mean_coefficients, PD_index]
        if plot_distances_matrix:
            distance_matrix = squareform(pdist(mean_coefficients, metric='euclidean'))
            sns.heatmap(distance_matrix, cmap='viridis')
            plt.title(f'Pairwise Distance Heatmap for coefficients {animal_names[index]}')
            plt.xlabel('Coefficient Index')
            plt.ylabel('Coefficient Index')
            plt.show()
    with open('linear_regression_analysis.pkl', 'wb') as f:
        pickle.dump(linear_regression_results_dict, f)

    return linear_regression_results_dict


def plot_linear_regression_results(linear_regression_results_dict):
    r2_mean_per_date_all = []
    mean_non_zero_weights_all = []
    success_all = []
    activity_all = []

    for key, value in linear_regression_results_dict.items():
        r2_mean_per_date_all.append(value[1])
        mean_non_zero_weights_all.append(value[2])
        success_all.append(value[3])
        activity_all.append(value[4])
    # Compute the mean values of each metric
    mean_r2 = np.mean(r2_mean_per_date_all, axis=0)
    std_r2 = np.std(r2_mean_per_date_all, axis=0) / (len(r2_mean_per_date_all) - 1)

    mean_non_zero_weights = np.mean(mean_non_zero_weights_all, axis=0)
    std_non_zero_weights = np.std(mean_non_zero_weights_all, axis=0) / (len(mean_non_zero_weights_all) - 1)

    mean_success = np.mean(success_all, axis=0)
    std_success = np.std(success_all, axis=0) / (len(success_all) - 1)

    mean_activity = np.mean(activity_all, axis=0)
    std_activity = np.std(activity_all, axis=0) / (len(activity_all) - 1)

    # Plot the mean values in subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    # Plot for R2 Mean Per Date
    axs[0, 0].errorbar(range(len(mean_r2)), mean_r2, yerr=std_r2, fmt='o', color='blue', capsize=5)
    axs[0, 0].axvline(x=16, color='r', linestyle='--')
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('Mean Value')
    axs[0, 0].set_title('R2 Mean Per Date')

    # Plot for Mean Non-Zero Weights
    axs[0, 1].errorbar(range(len(mean_non_zero_weights)), mean_non_zero_weights, yerr=std_non_zero_weights, fmt='o',
                       color='green', capsize=5)
    axs[0, 1].axvline(x=16, color='r', linestyle='--')
    axs[0, 1].set_xlabel('Index')
    axs[0, 1].set_ylabel('Mean Value')
    axs[0, 1].set_title('Mean Non-Zero Weights')

    # Plot for Success
    axs[1, 0].errorbar(range(len(mean_success)), mean_success, yerr=std_success, fmt='o', color='red', capsize=5)
    axs[1, 0].axvline(x=16, color='r', linestyle='--')
    axs[1, 0].set_xlabel('Index')
    axs[1, 0].set_ylabel('Mean Value')
    axs[1, 0].set_title('Success')

    # Plot for Activity
    axs[1, 1].errorbar(range(len(mean_activity)), mean_activity, yerr=std_activity, fmt='o', color='purple', capsize=5)
    axs[1, 1].axvline(x=16, color='r', linestyle='--')
    axs[1, 1].set_xlabel('Index')
    axs[1, 1].set_ylabel('Mean Value')
    axs[1, 1].set_title('Activity')

    plt.suptitle('Mean Values of Metrics with Error Bars')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def PCA_analysis():
    pca_transformers = []
    # Assume you have a list of matrices
    for index, animal in enumerate(animals_data_list):
        imaging_list = []
        for i, key in enumerate(animal.keys()):
            imaging = np.array(animal[key]['imagingData.samples'])
            imaging_list.append(imaging)
        # concatenate over dates
        concatenated_matrix = np.concatenate(imaging_list, axis=2)
        # concatenate over trials
        reshaped_matrix = np.array(concatenated_matrix.reshape(concatenated_matrix.shape[0], -1))

        # Perform PCA to reduce dimensions to 3
        pca = PCA(n_components=3)
        pca = pca.fit(reshaped_matrix.T)
        pca_transformers.append(pca)

    for index, animal in enumerate(animals_data_list):
        date_dict = {}
        for i, key in enumerate(animal.keys()):
            trail_dict = []
            imaging = np.array(animal[key]['imagingData.samples'])
            for j in range(np.shape(imaging)[2]):
                trail_dict.append(pca_transformers[index].transform(np.array(imaging[:, :, j]).T))
            mean_res = np.mean(trail_dict, axis=0)
            date_dict[key] = mean_res

        for idx, key in enumerate(date_dict.keys()):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            pca_result = date_dict[key]
            colors = np.linspace(0, 1, len(pca_result))

            # Extract the PCA components
            xs = pca_result[:, 0]
            ys = pca_result[:, 1]
            zs = pca_result[:, 2]

            # Plot with gradient colors
            ax.plot(xs, ys, zs, color='k', linestyle='-', alpha=0.1)
            ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o')
            # Add legend, labels, and title
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            ax.set_title(f'3D PCA Plot for {animal_names[index]} {key}')

            # Show the plot
            plt.savefig(f'3d_pca_plot_{animal_names[index]}_{key}.png')
            plt.close(fig)


def diffusion_map_analysis(PD_indexes):
    for index, animal in enumerate(animals_data_list):
        cov_matrices = []

        for date in animal.keys():
            imaging = np.array(animal[date]['imagingData.samples'])
            concatenated_matrix = imaging.reshape(np.shape(imaging)[0], -1)
            cov_matrix = np.corrcoef(concatenated_matrix, rowvar=True)
            cov_matrices.append(cov_matrix)
            window_length = np.shape(imaging)[0]

        cov_matrices = np.array(cov_matrices)
        print(np.shape(cov_matrices))

        diffusion_representations, distances = get_diffusion_embedding(cov_matrices, window_length)
        print(np.shape(diffusion_representations))

        distances_test = distances.squeeze(0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(distances_test, annot=False, cmap='viridis')

        # Add labels
        plt.title(f'Heatmap of the distances Matrix for {animal_names[index]}')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')

        # Show the plot
        plt.show()

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        colors = np.linspace(0, 1, len(diffusion_representations[0, 0, :]))

        xs = diffusion_representations[0, 0, :]
        ys = diffusion_representations[0, 1, :]
        zs = diffusion_representations[0, 2, :]

        # Plot with gradient colors
        ax.plot(xs, ys, zs, color='k', linestyle='-', alpha=0.1)
        sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o')
        ax.scatter(xs[PD_indexes[index]], ys[PD_indexes[index]], zs[PD_indexes[index]], color='red', s=100, marker='o',
                   edgecolor='black', label=f'PD date')
        # Add legend, labels, and title
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('session number')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'3D Diffusion map Plot for {animal_names[index]}')
        ax.legend()


if __name__ == '__main__':

    for animal in animals_data_list:
        common_numbers = None
        for train, date in enumerate(animal.keys()):
            ROI_list = animal[date]['imagingData.roiNames']
            check_rows(ROI_list)
            if train == 0:
                common_numbers = set(np.unique(ROI_list))
            else:
                matrix_numbers = set(np.unique(ROI_list))
                common_numbers = common_numbers.intersection(matrix_numbers)
        print(len(common_numbers))
        for train, date in enumerate(animal.keys()):
            common_ROI_rows = find_rows_with_common_numbers(animal[date]['imagingData.roiNames'], common_numbers)
            animal[date]['imagingData.samples'] = extract_rows(animal[date]['imagingData.samples'], common_ROI_rows)
            animal[date]['imagingData.roiNames'] = extract_rows(animal[date]['imagingData.roiNames'], common_ROI_rows)
            animal[date]['imagingData.roiNames'], animal[date]['imagingData.samples'] = sort_matrices_by_A(
                animal[date]['imagingData.roiNames'], animal[date]['imagingData.samples'])

    PD_indexes = mean_and_correlation_analysis()

    # logistic_regression novice expert
    if os.path.exists(r'.\logistic_novice_expert.pkl'):
        with open('logistic_novice_expert.pkl', 'rb') as file:
            novice_expert_dict = pickle.load(file)
    else:
        novice_expert_dict = novice_expert_analysis()

    plot_novice_expert_results(novice_expert_dict)

    if os.path.exists(r'.\linear_regression_analysis.pkl'):
        with open('linear_regression_analysis.pkl', 'rb') as file:
            linear_regression_results_dict = pickle.load(file)
    else:
        linear_regression_results_dict = linear_regression_analysis(plot_distances_matrix=False)

    plot_linear_regression_results(linear_regression_results_dict)

    PCA_analysis()

    diffusion_map_analysis(PD_indexes)

