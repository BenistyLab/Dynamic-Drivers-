import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import glob
import re
from analysis_tools import Initial_Analysis, logistic_regression_expert_prediction


def load_dataset(data, field):
    split_result = field.split('.')
    num_splits = len(split_result)
    # Perform actions based on the number of splits
    if num_splits == 2:
        if split_result[1] == 'loc':
            y = data[split_result[0]][split_result[1]][0,0][0]
        else:
            y = data[split_result[0]][split_result[1]][0,0]
            y = y.astype(np.float32)
    elif num_splits == 3:
        if split_result[2] == 'eventTimeStamps':
            y = data[split_result[0]][split_result[1]][0,0][split_result[2]][0,0][0].flatten()
        elif split_result[2] == 'indicatorPerTrial':
            y = data[split_result[0]][split_result[1]][0,0][split_result[2]][0,0].flatten()
        else:
            y = data[split_result[0]][split_result[1]][0,0][split_result[2]][0,0]
    else:
        y = None
        print("Invalid field")

    return y


def explore_keys(data, prefix=""):
    fields = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            fields.extend(
                explore_keys(value, new_prefix))  # Extend the fields list with the results of the recursive call

    elif isinstance(data, np.ndarray) and (data.dtype.names is not None):
        for field in data.dtype.names:
            new_prefix = f"{prefix}.{field}" if prefix else field
            fields.extend(
                explore_keys(data[field], new_prefix))  # Extend the fields list with the results of the recursive call

    try:
        if len(data.shape) >= 2:
            if isinstance(data, np.ndarray) and data[0, 0].dtype.names is not None:
                for field in data[0, 0].dtype.names:
                    new_prefix = f"{prefix}.{field}" if prefix else field
                    fields.extend(explore_keys(data[0, 0][field],
                                               new_prefix))  # Extend the fields list with the results of the recursive call
    except AttributeError:
        pass

    if not fields:  # Base case: if fields list is empty, return the current prefix
        fields.append(prefix)

    return fields


def extract_roi_numbers_from_swc(file_path):
    try:
        # Define the column names based on the SWC format
        column_names = ['ID', 'Type', 'X', 'Y', 'Z', 'Parent', 'dist', 'label', 'vid']

        # Read the SWC file, handling the header and comments
        df = pd.read_csv(file_path, comment='#', sep='\s+', names=column_names, skiprows=1)

        # Extract numbers from 'roi' labels
        def extract_roi_number(label):
            match = re.search(r'roi(\d+)', label)
            if match:
                return int(match.group(1))
            return None

        # Apply the function to the 'label' column and filter out None values
        roi_numbers = df['label'].apply(extract_roi_number).dropna().astype(int).tolist()

        return roi_numbers

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return []


def find_swc_files_and_extract_roi_numbers(directory_path):
    swc_files = glob.glob(os.path.join(directory_path, '**', '*.swc'), recursive=True)
    all_roi_numbers = {}

    for file_path in swc_files:
        roi_numbers = extract_roi_numbers_from_swc(file_path)
        if roi_numbers:
            file_name = os.path.basename(file_path)
            key_name = os.path.splitext(file_name)[0]  # Remove the file extension
            all_roi_numbers[key_name] = roi_numbers

    return all_roi_numbers


def create_dataset(subdirectories_with_data_mat, main_directory, excel_directory,
                   animal_name='OTF77', single_cell=False):
    dataset_dictionary = {}
    for subdir in subdirectories_with_data_mat:
        name = subdir
        data_path = os.path.join(main_directory, subdir) + '\data.mat'
        data = scipy.io.loadmat(data_path)
        fields = explore_keys(data)
        date_dict = {}
        for field in fields:
            add_data = 0
            split_result = field.split('.')
            num_splits = len(split_result)
            if num_splits == 2 and split_result[0] == 'imagingData':
                add_data = 1
            elif num_splits == 3 and split_result[0] == 'BehaveData':
                add_data = 1
            else:
                continue
            if add_data:
                data_inst = load_dataset(data, field)
                dict_inst = {f'{field}': data_inst}
                date_dict.update(dict_inst)  # Use setdefault to initialize dataset_dictionary[name] if it doesn't exist
            else:
                continue
        dataset_dictionary[name] = date_dict

        if single_cell:
            swc_directory_path = os.path.join(main_directory, subdir) + '\Tree'
            roi_numbers_dict = find_swc_files_and_extract_roi_numbers(swc_directory_path)
            dataset_dictionary[name]['ROI_split'] = roi_numbers_dict

        df = pd.read_excel(excel_directory, sheet_name=f'{animal_name}')
        for key in dataset_dictionary.keys():
            reversed_date = '20' + key.replace("_", "")
            filtered_df = df[df['Session Name'].str.contains(reversed_date)]
            filtered_df.reset_index(drop=True, inplace=True)
            dataset_dictionary[key]['metadata'] = filtered_df

    return dataset_dictionary


class Import_Data():
    def __init__(self, main_directory, excel_directory, animal_name):
        self.main_directory = main_directory
        self.excel_directory = excel_directory
        self.animal_name = animal_name

    def get_fields(self):
        dir_list = []
        # Get a list of all subdirectories in the main directory
        subdirectories = [d for d in os.listdir(self.main_directory) if
                          os.path.isdir(os.path.join(self.main_directory, d))]

        # Filter subdirectories that contain a file named 'data.mat'
        subdirectories_with_data_mat = [d for d in subdirectories if
                                        'data.mat' in os.listdir(os.path.join(self.main_directory, d))]

        # Print the list of subdirectories with 'data.mat' file
        for subdir in subdirectories_with_data_mat:
            dir_list.append(os.path.join(self.main_directory, subdir))

        print("Data fields to choose from:")
        data_path = dir_list[0] + '\data.mat'
        data = scipy.io.loadmat(data_path)
        explore_keys(data)

    def get_data(self, single_cell=False):
        subdirectories = [d for d in os.listdir(self.main_directory) if
                          os.path.isdir(os.path.join(self.main_directory, d))]
        subdirectories_with_data_mat = [d for d in subdirectories if
                                        'data.mat' in os.listdir(os.path.join(self.main_directory, d))]
        return create_dataset(subdirectories_with_data_mat, self.main_directory, self.excel_directory,
                              self.animal_name, single_cell)

    def help(self):
        print("\nAttributes:")
        print("- main_directory (str): The main directory containing subdirectories with data files.")
        print("- excel_directory (str): The directory containing Excel files with metadata.")
        print("- animal_name (str): The name of the animal for which the data is being imported.\n")
        print("Methods:")
        print("- get_fields(): Retrieves and prints available data fields to choose from.")
        print("- get_data(): Imports and processes data from the specified directories.")
        print("Assumptions:")
        print("- This code assumes that in the directories there exists a file called data.mat containing all" +
              " the data from the BDA/TPA files")

def plot_logistic_regression_results(path):

    with open(path) as f:
        loaded_dict = pickle.load(f)

    predictions = []
    pd_index = []
    animals = []
    for animal in loaded_dict.keys():
        predictions.append(np.array(loaded_dict[animal].predictions_expert))
        pd_index.append(loaded_dict[animal].PD_index)
        animals.append(animal)

    max_len = max(len(pred) for pred in predictions)

    columns_to_plot = np.arange(0, 23, 1)
    # Create subplots
    num_plots = len(columns_to_plot)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Create subplots in a square grid
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

    # Plot each column in a separate subplot
    for i, col_index in enumerate(columns_to_plot):
        row = i // num_cols
        col = i % num_cols
        for j, pred in enumerate(predictions):
            x_values = np.arange(-pd_index[j], max_len - pd_index[j])
            if len(x_values) > len(pred[:, col_index]):
                x_values = x_values[:len(pred[:, col_index])]
            axes[row, col].plot(x_values, pred[:, col_index], label=f'{animals[j]}')
        axes[row, col].axvline(x=0, color='r', linestyle='--',
                               label='PD')  # Plot vertical line at index 13
        axes[row, col].set_title(f'Time {col_index / 2} - {col_index / 2 + 1} [sec]')
        axes[row, col].set_xlabel('Session')
        axes[row, col].set_ylabel('Percent of expert')
        axes[row, col].set_xticks([])
    fig.legend(labels=animals + ['PD'], loc='upper right', bbox_to_anchor=(1.1, 0.95))
    fig.suptitle('Prediction per segment for expert all animals')
    plt.tight_layout()
    plt.show()


def check_rows(matrix):
    for i, row in enumerate(matrix):
        assert np.all(row == row[0]), f"Row {i} does not contain the same number in all columns: {row}"


def find_rows_with_common_numbers(matrix, common_numbers):
    rows_with_common_numbers = []
    for i, row in enumerate(matrix):
        if any(number in common_numbers for number in row):
            rows_with_common_numbers.append(i)
    return rows_with_common_numbers


def extract_rows(matrix, row_indices):
    return matrix[row_indices, :]


def sort_matrices_by_A(A, B):
    # Ensure A contains identical numbers in each row
    assert np.all(A == A[:, 0:1]), "Each row in matrix A should contain identical numbers."

    # Extract the unique number from each row (since all elements in the row are the same)
    unique_numbers = A[:, 0]

    # Get the sorted indices based on the unique numbers
    sorted_indices = np.argsort(unique_numbers)

    # Sort A and B using the sorted indices
    sorted_A = A[sorted_indices]
    sorted_B = B[sorted_indices]

    return sorted_A, sorted_B