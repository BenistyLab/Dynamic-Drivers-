import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
from scipy.interpolate import interp1d
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def split_signal_into_segments(signals, segment_length, hop_length, label=None):
    segmented_signals = []
    labels = []
    for signal in signals:
        signal_length = len(signal)
        num_segments = int((signal_length - segment_length) / hop_length) + 1
        segments = []
        label_temp = []

        for j in range(num_segments):
            start_index = int(j * hop_length)
            end_index = int(start_index + segment_length)
            segment = signal[start_index:end_index]
            segments.append(np.array(segment))
            if label is not None:
                label_temp.append(label)

        segmented_signals.append(segments)
        labels.append(label_temp)

    return np.array(segmented_signals), np.array(labels)


def run_logistic_regression_with_kfold(X, y, k=5):
    # kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_shuffled)
    X_scaled = scaler.transform(X_shuffled)
    lambdas = [0.001, 0.01, 0.05, 0.1, 1., 100., 1000., 10000.]
    # i = 0
    # for train_index, test_index in kf.split(X_scaled):
    #     i += 1
    #     train_index = np.array(train_index)
    #     test_index = np.array(test_index)
    #
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     # Initialize logistic regression model
    #     model = LogisticRegressionCV(Cs=lambdas, penalty='l1', solver='liblinear',
    #                                  cv=5, random_state=42, max_iter=200)
    #
    #     # Fit the model on the training data
    #     model.fit(X_train, y_train)
    #
    #     # Predict on the test data
    #     y_pred = model.predict(X_test)
    #
    #     # Calculate accuracy
    #     accuracy = accuracy_score(y_test, y_pred)
    #     accuracies.append(accuracy)

    model_fin = LogisticRegressionCV(Cs=lambdas, penalty='l1', solver='liblinear',
                                     cv=5, random_state=42, max_iter=200)
    model_fin.fit(X_scaled, y_shuffled)

    mean_accuracy = 0 # np.mean(accuracies) TODO: fix and change no need for accuracy return
    return mean_accuracy, model_fin, scaler


def interpolate_points(data, expert_index, points_before=15, points_after=10):
    if expert_index == -1:
        sessions_before = np.linspace(0, 1, len(data))
        sessions_before_interp = np.linspace(0, 1, points_before)
        interp_func_b4 = interp1d(sessions_before, data, kind='linear')
        interpolated_data = interp_func_b4(sessions_before_interp)
    else:
        sessions_before = np.linspace(0, 1, expert_index)
        sessions_before_interp = np.linspace(0, 1, points_before)
        sessions_after = np.linspace(0, 1, len(data) - expert_index)
        sessions_after_interp = np.linspace(0, 1, points_after)
        interp_func_b4 = interp1d(sessions_before, data[:expert_index], kind='linear')
        interp_func_after = interp1d(sessions_after, data[expert_index:], kind='linear')
        interpolated_data_before = interp_func_b4(sessions_before_interp)
        interpolated_data_after = interp_func_after(sessions_after_interp)

        interpolated_data = np.concatenate((interpolated_data_before, interpolated_data_after))

    return interpolated_data


def interpolate_heatmap(heatmap, expert_index, points_before=15, points_after=10):
    # Number of columns in the heatmap
    num_columns = heatmap.shape[1]

    # Initialize an array to hold the interpolated heatmap
    if expert_index == -1:
        interpolated_heatmap = np.zeros((points_before, num_columns))
    else:
        interpolated_heatmap = np.zeros((points_before + points_after, num_columns))

    # Loop through each column and interpolate
    for i in range(num_columns):
        column_data = heatmap[:, i]
        interpolated_column = interpolate_points(column_data, expert_index, points_before, points_after)
        interpolated_heatmap[:, i] = interpolated_column

    return interpolated_heatmap


def run_linear_regression_lever_pull_with_kfold(data, animal_name, k=5, tone=120,
                                                regularization_type='L1', interp_points=True, plot=True):

    r2_mean_per_date = []
    mse_mean_per_date = []
    mean_non_zero_weights = []
    success = []
    imaging_mean = []
    mean_coefficients = []
    PD_index = 0
    for index, date in enumerate(data.keys()):
        df = data[date]['metadata']
        if df['Session Type'][0] == 'PD' and PD_index == 0:
            PD_index = index
        if 'BehaveData.failure.indicatorPerTrial' in data[date].keys():
            fail_indicator = data[date]['BehaveData.failure.indicatorPerTrial']
            success.append(sum(1 - fail_indicator)/len(fail_indicator))
        elif 'BehaveData.success.indicatorPerTrial' in data[date].keys():
            success_indicator = data[date]['BehaveData.success.indicatorPerTrial']
            success.append(sum(success_indicator) / len(success_indicator))
        else:
            print("no success/fail indicator for this data")
        coefficients = []
        imaging = data[date]['imagingData.samples']
        imaging_mean.append(np.mean(imaging))
        lever_pull = data[date]['BehaveData.position.eventTrace']

        if regularization_type == 'L1':
            model = LassoCV(cv=5, tol=0.001)
        elif regularization_type == 'L2':
            model = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5)
        elif regularization_type == 'None':
            model = LinearRegression()
        else:
            raise ValueError("Invalid regularization type. Choose 'L1', 'L2', or 'None'.")
        X_train = imaging[:, tone - 5:, :]
        X_train = X_train.reshape((X_train.shape[0], -1))
        y_train = lever_pull[tone - 5:, :]
        y_train = y_train.flatten()
        # Train the model
        model.fit(X_train.T, y_train)
        weights = model.coef_
        # Evaluate the model
        y_pred = model.predict(X_train.T)
        coefficients.append(np.array(model.coef_))

        r2_mean_per_date.append(max(r2_score(y_train, y_pred), 0))
        mse_mean_per_date.append(mean_squared_error(y_train, y_pred))
        mean_non_zero_weights.append(np.count_nonzero(weights)/len(imaging))
        mean_coefficients.append(np.mean(coefficients, axis=0))

    if interp_points:
        # interpolate points
        r2_mean_per_date = interpolate_points(r2_mean_per_date, PD_index-1)
        imaging_mean = interpolate_points(imaging_mean, PD_index-1)
        mean_non_zero_weights = interpolate_points(mean_non_zero_weights, PD_index-1)
        success = interpolate_points(success, PD_index-1)
        PD_index = 16
    if plot:
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        # Plot nonzero weights
        axs[0,0].plot(mean_non_zero_weights)
        axs[0,0].axvline(x=PD_index, color='r', linestyle='--')
        axs[0,0].set_title('Percent of Nonzero Weights')
        axs[0,0].set_xlabel('Session')
        axs[0,0].set_ylabel('Nonzero Count')

        # Plot Success
        axs[1,0].plot(success)
        axs[1,0].axvline(x=PD_index, color='r', linestyle='--')
        axs[1,0].set_title('Success percentage')
        axs[1,0].set_xlabel('Session')
        axs[1,0].set_ylabel('Success')

        # Plot R^2 score
        axs[0,1].plot(r2_mean_per_date)
        axs[0,1].axvline(x=PD_index, color='r', linestyle='--')
        axs[0,1].set_title('R^2 Score')
        axs[0,1].set_xlabel('Session')
        axs[0,1].set_ylabel('R^2 Score')

        # Plot mean activity score
        axs[1, 1].plot(imaging_mean)
        axs[1, 1].axvline(x=PD_index, color='r', linestyle='--')
        axs[1, 1].set_title('Mean ROI activity')
        axs[1, 1].set_xlabel('Session')
        axs[1, 1].set_ylabel('Activity')

        fig.suptitle(f'Linear regression for {animal_name}')
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show()

    return (mse_mean_per_date, r2_mean_per_date, mean_non_zero_weights, success,
            imaging_mean, mean_coefficients, PD_index)


class Initial_Analysis():
    def __init__(self, dataset):
        means = []
        mean_trace = []
        self.session_types = []
        self.means_session_type = {}
        self.mean_sig_session_type = {}
        for key in dataset.keys():
            imaging = dataset[key]['imagingData.samples']
            means.append(np.mean(imaging))
            mean_trace.append(np.mean(np.mean(imaging, axis=2), axis=0))
            df = dataset[key]['metadata']
            self.session_types.append(df['Session Type'][0])

        for index, session in enumerate(self.session_types):
            if session in self.means_session_type:
                self.means_session_type[session].append(means[index])
                self.mean_sig_session_type[session].append(mean_trace[index])
            else:
                self.means_session_type[session] = [means[index]]
                self.mean_sig_session_type[session] = [mean_trace[index]]
        for i in range(len(self.session_types)):
            if self.session_types[i] == 'PD':
                self.expert_index = i - 1
                break
        self.num_traces = len(mean_trace)

        self.corr_matrix = np.zeros((self.num_traces, self.num_traces))
        for i in range(self.num_traces):
            for j in range(self.num_traces):
                self.corr_matrix[i, j] = np.corrcoef(mean_trace[i], mean_trace[j])[0, 1]

    def get_mean_value(self):
        mean_value = []
        for key in self.means_session_type.keys():
            mean_value.append(np.mean(self.means_session_type[key]))
        return mean_value

    def get_mean_signal(self):
        mean_signal = []
        for key in self.mean_sig_session_type.keys():
            mean_signal.append(np.mean(self.mean_sig_session_type[key]))
        return mean_signal

    def get_corr_with_expert_signal(self):
        return self.corr_matrix[self.expert_index]

    def plot_corr_matrix(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation Coefficient')
        plt.title('Correlation Coefficients between Traces')
        plt.xlabel('Trace Index')
        plt.ylabel('Trace Index')
        plt.xticks(np.arange(self.num_traces), np.arange(self.num_traces) + 1)  # Adjust x-axis ticks
        plt.yticks(np.arange(self.num_traces), np.arange(self.num_traces) + 1)  # Adjust y-axis ticks
        plt.show()
        return

    def get_corr_by_session_type(self):
        corr_w_expert = self.get_corr_with_expert_signal()
        corr = []
        corr_session_type = {}
        for index, session in enumerate(self.session_types):
            if session in corr_session_type:
                if index != self.expert_index:
                    corr_session_type[session].append(corr_w_expert[index])
                else:
                    continue
            else:
                corr_session_type[session] = [corr_w_expert[index]]
        for key in corr_session_type.keys():
            corr.append(np.mean(corr_session_type[key]))
        return corr

    def help(self):
        print("This class performs initial analysis on a given dataset.")
        print("You can create an object of this class by passing the dataset as an argument to the constructor.")
        print("After creating an object, you can use the following methods:")
        print("1. get_mean_value() - Returns the mean value for each session type.")
        print("2. get_mean_signal() - Returns the mean signal for each session type.")
        print("3. get_corr_with_expert_signal() - Returns the correlation with the expert signal.")
        print("4. plot_corr_matrix() - Plots the correlation matrix between traces.")
        print("5. get_corr_by_session_type() - Returns the correlation by session type.")


class logistic_regression_expert_prediction():
    def __init__(self,dataset,animal_name, test=False, interp=True):
        self.PD_index = 0
        self.animal_name = animal_name

        for index, key in enumerate(dataset.keys()):
            if index == 0:
                imaging_novice = dataset[key]['imagingData.samples']
            df = dataset[key]['metadata']
            if df['Session Type'][0] == 'PD':
                self.PD_index = index
                break
            imaging_expert = dataset[key]['imagingData.samples']

        self.predictions_expert, self.accuracy_of_models_expert, self.models_expert, self.scalers_expert =\
            self.novice_expert_analysis(imaging_novice, imaging_expert, dataset)
        if interp:
            self.predictions_expert_interp = interpolate_heatmap(np.array(self.predictions_expert), self.PD_index - 1)

        if not test:
            nov_flag = 0
            for index, key in enumerate(dataset.keys()):
                df = dataset[key]['metadata']
                if df['Session Type'][0] == 'PD' and nov_flag == 0:
                    nov_flag = 1
                    imaging_novice_PD = dataset[key]['imagingData.samples']
                imaging_expert_PD = dataset[key]['imagingData.samples']

            self.predictions_expert_PD, self.accuracy_of_models_expert_PD, self.models_expert_PD, self.scalers_expert_PD = \
                self.novice_expert_analysis(imaging_novice_PD, imaging_expert_PD, dataset)
            if interp:
                self.predictions_expert_PD_interp = interpolate_heatmap(np.array(self.predictions_expert_PD), self.PD_index - 1)

            self.predictions_expert_v_PD, self.accuracy_of_models_expert_v_PD, self.models_expert_v_PD, self.scalers_expert_v_PD = \
                self.novice_expert_analysis(imaging_expert, imaging_expert_PD, dataset)
            if interp:
                self.predictions_expert_v_PD_interp = interpolate_heatmap(np.array(self.predictions_expert_v_PD), self.PD_index - 1)


    def plot_heat_map(self, reference='expert'):
        if reference == 'expert':
            predictions = self.predictions_expert
            title = f'Prediction of Expert for novice vs expert {self.animal_name}'
        elif reference == 'PD':
            predictions = self.predictions_expert_PD
            title = f'Prediction of Expert for novice PD vs expert PD {self.animal_name}'
        else:
            predictions = self.predictions_expert_v_PD
            title = f'Prediction of Expert for expert vs expert PD {self.animal_name}'
        plt.figure(figsize=(8, 6))
        plt.imshow(predictions, cmap='inferno', aspect='auto')
        plt.colorbar(label='Value')
        plt.title(title)
        plt.xlabel('Segment Number')
        plt.ylabel('Session Number')
        plt.show()

    def novice_expert_analysis(self, imaging_novice, imaging_expert, dataset):
        accuracy_of_models_expert = []
        models_expert = []
        scalers_expert = []
        predictions_expert = []
        imaging_novice = np.array(np.mean(imaging_novice, axis=0)).T
        imaging_expert = np.array(np.mean(imaging_expert, axis=0)).T

        segment_length = len(imaging_novice[0]) / (len(imaging_novice[0])/30)
        hop_length = segment_length / 2
        labels = []
        segmented_signals = []

        for i in range(len(imaging_novice[:, 0])):
            temp_sig, temp_lab = split_signal_into_segments(imaging_novice,
                                                            segment_length, hop_length, label=0)
            labels.extend(temp_lab)
            segmented_signals.extend(temp_sig)

        for i in range(len(imaging_expert[:, 0])):
            temp_sig, temp_lab = split_signal_into_segments(imaging_expert,
                                                            segment_length, hop_length, label=1)
            labels.extend(temp_lab)
            segmented_signals.extend(temp_sig)

        labels = np.array(labels)
        segmented_signals = np.array(segmented_signals)

        for i in tqdm(range(len(segmented_signals[0])), desc="Processing segments"):
            X = segmented_signals[:, i, :]
            y = labels[:, i]
            mean_temp, model_temp, temp_scaler = run_logistic_regression_with_kfold(X, y, k=5)
            accuracy_of_models_expert.append(mean_temp)
            models_expert.append(model_temp)
            scalers_expert.append(temp_scaler)

        for index, key in enumerate(dataset.keys()):
            imaging = dataset[key]['imagingData.samples']
            imaging = np.array(np.mean(imaging, axis=0)).T
            segment_length = len(imaging[0]) / (len(imaging_novice[0])/30)
            hop_length = segment_length / 2
            segmented_signals = []
            probs = []

            for i in range(len(imaging[:, 0])):
                temp_sig, _ = split_signal_into_segments(imaging,
                                                         segment_length, hop_length)
                segmented_signals.extend(temp_sig)

            segmented_signals = np.array(segmented_signals)

            for i, model in enumerate(models_expert):
                X = segmented_signals[:, i, :]
                X_scaled = scalers_expert[i].transform(X)
                predict = model.predict(X_scaled)
                probs.append(np.sum(predict) / len(predict))
            #print(f"For {key}, The predicted probabilities are: {probs}")
            predictions_expert.append(probs)

        return predictions_expert, accuracy_of_models_expert, models_expert, scalers_expert

    def plot_prediction_per_segment(self, reference='expert'):
        if reference == 'expert':
            predictions = self.predictions_expert
            title = f'Prediction per segment for novice vs expert {self.animal_name}'
        elif reference == 'PD':
            predictions = self.predictions_expert_PD
            title = f'Prediction per segment for novice PD vs expert PD {self.animal_name}'
        else:
            predictions = self.predictions_expert_v_PD
            title = f'Prediction per segment for expert vs expert PD {self.animal_name}'

        columns_to_plot = np.arange(0, 9, 1)
        predictions = np.array(predictions)
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
            axes[row, col].plot(predictions[:, col_index])
            axes[row, col].axvline(x=self.PD_index, color='r', linestyle='--',
                                   label='PD')  # Plot vertical line at index 13
            axes[row, col].set_title(f'Time {col_index / 2} - {col_index / 2 + 1} [sec]')
            axes[row, col].set_xlabel('Session')
            axes[row, col].set_ylabel('Percent of expert')
            axes[row, col].legend()  # Show legend

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_model_accuracy(self, reference='expert'):
        if reference == 'expert':
            accuracy_of_models = self.accuracy_of_models_expert
        else:
            accuracy_of_models = self.accuracy_of_models_expert_PD

        x_values = range(1, len(accuracy_of_models) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(x_values, accuracy_of_models, marker='o', linestyle='-')
        plt.xlabel('Segment Number')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Segment Number')
        plt.grid(True)
        plt.show()

    def help(self):
        print("""
        logistic_regression_expert_prediction class:
        --------------------------------------------
        This class performs logistic regression analysis on expert and novice imaging data to predict
        the probability to be closer to expert or novice stages for before and after PD.

        Methods:
        1. __init__(dataset): Constructor method to initialize the class instance.
        2. plot_heat_map(reference='expert'): Plot a heatmap of the predicted probabilities.
        3. plot_prediction_per_segment(reference='expert'): Plot predictions per segment.
        4. plot_model_accuracy(reference='expert'): Plot model accuracy over segments.
        """)


