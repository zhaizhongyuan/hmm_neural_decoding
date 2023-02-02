import os
import joblib
import numpy as np

def load_predictions(path, name):
    with open(os.path.join(path, str.join('', (name, '_predictions.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


def weighted_smoothing(predictions, size):
    predictions_new = predictions.copy()
    group_start = [0]
    group_start = np.hstack((group_start, np.where(np.diff(predictions) != 0)[0] + 1))
    for i in range(len(group_start) - 3):
        if group_start[i + 2] - group_start[i + 1] < size:
            if predictions_new[group_start[i + 2]] == predictions_new[group_start[i]] and \
                    predictions_new[group_start[i]:group_start[i + 1]].shape[0] >= size and \
                    predictions_new[group_start[i + 2]:group_start[i + 3]].shape[0] >= size:
                predictions_new[group_start[i]:group_start[i + 2]] = predictions_new[group_start[i]]
    for i in range(len(group_start) - 3):
        if group_start[i + 1] - group_start[i] < size:
            predictions_new[group_start[i]:group_start[i + 1]] = predictions_new[group_start[i] - 1]
    return predictions_new


class bsoid_loader:

    def __init__(self, working_dir, prefix):
        _, _, filenames, self.filtered_data, self.new_predictions = load_predictions(working_dir, prefix)
        ## 112221
        # self.f_index = 0  # first file 16 hours
        ## 112321
        self.f_index = 0  # first file 4 hours
        # self.f_index = 1  # second file 15 hours
        ## 112721, 112821, 112921
        # self.f_index = 0  # first files [14, 11 and 10 hours]


        filter_nest = True
        if filter_nest:
            ## 112221
            # time_of_nest = [2, 0, 0]  # hour, minute, second
            # x_tobefiltered = [360, 540]
            # y_tobefiltered = [550, 720]

            ## 112321 recording1
            time_of_nest = [3, 45, 0]  # hour, minute, second
            x_tobefiltered = [900, 1050]
            y_tobefiltered = [0, 150]

            ## 112321 recording2
            # time_of_nest = [0, 0, 0]  # hour, minute, second
            # x_tobefiltered = [900, 1050]
            # y_tobefiltered = [0, 150]

            ## 112721, 112821, 112921
            # time_of_nest = [0, 30, 0]  # hour, minute, second
            # x_tobefiltered = [900, 1050]
            # y_tobefiltered = [0, 150]


            index_tobefiltered = np.where((x_tobefiltered[0] <
                                           self.filtered_data[self.f_index][
                                               round(time_of_nest[0]*60*60*60 +
                                                     time_of_nest[1]*60*60 +
                                                     time_of_nest[2]*60):, 10]) &
                                          (self.filtered_data[self.f_index][
                                               round(time_of_nest[0]*60*60*60 +
                                                     time_of_nest[1]*60*60 +
                                                     time_of_nest[2]*60):, 10] < x_tobefiltered[1]) &
                                          (y_tobefiltered[0] < self.filtered_data[self.f_index][
                                               round(time_of_nest[0]*60*60*60 +
                                                     time_of_nest[1]*60*60 +
                                                     time_of_nest[2]*60):, 11]) &
                                          (self.filtered_data[self.f_index][
                                               round(time_of_nest[0]*60*60*60 +
                                                     time_of_nest[1]*60*60 +
                                                     time_of_nest[2]*60):, 11] < y_tobefiltered[1]))[0]
            print('mouse stayed in nest for {}% of time'.format(
                100 * index_tobefiltered.shape[0] /
                self.filtered_data[self.f_index][:, 10].shape[0]))
            index_tobefiltered = index_tobefiltered[index_tobefiltered < self.new_predictions[self.f_index].shape[0]]
            self.new_predictions[self.f_index][index_tobefiltered] = -1
        self.smoothed_predictions = []

    def main(self, smooth_window=6):
        print(f'smooth window: {smooth_window}')
        print('File #{} (a {} body parts by {} frames) '
              'has {} classes'.format(self.f_index, int(self.filtered_data[self.f_index].shape[1] / 2),
                                      self.filtered_data[self.f_index].shape[0],
                                      len(np.unique(self.new_predictions[self.f_index]))))
        self.smoothed_predictions = weighted_smoothing(self.new_predictions[self.f_index], smooth_window)
        return self.f_index, self.filtered_data, self.smoothed_predictions