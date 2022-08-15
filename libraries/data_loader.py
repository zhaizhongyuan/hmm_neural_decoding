import os
import joblib
import numpy as np
import scipy.stats as stats

class data_loader:
    def __init__(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), 'rb') as file:
            spike_times_all_neurons = joblib.load(file)

        # Convert to numpy array
        spike_times_all_neurons = np.array(spike_times_all_neurons)

        # Transpose into (71999, 90)
        spike_times_all_neurons = spike_times_all_neurons.T

        # Truncate two hour m1 recording to one hour l5 recording into (36000, 46), 
        # 0th to 45th L5/6, 46th to 89th L2/3, 89th being the most shallow
        spike_times_l5_neurons_1h = spike_times_all_neurons[:36000, :46]

        # Z-score each neuron, i.e. firing rates of each neuron across time sum to 0
        spike_times_l5_neurons_1h_zscored = stats.zscore(spike_times_l5_neurons_1h) # Default axis is 0

        self.data = spike_times_l5_neurons_1h_zscored

    def load_data(self):
        print(f'Spike times are binned into shape for hmm: {self.data.shape}')
        return self.data