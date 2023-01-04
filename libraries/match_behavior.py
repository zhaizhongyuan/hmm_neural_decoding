import os
import sys
sys.path.append(os.getcwd()) # Append the path within which the user-defined class is in

import pickle

import numpy as np

from typing import List, Tuple

from data_loader import data_loader
from behavior_loader import bsoid_loader

class match_behavior:
    def __init__(self, data_dir: str, model_dir: str) -> None:
        """
        Initialization of a behavior matcher.
    
        Parameters:
        data_dir (str): Path to the /processed_data folder
        model_dir(str): Path to the /hmm_models folder
    
        Returns:
        None

        """
        self.data_dir = data_dir
        self.model_dir = model_dir

        return


    def load_data(self, neural_file: str, sample_rate: int) -> None:
        """
        Load neural data.
    
        Parameters:
        neural_file (str): Name of a neural data file containing counts of spikes
        sample_rate (int): Bin width of the spike count
    
        Returns:
        data (numpy.ndarray): Has shape (N, D), where N is the # of counts, D is # of neurons

        """
        self.sample_rate = sample_rate

        d_l = data_loader(self.data_dir, neural_file)
        self.data = d_l.load_data()

        return


    def load_behavior(self, behavior_file: str, behavior_names: List[str], frame_rate: int, delay: float) -> None:
        self.frame_rate = frame_rate

        b_l = bsoid_loader(self.data_dir, behavior_file)
        _, _, smoothed_predictions = b_l.main()
        # Correct prediction start
        behavior_start = int(delay * frame_rate)  # start of behavior
        smoothed_predictions_1h = smoothed_predictions[behavior_start:(behavior_start+3600*60)]
        self.behavior_predictions = smoothed_predictions_1h
        self.behavior_names = behavior_names

        return

    
    def load_hmm(self, K: int, mode: str) -> None:
        hmm_file = '1h_l5_possion_' + mode + f'_{K}_latents' + '.hmm'
        self.hmm_model = pickle.load(open(os.path.join(self.model_dir, hmm_file), "rb"))
        print('-'*50)
        print(f'A {K}-state hmm is used to match with behaviors.')
        print('-'*50)

        return


    def hmm_states_on_data(self) -> np.ndarray:
        # hmm states identified after each 100 ms
        hmm_states = self.hmm_model.predict(self.data)
        print('Per-timestamp hmm states are identified.')

        return hmm_states


    def hmm_state_to_behavior(self, lead_seconds: float, lag_seconds: float, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
        
        def most_freq_behav_idx(uniq_behavs, uniq_behavs_cnts):
            uniq_behavs_indcs = np.argsort(-uniq_behavs_cnts)  
            for uniq_behavs_indc in uniq_behavs_indcs:
                idt_behav = int(uniq_behavs[uniq_behavs_indc])
                if self.behavior_names[idt_behav] != 'insignificant':
                    return idt_behav
            return -1
        
        hmm_states = self.hmm_model.predict(self.data)
        end_behav_idx = len(self.behavior_predictions)
        idt_behav_for_states = np.ones(len(np.unique(hmm_states)))
        idt_behav_for_states_names = []
        for i, hmm_state in enumerate(np.unique(hmm_states)):
            state_idcs = np.where(hmm_states == hmm_state)[0]
            behav_idcs = state_idcs * int(self.frame_rate/self.sample_rate)
            idt_behavs_list = [] # identified behaviors list
            for behav_idx in behav_idcs:
                # Start index is 100ms ahead on-site
                start_idx = behav_idx - int(self.frame_rate/self.sample_rate * (lead_seconds/0.1))
                if start_idx < 0:
                    start_idx = 0
                # End index is 300ms behind on-site
                end_idx = behav_idx + int(self.frame_rate/self.sample_rate * (lag_seconds/0.1))
                if end_idx > end_behav_idx:
                    end_idx = end_behav_idx
                # Access identified behaviors using behavior indices
                behav_idcs_rng = np.arange(start_idx, end_idx)
                behavs = self.behavior_predictions[behav_idcs_rng]
                uniq_behavs, uniq_behavs_cnts = np.unique(behavs, return_counts=True)
                # Winner takes all
                idt_behav = most_freq_behav_idx(uniq_behavs, uniq_behavs_cnts)
                if idt_behav != -1:
                    idt_behavs_list.append(idt_behav)
            idt_behavs = np.array(idt_behavs_list)  
            uniq_idt_behavs, uniq_idt_behavs_cnts = np.unique(idt_behavs, return_counts=True)
            idt_behav_for_states[i] = most_freq_behav_idx(uniq_idt_behavs, uniq_idt_behavs_cnts)
            idt_behav_for_states_names.append(self.behavior_names[int(idt_behav_for_states[i])])
            idt_behav_for_states = idt_behav_for_states.astype(int)
        if verbose:
            print('Identified behaviors are: ' + str(idt_behav_for_states_names))
            print(50*'-')
        print(f'{len(np.unique(idt_behav_for_states))} unique behaviors identified.')
        print(50*'-')
        uniq_idt_behav_for_states, uniq_idt_behav_for_states_cnts = np.unique(idt_behav_for_states, return_counts=True)
        uniq_idt_behavs_for_states_indcs = np.argsort(-uniq_idt_behav_for_states_cnts)
        for uniq_idt_behavs_for_states_indc in uniq_idt_behavs_for_states_indcs:
            print(self.behavior_names[int(uniq_idt_behav_for_states[int(uniq_idt_behavs_for_states_indc)])] + f': {int(uniq_idt_behav_for_states_cnts[uniq_idt_behavs_for_states_indc])}')

        return idt_behav_for_states, idt_behav_for_states_names