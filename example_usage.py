import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from scipy import io

# this is the script in this repository:
from hrv_preprocessor import hrv_whole_recording, save_hrv_dataframes

# if this script was not in the same directory as hrv_preprocessor.py, and hrv_preprocessor.py was in dir "hrv_preprocessor":
#from hrv_preprocessor.hrv_preprocessor import hrv_whole_recording, save_hrv_dataframes

if __name__ == "__main__":

   
   # CSPC2020 ECG dataset ("TrainingSet"); 400Hz, ~24h in length, .mat format, subjects A01, A02, ... A10
   ecg_srate = 400
   print("Reading Sample Data ...")
   ecg = io.loadmat("TrainingSet/data/A01.mat")["ecg"]

   # ecg has to be either a 1D array, or in shape (n_rows, channel_length) (each row has to be ecg channel)
   ecg = ecg.T # this data had each ecg channel as a column, so we transpose

   segment_length_min = 5.0

   # instatiate rng with a seed to ensure reproducability (DVC method has some randomness)
   rng = np.random.default_rng(seed=1905)

   # produce dataframes with hrv metrics per segment
   time_dom_df, freq_dom_df, modification_report_df = hrv_whole_recording(ecg, ecg_srate, segment_length_min, verbose = True,
            save_plots=True, save_plots_dir="saved_plots",
            use_emd=True, use_reflection=True, use_segmenter="engZee", remove_noisy_beats=True, remove_noisy_RRI=True, rri_in_ms = True,
            QRS_MAX_DIST_THRESH = 0.30, DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER = 0.04, DBSCAN_MIN_SAMPLES = 100, rng=rng) 

   # save the dataframes as .csv files; can load in later (pd.read_csv) and process
   save_hrv_dataframes(time_dom_df, freq_dom_df, modification_report_df, save_dfs_dir="out")

   
   # example plotting
   timevec = (np.array(range(0, len(freq_dom_df))) * segment_length_min) / 60 # each segment is n mins, so can work out hours
   plt.clf() # necessary, as biosppy makes plots in the background ...
   plt.plot(timevec, freq_dom_df["fft_ratio"]) # look at pyHRV documentation to see what columns = what properties
   plt.title("LF/HF Ratio per segment over entire recording for Subject A01")
   plt.ylabel("LF/HF Ratio")
   plt.xlabel("Time (h)")
   plt.xlim([0, max(timevec)]) # in A01, the last few segments give NaN, so ensure this isn't chopped off automatically
   plt.savefig("example_plot")


