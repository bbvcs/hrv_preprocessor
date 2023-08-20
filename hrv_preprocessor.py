import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, interpolate, signal
import pywt

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import biosppy
import pyhrv
from dtw import * # NOTE; this is the library "dtw-python", NOT "dtw"!
import emd

import os
import math
import time 

import warnings

# keys returned by pyHRV for HRV metrics
time_dom_keys = np.array(['nni_counter', 'nni_mean', 'nni_min', 'nni_max', 'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'nni_diff_mean', 'nni_diff_min', 'nni_diff_max', 'sdnn', 'sdnn_index', 'sdann', 'rmssd', 'sdsd', 'nn50', 'pnn50', 'nn20', 'pnn20', 'nni_histogram',     'tinn_n', 'tinn_m', 'tinn', 'tri_index'])
freq_dom_keys = np.array(['fft_bands', 'fft_peak', 'fft_abs', 'fft_rel', 'fft_log', 'fft_norm', 'fft_ratio', 'fft_total', 'fft_plot', 'fft_nfft', 'fft_window', 'fft_resampling_frequency', 'fft_interpolation'])

def find_nearest(array, value):
	# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array (modified)
	idx = np.nanargmin((np.abs(array - value)))

	return idx

def extract_from_pyHRV_tuple(tuple_as_str):
    # for example, fft_log entries look like : '(8.901840598362377, 5.9300714514495, 5.228194636918156)' (strings, not tuple)
    # so convert to a tuple of (8.901840598362377, 5.9300714514495, 5.228194636918156)
    # (I think this is (VLF, LF, HF)

    tuple_as_str = tuple_as_str[1:-2] # remove brackets
    tuple_as_str = tuple_as_str.split(", ")

    return tuple([float(s) for s in tuple_as_str])


def hrv_per_segment(ecg_segment, ecg_srate, segment_length_min, timevec=None, segment_idx=0,
					save_plots=False, save_plots_dir='saved_plots', save_plot_filename=math.floor(time.time()),
					use_emd=True, use_reflection=True, use_segmenter="engzee", remove_noisy_beats=True, remove_noisy_RRI=True, rri_in_ms = True,
					QRS_MAX_DIST_THRESH = 0.30, DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER = 0.25, DBSCAN_MIN_SAMPLES = 100, rng=np.random.default_rng()): 
	"""
	Calculate HRV metrics for a segment of ECG, returning a tuple of ReturnTuples containing HRV Metrics and a Modification Report for this segment.

	Args:
		ecg_segment:                        (1D NumPy Array)    A segment (tested with 5min, likely to work for other lengths) of consecutive ECG samples.
		ecg_srate:                          (int)               Sample rate that segment was recorded at (int)
		segment_length_min:                 (float)             The length of the segment, in minutes (e.g 5.0 for 5 minutes)
		timevec:                            (1D NumPy Array)    A vector with a time value for each ecg_segment sample - defaults to range(0, len(ecg_segment))  
		segment_idx:                        (int)               The index of the segment within the recording (for writing row in dataframes) - handled automatically by hrv_whole_recording, ignore if using this directly.
		save_plots:                         (bool)              If true, save a plot of the ECG signal and corresponding HRV w/ corrections visualised.
		save_plots_dir:                     (string)            A location on disk to save plots to - will be created if doesn't already exist
		save_plot_filename:                 (string)            The filename the plot is saved to, and the title displayed on the plot.
		use_emd:                            (bool)              If True, use Empirical Mode Decomposition to remove LF noise (drift) from ECG.
		use_reflection:                     (bool)              If True, use reflection to remove edge effects when segmenting rpeaks. 
		use_segmenter:                      (string)            The bioSPPy ECG Segmenter algorithm to use to detect R peak in the ECG. (e.g "engZee", "hamilton")              
		remove_noisy_beats:                 (bool)              Try to remove QRS segments that are too distant from the average QRS beat for this segment.
		remove_noisy_RRI:                   (bool)              Try to remove spikes in the RRI signal, which may be caused by removal of noisy QRS, ectopic beats, or segmenter algorithm errors (false positive R peak)
		rri_in_ms:                          (bool)              If True, return RRI in milliseconds, otherwise in seconds.
		QRS_MAX_DIST_THRESH:                (float)             Maximum normalised distance (0-1) for a beat from the average to be considered valid (see remove_noisy_beats)
		DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER  (float)             Episilon of DBSCAN algorithm for finding RRI outliers (see remove_noisy_RRI) uses the mean of RRI in that segment multiplied by this value.
		DBSCAN_MIN_SAMPLES                  (int)               Min samples/min points parameter for DBSCAN when finding RRI outliers (see remove_noisy_RRI).
		rng				    (numpy.random._generator.Generator) Provided as param so the same RNG can be used for all calls to hrv_per_segment(), if reproducability is important.

	Returns:
		- rpeaks, original rri and rri (corrected unless disabled by param, if so same as original rri) used for HRV calculation
			- if remove_noisy_beats is true, returned rpeaks will have had outliers removed
			- if remove_noisy_RRI is true, returned RRI will have had spikes interpolated
		- a tuple of (freq_dom_hrv, time_dom_hrv, modification_report) where:
			- freq/time_dom_hrv are pyHRV ReturnTuples containing HRV metrics as calculated by pyHRV for this ECG.
			- modification_report is a dict containing some info on what has been doen to the segment (noise removal), whether it was excluded, some notes etc
				- notes are a brief string summary of why HRV could/should not be calculated for this ECG (see <EXIT_CONDITION>'s')
			- in most cases, you will get (freq_dom_hrv, time_dom_hrv, None) (no problems with segment), or (np.NaN, np.NaN, note) (segment excluded for whatever reason, see note)
				- not always; a rare error in calculating time_dom_hrv means (freq_dom_hrv, np.NaN, note) can be returned.
		
	"""

	if save_plots:
		fig, axs = plt.subplots(3, 1, sharex=True)
	
		if not os.path.exists(save_plots_dir):
			print("Setting up directory for saving plots at {}".format(save_plots_dir))
			os.makedirs(save_plots_dir, exist_ok=True)




	freq_dom_hrv = np.NaN
	time_dom_hrv = np.NaN
	
	# keep a report of what happens to this to this segment
	modification_report = {}
	modification_report["seg_idx"] = segment_idx
	modification_report["excluded"] = False
	modification_report["n_rpeaks_noisy"] = np.NaN
	modification_report["n_RRI_detected"] = np.NaN 
	modification_report["n_RRI_suprathresh"] = np.NaN
	modification_report["suprathresh_values"] = np.NaN
	modification_report["notes"] = ""
 
	# <EXIT_CONDITION>
	if (False not in pd.isnull(ecg_segment)):

		modification_report["excluded"] = True
		modification_report["notes"] = "ALL data is NaN"
 
		return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
	# </EXIT_CONDITION>



	# we might have multiple ecg channels in our ecg segment
	if len(ecg_segment.shape) > 1: # ndarray of one or multiple channels (shape = (k, n), where k is n of channel length and n is various channel length)
		n_ecg_channels = ecg_segment.shape[0]

		if n_ecg_channels == 1: # 1 ecg channel like [[ecg,]] (shape = (1, n))
			ecg_segment = ecg_segment[0, :] # convert it into non-nested 1d numpy array

	else: # non-nested 1D numpy array, or python array (shape = (n,))
		n_ecg_channels = 1

	
	rri_time_multiplier = 1000 if rri_in_ms else 1 # do we want RRI in ms or s
	if timevec is None:
		timevec = np.array(range(0, ecg_segment.shape[-1]))/ecg_srate * rri_time_multiplier
	
	
	if save_plots and n_ecg_channels == 1:
		axs[0].plot(timevec, ecg_segment, c="lightgrey", label="Raw ECG Signal")

	# Apply Empirical Mode Decomposition (EMD) to detrend the ECG Signal (remove low freq drift)
	if use_emd and n_ecg_channels==1: 
		
		try:  
			# perform EMD on the ecg_segment, and take ecg_segment as sum of IMFS 1-3; this is to remove low frequency drift from the signal, hopefully help R peak detection
			imfs = emd.sift.sift(ecg_segment).T
		except Exception as e:
			# <EXIT_CONDITION>
			modification_report["excluded"] = True
			modification_report["notes"] = e

			return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
			# </EXIT_CONDITION>
			

		# <EXIT_CONDITION>
		# if not enough imfs can be detected (this can happen if the data is mostly zeros)
		if len(imfs) < 3:

			modification_report["excluded"] = True
			modification_report["notes"] = "Less than 3 IMFs were produced by EMD"

			return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>


		ecg_emd = sum(imfs[[0, 1, 2]])

		if save_plots:
			axs[0].plot(timevec, ecg_emd, c="lightcoral", label="ECG Signal w/ EMD Applied")

		# replace the ecg_segment with the detrended signal
		ecg_segment = ecg_emd



	""" Get ECG RPeaks """

	segmenters = { # TODO what about extra args (e.g SSF) - include parameter to function "segmenter_args" (*kwargs), pass that and defaults below into segmenter?
		"engzee" : biosppy.signals.ecg.engzee_segmenter,
		"hamilton": biosppy.signals.ecg.hamilton_segmenter,
		"christov": biosppy.signals.ecg.christov_segmenter,
		"gamboa": biosppy.signals.ecg.gamboa_segmenter,
		"ssf" : biosppy.signals.ecg.ssf_segmenter,
	}
	chosen_segmenter = segmenters[use_segmenter.lower()]


	# how many rpeaks should we expect the alg to detect for a reflected piece of ecg_segment?
		# if lowest bpm ever achieved was 27, expect 27 peaks per min, 27*5 for 5 min
		# then use reflection order to work out how many we might expect in the length of reflected ECG we have
	min_rpeaks = (27*segment_length_min)

	if use_reflection:
		
		if n_ecg_channels == 1:
			# with reflection to remove edge effects
			reflection_order = math.floor(len(ecg_segment) / 2)
			ecg_reflected = np.concatenate(
				(ecg_segment[reflection_order:0:-1], ecg_segment, ecg_segment[-2:len(ecg_segment) - reflection_order - 2:-1]))
			

			# get rpeak locations, using a "segmenter algorithm" (algorithm to detect R peaks in ECG)
			rpeaks = chosen_segmenter(signal=ecg_reflected, sampling_rate=ecg_srate)["rpeaks"]
			
			# NOTE: biosppy provides other segmenters. method "biosppy.signals.ecg.ecg()" uses the hamilton segmenter.
			# christov and hamilton are likely valid alternatives to engzee segmenter, but I haven't thoroughly tested.
			# others (e.g ssf and gamboa) didn't seem great
			# how many rpeaks should we expect the alg to detect for a reflected piece of ecg_segment?
				# if lowest bpm ever achieved was 27, expect 27 peaks per min, 27*5 for 5 min
				# then use reflection order to work out how many we might expect in the length of reflected ECG we have
			min_rpeaks = (27*segment_length_min)
			min_rpeaks_in_reflected = min_rpeaks * (len(ecg_segment) / reflection_order) 

			"""
			# if there isn't "enough" rpeaks, it may be possible that a certain segmenter is the problem
			j = 0
			while len(rpeaks) < min_rpeaks_in_reflected:

				chosen_segmenter = segmenters[sorted(list(segmenters.keys()))[j]] 
				if chosen_segmenter != segmenters[use_segmenter]: # if it's not the one we tried in the first place
					rpeaks = chosen_segmenter(signal=ecg_reflected, sampling_rate=ecg_srate)["rpeaks"]
				j+=1

				# <EXIT_CONDITION>
				# if none helped, then exit
				if len(rpeaks) < min_rpeaks_in_reflected and j == len(list(segmenters.keys())):
			 
					modification_report["excluded"] = True
					modification_report["notes"] = "Segmenters detected no Rpeaks"

					return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
				# </EXIT_CONDITION>
			"""
			
			# <exit_condition>
			if len(rpeaks) < min_rpeaks_in_reflected:
		 
				modification_report["excluded"] = True
				modification_report["notes"] = f"segmenter ({use_segmenter}) detected not enough rpeaks ({len(rpeaks)} < {min_rpeaks_in_reflected}) in reflected rpeaks"

				return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
			# </exit_condition>


			# need to chop off the reflected parts before and after original signal
			original_begins = reflection_order
			original_ends = original_begins + len(ecg_segment)-1

			rpeaks_begins = find_nearest(rpeaks, original_begins)
			rpeaks_ends = find_nearest(rpeaks, original_ends)
			rpeaks = rpeaks[rpeaks_begins:rpeaks_ends]

			# get their position in the original
			rpeaks = rpeaks - original_begins

			# find_nearest may return the first as an element before original_begins
			# as we flipped, the last r peak of the flipped data put before original
			# will be the negative of the first r peak in the original data
			# as we are using argmin(), this will be returned first
			# so, remove any negative indices (r peaks before original begins)
			rpeaks = rpeaks[rpeaks > 0]

		else: 	# if we have multiple ECG channels, we should decide which one will be best suited and use that
	
			# arrays for determining ecg channel to use	
			rpeaks_candidates = []
			candidates_scores = []
			
			# arrays for error checking etc once we've decided winning list
			len_reflected_rpeaks = [] 
			modification_reports_local = []
			ecg_emds = []
	
			for ch in range(0, n_ecg_channels):
				ecg_channel = ecg_segment[ch, :]		
				modification_report_local = modification_report.copy()
				

	
				reflection_order = math.floor(len(ecg_channel) / 2)
				ecg_reflected = np.concatenate(
					(ecg_channel[reflection_order:0:-1], ecg_channel, ecg_channel[-2:len(ecg_segment) - reflection_order - 2:-1]))
				
				if not all(np.isnan(ecg_reflected)): # this code mostly duplicated from somewhere below
					# Apply Empirical Mode Decomposition (EMD) to detrend the ECG Signal (remove low freq drift)
					if use_emd: 
						
						try:  
							# perform EMD on the ecg_segment, and take ecg_segment as sum of IMFS 1-3; this is to remove low frequency drift from the signal, hopefully help R peak detection
							imfs = emd.sift.sift(ecg_channel).T
						except Exception as e:
							# <EXIT_CONDITION>
							modification_report_local["excluded"] = True
							modification_report_local["notes"] = e

							modification_reports_local.append(modification_report_local)
							continue
							# </EXIT_CONDITION>

						# <EXIT_CONDITION>
						# if not enough imfs can be detected (this can happen if the data is mostly zeros)
						if len(imfs) < 3:

							modification_report_local["excluded"] = True
							modification_report_local["notes"] = "Less than 3 IMFs were produced by EMD"

							modification_reports_local.append(modification_report_local)
							continue
						# </EXIT_CONDITION>


						ecg_emd = sum(imfs[[0, 1, 2]])
						ecg_emds.append(ecg_emd)

						# replace the ecg_segment with the detrended signal
						ecg_channel = ecg_emd
						modification_reports_local.append(None)

					rpeaks = chosen_segmenter(signal=ecg_reflected, sampling_rate=ecg_srate)["rpeaks"]
					len_reflected_rpeaks.append(len(rpeaks))

					original_begins = reflection_order
					original_ends = original_begins + len(ecg_channel)-1

					rpeaks_begins = find_nearest(rpeaks, original_begins)
					rpeaks_ends = find_nearest(rpeaks, original_ends)
					rpeaks = rpeaks[rpeaks_begins:rpeaks_ends]
					rpeaks = rpeaks - original_begins
					rpeaks = rpeaks[rpeaks > 0]
					
					rpeaks_candidates.append(rpeaks)

					# get each QRS 	
					beats = biosppy.signals.ecg.extract_heartbeats(ecg_channel, rpeaks, ecg_srate)["templates"] # get ECG signal a small amount of time around detected Rpeaks
	
					# create comparison wavelet, that looks like desired ECG waveform, according to https://uk.mathworks.com/help/wavelet/ug/r-wave-detection-in-the-ecg.html
					_, comp_wl_y, comp_wl_x = pywt.Wavelet("sym4").wavefun(5)
					comp_wl_y = -2*comp_wl_y

					# determine how far each beat is from comparison wavelet
					beats_distance = np.array([dtw(stats.zscore(beats[x]), stats.zscore(comp_wl_y), keep_internals=True).normalizedDistance for x in range(0, len(beats))])
					
					candidate_score = np.mean(beats_distance)
					candidates_scores.append(candidate_score)
			

			# <exit_condition>
			if len(candidates_scores) == 0:
		 
				modification_report["excluded"] = True
				modification_report["notes"] = f"candidates_scores empty despite {n_ecg_channels} ECG channels present - probably all NaN or 0"

				return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
			# </exit_condition>

			winner_idx = np.argmin(candidates_scores) # we want the ecg with lowest mean distance from comparison wavelet			
			print(f"WINNER: {winner_idx}")
			rpeaks = rpeaks_candidates[winner_idx]
 
			if save_plots:
				axs[0].plot(timevec, ecg_segment[winner_idx, :], c="lightgrey", label="Raw ECG Signal")
			
			# emd-specific code
			if use_emd and modification_reports_local[winner_idx] != None:
				return None, None, None, freq_dom_hrv, time_dom_hrv, modification_reports_local[winner_idx]
			if use_emd and save_plots:
				axs[0].plot(timevec, ecg_emds[winner_idx], c="lightcoral", label="ECG Signal w/ EMD Applied")
			

			# for compatibility with later code
			ecg_segment = ecg_segment[winner_idx, :] if not use_emd else ecg_emds[winner_idx]
			reflection_order = math.floor(len(ecg_segment) / 2)
			
			min_rpeaks = (27*segment_length_min)
			min_rpeaks_in_reflected = min_rpeaks * (len(ecg_segment) / reflection_order) 
			
			# <exit_condition>
			if len_reflected_rpeaks[winner_idx] < min_rpeaks_in_reflected:
		 
				modification_report["excluded"] = True
				modification_report["notes"] = f"segmenter ({use_segmenter}) detected not enough rpeaks ({len_reflected_rpeaks[winner_idx]} < {min_rpeaks_in_reflected}) in reflected rpeaks of winning ECG channel"

				return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
			# </exit_condition>

	
	elif not use_reflection: 
	  
		if n_ecg_channels == 1: 
			rpeaks = chosen_segmenter(signal=ecg_segment, sampling_rate=ecg_srate)["rpeaks"]
	 
		else:
		
			rpeaks_candidates = []
			candidates_scores = []
	
			modification_reports_local = []
			ecg_emds = []

			for ch in range(0, n_ecg_channels):
				ecg_channel = ecg_segment[ch, :]		
				modification_report_local = modification_report.copy()
				

				if not all(np.isnan(ecg_channel)): # this code mostly duplicated from somewhere below AND in use_reflection above
					if use_emd: 
						
						try:  
							# perform EMD on the ecg_segment, and take ecg_segment as sum of IMFS 1-3; this is to remove low frequency drift from the signal, hopefully help R peak detection
							imfs = emd.sift.sift(ecg_channel).T
						except Exception as e:
							# <EXIT_CONDITION>
							modification_report_local["excluded"] = True
							modification_report_local["notes"] = e

							modification_reports_local.append(modification_report_local)
							continue
							# </EXIT_CONDITION>

						# <EXIT_CONDITION>
						# if not enough imfs can be detected (this can happen if the data is mostly zeros)
						if len(imfs) < 3:

							modification_report_local["excluded"] = True
							modification_report_local["notes"] = "Less than 3 IMFs were produced by EMD"

							modification_reports_local.append(modification_report_local)
							continue
						# </EXIT_CONDITION>


						ecg_emd = sum(imfs[[0, 1, 2]])

						ecg_emds.append(ecg_emd)

						# replace the ecg_segment with the detrended signal
						ecg_channel = ecg_emd
						modification_reports_local.append(None)

					rpeaks = chosen_segmenter(signal=ecg_channel, sampling_rate=ecg_srate)["rpeaks"]
					rpeaks_candidates.append(rpeaks)

					# get each QRS 	
					beats = biosppy.signals.ecg.extract_heartbeats(ecg_channel, rpeaks, ecg_srate)["templates"] # get ECG signal a small amount of time around detected Rpeaks
	
					# create comparison wavelet, that looks like desired ECG waveform, according to https://uk.mathworks.com/help/wavelet/ug/r-wave-detection-in-the-ecg.html
					_, comp_wl_y, comp_wl_x = pywt.Wavelet("sym4").wavefun(5)
					comp_wl_y = -2*comp_wl_y

					# determine how far each beat is from comparison wavelet
					beats_distance = np.array([dtw(stats.zscore(beats[x]), stats.zscore(comp_wl_y), keep_internals=True).normalizedDistance for x in range(0, len(beats))])
					
					candidate_score = np.mean(beats_distance)
					candidates_scores.append(candidate_score)

			# <exit_condition>
			if len(candidates_scores) == 0:
		 
				modification_report["excluded"] = True
				modification_report["notes"] = f"candidates_scores empty despite {n_ecg_channels} ECG channels present - probably all NaN or 0"

				return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
			# </exit_condition>
			winner_idx = np.argmin(candidates_scores) # we want the ecg with lowest mean distance from comparison wavelet			
			print(f"WINNER: {winner_idx}")
			rpeaks = rpeaks_candidates[winner_idx]
 
			if save_plots:
				axs[0].plot(timevec, ecg_segment[winner_idx, :], c="lightgrey", label="Raw ECG Signal")
			
			# emd-specific code
			if use_emd and modification_reports_local[winner_idx] != None:
				return None, None, None, freq_dom_hrv, time_dom_hrv, modification_reports_local[winner_idx]
			if use_emd and save_plots:
				axs[0].plot(timevec, ecg_emds[winner_idx], c="lightcoral", label="ECG Signal w/ EMD Applied")
			
			# for compatibility with later code
			ecg_segment = ecg_segment[winner_idx, :] if not use_emd else ecg_emds[winner_idx]
	
 
		"""
		j = 0
		while len(rpeaks) < min_rpeaks:

			chosen_segmenter = segmenters[sorted(list(segmenters.keys()))[j]] 
			if chosen_segmenter != segmenters[use_segmenter]: 
				rpeaks = chosen_segmenter(signal=ecg_segment, sampling_rate=ecg_srate)["rpeaks"]
			j+=1

			# <EXIT_CONDITION> (DUPLICATE)
			# if none helped, then exit
			if len(rpeaks) < min_rpeaks and j == len(list(segmenters.keys())):
		 
				modification_report["excluded"] = True
				modification_report["notes"] = "Segmenters detected no Rpeaks"

				return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
			# </EXIT_CONDITION>
		"""
		# <EXIT_CONDITION>
		if len(rpeaks) < min_rpeaks:
	 
			modification_report["excluded"] = True
			modification_report["notes"] = f"Segmenter ({use_segmenter}) detected not enough Rpeaks ({len(rpeaks)} < {min_rpeaks})"

			return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>


	# <EXIT_CONDITION>
	if (True in pd.isnull(ecg_segment)):

		modification_report["excluded"] = True
		modification_report["notes"] = "At least 1 (but not all) datapoint is NaN"
 
		return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
	# </EXIT_CONDITION>
   
	# <EXIT_CONDITION>
	# if there isn't enough data in the segment to calculate LF/HF
	if len(ecg_segment) < ecg_srate * (60 * 2):

		modification_report["excluded"] = True
		modification_report["notes"] = "Not enough data recorded in this segment interval"
 
		return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
	# </EXIT_CONDITION>

		
	# correct candidate rpeaks to the maximum ECG value within a time tolerance (0.05s by default)
	rpeaks = biosppy.signals.ecg.correct_rpeaks(ecg_segment, rpeaks, sampling_rate = ecg_srate, tol = 0.05)["rpeaks"]

	if remove_noisy_beats:
		""" Attempt to remove noise that has been incorrectly identified as QRS """

		# look for noise in the ECG signal by checking if each detected QRS complex is similar enough to the average QRS in this segment
		beats = biosppy.signals.ecg.extract_heartbeats(ecg_segment, rpeaks, ecg_srate)["templates"] # get ECG signal a small amount of time around detected Rpeaks
		avg_beat = np.nanmean(beats, axis=0) # produce the average/'typical' beat within the segment
		
		# produce a vector of 1 value per QRS of how similar it is to the avg
		# use dynamic time warping (DTW)
		# use z-score to eliminate difference in amplitude
		beats_distance = np.array([dtw(stats.zscore(beats[x]), stats.zscore(avg_beat), keep_internals=True).normalizedDistance for x in range(0, len(beats))])

		# how many beats are too distant from the average
		noisy_beats_idx = np.where(beats_distance > QRS_MAX_DIST_THRESH)[0]

		# get indices of longest consecutive run of valid rpeaks
		run_start = run_end = 0
		runs = [] # (start, end)
		on_noise = True
		for j in range(0, len(rpeaks)):

		  
			if j in noisy_beats_idx:
				if on_noise:
					# we're still on noise
					pass
				else:
					# run has ended
					run_end = j
					runs.append((run_start, run_end))
					on_noise = True
			
			else: # a run has begun/continues

				if on_noise: # a run has begun
					run_start = j
					on_noise = False

				if j == len(rpeaks) - 1:
					# we've reached end of segment
					run_end = j
					runs.append((run_start, run_end))
						
		#print(runs)
		

		# <EXIT_CONDITION>
		# discard as NaN if no valid rpeaks were found
		if len(runs) == 0:

			modification_report["excluded"] = True
			modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
			modification_report["notes"] = f"No runs detected - so likely signal was all noise."

			return rpeaks, None, None, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>


		run_lengths = [np.abs(run[1] - run[0]) for run in runs] 
		longest_consecutive = runs[run_lengths.index(max(run_lengths))]
		#print(f"Longest run = {longest_consecutive[0]} -> {longest_consecutive[1]}")
		

		noisy_rpeaks = rpeaks[noisy_beats_idx] # keep a copy for plotting
		rpeaks = np.delete(rpeaks, noisy_beats_idx)
		

		# <EXIT_CONDITION>
		# if too great a percentage were due to noise
		snr = len(noisy_beats_idx) / len(beats)
		if snr > 0.40:

			modification_report["excluded"] = True
			modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
			modification_report["notes"] = f"Noisy beats {snr}"

			return rpeaks, None, None, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>

	   
		# <EXIT_CONDITION>
		# if there isn't enough 
		if len(rpeaks) < min_rpeaks:
					
			modification_report["excluded"] = True
			modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
			modification_report["notes"] = "No rpeaks left after noisy rpeaks removed"

			return rpeaks, None, None, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>





	""" Calculate and correct R-R Intervals """
	
	rri = (np.diff(rpeaks) / ecg_srate) * rri_time_multiplier 

	
	# <EXIT_CONDITION>
	# if there is less than 2m of RRI
	if sum(rri) < (rri_time_multiplier * 120): 

		modification_report["excluded"] = True
		modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
		modification_report["n_RRI_detected"] = len(rri)
		modification_report["notes"] = f"Sum of RRI ({sum(rri)}) was less than 2Mins"

		return rpeaks, rri, None, freq_dom_hrv, time_dom_hrv, modification_report
	# </EXIT_CONDITION>
	
	
	# <EXIT_CONDITION>
	# if there is only a bit of noise, is there enough consecutive for LFHF?
	# i guess this covers spread? slightly
	# TODO surely we should only keep the longest consecutive
		# - if it was e.g 3m 1m and 1m runs for example, pick only 3m? (use the 3m as the RRI for next)
	if snr > 0.20:
		k = np.sum(rri[longest_consecutive[0]:longest_consecutive[1]])

		if k < (rri_time_multiplier * 120):

			modification_report["excluded"] = True
			modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
			modification_report["n_RRI_detected"] = len(rri)
			modification_report["notes"] = "Sum of RRI in LONGEST CONSECUTIVE was less than 2Mins"

			return rpeaks, rri, None, freq_dom_hrv, time_dom_hrv, modification_report
	# </EXIT_CONDITION>


	if not remove_noisy_RRI:
		rri_corrected = rri # for compatibility with plots

	elif remove_noisy_RRI:
		# Often RRI contain spikes, when algorithm error has resulted in a beat being missed/noise has mean extra beat detected
		# These spikes must be removed as they will affect HRV metrics
		rri_corrected = np.copy(rri)

		# produce a poincare reccurence represntation of the segment, where each RRI is paired with the next
		poincare = np.array([rri[:-1],rri[1:]], dtype=np.float32)

		# in poincare plot, outliers should be far from a main cluster of valid RRIs. So use DBSCAN to detect outliers
		eps = (np.mean(rri_corrected) * DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER)	
		db = DBSCAN(eps = eps, min_samples=DBSCAN_MIN_SAMPLES).fit(poincare.T)
		labels = db.labels_ # -1 is outliers, >= 0 is a valid cluster

		# every RRI except first & last will appear twice in poincare representation
		# if both appearances are an outlier, then it is probably a spike, so we say this RRI is an outlier
		poincare_outliers = np.zeros(len(rri_corrected))
		for j in range(0, len(poincare_outliers)):
			if j == 0:
				if labels[j] == -1:
					poincare_outliers[j] = 1                         

			if j == len(poincare_outliers)-1:
				if labels[-1] == -1:
					poincare_outliers[j] = 1  

			else:
				if (labels[j-1] == -1) and (labels[j] == -1):
					poincare_outliers[j] = 1

		if save_plots:
			# plot poincare representation w/ outliers
			fig2, ax2 = plt.subplots()
			fig2.suptitle(f"{segment_idx}")
			ax2.set_title(f"Mean RRI (corrected): {np.mean(rri_corrected)}, eps multiplier = {DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER}, eps = {eps}")
			labels_text = ["Valid" if label >= 0 else "Outlier" for label in labels]
			sns.scatterplot(x=rri[:-1], y=rri[1:], hue=labels_text, palette={"Valid": "#000000", "Outlier": "#FF0000"}, ax=ax2)
			fig2.savefig(f"{save_plots_dir}/{save_plot_filename}_POINCARE", dpi=100)

		# get idx of outliers in rri
		#outlier_idx = np.where(poincare_outliers == 1)[0]
		poincare_outlier_idx = np.where(poincare_outliers == 1)[0]

		"""
		# produce a copy without the RRIs exceeding the threshold, for use in interpolation
		rri_corrected_supra_removed = np.delete(rri_corrected, outlier_idx)
		rri_corrected_supra_idx_removed = np.delete(np.array(range(0, len(rri_corrected))), outlier_idx)
		
		# <EXIT_CONDITION>
		# if too many have been detected as outliers
		if sum(rri_corrected_supra_removed) < (rri_time_multiplier * 120):

			modification_report["excluded"] = True
			modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
			modification_report["n_RRI_detected"] = len(rri)
			modification_report["notes"] = f"Sum of corrected RRI (outliers removed) ({sum(rri_corrected_supra_removed)}) was less than 2Mins"

			return rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>
			
		# interpolate points above threshold
		rri_corrected[outlier_idx] = np.interp(outlier_idx, rri_corrected_supra_idx_removed, rri_corrected_supra_removed)
		"""


		""" DVC METHOD """
		
		def dvc_conditions(rri_before, rri_j, rri_after, e_10):
			# physiological dvc_conditions defined in Table 1
			e_r = np.abs((rri_after-rri_j)/rri_j)
			e_l = np.abs((rri_j - rri_before)/rri_before)
			if (((0.3*rri_time_multiplier) < rri_j) and  (rri_j < (1.3*rri_time_multiplier))) and \
				((e_r <= e_10) and (e_10 <= 0.4)) and((e_l <= e_10) and (e_10 <= 0.4)): 
					
					return True
			return False
		
		def dvc_conditions_rightonly(rri_j, rri_after, e_10):
			# physiological dvc_conditions defined in Table 1
			e_r = np.abs((rri_after-rri_j)/rri_j)
			if (((0.3*rri_time_multiplier) < rri_j) and  (rri_j < (1.3*rri_time_multiplier))) and \
				((e_r <= e_10) and (e_10 <= 0.4)): 
					
					return True
			return False
		

		# DVC-ONLY
		"""		
		outlier_idx = []
		# filtering outlier rri
		for j in range(0, len(rri_corrected)-1): # TODO last RRI excluded
			rri_l = rri_corrected[j-1]
			rri_j = rri_corrected[j]
			rri_r = rri_corrected[j+1]

			if rri_j > (1.3 * rri_time_multiplier):
				outlier_idx.append(j)
			elif rri_j < (0.3 * rri_time_multiplier):

				e_10 = 1/10 * np.sum([np.abs((rri_corrected[k]-rri_corrected[k-1])/rri_corrected[k-1]) for k in range(j-10, j)]) # TODO is this getting 10
				e_10 = min(e_10, 0.4)

				# merge 
				right_merge = rri_j + rri_r
				if dvc_conditions(rri_l, right_merge, rri_r, e_10):
					outlier_idx.append(j)
					rri_corrected[j+1] = right_merge			
				else:
					left_merge = rri_j + rri_l
					if dvc_conditions(rri_l, left_merge, rri_r, e_10):
						outlier_idx.append(j-1)
						rri_corrected[j] = left_merge
					else:
						outlier_idx.append(j)
						outlier_idx.append(j+1)

		"""

		# DVC + POINCARE
		outlier_idx = []
		"""
		# filtering outlier rri
		for j in range(0, len(rri_corrected)-1): # TODO last RRI excluded
			rri_l = rri_corrected[j-1]
			rri_j = rri_corrected[j]
			rri_r = rri_corrected[j+1]
			
			if j in poincare_outlier_idx:
				if rri_j < (0.3 * rri_time_multiplier):

					e_10 = 1/10 * np.sum([np.abs((rri_corrected[k]-rri_corrected[k-1])/rri_corrected[k-1]) for k in range(j-10, j)]) # TODO is this getting 10
					e_10 = min(e_10, 0.4)

					# try right merge first
					right_merge = rri_j + rri_r
					if dvc_conditions(rri_l, right_merge, rri_r, e_10):
						outlier_idx.append(j)
						rri_corrected[j+1] = right_merge			
					else:
						# if right doesn't work, try left
						left_merge = rri_j + rri_l
						if dvc_conditions(rri_l, left_merge, rri_r, e_10):
							outlier_idx.append(j-1)
							rri_corrected[j] = left_merge
						else: 
							# if both merge results are > 1.3s ...
							if (right_merge > 1.3*rri_time_multiplier) \
								and (left_merge > 1.3*rri_time_multiplier):	
								outlier_idx.append(j)
								outlier_idx.append(j+1)
							# if both merge results are in physiological boundries ...
							elif (right_merge < 1.3*rri_time_multiplier) \
								and (right_merge > 0.3*rri_time_multiplier):	

				else:
					outlier_idx.append(j)
		"""
		for j in range(0, len(rri_corrected)):
			rr_i = rri_corrected[j]

	
			if j in poincare_outlier_idx:

				if (j < 2) or (j >= len(rri_corrected)-2):
					outlier_idx.append(j)
					continue

				# implementation of page 8 pseudocode
				if rr_i < (0.3 * rri_time_multiplier):
					e_10 = 1/10 * np.sum([np.abs((rri_corrected[k]-rri_corrected[k-1])/rri_corrected[k-1]) for k in range(j-10, j)]) # TODO is this getting 10
					e_10 = min(e_10, 0.4)
					
					rr_r = rr_i + rri_corrected[j+1]  
					er_r = np.abs((rri_corrected[j+2]-rr_r)/rr_r)
					er_l = np.abs((rr_r - rri_corrected[j-1])/rri_corrected[j-1])
					etot_r = er_r + er_l

					if (rr_r < (1.3*rri_time_multiplier)) and (er_l <= e_10) and (er_r <= e_10):
						outlier_idx.append(j)
						rri_corrected[j+1] = rr_r
					else:
						rr_l = rr_i + rri_corrected[j-1]  
						el_r = np.abs((rri_corrected[j+1]-rr_l)/rr_l)
						el_l = np.abs((rr_l - rri_corrected[j-2])/rri_corrected[j-2])
						etot_l = el_r + el_l

						if (rr_l < (1.3*rri_time_multiplier)) and (el_l <= e_10) and (el_r <= e_10):
							outlier_idx.append(j-1)
							rri_corrected[j] = rr_l

						elif (rr_r > (1.3*rri_time_multiplier)) and (rr_l > (1.3*rri_time_multiplier)):
							outlier_idx.append(j)
							outlier_idx.append(j+1)
						
						elif (rr_r < (1.3*rri_time_multiplier)) and (rr_l > (1.3*rri_time_multiplier)):
							rri_corrected[j+1] = rr_r
							outlier_idx.append(j)

						elif (rr_r > (1.3*rri_time_multiplier)) and (rr_l < (1.3*rri_time_multiplier)):
							rri_corrected[j-1] = rr_l
							outlier_idx.append(j)

						elif (rr_r < (1.3*rri_time_multiplier)) and (rr_l < (1.3*rri_time_multiplier)) \
							and (etot_r > 0.4) and (etot_l > 0.4):
							
							if min(etot_r, etot_l) == etot_r:
								rri_corrected[j+1] = rr_r
								outlier_idx.append(j)
							else:
								rri_corrected[j-1] = rr_l
								outlier_idx.append(j)
						else:# this is only case I've added that wasn't covered by paper/pseudocode
							
							outlier_idx.append(j)
							outlier_idx.append(j+1)
				else:
					outlier_idx.append(j)
		
		Ts = (rpeaks / ecg_srate * rri_time_multiplier) #= np.cumsum(rri_corrected)  
		Ts_original = Ts.copy() # for plotting original RRIs	
	
		outlier_idx = np.array(sorted(set(outlier_idx))) # remove duplicates

		MAX_E10 = 0.4 # TODO should move this up; outlier detection also uses max 0.4

		if len(outlier_idx) > 0:
			rri_corrected[outlier_idx] = np.NaN # drop outliers - do after making timevec so corrected rri adjacent to those removed are accounted for 
			Ts[1:][outlier_idx] = np.NaN  
		
			# <EXIT_CONDITION>
			if np.nansum(rri_corrected) < (rri_time_multiplier * 180):

				modification_report["excluded"] = True
				modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
				modification_report["n_RRI_detected"] = len(rri)
				modification_report["notes"] = f"Sum of RRI with outliers deleted ({np.nansum(rri_corrected)}) was less than 3Mins (before correction!)"

				return rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report
			# </EXIT_CONDITION>
		
			# find 'gaps' (runs of NaNs)	
			runs = []
			on_NaN = False
			p = len(rri_corrected)-1
			while p >= 0:
				if np.isnan(rri_corrected[p]) and not on_NaN:
					NaN_end = p
					on_NaN = True
				elif on_NaN and not np.isnan(rri_corrected[p]):
					NaN_start = p+1
					runs.append((NaN_start, NaN_end))
					on_NaN = False

				if p == 0:
					if on_NaN:
						NaN_start = 0 
						runs.append((NaN_start, NaN_end))

				p -= 1

			runs = list(reversed(runs))
			
			total_gap_length_difference = 0

			for gap in runs:
				gap_end = gap[1]
				gap_start = gap[0]
				gap_length = (gap_end - gap_start) + 1
		
				# get data we will use to randomly generate new RRIs (10 RRIs before gap)	
				if gap_start >= 10:
					gauss_data = rri_corrected[gap_start-10:gap_start]
				elif gap_start == 0:
					gauss_data = rri_corrected[gap_end:gap_end+10]
				else:
					# TODO could we use the 10 after the gap? Or e.g if there were 8 before, use 8+2 after gap?
					# what if length of gauss data is 1?
					gauss_data = rri_corrected[0:gap_start]
					

	
				# TODO what if all NaN?
				# TODO should we replace NaNs with equal num of extra RRI before/after gap?
				gauss_data = gauss_data[~pd.isnull(gauss_data)]
				mu = np.mean(gauss_data)
				sigma = np.std(gauss_data)	

				# determien data surrounding the gap we will use as basis to fill in 
				new_Ts = []
				new_RRs = []
				
				if gap_end == len(rri_corrected)-1:
					# if final rri is erroneous, re_generate an RR for the original time point, and bring gap in 
					#Ts[1:][gap_end] = Ts_original[1:][len(Ts_original) - (len(Ts) - gap_end)] # undo NaN (have to account for changing gap positions as we fix successive gaps, and that original Ts doesn't change)
					Ts[1:][gap_end] = Ts[1:][gap_start-1] + np.sum(rng.normal(mu, sigma, gap_length)) # using the approximate number of RRIs we need to replace (gap_length), create a plausible timestamp for the end
					rri_corrected[gap_end] = rng.normal(mu, sigma, 1)[0]
					gap_end -=1
					# TODO if gap is last point only?

				T_end = Ts[1:][gap_end+1]

				if gap_start > 0:
					T_start = Ts[1:][gap_start-1]
				else:
					T_start = Ts[0]				

				RR_end = rri_corrected[gap_end+1]
				RR_start = rri_corrected[gap_start-1] 



				# generate new Timestamps and RRIs to fill the gap
				T = T_end - RR_end 
				new_Ts.insert(0, T)

				e_10 = 1/10 * np.sum([np.abs((gauss_data[k]-gauss_data[k-1])/gauss_data[k-1]) for k in range(1, len(gauss_data))])
				e_10 = min(e_10, MAX_E10)
				
				#while ((T - T_start) >= (1.3 * rri_time_multiplier)) or ((gap_length == 1) and ((T-T_start) == rri[gap_start-total_gap_length_difference])):
				if not ((T-T_start) == 0):
					#while ((T - T_start) >= (1.3 * rri_time_multiplier)) or ((len(new_RRs) == 0) and ((T-T_start) < (1.3*rri_time_multiplier)) and ((T-T_start) > (mu+(sigma*3))) not dvc_conditions(RR_start, (T-T_start), RR_end, 0.4)):# or ((gap_length == 1) and ((T-T_start) == rri[gap_start-total_gap_length_difference])):
					#while (T - T_start)-mu >= sigma*3:
					while (T - T_start) >= (mu + (sigma*3)):	
						
						RR = rng.normal(mu, sigma, 1)[0]
						tries = 0
						while not dvc_conditions_rightonly(RR, new_RRs[0] if len(new_RRs) > 1 else RR_end, e_10):
							tries += 1
							RR = rng.normal(mu, sigma, 1)[0] # generate until meets condition TODO potential for infinite loop, apply same increase to variance?
							if tries % 4 == 0:
								if e_10 < MAX_E10:
									e_10 = e_10 + 0.05
									e_10 = min(e_10, MAX_E10)
								else:
									break

						new_RRs.insert(0, RR)
						
						T = T - RR
						new_Ts.insert(0, T)

				RR = T - T_start
				new_RRs.insert(0, RR)

				# check 2 most recently generated RRI, re-generate if needs be to fit distribution
				print(new_RRs)
				if not new_RRs == [0]: # new_RRs can be [0.0,] if right merge etc. no point in further action, we just want to remove this one 

					tries = 0
					rounds_at_max = 0
					e_10 = 1/10 * np.sum([np.abs((gauss_data[k]-gauss_data[k-1])/gauss_data[k-1]) for k in range(1, len(gauss_data))])
					e_10 = min(e_10, MAX_E10)
					while not dvc_conditions(RR_start, new_RRs[0], new_RRs[1] if len(new_RRs) > 1 else RR_end, e_10):
						tries += 1

						#new_RRs = new_RRs[2:] # clear last two RRs
						#new_Ts = new_Ts[1:]
						new_RRs = []
						new_Ts = [new_Ts[-1]]
						
						if len(new_Ts) > 0:
							T = new_Ts[0]
						else: 
							T = T_end - RR_end 
							new_Ts.insert(0, T)

						# re-fill after we have cleared last two 
						#while ((T - T_start) >= (1.3 * rri_time_multiplier)) or ((gap_length == 1) and ((T-T_start) == rri[gap_start-total_gap_length_difference])):
						#while ((T - T_start) >= (1.3 * rri_time_multiplier)) or ((len(new_RRs) == 0) and ((T-T_start) < (1.3*rri_time_multiplier)) and not dvc_conditions(RR_start, (T-T_start), RR_end, 0.4)):# or ((gap_length == 1) and ((T-T_start) == rri[gap_start-total_gap_length_difference])):
						#while (T - T_start)-mu >= (sigma*2):	
						while (T - T_start) >= (mu + (sigma* 3)): # TODO copied code from above
							
							e_10_rr = e_10
							RR = rng.normal(mu, sigma, 1)[0]
							tries_rr = 0
							while not dvc_conditions_rightonly(RR, new_RRs[0] if len(new_RRs) > 1 else RR_end, e_10_rr):
								tries_rr += 1
								RR = rng.normal(mu, sigma, 1)[0] 
								if tries_rr % 4 == 0:
									if e_10_rr < MAX_E10:
										e_10_rr = e_10_rr + 0.05
										e_10_rr = min(e_10_rr, MAX_E10)
									else:
										break
							new_RRs.insert(0, RR)
							
							T = T - RR
							new_Ts.insert(0, T)
						
						RR = T - T_start
						new_RRs.insert(0, RR)
						#print(f"\t{new_RRs}\tMU: {mu} SIGMA: {sigma}\ttries:{tries} rounds_at_max:{rounds_at_max}")	
						# every 4 tries, increase deviation by 5%
						if tries % 4 == 0:
							#print(f"{gap}, {tries}")
							if e_10 < MAX_E10:
								e_10 = e_10 + 0.05
								e_10 = min(e_10, MAX_E10)
							else:
								# we've tried 4 times with 0.4, no point in trying any further as can't go past 0.4
								
								if e_10 < mu/2: 
									sigma += (0.05 * sigma)
								
								if rounds_at_max > 1000:
									break
								rounds_at_max += 1
							
					print(f"\t{new_RRs}\tMU: {mu} SIGMA: {sigma}\ttries:{tries} rounds_at_max:{rounds_at_max}")	
					print(new_RRs)
					print("\n")

				for k, value in enumerate(new_RRs):
					if value <= 0:
						new_RRs[k] = np.NaN
						new_Ts[k] = np.NaN
				
				# replace the NaN runs with our new Ts and new RRs
				T_before_gap = np.insert(Ts[1:][0:gap_start], 0, Ts[0]) # gap_start/end refer to RRIs, first timestamp has no assoc. RRI, so add it on after indexing
				T_after_gap = Ts[1:][gap_end+1:]
				Ts = np.concatenate([T_before_gap, new_Ts, T_after_gap])
				
				RR_before_gap = rri_corrected[0:gap_start]
				RR_after_gap = rri_corrected[gap_end+1:]
				rri_corrected = np.concatenate([RR_before_gap, new_RRs, RR_after_gap])			

				# as we are going through gaps in order, if we end up adding more/less RRI than were in oroginal, need to update gap positions
				new_gap_length = len(new_RRs)
				gap_length_difference = new_gap_length - gap_length
				if gap_length_difference != 0:
					gap_index = runs.index(gap)
					for g in range(gap_index+1, len(runs)):
						runs[g] = (runs[g][0] + gap_length_difference, runs[g][1] + gap_length_difference)
				total_gap_length_difference += gap_length_difference	

			# remove any timestamps/RRI we were unable to correct # TODO is this currently only used to remove 0 values?
			rri_corrected = rri_corrected[~pd.isnull(rri_corrected)]
			Ts = Ts[~pd.isnull(Ts)]
			if len(Ts) != len(rri_corrected)+1:
				raise Exception("length of timestamps is not proportional to RRI.")


		

	modification_report["excluded"] = False
	modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
	modification_report["n_RRI_detected"] = len(rri) # how many RRI were detected for the segment originally
	modification_report["n_RRI_suprathresh"] = len(outlier_idx)
	modification_report["suprathresh_values"] = rri[outlier_idx] if len(outlier_idx) > 0 else []
	modification_report["notes"] = ""

	if save_plots:
		axs[0].scatter(timevec[rpeaks], ecg_segment[rpeaks], c='springgreen', label="Valid R Peaks")
		axs[0].scatter(timevec[noisy_rpeaks], ecg_segment[noisy_rpeaks], c="r", label="Noisy R Peaks (Removed)")
		axs[0].set_title(f"ECG Data with Detected R Peaks", loc="left")

		if len(outlier_idx) > 0:
			#axs[1].plot(timevec[rpeaks][:-1], rri, c="dimgray", label="HRV")
			#axs[1].plot(timevec[rpeaks][:-1], rri_corrected, c="crimson", label="Outlier-Corrected HRV") # KEEP IF USING MY_METHOD
			axs[1].plot(Ts_original[1:], rri, c="dimgray", label="HRV")
			axs[1].plot(Ts[1:], rri_corrected, c="crimson", label="Outlier-Corrected HRV") # DVC METHOD

			axs[2].plot(Ts[1:], rri_corrected, c="crimson")
			axs[2].set_ylabel("ms")
			axs[2].set_xlabel("Time (ms)")
			axs[2].set_title("Corrected HRV Signal (Zoomed)", loc="left")		
		else:
			axs[1].plot(timevec[rpeaks][:-1], rri, c="crimson", label="HRV")
			axs = axs[:2]

		axs[1].set_title(f"HRV Signal", loc="left")

		fig.suptitle(save_plot_filename)

		axs[0].set_ylabel("uV")
		axs[1].set_ylabel("ms")

		axs[1].set_xlabel("Time (ms)") # TODO is this correct (previous "Datapoint No.")
		

		axs[0].legend()
		axs[1].legend()
		
		# save to image for inspection
		# should be 1280x720 (720p)
		fig.set_size_inches(12.80, 7.2)
		fig.savefig(f"{save_plots_dir}/{save_plot_filename}", dpi=100)


	""" Calculate HRV Metrics """
  
	try:
		freq_dom_hrv = pyhrv.frequency_domain.welch_psd(nni=rri_corrected, show=False)
	except Exception:	
		modification_report["notes"] += ". Frequency domain calc failed - INVESTIGATE"
		freq_dom_hrv = np.NaN

	try:
		time_dom_hrv = pyhrv.time_domain.time_domain(nni=rri_corrected, sampling_rate = ecg_srate, show=False, plot=False)
	except ZeroDivisionError: # temporary until bug fixed in sdnn_index()
		modification_report["notes"] += ". Zero Division Error (probably bug in sdnn_index()), so time domain excluded."
		time_dom_hrv = np.NaN

	plt.close("all")
	
	return rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report



def hrv_whole_recording(ecg, ecg_srate, segment_length_min, verbose = True,
		save_plots=False, save_plots_dir=None,
		use_emd=True, use_reflection=True, use_segmenter="engZee", remove_noisy_beats=True, remove_noisy_RRI=True, rri_in_ms = True,
		QRS_MAX_DIST_THRESH = 0.30, DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER = 0.25, DBSCAN_MIN_SAMPLES=100, rng=np.random.default_rng()): 
	"""
	Break a long-term ECG recording into n-minute segments, calculate HRV metrics, and return results in separate Pandas DataFrames.

	Args:
		ecg:                                (NumPy ndarray)     Long-term ECG recording, that we will break into segments.
		ecg_srate:                          (int)               Sample rate of the ECG recording
		segment_length_min:                 (float)             The length of each segment, in minutes (e.g 5.0 for 5 minutes, 0.5 for 30 seconds)
		verbose:                            (bool)              Print progress
		(for other parameters, see hrv_per_segment())

	Returns:
		time_dom_df:                        (Pandas DataFrame)  A DataFrame containing Time Domain HRV Metrics per segment (or NaN if calculation wasn't possible)
		freq_dom_df:                        (Pandas DataFrame)  A DataFrame containing Frequency Domain HRV Metrics per segment (or NaN if calculation wasn't possible)
		modification_report_df:             (Pandas DataFrame)  A DataFrame containing information on what modifications were performed to a segment, whether it was excluded, and any details if it was excluded.
	  
	"""


	if True in np.isnan(ecg):
		raise Exception("ECG must be a consecutive recording, with no NaN.")


	# store 
	time_dom_hrvs = []
	freq_dom_hrvs = []
	modification_reports = [] 

	onsets = np.arange(0, len(ecg), (segment_length_min * 60) * ecg_srate, dtype=int)

	for i in range(0, len(onsets)-1):
		if verbose:
			print(f"\r{i}/{len(onsets)-1}", end="")

		segment = ecg[onsets[i]:onsets[i+1]] # TODO will this overflow

		rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report = hrv_per_segment(
					segment, ecg_srate, segment_length_min, timevec=None, segment_idx = i,
					save_plots=save_plots, save_plots_dir=save_plots_dir, save_plot_filename=f"Segment #{i}",
					use_emd=use_emd, use_reflection=use_reflection, use_segmenter=use_segmenter, remove_noisy_beats=remove_noisy_beats, remove_noisy_RRI=remove_noisy_RRI, rri_in_ms = rri_in_ms,
					QRS_MAX_DIST_THRESH = QRS_MAX_DIST_THRESH, DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER = DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER, DBSCAN_MIN_SAMPLES=DBSCAN_MIN_SAMPLES, rng=rng
					)

		time_dom_hrvs.append(time_dom_hrv)
		freq_dom_hrvs.append(freq_dom_hrv)
		modification_reports.append(modification_report)
	
	

	segment_labels = np.array(range(0, len(onsets)-1))

	time_dom_df, freq_dom_df, modification_report_df = produce_hrv_dataframes(time_dom_hrvs, freq_dom_hrvs, modification_reports, segment_labels)


	return time_dom_df, freq_dom_df, modification_report_df


def produce_hrv_dataframes(time_dom_hrvs, freq_dom_hrvs, modification_reports, segment_labels):
	 
	time_dom_df = pd.DataFrame(time_dom_hrvs, index=segment_labels, columns=time_dom_keys)
	freq_dom_df = pd.DataFrame(freq_dom_hrvs, index=segment_labels, columns=freq_dom_keys)

	# make some adjustments
	time_dom_df.drop(columns=["tinn_n", "tinn_m", "tinn", "nni_histogram"], inplace=True) # pyHRV throws warning that tinn calculation is faulty. 
	freq_dom_df.drop(columns=["fft_bands", "fft_plot"], inplace=True)

	splittable_columns = ['fft_peak', 'fft_abs', 'fft_rel', 'fft_log', 'fft_norm']	
	for splittable_column in splittable_columns:
		# vlf, lf and hf values for each of these properties are stored in tuple form (vlf, lf, hf) - not great


		if splittable_column == "fft_norm": # TODO why does this only have 2 entries?
			continue

		# set up arrays for each frequency band
		vlf = np.full(freq_dom_df.shape[0], fill_value=np.NaN, dtype=np.float64)
		lf = np.full(freq_dom_df.shape[0], fill_value=np.NaN, dtype=np.float64)
		hf = np.full(freq_dom_df.shape[0], fill_value=np.NaN, dtype=np.float64)

		i = 0
		for entry in freq_dom_df[splittable_column]:
			if not pd.isnull(entry):
				vlf[i] = entry[0]
				lf[i] = entry[1]
				hf[i] = entry[2]
			i += 1

		freq_dom_df[f"{splittable_column}_VLF"] = vlf
		freq_dom_df[f"{splittable_column}_LF"] = lf
		freq_dom_df[f"{splittable_column}_HF"] = hf

	freq_dom_df.drop(columns=splittable_columns, inplace=True)


	modification_report_df = pd.DataFrame({"segment_idx":   [i["seg_idx"] for i in modification_reports], 
							"excluded":                 [i["excluded"] for i in modification_reports],
							"n_rpeaks_noisy":           [i["n_rpeaks_noisy"] for i in modification_reports],
							"n_RRI_detected":           [i["n_RRI_detected"] for i in modification_reports], 
							"n_RRI_suprathresh":        [i["n_RRI_suprathresh"] for i in modification_reports], 
							"suprathresh_RRI_values":   [i["suprathresh_values"] for i in modification_reports],
							"notes":                    [i["notes"] for i in modification_reports]})
	return time_dom_df, freq_dom_df, modification_report_df


def save_hrv_dataframes(time_dom_df, freq_dom_df, modification_report_df, save_dfs_dir="out"):

	if not os.path.exists(save_dfs_dir):
	   print("Setting up directory for HRV Output: at '{}'".format(save_dfs_dir))
	   os.makedirs(save_dfs_dir, exist_ok=True)

	time_dom_df.to_csv(f"{save_dfs_dir}/TIMEDOM.csv")
	freq_dom_df.to_csv(f"{save_dfs_dir}/FREQDOM.csv")
	modification_report_df.to_csv(f"{save_dfs_dir}/MODIFICATION_REPORT.csv")


def load_hrv_dataframes(save_dfs_dir="out"):
   
	time_dom_df = pd.read_csv(f"{save_dfs_dir}/TIMEDOM.csv")
	freq_dom_df = pd.read_csv(f"{save_dfs_dir}/FREQDOM.csv")
	modification_report_df = pd.read_csv(f"{save_dfs_dir}/MODIFICATION_REPORT.csv")

	return time_dom_df, freq_dom_df, modification_report_df

