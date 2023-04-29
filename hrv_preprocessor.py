import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, interpolate, signal

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


def find_nearest(array, value):
			# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array (modified)
			idx = np.nanargmin((np.abs(array - value)))

			return idx

def hrv_per_segment(ecg_segment, ecg_srate, segment_length_min, timevec=None, segment_idx=0,
					save_plots=False, save_plots_dir='saved_plots', save_plot_filename=math.floor(time.time()),
					use_emd=True, use_reflection=True, use_segmenter="engzee", remove_noisy_beats=True, remove_noisy_RRI=True, rri_in_ms = True,
					QRS_MAX_DIST_THRESH = 0.30, DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER = 0.25, DBSCAN_MIN_SAMPLES = 100): 
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

	if save_plots and not os.path.exists(save_plots_dir):
		print("Setting up directory for saving plots at {}".format(save_plots_dir))
		os.makedirs(save_plots_dir, exist_ok=True)


	rri_time_multiplier = 1000 if rri_in_ms else 1 # do we want RRI in ms or s
	if timevec is None:
		timevec = np.array(range(0, len(ecg_segment)))


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
	# if there isn't enough data in the segment to calculate LF/HF
	if len(ecg_segment) < ecg_srate * (60 * 2):

		modification_report["excluded"] = True
		modification_report["notes"] = "Not enough data recorded in this segment interval"
 
		return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
	# </EXIT_CONDITION>



	""" Apply Empirical Mode Decomposition (EMD) to detrend the ECG Signal (remove low freq drift) """

	if use_emd:  
		
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
			fig, axs = plt.subplots(2, 1, sharex=True)
			axs[0].plot(timevec, ecg_segment, c="lightgrey", label="Raw ECG Signal")
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
		rpeaks = chosen_segmenter(signal=ecg_reflected, sampling_rate=ecg_srate)["rpeaks"]
		# <EXIT_CONDITION>
		if len(rpeaks) < min_rpeaks_in_reflected:
	 
			modification_report["excluded"] = True
			modification_report["notes"] = f"Segmenter ({use_segmenter}) detected not enough Rpeaks ({len(rpeaks)} < {min_rpeaks_in_reflected}) in reflected Rpeaks"

			return None, None, None, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>


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

	elif not use_reflection: 
	   
		rpeaks = chosen_segmenter(signal=ecg_segment, sampling_rate=ecg_srate)["rpeaks"]
  
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



		
	# correct candidate rpeaks to the maximum ECG value within a time tolerance (0.05s by default)
	rpeaks = biosppy.signals.ecg.correct_rpeaks(ecg_segment, rpeaks, sampling_rate = ecg_srate, tol = 0.05)["rpeaks"]

	if remove_noisy_beats:
		""" Attempt to remove noise that has been incorrectly identified as QRS """

		# look for noise in the ECG signal by checking if each detected QRS complex is similar enough to the average QRS in this segment
		beats = biosppy.signals.ecg.extract_heartbeats(ecg_segment, rpeaks, ecg_srate)["templates"] # get ECG signal a small amount of time around detected Rpeaks
		avg_beat = np.mean(beats, axis=0) # produce the average/'typical' beat within the segment
		
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
		db = DBSCAN(eps = (np.mean(rri_corrected) * DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER), min_samples=DBSCAN_MIN_SAMPLES).fit(poincare.T)
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
			labels_text = ["Valid" if label >= 0 else "Outlier" for label in labels]
			sns.scatterplot(x=rri[:-1], y=rri[1:], hue=labels_text, palette={"Valid": "#000000", "Outlier": "#FF0000"}, ax=ax2)
			fig2.savefig(f"{save_plots_dir}/{save_plot_filename}_POINCARE", dpi=100)

		# get idx of outliers in rri
		outlier_idx = np.where(poincare_outliers == 1)[0]

		# produce a copy without the RRIs exceeding the threshold, for use in interpolation
		rri_corrected_supra_removed = np.delete(rri_corrected, outlier_idx)
		rri_corrected_supra_idx_removed = np.delete(np.array(range(0, len(rri_corrected))), outlier_idx)
		
		# <EXIT_CONDITION>
		# if too many have been detected as outliers
		if sum(rri_corrected_supra_removed) < (rri_time_multiplier * 120):

			modification_report["excluded"] = True
			modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
			modification_report["n_RRI_detected"] = len(rri)
			modification_report["notes"] = f"Sum of corrected RRI (outliers removed) ({sum(rri_corrected)}) was less than 2Mins"

			return rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report
		# </EXIT_CONDITION>
			
		# interpolate points above threshold
		rri_corrected[outlier_idx] = np.interp(outlier_idx, rri_corrected_supra_idx_removed, rri_corrected_supra_removed)

	modification_report["excluded"] = False
	modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
	modification_report["n_RRI_detected"] = len(rri) # how many RRI were detected for the segment originally
	modification_report["n_RRI_suprathresh"] = len(outlier_idx)
	modification_report["suprathresh_values"] = rri[outlier_idx]
	modification_report["notes"] = ""

	if save_plots:
		axs[0].scatter(timevec[rpeaks], ecg_segment[rpeaks], c='springgreen', label="Valid R Peaks")
		axs[0].scatter(timevec[noisy_rpeaks], ecg_segment[noisy_rpeaks], c="r", label="Noisy R Peaks (Removed)")
		axs[0].set_title(f"ECG Data with Detected R Peaks")

		if len(outlier_idx) > 0:
			axs[1].plot(timevec[rpeaks][:-1], rri, c="dimgray", label="Pre-processed HRV")
			axs[1].plot(timevec[rpeaks][:-1], rri_corrected, c="crimson", label="Processed HRV")
		else:
			axs[1].plot(timevec[rpeaks][:-1], rri, c="crimson", label="HRV")

		axs[1].set_title(f"HRV Signal")

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
  
	freq_dom_hrv = pyhrv.frequency_domain.welch_psd(nni=rri_corrected, show=False)
	try:
		time_dom_hrv = pyhrv.time_domain.time_domain(nni=rri_corrected, sampling_rate = ecg_srate, show=False, plot=False)
	except ZeroDivisionError: # temporary until bug fixed in sdnn_index()
		modification_report["notes"] = "Zero Division Error (probably bug in sdnn_index()), so time domain excluded."
		time_dom_hrv = np.NaN

	plt.close("all")

	return rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report



def hrv_whole_recording(ecg, ecg_srate, segment_length_min, verbose = True,
		save_plots=False, save_plots_dir=None,
		use_emd=True, use_reflection=True, use_segmenter="engZee", remove_noisy_beats=True, remove_noisy_RRI=True, rri_in_ms = True,
		QRS_MAX_DIST_THRESH = 0.30, DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER = 0.25, DBSCAN_MIN_SAMPLES=100): 
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
					QRS_MAX_DIST_THRESH = QRS_MAX_DIST_THRESH, DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER = DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER, DBSCAN_MIN_SAMPLES=DBSCAN_MIN_SAMPLES
					)

		time_dom_hrvs.append(time_dom_hrv)
		freq_dom_hrvs.append(freq_dom_hrv)
		modification_reports.append(modification_report)
	
	

	segment_labels = np.array(range(0, len(onsets)-1))

	time_dom_df = pd.DataFrame(time_dom_hrvs, index=segment_labels, columns=list(time_dom_hrvs[0].keys()))
	freq_dom_df = pd.DataFrame(freq_dom_hrvs, index=segment_labels, columns=list(freq_dom_hrvs[0].keys()))
	

   
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


