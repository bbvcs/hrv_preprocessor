# HRV Calculation w/ Preprocessing in Python
Python Script to calculate Time and Frequency Domain HRV metrics for segments of a long-term recording of ECG (e.g, HRV per 5min ECG segment over 7 days of continuous ECG recording), with certain preprocessing techniques (detailed below) to attempt to clean-up each segment prior to HRV processing. HRV metrics are saved to a .csv file per-segment, alongside a .csv providing some limited information on how each segment was modified.

DISCLAIMER: This is a work in progress; the techniques used to attempt to remove noise & ectopy are far from perfected and certainly not tested enough - for example, some of the constants involved are fairly arbitrary. Please feel free to improve this. Use at your own risk, for educational/research purposes only.



### Data
The script was originally developed with and tested on private ECG data ranging from 24h to 7 days in length, 120Hz, sourced from wearable devices.

To demonstrate, publicly available [1] ECG data (~24h in length, 400Hz) from the "The 3rd China Physiological Signal Challenge 2020" was used. This data can be downloaded in a 284 MB .zip file from [here](http://2020.icbeb.org/CSPC2020) (see Notes section, "[2020-8-18]")

Example outputs from Subject "A01" are available in the "*out*" directory of this repository, as well as an example plot (produced using *pyplot*) of the Frequency Domain LF/HF Ratio of this subject. All of these are produced by running the default *example_usage/py* script.



### Setup and Running
I would recommend running on an Ubuntu/Debian based OS (should work with any Linux OS and macOS though), with at least Python 3.8 installed.

1. Download the TrainingSet Data from the link above (see **Data**), and extract the .zip into the same directory as the scripts; there should be a directory "*TrainingSet*", with a subdirectory "*data*"
	- If you want to use your own data, inspect "**example_usage.py**", and modify so it loads your ECG data as a 1D NumPy array "*ecg*"
2. Enter the following commands: `pipenv shell`, followed by `pipenv update`. This will setup a pipenv virtual environment, and install all necessary packages for the python scripts. Pipenv combines the "pip" python package manager and the "venv" virtual environment tool. Pipenv uses a "Pipfile" to keep track of dependencies for the project, this file is provided in this repository. For more info on pipenv (and how to install if you don't have it): https://pipenv.pypa.io/en/latest/
3. Once you're in a pipenv shell, run *example_usage.py* using `python3.8 -i example_usage.py` (`-i` is optional, but will allow you to have the dataframes in an interactive shell once complete, so you can try out some more plots)

### How Preprocessing and HRV Calculation Works
1. Break the ECG into n-minute long segments (by default, 5min)
2. For each segment:
    - Use Empirical Mode Decomposition (EMD) to remove low-frequency drift from/straighten-out the signal, intended to help the R peak detection algorithm.
    - Use an R peak detection algorithm provided by *biosppy* (EngZee by default, but many others are available - Christov & Hamilton's also look good. Can be set using a parameter. See biosppy.signals.ecg documentation for more detail.)
    - Use *biosppy*'s rpeak correction - this basically just moves the detected R peak to a local maximum in the ECG signal, useful as sometimes the detection algorithm misses it slightly (e.g places it on the T wave)
    - Attempt to remove R peaks that may have mistakenly been placed on a noisy QRS complex.
        - *biosppy* provides a function to get the "templates" - the ECG snippets associated with each R peak it detected.
        - store all of these in an array, then determine the "average QRS" for the segment, by taking the mean of each template.
        - Use Dynamic Time Warping (DTW) to get a distance value for each beat from this average - the noisy beats should correlate poorly, and can be removed using a threshold (currently, any that have a distance value above 0.30 are removed.)
    - Use the R peak locations to calculate the R-R interval (RRI) series for this segment.
    - Noise/Misidentified R peaks/Ectopic beats etc all cause spikes in the RRI series, which are detrimental to HRV calculation - so we must try to correct these.
        - Iterate forwards over the RRI series, for each point:
            - Calculate the mean of the RRI either side of it.
                - if the point behind was determined to be an outlier/spikes, take the point before that instead.
                - Only try this 5 times - If all 5 behind the point are outliers, just use the point immediately behind it.
            - Calculate the difference between this RRI and the surrounding mean.
            - If this difference is greater than a percentage (30%) of the surrounding mean, then this RRI is an outlier.
        - Do the same, iterating backwards over the series - so now if the point in front of an RRI was previously determined as an outlier, we can take the next point instead, but again only try for 5 points.
        - Remove duplicates from the list of outlier points built up over both runs
        - Interpolate points we have determined as outliers.  

3. NaN Exit Conditions
    - There are some cases where we would want to discard a segment from HRV calculation.
    - The following are the current conditions for this, and are fairly arbitrary:
        - If, for whatever reason, not enough ECG readings occurred in the segment interval (lost data due to recording issues)
        - If EMD produced less than 3 IMFs.
        - The segmenter algorithms (multiple are tried in case of a fault in any particular one) didn't give an R Peak count above the *minimum*.
        - All R Peaks were determined to be associated with noise.
        - If more than 40% of all R Peaks were determined to be associated with noise.
        - If the number of R Peaks did not meet the *minimum* after removal of those associated with noise.
        - The sum of the RRI is less than 2 minutes, the minimum for calculation of LF HRV Frequency Domain Metrics.
        - If more than 20% of all R Peaks were determined to be associated with noise, check that the sum of the RRI related to the longest consecutive run of Valid R Peaks (not associated with noise) is greater than 2 minutes - if not, remove.
            - I did this to try to remove segments where the noise is spread evenly/frequently throughout a segment, meaning interpolation occurs too often, so won't resemble real data.
            - If the noise is only contained to a small part of the segment, and most of the data is valid, it may still be OK for HRV.
    - Note the "*minimum*" number of R Peaks is currently 27 per minute (27bpm).
    - Usually, a dictionary of time/frequency domain HRV metrics is saved per segment. Discarding a segment is currently implemented by saving a NaN instead.
    - These definitely require improvement and refinement:
        - A histogram could be used to determine the reasonable number of R Peaks per beat for each subject.
        - If, for example, the longest consecutive run of valid R Peaks is 2 minutes, and all other consecutive runs are less than 2 minutes, we should discard the rest of the data in this segment, only keeping the consecutive run.
	 	- and much more!
    - A note of the problem that caused the segment to be removed is stored in the **Modifications CSV**.
    
4. Return Values
    - CSVs for Segment Start Time ID -> Time/Frequency Domain Metrics (seperately)
        - Note - ignore Time Domain SDNN and TINN; these cannot be calculated currently (SDNN requires 24 hour long data, *pyHRV* throws a warning that a bug is causing TINN values to be incorrectly calculated) (I'll remove these at some point, and make the HRV CSV's more nicely formatted)
    - **Modifications CSV**; for each Segment, was it excluded or not (T/F), what was the number of noisy QRS, what was the number of RRI determined as spikes/outliers in the HRV Signal, and some brief notes if it was excluded.
	- Note, `hrv_per_segment()` by default (can be disabled via a parameter) saves a diagram of the ECG & HRV per segment (with corrections visible) to saved_plots; useful for brief inspection to see how the code performed, but due to a reduction in size when saving a plot to image and lack of zooming, not very useful for debugging - change this to `plt.show()` for this.

### Methods available:
This script can be used like a library - see how this is done in `example_usage.py`.

- `hrv_per_segment()`: 
	- Calculate HRV metrics for a segment of ECG, returning a tuple of ReturnTuples containing HRV Metrics and a Modification Report for this segment. 
	- This is used by `hrv_whole_recording()`; but it may be useful to call if segmentation of your ECG is more complicated.

- `hrv_whole_recording()`:
	- The main function; takes a long-term ECG recording, breaks it into n-minute segments, calculates HRV metrics, and returns results in separate Pandas DataFrames.
	- This method assumed a continuous ECG recording not interrupted by NaNs, at a constant sample rate, but it can be modified if this isn't the case.
- `save_hrv_dataframes()`:
	- Takes the output of `hrv_whole_recording()` and saves to .csv's - useful if you want to load the HRV data later on without re-calculating.

These methods have many parameters; see docstrings in the code, as well as `example_usage.py` for information on how to call them.

### References
[1] Z. Cai, C. Liu, G. Hongxiang, X. Wang, "An Open-Access Long-Term Wearable ECG Database for Premature Ventricular Contractions and Supraventricular Premature Beat Detection", *Journal of Medical Imaging and Health Informatics*, Nov. 2020. [Online] Available: [https://www.researchgate.net/publication/345142269_An_Open-Access_Long-Term_Wearable_ECG_Database_for_Premature_Ventricular_Contractions_and_Supraventricular_Premature_Beat_Detection](https://www.researchgate.net/publication/345142269_An_Open-Access_Long-Term_Wearable_ECG_Database_for_Premature_Ventricular_Contractions_and_Supraventricular_Premature_Beat_Detection)


### Author
Billy C. Smith  
bcsm@posteo.net  
MSc Student @ Newcastle University  
27/09/2022


