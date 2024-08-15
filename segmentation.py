import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
from scipy.signal import butter, lfilter
from time import gmtime, strftime
from denoise import wavelet_noising 

LOG_DIR = "./"
PLOT_DIR = "./"

class QRSDetectorOffline:
    """
    Python Offline ECG QRS Detector based on the Pan-Tomkins algorithm.
    This class processes ECG data to detect QRS complexes, which correspond to heartbeats.
    """

    def __init__(self, ecg_data_path, verbose=True, log_data=False, plot_data=False, show_plot=False, lead=0):
        """
        Initializes the QRSDetectorOffline class.
        
        Parameters:
            ecg_data_path (str): Path to the ECG dataset.
            verbose (bool): Flag to print the detection results.
            log_data (bool): Flag to log the detection results to a file.
            plot_data (bool): Flag to plot the detection results to a file.
            show_plot (bool): Flag to show the generated plot.
            lead (int): The ECG lead to process.
        """
        # Configuration parameters.
        self.ecg_data_path = ecg_data_path
        self.lead = lead  # Set ECG lead

        self.signal_frequency = 500  # ECG device frequency in samples per second.
        self.filter_lowcut = 0.01
        self.filter_highcut = 15.0
        self.filter_order = 1

        self.integration_window = 15  # Adjust proportionally when changing frequency (in samples).
        self.findpeaks_limit = 0.35
        self.findpeaks_spacing = 50  # Adjust proportionally when changing frequency (in samples).
        self.refractory_period = 120  # Adjust proportionally when changing frequency (in samples).
        self.qrs_peak_filtering_factor = 0.125
        self.noise_peak_filtering_factor = 0.125
        self.qrs_noise_diff_weight = 0.25

        # Loaded ECG data and processing results.
        self.ecg_data_raw = None
        self.filtered_ecg_measurements = None
        self.differentiated_ecg_measurements = None
        self.squared_ecg_measurements = None
        self.integrated_ecg_measurements = None
        self.detected_peaks_indices = None
        self.detected_peaks_values = None
        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        # Detection results.
        self.qrs_peaks_indices = np.array([], dtype=int)
        self.noise_peaks_indices = np.array([], dtype=int)
        self.ecg_data_detected = None

        # Run the detector.
        self.load_ecg_data()
        self.detect_peaks()
        self.detect_qrs()

        if verbose:
            self.print_detection_data()

        if log_data:
            self.log_path = f"{LOG_DIR}QRS_offline_detector_log_{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}.csv"
            self.log_detection_data()

        if plot_data:
            self.plot_path = f"{PLOT_DIR}QRS_offline_detector_plot_{strftime('%Y_%m_%d_%H_%M_%S', gmtime())}.png"
            self.plot_detection_data(show_plot=show_plot)

    def load_ecg_data(self):
        """Load the ECG dataset from a file."""
        self.ecg_data_raw = np.load(self.ecg_data_path)[1000:4000, :]

    def detect_peaks(self):
        """Extract peaks from the loaded ECG measurements data."""
        ecg_measurements = self.ecg_data_raw[:, self.lead]

        # Apply bandpass filter (0-15 Hz).
        self.filtered_ecg_measurements = self.bandpass_filter(ecg_measurements, lowcut=self.filter_lowcut,
                                                              highcut=self.filter_highcut, signal_freq=self.signal_frequency,
                                                              filter_order=self.filter_order)
        self.filtered_ecg_measurements[:5] = self.filtered_ecg_measurements[5]

        # Apply derivative.
        self.differentiated_ecg_measurements = np.ediff1d(self.filtered_ecg_measurements)

        # Square the signal to intensify the values.
        self.squared_ecg_measurements = self.differentiated_ecg_measurements ** 2

        # Apply moving-window integration.
        self.integrated_ecg_measurements = np.convolve(self.squared_ecg_measurements, np.ones(self.integration_window))

        # Detect peaks.
        self.detected_peaks_indices = self.findpeaks(data=self.integrated_ecg_measurements,
                                                     limit=self.findpeaks_limit,
                                                     spacing=self.findpeaks_spacing)

        self.detected_peaks_values = self.integrated_ecg_measurements[self.detected_peaks_indices]

    def detect_qrs(self):
        """Classify detected ECG peaks as either noise or QRS complexes."""
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_indices, self.detected_peaks_values):
            try:
                last_qrs_index = self.qrs_peaks_indices[-1]
            except IndexError:
                last_qrs_index = 0

            # Check for the refractory period.
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:
                if detected_peaks_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Update QRS peak value for threshold adjustment.
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                                          (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Update noise peak value for threshold adjustment.
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                                            (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # Adjust threshold.
                self.threshold_value = self.noise_peak_value + \
                                       self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        # Mark QRS detections.
        measurement_qrs_detection_flag = np.zeros([len(self.ecg_data_raw[:, 1]), 1])
        measurement_qrs_detection_flag[self.qrs_peaks_indices] = 1
        self.ecg_data_detected = np.append(self.ecg_data_raw, measurement_qrs_detection_flag, 1)

    def print_detection_data(self):
        """Print the QRS and noise peaks detection results."""
        print("QRS peaks indices")
        print(self.qrs_peaks_indices)
        print("Noise peaks indices")
        print(self.noise_peaks_indices)

    def log_detection_data(self):
        """Log the detected ECG data and results to a file."""
        with open(self.log_path, "wb") as fin:
            fin.write(b"timestamp,ecg_measurement,qrs_detected\n")
            np.savetxt(fin, self.ecg_data_detected, delimiter=",")

    def plot_detection_data(self, show_plot=False):
        """Plot the ECG detection results."""
        def plot_data(axis, data, title='', fontsize=10):
            axis.set_title(title, fontsize=fontsize)
            axis.grid(which='both', axis='both', linestyle='--')
            axis.plot(data, color="salmon", zorder=1)

        def plot_points(axis, values, indices):
            axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)

        plt.close('all')
        fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

        plot_data(axis=axarr[0], data=self.ecg_data_raw[:, 0], title='Raw ECG measurements')
        plot_data(axis=axarr[1], data=self.filtered_ecg_measurements, title='Filtered ECG measurements')
        plot_data(axis=axarr[2], data=self.differentiated_ecg_measurements, title='Differentiated ECG measurements')
        plot_data(axis=axarr[3], data=self.squared_ecg_measurements, title='Squared ECG measurements')
        plot_data(axis=axarr[4], data=self.integrated_ecg_measurements, title='Integrated ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[4], values=self.integrated_ecg_measurements, indices=self.qrs_peaks_indices)
        plot_data(axis=axarr[5], data=self.ecg_data_detected[:, self.lead], title='Raw ECG measurements with QRS peaks marked (black)')
        plot_points(axis=axarr[5], values=self.ecg_data_detected[:, self.lead], indices=self.qrs_peaks_indices)

        plt.tight_layout()
        fig.savefig(self.plot_path)

        if show_plot:
            plt.show()

        plt.close()

    def bandpass_filter(self, data, lowcut, highcut, signal_freq, filter_order):
        """Create and apply a Butterworth bandpass filter."""
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def findpeaks(self, data, spacing=1, limit=None):
        """
        Janko Slavic's peak detection algorithm implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind


if __name__ == "__main__":
    # Define the lead you want to process (0-based index, e.g., 0 for Lead I)
    lead = 0  # Change this value to select the specific lead

    # Define kernel for image processing
    kernel = np.ones((4, 4), np.uint8)

    # Load ECG data and process every lead for each file
    data = os.listdir('/mnt/data/ECG-npy')
    for ecgdata in data:
        basename, suffix = os.path.splitext(ecgdata)
        data_path = os.path.join("/mnt/data/ECG-npy/", ecgdata)
        rdata = np.load(data_path)[:, lead]

        # Denoise ECG signal
        rdata = wavelet_noising(rdata)
        wave = pywt.wavedec(rdata, 'db8', level=7)
        rdata = pywt.waverec(np.multiply(wave, [0, 1, 1, 1, 1, 1, 1, 1]).tolist(), wavelet='db8')

        # Detect QRS complexes
        qrs_detector = QRSDetectorOffline(ecg_data_path=data_path, verbose=False, log_data=False, plot_data=False, show_plot=False, lead=lead)
        QRS_ind = qrs_detector.qrs_peaks_indices

        # Plot and save the segment of ECG with detected QRS complexes
        if len(QRS_ind) >= 4:
            low_index = QRS_ind[0]
            high_index = QRS_ind[2]
            data_segment = rdata[low_index:high_index]
            plt.plot(data_segment)
            plt.xticks([])
            plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            fignamepath = os.path.join(f'/mnt/data1/ECG_data/ECG-cv/ECG-cv{lead}', basename + '.png')
            plt.savefig(fignamepath)
            plt.close()

            # Process and save the image
            im_gray = cv2.imread(fignamepath, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.erode(im_gray, kernel, iterations=1)
            im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(fignamepath, im_gray)
