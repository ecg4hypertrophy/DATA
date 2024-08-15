import numpy as np
import pywt
import os
import math

def sgn(num):
    """Return the sign of the input number."""
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0

def wavelet_noising(signal_data):
    """Apply wavelet denoising on the input signal."""
    data = signal_data
    w = pywt.Wavelet('db8')
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)

    length0 = len(data)

    # Calculate threshold
    abs_cd1 = np.abs(cd1)
    median_cd1 = np.median(abs_cd1)
    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))

    # Soft-thresholding
    def soft_thresholding(coefficients, lamda, a=0.5):
        return np.array([
            sgn(coeff) * (abs(coeff) - a * lamda) if abs(coeff) >= lamda else 0.0 
            for coeff in coefficients
        ])

    cd1 = soft_thresholding(cd1, lamda)
    cd2 = soft_thresholding(cd2, lamda)
    cd3 = soft_thresholding(cd3, lamda)
    cd4 = soft_thresholding(cd4, lamda)
    cd5 = soft_thresholding(cd5, lamda)

    usecoeffs = [ca5, cd5, cd4, cd3, cd2, cd1]
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs

def normalize(data):
    """Normalize the input data to the range [0, 1]."""
    data = data.astype('float')
    mx = np.max(data, axis=0).astype(np.float64)
    mn = np.min(data, axis=0).astype(np.float64)
    return np.true_divide(data - mn, mx - mn, out=np.zeros_like(data - mn), where=(mx - mn) != 0)

# Process the ECG data files
data_files = os.listdir('/mnt/data/ECG-npy')
for file in data_files:
    rdata = np.load("/mnt/data/ECG-npy/" + file)
    for k in range(12): 
        # Apply wavelet denoising
        rdata[:, k] = wavelet_noising(rdata[:, k])
        # Further wavelet processing
        wave = pywt.wavedec(rdata[:, k], 'db8', level=7)
        rdata[:, k] = pywt.waverec(np.multiply(wave, [0, 1, 1, 1, 1, 1, 1, 1]).tolist(), wavelet='db8')
    # Save the processed data
    np.save(os.path.join('/mnt/data/ECG_data/denoise_data/', file), rdata)

