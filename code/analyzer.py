import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
from scipy.signal import find_peaks
from skimage import filters

def process_csd(csd_array, plot=False, fontsize = 14):
    """
    Process a charge stability diagram to remove noise and artifacts.

    Parameters:
        csd_array (ndarray): 2D array representing the charge stability diagram.
        plot (bool): Whether to plot the original and processed CSD.

    Returns:
        ndarray: 2D array representing the processed charge stability diagram.
    """
    csd_array = gaussian_filter(csd_array, sigma=0.5)
    processed_csd = np.abs(filters.sobel(csd_array, axis=1)) + 0.1*np.abs(filters.sobel(csd_array, axis=0))
    #gaussian
   
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].pcolormesh(csd_array)
        ax[0].set_title("Original CSD", fontsize=fontsize)
        
        ax[1].pcolormesh(processed_csd)
        ax[1].set_title("Processed CSD", fontsize=fontsize)

        for k in range(2):
            ax[k].set_xlabel("$V_1$ (1/V)")
        ax[0].set_ylabel("$V_2$ (1/V)")
    return processed_csd

def get_fft_magnitude(csd_array, v1, v2):
    """
    Get the magnitude of the 2D Fourier Transform of a charge stability diagram.

    Parameters:
        csd_array (ndarray): 2D array representing the charge stability diagram.
        v1 (ndarray): Vector of gate voltage values for axis 1.
        v2 (ndarray): Vector of gate voltage values for axis 2.

    Returns:
        tuple: 2D array representing the magnitude of the 2D Fourier Transform, 
               and frequency vectors for both axes.
    """
    fft_result = np.fft.fftshift(np.fft.fft2(csd_array))
    fft_magnitude = np.abs(fft_result)
    freq_v1 = np.fft.fftshift(np.fft.fftfreq(len(v1), d=np.abs(v1[1] - v1[0])))
    freq_v2 = np.fft.fftshift(np.fft.fftfreq(len(v2), d=np.abs(v2[1] - v2[0])))
    return fft_magnitude, freq_v1, freq_v2

def extract_jet_points(fft_magnitude, freq_v1, freq_v2):
    """
    Extract the two jet points from the 2D Fourier Transform of a charge stability diagram.

    Parameters:
        fft_magnitude (ndarray): 2D array representing the magnitude of the 2D Fourier Transform.
        freq_v1 (ndarray): Vector of frequency values for axis 1.
        freq_v2 (ndarray): Vector of frequency values for axis 2.

    Returns:
        tuple: Smoothed FFT, coordinates of the two jet points, cut through FFT, 
               horizontal cut values, and angles of the jets.
    """
    smoothened_fft = gaussian_filter(fft_magnitude, sigma=3)
    interpolated_fft = interp2d(freq_v1, freq_v2, smoothened_fft)


    cut_space = np.linspace(0, 1, 101)
    zero_freq_v1_idx = np.where(freq_v1 == 0)[0][0]
    zero_freq_v2_idx = np.where(freq_v2 == 0)[0][0]
    startx = np.array([freq_v1[int(zero_freq_v1_idx / 2)], freq_v2[-1]])
    endx = np.array([freq_v1[-1], freq_v2[int(zero_freq_v2_idx / 2)]])
    xcut = np.array([(1 - x) * startx + x * endx for x in cut_space])

    hcut1 = np.array([interpolated_fft(*x) for x in xcut]).flatten()
    try: 
        peaks1, _ = find_peaks(hcut1, height=np.max(hcut1) / 4)
        peaks = [[xcut[peaks1[0], 0], xcut[peaks1[0], 1]],
             [xcut[peaks1[1], 0], xcut[peaks1[1], 1]]]
    except:
        plt.pcolormesh(freq_v1, freq_v2, smoothened_fft)
        plt.set_title("FFT", fontsize=12)
        plt.set_xlabel("1/V1 (1/V)")
        plt.set_ylabel("1/V2 (1/V)")


    thetas = [np.arctan2(*peaks[1][::-1]), np.arctan2(*peaks[0][::-1])]
    return smoothened_fft, peaks, xcut, hcut1, thetas

def get_control_plots(fft, smoothed_fft, freq_v1, freq_v2, peaks, xcut, hcut1, thetas, fontsize = 14):
    """
    Generate control plots for the analysis of a charge stability diagram.

    Parameters:
        fft (ndarray): 2D array representing the FFT magnitude.
        smoothed_fft (ndarray): 2D array representing the smoothed FFT magnitude.
        v1 (ndarray): Vector of gate voltage values for axis 1.
        v2 (ndarray): Vector of gate voltage values for axis 2.
        freq_v1 (ndarray): Vector of frequency values for axis 1.
        freq_v2 (ndarray): Vector of frequency values for axis 2.
        peaks (list): Coordinates of the jet points.
        xcut (ndarray): 2D array representing the cut through the FFT magnitude.
        hcut1 (ndarray): Horizontal cut values.
        thetas (list): Angles of the jets.
    """
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    plt.subplots_adjust(wspace=0.2)
    x = np.linspace(0, 1000, 100)

    # Barge FFT
    #ax[0].plot(*peaks[0], "o", ms=5)
    #ax[0].plot(*peaks[1], "o", ms=5)
    #ax[0].plot(x, x * np.tan(thetas[0]), "r")
    #ax[0].plot(x / np.tan(thetas[1]), x, "b")
    #ax[0].plot(*xcut.T, "y")
    ax[0].set_xlim(freq_v1[0], freq_v1[-1])
    ax[0].set_ylim(freq_v2[0], freq_v2[-1])
    ax[0].pcolormesh(freq_v1, freq_v2, fft)
    ax[0].set_title("FFT", fontsize=fontsize)
    ax[0].set_xlabel("1/V1 (1/V)")
    ax[0].set_ylabel("1/V2 (1/V)")
    # Smoothen FFT
    
    ax[1].pcolormesh(freq_v1, freq_v2, smoothed_fft)
    ax[1].plot(*xcut.T, "y")
    ax[1].set_title("Smoothed FFT", fontsize=fontsize)
    ax[1].set_xlabel(r"$1/V_1$ (1/V)")
    ax[1].set_ylabel(r"$1/V_2$ (1/V)")
    
    # CUt
    ax[2].plot(xcut[:, 0], hcut1)
    for peak in peaks:
        ax[2].vlines(peak[0], 0, np.max(hcut1), "k")
    ax[2].set_title("FFT through cut", fontsize=fontsize)
    ax[2].set_xlabel(r"$a/V_1 + b/V_2$ (1/V)")

def get_period_along_jet(fft_magnitude, freq_v1, freq_v2, thetas, plot=False):
    """
    Get the period of the Coulomb diamond along the jets.

    Parameters:
        fft_magnitude (ndarray): 2D array representing the magnitude of the 2D Fourier Transform.
        freq_v1 (ndarray): Vector of frequency values for axis 1.
        freq_v2 (ndarray): Vector of frequency values for axis 2.
        thetas (list): Angles of the jets.
        plot (bool): Whether to plot the period along the jets.

    Returns:
        tuple: Period along the jets.
    """
    zero_freq_v1_idx = np.where(freq_v1 == 0)[0][0]
    zero_freq_v2_idx = np.where(freq_v2 == 0)[0][0]
    fft_interp = interp2d(freq_v1, freq_v2, fft_magnitude)
    cutx = np.array([fft_interp(x, x * np.tan(thetas[0])) for x in freq_v1[zero_freq_v1_idx:]]).flatten()
    cuty = np.array([fft_interp(x / np.tan(thetas[1]), x) for x in freq_v2[zero_freq_v2_idx:]]).flatten()

    try:
        peaksx, _ = find_peaks(cutx)
        peaksy, _ = find_peaks(cuty)
    except:
        print("Peaks along jets not found!")
        return None

    dominant_freq_v1 = freq_v1[zero_freq_v1_idx:][peaksx[0]]
    dominant_freq_v2 = freq_v2[zero_freq_v2_idx:][peaksy[0]]
    period_v1 = 1 / dominant_freq_v1
    period_v2 = 1 / dominant_freq_v2
    print(period_v1, period_v2)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        plt.subplots_adjust(wspace=0.3)
        ax[0].plot(freq_v1[zero_freq_v1_idx:], cutx, "r")
        ax[0].vlines(dominant_freq_v1, 0, np.max(cutx) / 2, "k")
        ax[0].set_xlabel("$1/V_2$ along jet 1 (1/V)")
        ax[0].set_ylabel("FFT magnitude")
        
        ax[1].plot(freq_v2[zero_freq_v2_idx:], cuty, "b")
        ax[1].vlines(dominant_freq_v2, 0, np.max(cuty) / 2, "k")
        ax[1].set_xlabel("$1/V_1$ along jet 2 (1/V)")

    return period_v1, period_v2



def analyze_csd(csd_array, v1, v2, plot=True):
    """
    Analyze a charge stability diagram to extract Coulomb diamond sizes and cross-capacitance,
    excluding the zero-frequency peak.

    Parameters:
        csd_array (ndarray): 2D array representing the charge stability diagram.
        v1 (ndarray): Vector of gate voltage values for axis 1.
        v2 (ndarray): Vector of gate voltage values for axis 2.
        plot (bool): Whether to plot the analysis steps.

    Returns:
        dict: Contains Coulomb diamond sizes and cross-capacitance angle.
    """
    # Step 1: Process the CSD
    csd_array = process_csd(csd_array, plot=plot)

    # Step 2: Apply 2D Fourier Transform
    fft_magnitude, freq_v1, freq_v2 = get_fft_magnitude(csd_array, v1, v2)

    # Step 3: Jets extraction
    smoothed_fft, peaks, xcut, hcut1, thetas = extract_jet_points(fft_magnitude, freq_v1, freq_v2)

    if plot:
        get_control_plots(fft_magnitude, smoothed_fft, freq_v1, freq_v2, peaks, xcut, hcut1, thetas)

    periods_v = get_period_along_jet(fft_magnitude, freq_v1, freq_v2, thetas, plot=plot)

    # Return Results
    results = {
        "coulomb_diamond_size_v1": periods_v[0],
        "coulomb_diamond_size_v2": periods_v[1],
        "cross_angle1": thetas[0],
        "cross_angle2": np.pi/2-thetas[1],
    }
    return results


