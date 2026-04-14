import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

image_names = [
    'faint_pit.tiff',
    'big-pit1.tiff'
]

def plot_hist_derivatives(image : cv2.typing.MatLike, axs : np.ndarray, row_label : str) -> None:
    # raw histogram data (freq)
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
    x_bins = bin_edges[:-1] # drop last edge
    
    """
    ==================== DERIVATIVES ====================
    """
    
    # central difference (equivalent to 1D Sobel operator)
    deriv_sobel = np.gradient(hist)
    
    # savitzky-golay filter (polyorder < window length, first derivative)
    # applies element-wise convolution
    deriv_savgol = savgol_filter(hist, window_length=15, polyorder=3, deriv=1)
    
    # 1D gaussian derivative (sigma for smoothing radius, first derivative)
    deriv_gaussian = gaussian_filter1d(hist, sigma=3, order=1)
    
    # univariate spline derivative (smoothing factor)
    spline = UnivariateSpline(x_bins, hist, s=10000)
    deriv_spline = spline.derivative()(x_bins)
    
    """
    ==================== PLOTTING ====================
    """
    
    # image
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title(f"{row_label} Image")
    axs[0].axis('off')
    
    # sobel
    axs[1].plot(x_bins, hist, color='gray', alpha=0.5, label='Raw Hist')
    ax1_twin = axs[1].twinx()
    ax1_twin.plot(x_bins, deriv_sobel, color='yellow', label='Sobel Deriv')
    axs[1].set_title(f"{row_label}: Sobel 1D")
    
    # savitzky-golay
    axs[2].plot(x_bins, hist, color='gray', alpha=0.5)
    ax2_twin = axs[2].twinx()
    ax2_twin.plot(x_bins, deriv_savgol, color='blue', label='S-G Deriv')
    axs[2].set_title(f"{row_label}: Savitzky-Golay")
    
    # gaussian
    axs[3].plot(x_bins, hist, color='gray', alpha=0.5)
    ax3_twin = axs[3].twinx()
    ax3_twin.plot(x_bins, deriv_gaussian, color='green', label='Gaussian Deriv')
    axs[3].set_title(f"{row_label}: Gaussian 1D")
    
    # spline derivative
    axs[4].plot(x_bins, hist, color='gray', alpha=0.5)
    ax4_twin = axs[4].twinx()
    ax4_twin.plot(x_bins, deriv_spline, color='red', label='Spline Deriv')
    axs[4].set_title(f"{row_label}: Smoothing Spline")
    
    # format axes for derivative plots
    for ax in axs[1:]:
        ax.set_xlabel("Gray Value")
        ax.set_xlim(0, 255)
        ax.set_yticks([]) # hide primary y-axis ticks
    

def setup_plot(image_name : str) -> None:
    image = cv2.imread(f'images/{image_name}', cv2.IMREAD_COLOR)
    assert image is not None, f"Image '{image_name}' could not be loaded."
    
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    hs = min(h, w) // 2
    cropped = image[cy - hs:cy + hs, cx - hs:cx + hs] # cropped to min(h, w)^2, keeping center the same
    
    # grayscale
    image_float = cropped.astype(np.float32)
    gray_float = np.mean(image_float, axis=2)
    gray : cv2.typing.MatLike = gray_float.astype(np.uint8)
    gaussian_gray = cv2.GaussianBlur(gray, (0, 0), 9.5)
    
    # single figure: 2 rows and 5 columns
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    ext = image_name.find('.')
    fig.suptitle(f"Histogram Derivative Analysis: {image_name[:ext]}", fontsize=18)
    
    plot_hist_derivatives(gray, axs[0, :], "Raw")
    plot_hist_derivatives(gaussian_gray, axs[1, :], "Blurred")
    plt.tight_layout()
    fig.savefig(f'figures/{image_name[:ext]}_Deriv_Analysis.png')
    

if __name__ == "__main__":
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 5, "Please limit the number of images to 5"
    
    for filename in image_names:
        setup_plot(filename)
    
    plt.show()
