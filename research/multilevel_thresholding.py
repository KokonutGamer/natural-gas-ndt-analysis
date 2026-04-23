import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

image_names = [
    # 'faint_pit.tiff',
    # 'big-pit1.tiff',
    # '1018Steel_Section_A1_largerMP.tiff',
    # '1018Steel_Section_A1_Raw.tiff',
    # '1018Steel_Section_A6_smallerMP.tiff',
    # 'Aluminum_MP_1.tiff',
    # 'Aluminum_MP_2.tiff',
    # 'BlackIron_OMS_corrosion-crack.tif',
    # 'cc-8.tif',
    'fs-1.bmp',
    'fs-2.bmp',
    'fs-3.bmp',
    'fs-4.bmp',
    'fs-5.bmp',
]

def apply_multilevel_thresholding(image : cv2.typing.MatLike, title : str) -> None:
    """
    ==================== HISTOGRAM ====================
    """
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
    x_bins = bin_edges[:-1]
    sigma = 5.0
    hist = gaussian_filter1d(hist, sigma=sigma)
    
    """
    ==================== DERIVATIVES ====================
    """
    d1 = np.gradient(hist)
    d2 = np.gradient(d1)
    
    # find local maxima
    peaks, _ = find_peaks(hist, prominence=50)
    k = len(peaks)
    print(f"Found {k} local maxima at gray values: {peaks}")
    
    valleys = []
    if k > 1:
        for i in range(k - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            valley = p1 + np.argmin(hist[p1:p2])
            valleys.append(valley)
    print(f"Calculated {len(valleys)} thresholds at: {valleys}")
    
    """
    ==================== MULTILEVEL THRESHOLDING ====================
    """
    if len(valleys) > 0:
        binned = np.digitize(image, valleys)
        display_scale = 255 / len(valleys)
        thresholded_image = (binned * display_scale).astype(np.uint8)
    else:
        thresholded_image = image.copy()
    
    """
    ==================== PLOTTING ====================
    """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{title} - Multilevel Thresholding (k={k})", fontsize=16)

    # original image
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # histogram, first derivative, second derivative
    axs[1].plot(x_bins, hist, color='blue', label='Smoothed Hist')
    axs[1].plot(peaks, hist[list(peaks)], "x", color='red', markersize=10, label='Peaks')
    
    # vertical lines for thresholds
    for v in valleys:
        axs[1].axvline(v, color='green', linestyle='--', alpha=0.7)
        
    axs[1].set_title("Histogram & Thresholds")
    axs[1].legend()

    # derivatives
    axs[2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[2].plot(x_bins, d1, color='orange', label='1st Deriv')
    axs[2].plot(x_bins, d2, color='purple', label='2nd Deriv')
    axs[2].set_title("1st & 2nd Derivatives")
    axs[2].legend()

    # thresholded image
    axs[3].imshow(thresholded_image, cmap="gray")
    axs[3].set_title(f"Thresholded ({len(valleys) + 1} Levels)")
    axs[3].axis('off')

    for ax in axs[1:3]:
        ax.set_xlabel("Gray Value")
        ax.set_xlim(0, 255)

    plt.tight_layout()
    plt.savefig(f'figures/multilevel-thresholding/{title}_Thresholded.png')
    
def setup_plot(image_name: str) -> None:
    image = cv2.imread(f'images/{image_name}', cv2.IMREAD_COLOR)
    assert image is not None, f"Image '{image_name}' could not be loaded."

    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    half_size = min(h, w) // 2
    cropped = image[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    gray_float = np.mean(cropped.astype(np.float32), axis=2)
    gray = gray_float.astype(np.uint8) 

    # We use the Gaussian blurred image for the cleanest segmentation
    # gaussian_gray = cv2.GaussianBlur(gray, (0, 0), 9.5)

    ext = image_name.find('.')
    base_name = image_name[:ext]
    apply_multilevel_thresholding(gray, base_name)

if __name__ == "__main__":
    assert len(image_names) > 0, "No image names were provided"
    
    for filename in image_names:
        setup_plot(filename)
    
    plt.show()
