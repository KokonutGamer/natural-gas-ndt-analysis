import cv2
import numpy as np
import matplotlib.pyplot as plt

image_names = [
    'cc-8.tif',
]

def image_hist(image : cv2.typing.MatLike, title : str) -> None:
    """
    ==================== HISTOGRAM ====================
    """
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
    x_bins = bin_edges[:-1]
    
    """
    ==================== PERCENTILES ====================
    """
    # save fig later
    # fig.savefig(f'figures/{title}.png')
    thresh = np.percentile(image, 5.)
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)
    
    """
    ==================== PLOTTING ====================
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    fig.suptitle(title)
    
    # original image
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # histogram
    ax[1].plot(x_bins, hist, color='blue', label='Hist')
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Gray Value")
    ax[1].legend()
    
    # thresholded image
    ax[2].imshow(binary, cmap="gray")
    ax[2].axis('off')
    
    plt.tight_layout()

def setup_plot(image_name : str) -> None:
    image = cv2.imread(f'images/{image_name}', cv2.IMREAD_COLOR)
    assert image is not None, f"Image '{image_name}' could not be loaded."

    # Crop image
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    hs = min(h, w) // 2
    cropped = image[cy - hs:cy + hs, cx - hs:cx + hs] 

    # Convert to float and average channels
    image_float = cropped.astype(np.float32)
    gray_float = np.mean(image_float, axis=2)

    """
    ==================== PREPROCESSING ====================
    """
    gray_float = cv2.GaussianBlur(gray_float, (5, 5), 1.5)
    image_hist(gray_float, f"{image_name[:image_name.find('.')]} Distribution of Gray (Gaussian)")

if __name__ == '__main__':
    assert image_names is not None, "image_names is not instantiated."
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 5, "Please limit the number of images to 5"
    
    for filename in image_names:
        setup_plot(filename)
    
    plt.show()
