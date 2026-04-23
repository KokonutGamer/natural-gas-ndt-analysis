import cv2
import numpy as np
import matplotlib.pyplot as plt

image_names = [
    'faint_pit.tiff',
    'big-pit1.tiff',
    '1018Steel_Section_A1_Raw.tiff'
]

def fft_1d_axes(image_name : str) -> None:
    image = cv2.imread(f'images/{image_name}', cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"{image_name} could not be found."
    
    # ---------------------------------------------------------
    # 1. FFT along the X-axis ONLY (rows)
    # ---------------------------------------------------------
    f_x = np.fft.fft(image, axis=1) 
    # Shift the zero-frequency component to the center of the X-axis
    fshift_x = np.fft.fftshift(f_x, axes=1) 
    # Added 1e-8 to avoid log(0) warnings if a row is perfectly flat
    magnitude_spectrum_x = 20 * np.log(np.abs(fshift_x) + 1e-8)
    
    # ---------------------------------------------------------
    # 2. FFT along the Y-axis ONLY (columns)
    # ---------------------------------------------------------
    f_y = np.fft.fft(image, axis=0)
    # Shift the zero-frequency component to the center of the Y-axis
    fshift_y = np.fft.fftshift(f_y, axes=0)
    magnitude_spectrum_y = 20 * np.log(np.abs(fshift_y) + 1e-8)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f"{image_name} 1D Directional FFTs")
    
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original Image")
    
    # The X-axis FFT will show horizontal frequencies. The bright 
    # vertical line in the center represents the DC (average) component of each row.
    ax[1].imshow(magnitude_spectrum_x, cmap="gray")
    ax[1].set_title("FFT along X-axis (Rows)")
    
    # The Y-axis FFT will show vertical frequencies. The bright
    # horizontal line in the center represents the DC component of each column.
    ax[2].imshow(magnitude_spectrum_y, cmap="gray")
    ax[2].set_title("FFT along Y-axis (Columns)")

if __name__ == "__main__":
    for filename in image_names:
        fft_1d_axes(filename)
    
    plt.tight_layout()
    plt.show()
