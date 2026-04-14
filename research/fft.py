import cv2
import numpy as np
import matplotlib.pyplot as plt

image_names = [
    'faint_pit'    
]

def fft(image_name : str) -> None:
    image = cv2.imread(f'images/{image_name}.tiff', cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"{image_name} could not be found."
    
    np.mean(image)
    np.std(image)
    print(f"SNR = {np.mean(image)} / {np.std(image)} = {np.mean(image) / np.std(image)}")
    
    f = np.fft.fft2(image) # 2D FFT
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    h, w = image.shape
    cy, cx = h // 2, w // 2
    fshift[cy - 30:cy + 31, cx - 30:cx + 31] = 0
    f_ishift = np.fft.ifftshift(fshift)
    hp = np.fft.ifft2(f_ishift)
    hp = np.real(hp)
    
    hp_8 = hp.astype(np.uint8)
    
    print(f"HP max = {np.max(hp_8)}, HP min = {np.min(hp_8)}")
    print(f"HP SNR = {np.mean(hp_8)} / {np.std(hp_8)} = {np.mean(hp_8) / np.std(hp_8)}")
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    fig.suptitle(f"{image_name} FFT")
    ax[0].imshow(image, cmap="gray")
    ax[1].imshow(magnitude_spectrum, cmap="gray")
    ax[2].imshow(hp, cmap="gray")

if __name__ == "__main__":
    for filename in image_names:
        fft(filename)
    
    plt.tight_layout()
    plt.show()
