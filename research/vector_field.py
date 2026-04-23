import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk

image_names = [
    'faint_pit.tiff',
    'big-pit1.tiff'
]

def setup_plot(image_name : str, fig_num : int) -> None:
    image = cv2.imread(f'images/{image_name}', cv2.IMREAD_COLOR)
    assert image is not None, f"Image '{image_name}' could not be loaded."
    
    # crop image
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    hs = min(h, w) // 2
    cropped = image[cy - hs:cy + hs, cx - hs:cx + hs] # cropped to min(h, w)^2, keeping center the same
    
    # convert to grayscale
    image_float = cropped.astype(np.float32)
    gray : cv2.typing.MatLike = np.mean(image_float, axis=2)
    
    # apply circular median filter
    footprint = disk(radius=5)
    gray = median(gray, footprint)
    # gray = cv2.GaussianBlur(gray, (0, 0), 9.5)
    
    smooth_k = np.array([1, 2, 1], dtype=np.float32) # smoothing kernel
    diff_k = np.array([-1, 0, 1], dtype=np.float32) # Sobel operator
    
    # calculate gradients
    gx = cv2.sepFilter2D(gray, cv2.CV_64F, diff_k, smooth_k) # diff in x, smooth in y
    gy = cv2.sepFilter2D(gray, cv2.CV_64F, smooth_k, diff_k) # smooth in x, diff in y
    
    step = 10
    y, x = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    x_ds = x[::step, ::step]
    y_ds = y[::step, ::step]
    u = gx[::step, ::step]
    v = gy[::step, ::step]
    
    # normalize vectors
    """
    magnitude = np.hypot(u, v)
    magnitude[magnitude == 0] = 1.0
    u_norm = u / magnitude
    v_norm = v / magnitude
    """
    
    # plot vector field
    plt.figure(fig_num, figsize=(10, 10))
    plt.imshow(gray, cmap='gray', origin='upper')
    plt.quiver(x_ds, y_ds, u, -v, color='red', angles='xy', scale_units='xy', scale=10)
    plt.title(f'{image_name[:image_name.find('.')]} Gradient Vector Field')
    plt.axis('off')
    

if __name__ == "__main__":
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 5, "Please limit the number of images to 5"
    
    for i in range(len(image_names)):
        setup_plot(image_names[i], i)
    
    plt.show()
