import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage.morphology import disk

image_names = [
    # 'faint_pit.tiff',
    # 'big-pit1.tiff',
    # 'large_pit_1.tiff',
    # 'large_pit_2.tiff',
    # 'large_pit_3.tiff',
    # 'large_pit_4.tiff',
    # 'scratch_1.tiff',
    # 'scratch_2.tiff',
    # 'scratch_3.tiff',
    # 'scratch_4.tiff',
    # 'faint_1.tiff',
    # 'faint_2.tiff',
    # 'faint_3.tiff',
    # 'faint_4.tiff',
    # 'faint_5.tiff',
    # '1018Steel_Section_A1_Raw.tiff',
    # '1018Steel_Section_A1_largerMP.tiff',
    # '1018Steel_Section_A6_smallerMP.tiff',
    # 'Aluminum_MP_1.tiff',
    # 'BlackIron_OMS_corrosion-crack.tif',
    'cc-8.tif',
]

def plot_3d_surface(image_name : str, fig_num : int) -> None:
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
    
    # Filter the float image
    radius = 5
    circular_footprint = disk(radius)
    gray_filtered = median_filter(gray_float, footprint=circular_footprint)
    gray_blurred = cv2.GaussianBlur(gray_filtered, (0, 0), 9.5)
    
    # --- 3D PLOTTING ---
    # 1. Downsample for performance. A step of 4 to 10 is usually good.
    step = 5  
    z = gray_blurred[::step, ::step]
    
    # 2. Create X and Y grids that match the downsampled Z array
    y_grid, x_grid = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    
    # Multiply by the step size so the axis ticks represent actual pixel coordinates
    x = x_grid * step
    y = y_grid * step
    
    # 3. Set up the 3D figure
    fig = plt.figure(fig_num, figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 4. Plot the surface
    # 'viridis' maps lowest values to dark purple and highest to bright yellow.
    # Replace with cmap='gray' for standard grayscale mapping.
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # 5. Add a color bar to help read the intensities
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Intensity (Float32)')
    
    # Formatting
    ax.set_title(f"{image_name[:image_name.find('.')]} 3D Surface Map")
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Intensity')
    
    # Invert the Y-axis so the coordinate origin (0,0) remains top-left
    ax.invert_yaxis()

if __name__ == "__main__":
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 5, "Please limit the number of images to 5"
    
    for i in range(len(image_names)):
        plot_3d_surface(image_names[i], i)
    
    plt.show()