import cv2
import numpy as np
import matplotlib.pyplot as plt

image_names = [
    # 'large_pit_1.tiff',
    # 'large_pit_2.tiff',
    # 'large_pit_3.tiff',
    # 'large_pit_4.tiff',
    # 'scratch_1.tiff',
    # 'scratch_2.tiff',
    # 'scratch_3.tiff',
    # 'scratch_4.tiff',
    # 'big-pit1.tiff',
    # 'faint_pit.tiff'
    # 'BlackIron_OMS_corrosion-crack.tif'
    'cc-8.tif',
]

def image_hist(image : cv2.typing.MatLike, title: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(image, cmap="gray")
    ax[1].hist(image.ravel(), bins=256, range=(0., 256.))
    fig.suptitle(title)
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Gray Value")
    fig.savefig(f'figures/{title}.png')

def setup_plot(image_name : str) -> None:
    image = cv2.imread(f'images/{image_name}', cv2.IMREAD_COLOR)

    assert image is not None, f"Image '{image_name}' could not be loaded. Check the file path."

    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    half_size = min(h, w) // 2
    cropped = image[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    """
    ==================== SIMPLE AVERAGE FOR GRAYSCALE ====================
    """
    image_float = cropped.astype(np.float32)
    gray_float = np.mean(image_float, axis=2)
    gray : cv2.typing.MatLike = gray_float.astype(np.uint8) # final image, equal-weighted average

    ext = image_name.find('.')

    # unprocessed
    image_hist(gray, f"{image_name[:ext]} Distribution of Gray (Unprocessed)")

    gaussian_gray = cv2.GaussianBlur(gray, (0, 0), 9.5)

    # gaussian
    image_hist(gaussian_gray, f"{image_name[:ext]} Distribution of Gray (Gaussian)")


if __name__ == "__main__":
    
    assert len(image_names) > 0, "No image names were provided"
    assert len(image_names) <= 5, "Please limit the number of images to 5"
    
    for filename in image_names:
        setup_plot(filename)
    plt.tight_layout()
    plt.show()
