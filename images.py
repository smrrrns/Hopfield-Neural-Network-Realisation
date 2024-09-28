import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from scipy.ndimage import gaussian_filter
from scipy.stats import uniform
import os

# Constants
NOISE_PARAMS = {
    16: {'gaussian_sigma': 0.5, 'impulse_prob': 0.02},
    32: {'gaussian_sigma': 0.7, 'impulse_prob': 0.03},
    64: {'gaussian_sigma': 1.0, 'impulse_prob': 0.05}
}

def create_folders(sizes, noise_types):
    """
    Creates the necessary folders to store images.

    Parameters:
    - sizes (list[int]): List of image sizes to create folders for.
    - noise_types (list[str]): List of noise types to create folders for.

    This function creates the required directories to store the original, resized, and noisy images.
    """
    original_folder_path = os.path.join('train', 'original')
    if not os.path.exists(original_folder_path):
        os.makedirs(original_folder_path)

    for size in sizes:
        train_folder_path = os.path.join('train', str(size))
        test_folder_path = os.path.join('test', str(size))

        if not os.path.exists(train_folder_path):
            os.makedirs(train_folder_path)

        if not os.path.exists(test_folder_path):
            os.makedirs(test_folder_path)

        for noise_type in noise_types:
            noise_folder_path = os.path.join(test_folder_path, noise_type)
            if not os.path.exists(noise_folder_path):
                os.makedirs(noise_folder_path)

def save_original_image(img, label, i, original_folder_path):
    """
    Saves the original image.

    Parameters:
    - img (PIL.Image): The original image to save.
    - label (int): The label of the image.
    - i (int): The index of the image.
    - original_folder_path (str): The path to the folder where the original images are saved.

    This function saves the original image in the specified folder.
    """
    bmp_path = os.path.join(original_folder_path, f'image_{label}_{i}.bmp')
    img.save(bmp_path)

def save_resized_image(img, label, i, size, train_folder_path):
    """
    Saves the resized image.

    Parameters:
    - img (PIL.Image): The original image to resize.
    - label (int): The label of the image.
    - i (int): The index of the image.
    - size (int): The size to resize the image to.
    - train_folder_path (str): The path to the folder where the resized images are saved.

    This function resizes the original image to the specified size and saves it in the specified folder.
    """
    resized_img = img.resize((size, size), Image.Resampling.LANCZOS)
    resized_path = os.path.join(train_folder_path, f'image_{label}_{i}_{size}.bmp')
    resized_img.save(resized_path)
    return resized_img

def add_gaussian_noise(img_array, size):
    """
    Adds Gaussian noise to the image.

    Parameters:
    - img_array (numpy.ndarray): The image array to add noise to.
    - size (int): The size of the image.

    This function applies Gaussian noise to the image array and returns the noisy image.
    """
    gaussian_sigma = NOISE_PARAMS[size]['gaussian_sigma']
    gaussian_noisy_img = gaussian_filter(img_array, sigma=gaussian_sigma)
    return Image.fromarray(gaussian_noisy_img)

def add_impulse_noise(img_array, size):
    """
    Adds impulse noise to the image.

    Parameters:
    - img_array (numpy.ndarray): The image array to add noise to.
    - size (int): The size of the image.

    This function applies impulse noise to the image array and returns the noisy image.
    """
    impulse_prob = NOISE_PARAMS[size]['impulse_prob']
    impulse_noisy_img = img_array.copy()
    impulse_mask = uniform.rvs(size=(size, size)) < impulse_prob
    impulse_noisy_img[impulse_mask] = 255
    return Image.fromarray(impulse_noisy_img)

def save_noisy_images(resized_img_array, label, i, size, noise_types, test_folder_path):
    """
    Saves the noisy images.

    Parameters:
    - resized_img_array (numpy.ndarray): The resized image array.
    - label (int): The label of the image.
    - i (int): The index of the image.
    - size (int): The size of the image.
    - noise_types (list[str]): List of noise types to apply.
    - test_folder_path (str): The path to the folder where the noisy images are saved.

    This function applies the specified noise types to the resized image and saves the noisy images in the specified folder.
    """
    for noise_type in noise_types:
        if noise_type == 'gaussian':
            noisy_img = add_gaussian_noise(resized_img_array, size)
            noisy_path = os.path.join(test_folder_path, 'gaussian', f'image_{label}_{i}_{size}_gaussian.bmp')
        elif noise_type == 'impulse':
            noisy_img = add_impulse_noise(resized_img_array, size)
            noisy_path = os.path.join(test_folder_path, 'impulse', f'image_{label}_{i}_{size}_impulse.bmp')
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        noisy_img.save(noisy_path)

def load_images(N, sizes, noise_types):
    """
    Processes and saves the images.

    Parameters:
    - N (int): Number of images to load per digit.
    - sizes (list[int]): List of image sizes to resize the images to.
    - noise_types (list[str]): List of noise types to apply to the images.

    This function loads images from the MNIST dataset, resizes them to the specified sizes,
    and applies the specified noise types. The processed images are saved in the appropriate
    directories for training and testing.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    indices_by_label = {i: [] for i in range(10)}

    for idx, label in enumerate(y_train):
        indices_by_label[label].append(idx)

    random_indices = []
    for label in range(10):
        random_indices.extend(np.random.choice(indices_by_label[label], N, replace=False))

    create_folders(sizes, noise_types)

    original_folder_path = os.path.join('train', 'original')

    for i, idx in enumerate(random_indices):
        img = x_train[idx]
        label = y_train[idx]
        img = Image.fromarray(img)

        save_original_image(img, label, i, original_folder_path)

        for size in sizes:
            train_folder_path = os.path.join('train', str(size))
            test_folder_path = os.path.join('test', str(size))

            resized_img = save_resized_image(img, label, i, size, train_folder_path)
            resized_img_array = np.array(resized_img)

            save_noisy_images(resized_img_array, label, i, size, noise_types, test_folder_path)

    print(f"Saved {N * 10} images in train/16, train/32, train/64 folders.")
    print(f"Saved noisy images in test/16, test/32, test/64 folders with different noise types.")
