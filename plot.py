import matplotlib.pyplot as plt
import os
import random

from PIL import Image


from hopfield import process_images

def plot_accuracy_vs_threshold(thresholds, accuracies):
    """
    Plots the accuracy vs threshold.

    Parameters:
    - thresholds (list[int]): List of threshold values.
    - accuracies (list[float]): List of accuracy values corresponding to the thresholds.

    This function plots the accuracy of the restored images against the threshold values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, marker='o')
    plt.title('Accuracy vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()

def plot_accuracy_vs_median_filter_size(filter_sizes, accuracies):
    """
    Plots the accuracy vs median filter size.

    Parameters:
    - filter_sizes (list[int]): List of median filter sizes.
    - accuracies (list[float]): List of accuracy values corresponding to the filter sizes.

    This function plots the accuracy of the restored images against the median filter sizes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(filter_sizes, accuracies, marker='o')
    plt.title('Accuracy vs Median Filter Size')
    plt.xlabel('Median Filter Size')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()

def plot_restored_images(original_images, noisy_images, restored_images, titles, size, noise_type):
    """
    Plots the original, noisy, and restored images.

    Parameters:
    - original_images (list[PIL.Image]): List of original images.
    - noisy_images (list[PIL.Image]): List of noisy images.
    - restored_images (list[PIL.Image]): List of restored images.
    - titles (list[str]): List of titles for the images.
    - size (int): The size of the images.
    - noise_type (str): The type of noise applied to the images.

    This function plots the original, noisy, and restored images in a grid format with appropriate titles.
    """
    num_images = len(original_images)
    if len(titles) < num_images:
        titles += [f'Image {i + 1}' for i in range(len(titles), num_images)]

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))

    # Set figure title with size and noise type
    fig.suptitle(f'Size: {size}x{size}, Noise Type: {noise_type}', fontsize=16)

    # Set column titles
    axes[0, 0].set_title('Original', fontsize=14)
    axes[0, 1].set_title('Noisy', fontsize=14)
    axes[0, 2].set_title('Restored', fontsize=14)

    for i in range(num_images):
        axes[i, 0].imshow(original_images[i], cmap='gray')
        axes[i, 0].set_ylabel(titles[i], fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy_images[i], cmap='gray')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(restored_images[i], cmap='gray')
        axes[i, 2].axis('off')

    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust layout to make space for the figure title
    plt.show()

def plot_restored_images_one_number(digit, noise_type, sizes, titles=None):
    """
    Plots the original, noisy, and restored images for a specific digit and noise type.

    Parameters:
    - digit (int): The digit to plot.
    - noise_type (str): The type of noise applied to the images.
    - sizes (list[int]): List of image sizes to plot.
    - titles (list[str]): List of titles for the images (optional).

    This function plots the original, noisy, and restored images for a specific digit and noise type.
    """
    if titles is None:
        titles = [f'Image {i+1}' for i in range(len(sizes))]

    fig, axes = plt.subplots(len(sizes), 3, figsize=(11, 3 * len(sizes)))

    for i, size in enumerate(sizes):
        original_folder = os.path.join('train', str(size))
        noisy_folder = os.path.join('test', str(size), noise_type)
        restored_folder = os.path.join('outputs', str(size), noise_type)

        if not os.path.exists(original_folder):
            print(f"No original folder path found for size {size}.")
            return None
        if not os.path.exists(noisy_folder):
            print(f"No noisy folder path found for size {size}.")
            return None
        if not os.path.exists(restored_folder):
            print(f"No restored folder path found for size {size}.")
            return None

        original_files = [f for f in os.listdir(original_folder) if f.endswith('.bmp') and f'image_{digit}' in f]
        noisy_files = [f for f in os.listdir(noisy_folder) if f.endswith('.bmp') and f'image_{digit}' in f]
        restored_files = [f for f in os.listdir(restored_folder) if f.endswith('.jpeg') and f'image_{digit}' in f]

        if len(original_files) == 0 or len(noisy_files) == 0 or len(restored_files) == 0:
            print(f"No images found for digit {digit} and size {size} with noise type {noise_type}.")
            continue

        original_image = Image.open(os.path.join(original_folder, original_files[0]))
        noisy_image = Image.open(os.path.join(noisy_folder, noisy_files[0]))
        restored_image = Image.open(os.path.join(restored_folder, restored_files[0]))

        axes[i, 0].imshow(original_image, cmap='gray')
        axes[i, 0].set_title(f'{titles[i]} (Original {size}x{size})')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy_image, cmap='gray')
        axes[i, 1].set_title(f'{titles[i]} (Noisy {size}x{size})')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(restored_image, cmap='gray')
        axes[i, 2].set_title(f'{titles[i]} (Restored {size}x{size})')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def plot_restored_images_all(sizes, noise_types):
    """
    Plots all restored images for all sizes and noise types.

    Parameters:
    - sizes (list[int]): List of image sizes to plot.
    - noise_types (list[str]): List of noise types to plot.

    This function plots the original, noisy, and restored images for all specified sizes and noise types.
    """
    for size in sizes:
        for noise_type in noise_types:
            original_folder = os.path.join('train', str(size))
            noisy_folder = os.path.join('test', str(size), noise_type)
            restored_folder = os.path.join('outputs', str(size), noise_type)

            original_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.bmp')])
            noisy_files = sorted([f for f in os.listdir(noisy_folder) if f.endswith('.bmp')])
            restored_files = sorted([f for f in os.listdir(restored_folder) if f.endswith('.jpeg')])

            original_images = [Image.open(os.path.join(original_folder, f)) for f in original_files]
            noisy_images = [Image.open(os.path.join(noisy_folder, f)) for f in noisy_files]
            restored_images = [Image.open(os.path.join(restored_folder, f)) for f in restored_files]

            titles = [f'Image {i+1}' for i in range(len(original_images))]
            plot_restored_images(original_images, noisy_images, restored_images, titles, size, noise_type)

def plot_accuracy_vs_threshold_for_parameters(size_img, network_size, noise_types, max_iter, threshold_values, current_path='.', apply_median_filter=False, median_filter_size=3):
    """
    Plots the accuracy vs threshold for given parameters.

    Parameters:
    - size_img (int): The size of the images to process.
    - network_size (tuple[int, int]): The size of the Hopfield network.
    - noise (list[str]): List of noise types to process.
    - time_value (int): The maximum number of iterations for the Hopfield network.
    - threshold_values (list[int]): List of threshold values to test.
    - current_path (str): The current working directory.
    - apply_median_filter (bool): Whether to apply a median filter.
    - median_filter_size (int): The size of the median filter to apply.

    This function processes images using the Hopfield network for different threshold values and plots the accuracy vs threshold.
    """
    accuracies_threshold = []
    for threshold in threshold_values:
        accuracy = process_images(size_img, network_size, noise_types, max_iter, threshold, 3, current_path,
                                  apply_median_filter=apply_median_filter, median_filter_size=median_filter_size)
        accuracies_threshold.append(accuracy)
        if accuracy is None:
            return None

    if accuracies_threshold:
        plot_accuracy_vs_threshold(threshold_values, accuracies_threshold)

def plot_accuracy_vs_median_filter_size_for_parameters(size_img, network_size, noise_types, max_iter, median_filter_values, current_path='.', apply_median_filter=True, threshold=60):
    """
    Plots the accuracy vs median filter size for given parameters.

    Parameters:
    - size_img (int): The size of the images to process.
    - network_size (tuple[int, int]): The size of the Hopfield network.
    - noise (list[str]): List of noise types to process.
    - time_value (int): The maximum number of iterations for the Hopfield network.
    - median_filter_size_values (list[int]): List of median filter sizes to test.
    - current_path (str): The current working directory.
    - apply_median_filter (bool): Whether to apply a median filter.
    - threshold (int): The threshold value for image binarization.

    This function processes images using the Hopfield network for different median filter sizes and plots the accuracy vs median filter size.
    """
    accuracies_filter = []
    for filter_size in median_filter_values:
        accuracy = process_images(size_img, network_size, noise_types, max_iter, threshold, 3, current_path,
                                  apply_median_filter=apply_median_filter, median_filter_size=filter_size)
        accuracies_filter.append(accuracy)
        if accuracy is None:
            return None

    if accuracies_filter:
        plot_accuracy_vs_median_filter_size(median_filter_values, accuracies_filter)


#       Uncomment this code to plot the dependencies of accuracy on the threshold value and the size
#       of the median filter window, as well as to display the restoration results for a random digit

#       Do not run until the run() function is executed.

# img_size = 64
# net_size = (100, 100)
# noise = ['impulse']
# time_value = 100
# thresholds = [50, 60, 70, 80, 90]
# median_filter_size_values = [1, 3, 5, 7, 9]
#
# plot_accuracy_vs_threshold_for_parameters(img_size, net_size, noise, time_value, thresholds)
# plot_accuracy_vs_median_filter_size_for_parameters(img_size, net_size, noise, time_value, median_filter_size_values)
#
# img_sizes = [16, 32, 64]
# random_digit = random.randint(0, 9)
#
# plot_restored_images_one_number(random_digit, 'gaussian', img_sizes)
# plot_restored_images_one_number(random_digit, 'impulse', img_sizes)