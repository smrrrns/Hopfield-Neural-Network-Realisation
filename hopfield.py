import os
import numpy as np
import random
import joblib

from PIL import Image
from itertools import product
from scipy.ndimage import median_filter

def save_model(model, filename):
    """
    Saves the trained model to a file.

    Parameters:
    - model: The trained model to save.
    - filename (str): The filename to save the model to.

    This function saves the trained model to the specified file using joblib.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Loads the trained model from a file.

    Parameters:
    - filename (str): The filename to load the model from.

    Returns:
    - model: The loaded model.

    This function loads the trained model from the specified file using joblib.
    """
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def read_image_to_array(image_file, image_size, threshold, apply_median_filter=False, median_filter_size=3):
    """
    Reads an image, converts it to a binary array, and optionally applies a median filter.

    Parameters:
    - image_file (str): The path to the image file.
    - image_size (tuple[int, int]): The size to resize the image to.
    - threshold (int): The threshold value for image binarization.
    - apply_median_filter (bool): Whether to apply a median filter.
    - median_filter_size (int): The size of the median filter to apply.

    Returns:
    - binary_image (numpy.ndarray): The binary image array.

    This function reads an image, converts it to grayscale, optionally applies a median filter,
    resizes it to the specified size, and converts it to a binary array.
    """
    image = Image.open(image_file).convert('L')

    if apply_median_filter:
        image = np.array(image)
        image = median_filter(image, size=median_filter_size)
        image = Image.fromarray(image)

    image = image.resize(image_size)
    image_array = np.asarray(image, dtype=np.uint8)

    binary_image = np.zeros(image_array.shape, dtype=np.float64)
    binary_image[image_array > threshold] = 1
    binary_image[binary_image == 0] = -1

    return binary_image

def create_weight_matrix(flattened_image_vectors):
    """
    Creates a weight matrix from a list of flattened image vectors.

    Parameters:
    - flattened_image_vectors (list[numpy.ndarray]): List of flattened image vectors.

    Returns:
    - weight_matrix (numpy.ndarray): The weight matrix.

    This function creates a weight matrix for the Hopfield network from a list of flattened image vectors.
    """
    num_pixels = len(flattened_image_vectors[0])
    num_images = len(flattened_image_vectors)
    weight_matrix = np.zeros((num_pixels, num_pixels))

    for vector in flattened_image_vectors:
        weight_matrix += np.outer(vector, vector)

    np.fill_diagonal(weight_matrix, 0)
    weight_matrix /= num_images
    return weight_matrix

def convert_array_to_image(binary_array, output_file=None):
    """
    Converts a binary array back to an image and optionally saves it.

    Parameters:
    - binary_array (numpy.ndarray): The binary image array.
    - output_file (str): The filename to save the image to (optional).

    Returns:
    - image (PIL.Image): The converted image.

    This function converts a binary array back to an image and optionally saves it to the specified file.
    """
    image_array = np.zeros(binary_array.shape, dtype=np.uint8)
    image_array[binary_array == 1] = 255
    image_array[binary_array == -1] = 0
    image = Image.fromarray(image_array, mode="L")
    if output_file is not None:
        image.save(output_file)
    return image

def update_network_state(weight_matrix, state_vector, theta=0.5, max_iterations=100, stability_threshold=10):
    """
    Updates the state of the Hopfield network.

    Parameters:
    - weight_matrix (numpy.ndarray): The weight matrix of the Hopfield network.
    - state_vector (numpy.ndarray): The current state vector of the network.
    - theta (float): The threshold value for the network activation.
    - max_iterations (int): The maximum number of iterations to run.
    - stability_threshold (int): The number of iterations without change to consider the network stable.

    Returns:
    - state_vector (numpy.ndarray): The updated state vector of the network.

    This function updates the state of the Hopfield network until it reaches stability or the maximum number of iterations.
    """
    stable_count = 0
    for iteration in range(max_iterations):
        random_index = random.randint(0, len(state_vector) - 1)
        activation = np.dot(weight_matrix[random_index], state_vector) - theta
        new_state = 1 if activation > 0 else -1

        if state_vector[random_index] == new_state:
            stable_count += 1
        else:
            stable_count = 0
            state_vector[random_index] = new_state

        if stable_count >= stability_threshold:
            print(f"Network reached stability after {iteration} iterations.")
            break

    return state_vector

def train_hopfield_network(training_files, image_size=(100, 100), threshold=60):
    """
    Trains the Hopfield network.

    Parameters:
    - training_files (list[str]): List of paths to the training images.
    - image_size (tuple[int, int]): The size to resize the images to.
    - threshold (int): The threshold value for image binarization.

    Returns:
    - weight_matrix (numpy.ndarray): The trained weight matrix.

    This function trains the Hopfield network by creating a weight matrix from the training images.
    """
    print("Importing images and creating weight matrix....")

    flattened_image_vectors = [read_image_to_array(path, image_size, threshold).flatten() for path in training_files]
    weight_matrix = create_weight_matrix(flattened_image_vectors)
    print("Weight matrix is done!!")
    return weight_matrix

def restore_hopfield_network(weight_matrix, test_files, image_size, theta=0.5, max_iterations=100, threshold=60,
                             current_path=None, apply_median_filter=False, median_filter_size=3,
                             stability_threshold=10):
    """
    Restores images using the trained Hopfield network.

    Parameters:
    - weight_matrix (numpy.ndarray): The weight matrix of the Hopfield network.
    - test_files (list[str]): List of paths to the test images.
    - image_size (tuple[int, int]): The size to resize the images to.
    - theta (float): The threshold value for the network activation.
    - max_iterations (int): The maximum number of iterations to run.
    - threshold (int): The threshold value for image binarization.
    - current_path (str): The current working directory.
    - apply_median_filter (bool): Whether to apply a median filter.
    - median_filter_size (int): The size of the median filter to apply.
    - stability_threshold (int): The number of iterations without change to consider the network stable.

    This function restores images using the trained Hopfield network and saves the restored images.
    """
    counter = 0
    for file_path in test_files:
        binary_image = read_image_to_array(file_path, image_size, threshold, apply_median_filter, median_filter_size)
        original_shape = binary_image.shape
        state_vector = binary_image.flatten()
        updated_state_vector = update_network_state(weight_matrix, state_vector, theta, max_iterations,
                                                    stability_threshold)
        updated_state_vector = updated_state_vector.reshape(original_shape)

        size_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        noise_type = os.path.basename(os.path.dirname(file_path))
        output_folder = os.path.join(current_path, 'outputs', size_folder, noise_type)
        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(str(output_folder), f'image_{counter}.jpeg')
        convert_array_to_image(updated_state_vector, output_file=output_file)
        counter += 1

def calculate_restoration_accuracy(original_folder, restored_folder):
    """
    Calculates the accuracy of restored images compared to original images.

    Parameters:
    - original_folder (str): The path to the folder containing the original images.
    - restored_folder (str): The path to the folder containing the restored images.

    Returns:
    - average_accuracy (float): The average accuracy of the restored images.

    This function calculates the accuracy of the restored images by comparing them to the original images.
    """
    original_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.bmp')])
    restored_files = sorted([f for f in os.listdir(restored_folder) if f.endswith('.jpeg')])

    total_accuracy = 0
    for orig_file, rest_file in zip(original_files, restored_files):
        orig_path = os.path.join(original_folder, orig_file)
        rest_path = os.path.join(restored_folder, rest_file)

        orig_image = read_image_to_array(orig_path, image_size=(100, 100), threshold=60)
        rest_image = read_image_to_array(rest_path, image_size=(100, 100), threshold=60)

        accuracy = np.sum(orig_image == rest_image) / orig_image.size * 100
        total_accuracy += accuracy

    average_accuracy = total_accuracy / len(original_files)
    return average_accuracy

def process_images(image_size, network_size=(100, 100), noise_types=('gaussian', 'impulse'), max_iterations=100,
                   threshold=60, theta=0.5, current_path='.',
                   model_filename='hopfield_model.pkl', apply_median_filter=False, median_filter_size=3, model_save=False):
    """
    Processes images using the Hopfield network.

    Parameters:
    - image_size (int): The size of the images to process.
    - network_size (tuple[int, int]): The size of the Hopfield network.
    - noise_types (list[str]): List of noise types to process.
    - max_iterations (int): The maximum number of iterations for the Hopfield network.
    - threshold (int): The threshold value for image binarization.
    - theta (float): The threshold value for the network activation.
    - current_path (str): The current working directory.
    - model_filename (str): The filename to save or load the Hopfield model.
    - apply_median_filter (bool): Whether to apply a median filter.
    - median_filter_size (int): The size of the median filter to apply.
    - model_save (bool): Whether to save the Hopfield model.

    Returns:
    - accuracy (float): The accuracy of the restored images.

    This function processes images using the Hopfield network, restores them, and calculates the accuracy of the restored images.
    """
    train_folder_path = os.path.join('train', str(image_size))
    test_folder_path = os.path.join('test', str(image_size))

    if not os.path.exists(train_folder_path):
        print(f"No train folder path found for size {image_size}.")
        return None
    if not os.path.exists(test_folder_path):
        print(f"No test folder path found for size {image_size}.")
        return None


    training_files = [os.path.join(train_folder_path, f) for f in os.listdir(train_folder_path) if f.endswith('.bmp')]

    if not training_files:
        print(f"No training images found for size {image_size}.")
        return None

    if os.path.exists(model_filename):
        weight_matrix = load_model(model_filename)
    else:
        weight_matrix = train_hopfield_network(training_files, network_size, threshold)
        if model_save is True:
            save_model(weight_matrix, model_filename)

    accuracy = None
    for noise_type in noise_types:
        noise_folder_path = os.path.join(test_folder_path, noise_type)
        test_files = [os.path.join(str(noise_folder_path), f) for f in os.listdir(str(noise_folder_path)) if
                      f.endswith('.bmp')]

        if not test_files:
            print(f"No test images found for size {image_size} and noise type {noise_type}.")
            continue

        apply_filter = apply_median_filter and noise_type == 'impulse'

        restore_hopfield_network(weight_matrix, test_files, network_size, theta, max_iterations, threshold,
                                 current_path, apply_filter, median_filter_size)

        restored_folder = os.path.join('outputs', str(image_size), noise_type)
        accuracy = calculate_restoration_accuracy(train_folder_path, restored_folder)
        print(f"Accuracy for size {image_size} and noise type {noise_type}: {accuracy:.2f}%")

    return accuracy

def optimize_parameters(image_size, network_size, noise_types, time_value, threshold_values, theta_values,
                        current_path='.',
                        apply_median_filter=False, median_filter_size_values=None):
    """
    Optimizes parameters for the Hopfield network.

    Parameters:
    - image_size (int): The size of the images to process.
    - network_size (tuple[int, int]): The size of the Hopfield network.
    - noise_types (list[str]): List of noise types to process.
    - time_value (int): The time value for the optimization process.
    - threshold_values (list[int]): List of threshold values to test.
    - theta_values (list[float]): List of theta values to test.
    - current_path (str): The current working directory.
    - apply_median_filter (bool): Whether to apply a median filter.
    - median_filter_size_values (list[int]): List of median filter sizes to test.

    Returns:
    - best_params (dict): The best parameters for each noise type.

    This function optimizes the parameters for the Hopfield network by testing different combinations of threshold, theta, and median filter size.
    """
    if median_filter_size_values is None:
        median_filter_size_values = [3]
    best_accuracies = {}
    best_params = {}

    for noise_type in noise_types:
        best_accuracy = 0
        best_param_sets = []

        for threshold, theta, median_filter_size in product(threshold_values, theta_values,
                                                                  median_filter_size_values):
            model_filename = f'hopfield_model_{image_size}_{network_size[0]}_{threshold}_{theta}_{noise_type}_{median_filter_size}.pkl'
            print(
                f"Training with parameters: threshold={threshold}, theta={theta}, noise_type={noise_type}, median_filter_size={median_filter_size}")

            apply_filter = apply_median_filter and noise_type == 'impulse'

            process_images(image_size, network_size, [noise_type], time_value, threshold, theta, current_path, model_filename,
                           apply_filter, median_filter_size)

            restored_folder = os.path.join('outputs', str(image_size), noise_type)
            accuracy = calculate_restoration_accuracy(os.path.join('train', str(image_size)), restored_folder)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param_sets = [(threshold, theta, median_filter_size)]
            elif accuracy == best_accuracy:
                best_param_sets.append((threshold, theta, median_filter_size))

        best_accuracies[noise_type] = best_accuracy
        best_params[noise_type] = best_param_sets

    for noise_type in noise_types:
        print(
            f"Best accuracy for noise type {noise_type}: {best_accuracies[noise_type]:.2f}%")
        for param_set in best_params[noise_type]:
            print(
                f"  Parameters: threshold={param_set[0]}, theta={param_set[1]}, median_filter_size={param_set[2]}")

    return best_params
