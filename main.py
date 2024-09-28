from images import load_images
from hopfield import process_images
from plot import plot_restored_images_all

optimal_parameters = {
    (16, 'gaussian'): {'threshold': 70, 'theta': 2, 'median_filter_size': 3},
    (16, 'impulse'): {'threshold': 50, 'theta': 3, 'median_filter_size': 3},
    (32, 'gaussian'): {'threshold': 80, 'theta': 4, 'median_filter_size': 3},
    (32, 'impulse'): {'threshold': 60, 'theta': 3, 'median_filter_size': 3},
    (64, 'gaussian'): {'threshold': 75, 'theta': 2, 'median_filter_size': 3},
    (64, 'impulse'): {'threshold': 60, 'theta': 3, 'median_filter_size': 3}
}

def run(n, sizes, network_size, max_iterations, noise_types, optimal_params, model_save):
    """
    Runs the entire image processing pipeline.

    Parameters:
    - n (int): Number of images to load per digit.
    - sizes (list[int]): List of image sizes to resize the images to.
    - max_iterations (int): Maximum number of iterations for the Hopfield network.
    - noise_types (list[str]): List of noise types to apply to the images.
    - optimal_params (dict): Dictionary containing optimal parameters for each size and noise type.

    This function orchestrates the entire image processing pipeline. It loads the images, processes
    them using the Hopfield network with the specified optimal parameters, and plots the restored
    images for all sizes and noise types.
    """
    load_images(n, sizes, noise_types)

    for size in sizes:
        for noise_type in noise_types:
            params = optimal_params[(size, noise_type)]
            process_images(
                image_size=size,
                network_size=network_size,
                noise_types=[noise_type],
                max_iterations=max_iterations,
                threshold=params['threshold'],
                theta=params['theta'],
                current_path='.',
                model_filename=f'hopfield_model_{size}_{size}_{params["threshold"]}_{params["theta"]}_{noise_type}_{params["median_filter_size"]}.pkl',
                apply_median_filter=True if noise_type == 'impulse' else False,
                median_filter_size=params['median_filter_size'],
                model_save=model_save
            )

    plot_restored_images_all(sizes, noise_types)

# Run this function to start the process of downloading and processing images on optimal parameters.
run(1, [16, 32, 64], (100, 100), 100, ['gaussian', 'impulse'], optimal_parameters, model_save=True)