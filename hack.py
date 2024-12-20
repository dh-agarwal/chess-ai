"""Label mapping: {'black bishop': 0, 'black king': 1, 'black knight': 2, 'black pawn': 3, 'black queen': 4, 'black rook': 5, 'none': 6, 'white bishop': 7, 'white king': 8, 'white knight': 9, 'white pawn': 10, 'white queen': 11, 'white rook': 12}"""

import os
from PIL import Image
import numpy as np

def calculate_mean_std(image_dir):
    """Calculate the mean and standard deviation of images in a directory."""
    means = []
    stds = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path).convert('RGB')
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            means.append(image.mean(axis=(0, 1)))
            stds.append(image.std(axis=(0, 1)))

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    
    return mean, std

# Example usage
mean, std = calculate_mean_std('images/')
print(f'Mean: {mean}, Std: {std}')
