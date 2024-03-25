import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import torch

def save_predicted_image(input_image: torch.Tensor, predicted_image: torch.Tensor, target_image: torch.Tensor, current_epoch: int, iteration: int) -> None:
    """
    Save the predicted, target, and input images as a single PNG file.

    Args:
    - input_image (torch.Tensor): The input image tensor.
    - predicted_image (torch.Tensor): The predicted image tensor.
    - target_image (torch.Tensor): The target image tensor.
    - current_epoch (int): The current epoch number.
    - iteration (int): The current iteration number.

    Returns:
    - None
    """
    # Set epoch
    epoch = current_epoch

    # Convert tensors to numpy arrays and remove the first dimension
    input_image = input_image.permute(1, 2, 0).cpu().detach().numpy()
    predicted_image = torch.squeeze(predicted_image)
    predicted_image = predicted_image.permute(1, 2, 0).cpu().detach().numpy()
    target_image = target_image.permute(1, 2, 0).cpu().detach().numpy()

    predicted_image = np.clip(predicted_image, 0, 1)

    min_value = np.min(predicted_image)
    max_value = np.max(predicted_image)

    # Plot the input, predicted, and target images
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Predicted Image")
    plt.imshow(predicted_image)
    plt.subplot(1, 2, 2)
    plt.title("Target Image")
    plt.imshow(target_image)

    output_directory = "/images"

    filename = f'Epoch {epoch} Image {iteration}.png'
    output_path = os.path.join(output_directory, filename)

    plt.savefig(output_path)

    # Close the plot to prevent it from being displayed
    plt.close()
