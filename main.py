"""
Machine Learning Model Training Script

This script is responsible for training a machine learning model for image generation.
It utilizes a Generative Adversarial Network (GAN) architecture with a generator and discriminator.

Requirements:
- PyTorch
- models.py: Contains the Generator and Discriminator classes
- image_loader.py: Contains functions for loading images
- plot_images.py: Contains a function for saving predicted images

Usage:
1. Ensure all required modules are installed.
2. Provide the path to the dataset directory in the 'get_image_pointers' function.
3. Run the script to start model training.

"""

import torch
from torch import nn
import os

from models import Generator, Discriminator
from image_loader import get_image_pointers, load_image
from plot_images import save_predicted_image
import torch.optim.lr_scheduler as lr_scheduler


def train(pointers, generator, discriminator) -> tuple:
    """
    Train the generator and discriminator models.

    Args:
    - pointers (list): List of image file pointers.
    - generator (torch.nn.Module): The generator model.
    - discriminator (torch.nn.Module): The discriminator model.

    Returns:
    Tuple of trained generator and discriminator models.
    """
  
    epochs = 7

    torch.manual_seed(42)

    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=0.0000001, betas=(0.5, 0.999))
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=0.0000001, betas=(0.5, 0.999))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(epochs):

        # Set models to training mode
        generator.train()
        discriminator.train()

        for pointer in pointers:
            # Load input and target images
            batch_x, batch_y = load_image(pointer)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass through generator
            forward_pass = generator(torch.unsqueeze(batch_x, 0))

            # Calculate discriminator's output for real and fake images
            discriminator_pass_real = discriminator(torch.unsqueeze(batch_x, 0), torch.unsqueeze(batch_y, 0))
            discriminator_pass_fake = discriminator(torch.unsqueeze(batch_x, 0), forward_pass.detach())

            # Calculate discriminator's loss
            discriminator_real_loss = bce_loss(discriminator_pass_real, torch.ones_like(discriminator_pass_real))
            discriminator_fake_loss = bce_loss(discriminator_pass_fake, torch.zeros_like(discriminator_pass_fake))
            discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2

            # Update discriminator parameters
            discriminator.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Re-calculate discriminator's output for fake images
            discriminator_pass_fake = discriminator(torch.unsqueeze(batch_x, 0), forward_pass)

            # Calculate generator's loss
            generator_fake_loss = bce_loss(discriminator_pass_fake, torch.ones_like(discriminator_pass_fake))
            mae_loss = l1_loss(forward_pass, torch.unsqueeze(batch_y, 0)) * 100
            generator_loss = generator_fake_loss + mae_loss

            # Print losses and accuracy
            print(f'\033[91m Generator loss: {generator_loss} \033[0m')
            print(f'\033[91m Discriminator loss: {discriminator_loss} \033[0m')
            print(f'\033[91m Accuracy: {100 - mae_loss}% \033[0m')
            print("\033[91m ----------------------------------------------------------- \033[0m")

            # Save generator's loss to file
            if (pointers.index(pointer) + 1) % 10 == 0:
                with open('loss.txt', 'a') as file:
                    file.write(f'{mae_loss}\n')

            # Update generator parameters
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            # Save predicted images at certain intervals
            if (pointers.index(pointer) + 1) % 33 == 0:
                save_predicted_image(batch_x, forward_pass, batch_y, epoch, pointers.index(pointer))

        # Save models' states at the end of each epoch
        torch.save(generator.state_dict(), 'initial_model_checkpoint.pth')
        torch.save(discriminator.state_dict(), 'initial_discriminator_model_checkpoint.pth')

    # Set models to evaluation mode
    generator.eval()
    discriminator.eval()

    return generator, discriminator


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"\033[1m Training on {device} \033[0m")

    checkpoint_path = 'initial_model_checkpoint.pth'
    discriminator_checkpoint_path = 'initial_discriminator_model_checkpoint.pth'

    pointers = get_image_pointers("path_to_dataset")
    print(f"Loaded {len(pointers)} image pointers")

    generator = Generator().to(device)

    discriminator = Discriminator(3, [64, 128, 256, 512]).to(device)

    # Load models from existing checkpoints if available
    if os.path.exists(checkpoint_path):
        if os.path.exists(discriminator_checkpoint_path):
            discriminator.load_state_dict(torch.load(discriminator_checkpoint_path))
            generator.load_state_dict(torch.load(checkpoint_path))

            print('\033[93m' + "Loaded existing model checkpoint")
        else:
            generator.load_state_dict(torch.load(checkpoint_path))
            print('\033[93m' + 'Only found generator')

    else:
        if os.path.exists(discriminator_checkpoint_path):
            discriminator.load_state_dict(torch.load(discriminator_checkpoint_path))
            print('\033[93m' + "Trained a new generator model")

        else:
            print('\033[93m' + 'Training two new models')

    # Train the models
    trained_generator, trained_discriminator = train(pointers, generator, discriminator)

    # Save trained models' states
    torch.save(trained_generator.state_dict(), 'initial_model_checkpoint.pth')
    torch.save(trained_discriminator.state_dict(), 'initial_discriminator_model_checkpoint.pth')
