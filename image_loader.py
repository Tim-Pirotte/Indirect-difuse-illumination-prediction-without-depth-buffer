import os
import random
from PIL import Image
from torchvision import transforms


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print("", end=f'\r{prefix} |{bar}| {percent}% {suffix}')
    # Print New Line on Complete
    if iteration == total:
        print()
      

def get_image_pointers(path: str, limit: int = 100) -> list:
    """
    Retrieve a list of image file paths from the specified directory.

    Args:
    - path (str): The directory path where images are located.
    - limit (int): The maximum number of images to load.

    Returns:
    - list: A list containing image file paths.
    """
    image_pointers = []
    print_progress_bar(0, limit, "Loading images:")
    files = os.listdir(path)
    for root, dirs, files in os.walk(path):
        for filename in files:
            image_path = os.path.join(root, filename)
            image_pointers.append(image_path)
            print_progress_bar(len(image_pointers), limit, "Loading images:")
            if len(image_pointers) >= limit:
                return image_pointers
    return image_pointers


def load_image(image_path: str) -> tuple:
    """
    Load and preprocess an image from the specified file path.

    Args:
    - image_path (str): The file path of the image to load.

    Returns:
    - tuple: A tuple containing left and right image tensors.
    """
    transform = transforms.ToTensor()
    img = Image.open(image_path)
    width, height = img.size

    # Split the image into left and right halves
    left_half = img.crop((0, 0, width // 2, height))
    right_half = img.crop((width // 2, 0, width, height))

    # Randomly select a width for the cropped region
    random_width = random.randrange(960, 1920)
    random_height = float(random_width) * 0.5625

    # Randomly select the position of the cropped region
    left = random.randrange(0, (1920 - random_width))
    top = random.randrange(0, (1080 - int(random_height)))

    # Crop and resize the left half of the image
    left_half.crop((left, 0, random_width + left, int(random_height)))
    left_half = left_half.resize((960, 540), Image.NEAREST)
    right_half = right_half.resize((960, 540), Image.NEAREST)

    # Apply transformations to convert images to tensors
    left_tensor = transform(left_half)
    right_tensor = transform(right_half)

    return left_tensor, right_tensor
