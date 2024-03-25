# Indirect-difuse-illumination-prediction-without-depth-buffer

This project aims to generate indirect diffuse illumination based on direct illumination from images created in Blender. It utilizes machine learning techniques to predict indirect lighting effects, enhancing the realism and visual quality of rendered scenes.

## Overview

In computer graphics, indirect diffuse illumination is crucial in simulating realistic lighting conditions within a scene. While direct illumination accounts for light sources directly visible to surfaces, indirect illumination captures the light that bounces off surfaces and contributes to overall scene brightness.

This project focuses on generating medium-quality (960px x 540px) indirect diffuse illumination using machine-learning models trained on images generated in Blender. By leveraging deep learning techniques, the generator network predicts the indirect lighting effects based on the input direct illumination images. It does this without using a depth buffer or normal buffer.

## Key Features

- **Machine Learning Model**: Utilizes a deep learning model to predict indirect diffuse illumination from direct illumination images. This model can be pruned and quantized to optimise its performance for real-time applications.
- **Enhanced Realism**: Enhances the realism of rendered scenes by simulating indirect lighting effects accurately.
- **Seamless integration**: When trained on a robust dataset. The model can be used on existing games or old movies without the need for the developer to add the system to the application. 

## How It Works

The process involves training a machine learning model using pairs of direct illumination images and corresponding ground truth indirect illumination images. The trained model is capable of generating indirect diffuse illumination based solely on direct illumination inputs.

1. **Data Collection**: Gather pairs of direct and indirect illumination images generated in Blender with the generate_dataset script.
2. **Model Training**: Train the machine learning model using the collected image pairs to learn the direct and indirect illumination mapping.
3. **Inference**: Feed direct illumination images into the trained model to generate predicted indirect diffuse illumination.

## Getting Started

To get started with using the Indirect Diffuse Illumination Generator:

1. **Clone the Repository**: Clone this repository to your local machine using `git clone https://github.com/yourusername/indirect-diffuse-illumination.git`.
2. **Install Dependencies**: Install the required dependencies by running `pip install -r requirements.txt`.
3. **Train the Model**: Train the machine learning model using your dataset.
4. **Converting**: Convert the model using the 'PyTorch to Onnx' script and simplify the model using the 'onnx_to_onnx_simplified' script to enable faster inferencing. The latter will remove normalisation layers.
5. **Generate Indirect Illumination**: Use the trained model to generate indirect diffuse illumination from direct illumination images using the inferencing script in the map 'onnx'.

## Contributors

- Tim Pirotte (@Tim-Pirotte)

### Environment maps used in the dataset

- Sergej Majboroda
- Greg Zaal
- Dimitrios Savva
- Jarod Guest
- Alexander Scholten

## License

This project is licensed under this [LICENSE](LICENSE) file.
