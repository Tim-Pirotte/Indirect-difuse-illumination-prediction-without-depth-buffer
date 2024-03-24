import torch
from torch import nn


class Block(nn.Module):
    """
    Defines a basic convolutional block used in the Generator network.

    Args:
    - input_channels (int): Number of input channels.
    - output_channels (int): Number of output channels.
    - encoder (bool): If True, defines an encoder block. If False, defines a decoder block.
    - activation (str): Activation function to use. Defaults to "relu".
    - use_dropout (bool): If True, applies dropout regularization. Defaults to False.
    """
  
    def __init__(self, input_channels, output_channels, encoder=True, activation="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if encoder
            else nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=output_channels),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.encoder = encoder

    def forward(self, x):
        """
        Defines the forward pass of the block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
      
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    Defines the Generator network for image generation.

    Args:
    - input_channels (int): Number of input channels. Defaults to 3.
    - features (int): Number of features used in the network. Defaults to 64.
    """
  
    def __init__(self, input_channels=3, features=64):
        super(Generator, self).__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(num_groups=2, num_channels=features * 2),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(num_groups=2, num_channels=features * 4),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(num_groups=2, num_channels=features * 8),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(num_groups=2, num_channels=features * 8),
        )

        self.encoder5 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(num_groups=2, num_channels=features * 8),
        )

        self.encoder6 = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(num_groups=2, num_channels=features * 8),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1),
            nn.ReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(features * 8, features * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=features * 8),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(features * 8 * 2, features * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=features * 8),
            nn.ReLU()
        )

        self.decoder3 = Block(features * 8 * 2, features * 8, encoder=False, activation="relu", use_dropout=True)
        self.decoder4 = Block(features * 8 * 2, features * 8, encoder=False, activation="relu", use_dropout=False)
        self.decoder5 = Block(features * 8 * 2, features * 4, encoder=False, activation="relu", use_dropout=False)
        self.decoder6 = Block(features * 4 * 2, features * 2, encoder=False, activation="relu", use_dropout=False)
        self.decoder7 = Block(features * 2 * 2, features, encoder=False, activation="relu", use_dropout=False)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(features * 2, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Tanh for pixel values in the range [-1, 1]
        )

    def forward(self, x):
        """
        Defines the forward pass of the Generator network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
      
        initial_output = self.initial_down(x)
        encoder1 = self.encoder1(initial_output)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        encoder6 = self.encoder6(encoder5)

        bottleneck = self.bottleneck(encoder6)

        decoder1 = self.decoder1(bottleneck)

        decoder1 = torch.nn.functional.pad(decoder1, (0, 1), "constant", 0)

        decoder2 = self.decoder2(torch.cat([decoder1, encoder6], 1))

        decoder2 = torch.nn.functional.pad(decoder2, (0, 1), "constant", 0)

        decoder3 = self.decoder3(torch.cat([decoder2, encoder5], 1))

        decoder4 = self.decoder4(torch.cat([decoder3, encoder4], 1))

        decoder4 = torch.nn.functional.pad(decoder4, (0, 0, 1, 0), "constant", 0)

        decoder5 = self.decoder5(torch.cat([decoder4, encoder3], 1))

        decoder5 = torch.nn.functional.pad(decoder5, (0, 0, 1, 0), "constant", 0)

        decoder6 = self.decoder6(torch.cat([decoder5, encoder2], 1))

        decoder6 = torch.nn.functional.pad(decoder6, (0, 0, 1, 0), "constant", 0)

        decoder7 = self.decoder7(torch.cat([decoder6, encoder1], 1))

        return self.output(torch.cat([decoder7, initial_output], 1))


class CNN_block(nn.Module):
    """
    Defines a basic convolutional block used in the Discriminator network.

    Args:
    - input_channels (int): Number of input channels.
    - output_channels (int): Number of output channels.
    - stride (int): Stride value for convolution operation.
    """
  
    def __init__(self, input_channels, output_channels, stride):
        super(CNN_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 4, stride, bias=False, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        Defines the forward pass of the block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
      
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Defines the Discriminator network for distinguishing between real and fake images.

    Args:
    - input_channels (int): Number of input channels.
    - features (list): List of feature sizes for different layers.
    """
  
    def __init__(self, input_channels, features):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels * 2, features[0], kernel_size=4, stride=4, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        layers = []
        input_channels = features[0]
        for feature in features[1:]:
            stride = 1 if feature == features[-1] else 2

            layers.append(CNN_block(input_channels, feature, stride=stride))

            input_channels = feature

        layers.append(
            nn.Conv2d(
                input_channels,
                1,
                4,
                1,
                1,
                padding_mode='reflect'
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Defines the forward pass of the Discriminator network.

        Args:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor): Target tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
      
        x = torch.cat([x, y], 1)
        x = self.initial(x)
        return self.model(x)
