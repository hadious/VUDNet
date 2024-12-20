
import torch

# Custom flag for controlling skip connections
use_skip_connection = True  

class UNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # Input block using two convolutional layers
        self.entry_block = ConvBlock(config, in_channels=config['input_channels'], out_channels=config['unet_conv_filters'][0])

        # Contracting (downsampling) layers
        self.down_blocks = torch.nn.ModuleList()
        for idx in range(1, len(config['unet_conv_filters'])):
            self.down_blocks.append(
                DownsampleBlock(config=config, 
                                in_channels=config['unet_conv_filters'][idx - 1], 
                                out_channels=config['unet_conv_filters'][idx])
            )

        # Bridge between contraction and expansion paths
        self.middle_block = DownsampleBlock(config, in_channels=config['unet_conv_filters'][-1], out_channels=config['unet_conv_filters'][-1])

        # Expanding (upsampling) layers
        self.up_blocks = torch.nn.ModuleList()
        self.up_blocks.append(UpBlock(config=config, in_channels=config['unet_conv_filters'][-1], out_channels=config['unet_conv_filters'][-1]))

        for idx in range(len(config['unet_conv_filters']) - 1, 0, -1):
            self.up_blocks.append(UpBlock(config=config, in_channels=config['unet_conv_filters'][idx], out_channels=config['unet_conv_filters'][idx - 1]))

        # Output layer
        self.output_layer = OutputMap(in_channels=config['unet_conv_filters'][0], out_channels=config['n_classes'])

    def forward(self, x):
        """Forward pass through the network."""
        downsampled_features = [self.entry_block(x)]
        for block in self.down_blocks:
            downsampled_features.append(block(downsampled_features[-1]))

        x = self.middle_block(downsampled_features[-1])

        for idx, block in enumerate(self.up_blocks):
            x = block(x, downsampled_features[-(idx + 2)])

        return self.output_layer(x)


class OutputMap(torch.nn.Module):
    """Final convolution to map to the number of output classes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.final_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.final_conv(x)
        x = self.activation(x)
        return x


class ConvBlock(torch.nn.Module):
    """Block with two convolutional layers and batch normalization."""
    
    def __init__(self, config, in_channels, out_channels):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=config['conv_kernel_size'], 
                            stride=config['conv_stride_rate'], padding=config['conv_padding'], 
                            padding_mode=config['conv_padding_style'], bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=config['conv_kernel_size'], 
                            stride=config['conv_stride_rate'], padding=config['conv_padding'], 
                            padding_mode=config['conv_padding_style'], bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.conv_layers(x)


class DownsampleBlock(torch.nn.Module):
    """Block that downsamples the input using max pooling followed by a ConvBlock."""

    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block = ConvBlock(config, in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class UpBlock(torch.nn.Module):
    """Block that upsamples the input and performs a ConvBlock."""

    def __init__(self, config, in_channels, out_channels):
        super().__init__()

        self.padding_mode = config['conv_padding_style']
        self.upsample_layer = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if use_skip_connection:
            self.conv_block = ConvBlock(config, in_channels=in_channels, out_channels=out_channels)
        else:
            self.conv_block = ConvBlock(config, in_channels=in_channels + out_channels, out_channels=out_channels)

    def forward(self, x, skip_feature):
        x = self.upsample_layer(x)
        x = adjust_padding(x, skip_feature, padding_mode=self.padding_mode)
        if not use_skip_connection:
            x = torch.cat([x, skip_feature], dim=1)
        return self.conv_block(x)


def adjust_padding(x, target_feature, padding_mode='constant'):
    """Adjust padding so that the spatial dimensions match between x and target_feature."""
    if isinstance(target_feature, torch.Tensor):
        target_shape = target_feature.size()
    else:
        target_shape = target_feature

    pad_y = target_shape[2] - x.size()[2]
    pad_x = target_shape[3] - x.size()[3]

    if padding_mode == 'zeros':
        padding_mode = 'constant'

    return torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_mode)



# import torch

# skip_skip = True  

# class UNet(torch.nn.Module):

#     def __init__(self, options):
#         super().__init__()

#         self.input_block = DoubleConv(options, input_n=options['input_channels'], output_n=options['unet_conv_filters'][0])

#         self.contract_blocks = torch.nn.ModuleList()
#         for contract_n in range(1, len(options['unet_conv_filters'])):
#             self.contract_blocks.append(
#                 ContractingBlock(options=options,
#                                  input_n=options['unet_conv_filters'][contract_n - 1],
#                                  output_n=options['unet_conv_filters'][contract_n]))  # only used to contract input patch.

#         self.bridge = ContractingBlock(options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1])

#         self.expand_blocks = torch.nn.ModuleList()
#         self.expand_blocks.append(
#             ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1]))

#         for expand_n in range(len(options['unet_conv_filters']), 1, -1):
#             self.expand_blocks.append(ExpandingBlock(options=options,
#                                                      input_n=options['unet_conv_filters'][expand_n - 1],
#                                                      output_n=options['unet_conv_filters'][expand_n - 2]))

#         self.output = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes'])
#     def forward(self, x):
#         """Forward model pass."""
#         x_contract = [self.input_block(x)]
#         for contract_block in self.contract_blocks:
#             x_contract.append(contract_block(x_contract[-1]))
#         x_expand = self.bridge(x_contract[-1])
#         up_idx = len(x_contract)
#         for expand_block in self.expand_blocks:
#             x_expand = expand_block(x_expand, x_contract[up_idx - 1])
#             up_idx -= 1            

#         return self.output(x_expand)


# class FeatureMap(torch.nn.Module):
#     """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

#     def __init__(self, input_n, output_n):
#         super(FeatureMap, self).__init__()

#         self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))
#         self.activation = torch.nn.ReLU() 

#     def forward(self, x):
        
#         x = self.feature_out(x)
#         x = self.activation(x)
#         return x


# class DoubleConv(torch.nn.Module):
#     """Class to perform a double conv layer in the U-NET architecture. Used in unet_model.py."""

#     def __init__(self, options, input_n, output_n):
#         super(DoubleConv, self).__init__()

#         self.double_conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=input_n,
#                       out_channels=output_n,
#                       kernel_size=options['conv_kernel_size'],
#                       stride=options['conv_stride_rate'],
#                       padding=options['conv_padding'],
#                       padding_mode=options['conv_padding_style'],
#                       bias=False),
#             torch.nn.BatchNorm2d(output_n),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(in_channels=output_n,
#                       out_channels=output_n,
#                       kernel_size=options['conv_kernel_size'],
#                       stride=options['conv_stride_rate'],
#                       padding=options['conv_padding'],
#                       padding_mode=options['conv_padding_style'],
#                       bias=False),
#             torch.nn.BatchNorm2d(output_n),
#             torch.nn.ReLU()
#         )

#     def forward(self, x):
#         """Pass x through the double conv layer."""
#         x = self.double_conv(x)

#         return x


# class ContractingBlock(torch.nn.Module):
#     """Class to perform downward pass in the U-Net."""

#     def __init__(self, options, input_n, output_n):
#         super(ContractingBlock, self).__init__()

#         self.contract_block = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.double_conv = DoubleConv(options, input_n, output_n)

#     def forward(self, x):
#         """Pass x through the downward layer."""
#         x = self.contract_block(x)
#         x = self.double_conv(x)
#         return x


# class ExpandingBlock(torch.nn.Module):
#     """Class to perform upward layer in the U-Net."""

#     def __init__(self, options, input_n, output_n):
#         super(ExpandingBlock, self).__init__()

#         self.padding_style = options['conv_padding_style']
#         self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         if skip_skip:
#             self.double_conv = DoubleConv(options, input_n=input_n, output_n=output_n)
#         else:
#             self.double_conv = DoubleConv(options, input_n=input_n + output_n, output_n=output_n)
        


#     def forward(self, x, x_skip):
#         """Pass x through the upward layer and concatenate with opposite layer."""
#         x = self.upsample(x)

#         # Insure that x and skip H and W dimensions match.
#         x = expand_padding(x, x_skip, padding_style=self.padding_style)
#         if not skip_skip:
#             x = torch.cat([x, x_skip], dim=1)
#             # x = torch.cat([x, x], dim=1)
#             # x=x

#         return self.double_conv(x)
    
    
# def expand_padding(x, x_contract, padding_style: str = 'constant'):
#     """
#     Insure that x and x_skip H and W dimensions match.
#     Parameters
#     ----------
#     x :
#         Image tensor of shape (batch size, channels, height, width). Expanding path.
#     x_contract :
#         Image tensor of shape (batch size, channels, height, width) Contracting path.
#         or torch.Size. Contracting path.
#     padding_style : str
#         Type of padding.

#     Returns
#     -------
#     x : ndtensor
#         Padded expanding path.
#     """
#     # Check whether x_contract is tensor or shape.
#     if type(x_contract) == type(x):
#         x_contract = x_contract.size()

#     # Calculate necessary padding to retain patch size.
#     pad_y = x_contract[2] - x.size()[2]
#     pad_x = x_contract[3] - x.size()[3]

#     if padding_style == 'zeros':
#         padding_style = 'constant'

#     x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

#     return x