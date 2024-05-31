import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, Input_channel):
        super(VAE, self).__init__()
        self.Input_channel = Input_channel
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.Input_channel, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space
        self.fc1 = nn.Linear(128 * 40 * 30, 512)  # Smaller latent dimension
        self.fc2 = nn.Linear(128 * 40 * 30, 512)

        # Decoder
        self.decoder_input = nn.Linear(512, 128 * 40 * 30)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # Final output to 1 channel
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mu, logvar)
        z = self.decoder_input(z)
        z = z.view(-1, 128, 40, 30)
        return self.decoder(z), mu, logvar




skip_skip = True  
class UNet(torch.nn.Module):

    def __init__(self, options):
        super().__init__()

        self.input_block = DoubleConv(options, input_n=options['input_channels'], output_n=options['unet_conv_filters'][0])

        self.contract_blocks = torch.nn.ModuleList()
        for contract_n in range(1, len(options['unet_conv_filters'])):
            self.contract_blocks.append(
                ContractingBlock(options=options,
                                 input_n=options['unet_conv_filters'][contract_n - 1],
                                 output_n=options['unet_conv_filters'][contract_n]))  
        self.bridge = ContractingBlock(options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1])

        self.expand_blocks = torch.nn.ModuleList()
        self.expand_blocks.append(
            ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1]))

        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=options['unet_conv_filters'][expand_n - 1],
                                                     output_n=options['unet_conv_filters'][expand_n - 2]))

        self.output = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes'])
    def forward(self, x):
        
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1            

        return self.output(x_expand)


class FeatureMap(torch.nn.Module):
   

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))
        self.activation = torch.nn.ReLU() 

    def forward(self, x):
        
        x = self.feature_out(x)
        x = self.activation(x)
        return x


class DoubleConv(torch.nn.Module):
   
    def __init__(self, options, input_n, output_n):
        super(DoubleConv, self).__init__()

        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_n,
                      out_channels=output_n,
                      kernel_size=options['conv_kernel_size'],
                      stride=options['conv_stride_rate'],
                      padding=options['conv_padding'],
                      padding_mode=options['conv_padding_style'],
                      bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_n,
                      out_channels=output_n,
                      kernel_size=options['conv_kernel_size'],
                      stride=options['conv_stride_rate'],
                      padding=options['conv_padding'],
                      padding_mode=options['conv_padding_style'],
                      bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU()
        )

    def forward(self, x):
    
        x = self.double_conv(x)

        return x


class ContractingBlock(torch.nn.Module):
   

    def __init__(self, options, input_n, output_n):
        super(ContractingBlock, self).__init__()

        self.contract_block = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(options, input_n, output_n)

    def forward(self, x):
        x = self.contract_block(x)
        x = self.double_conv(x)
        return x


class ExpandingBlock(torch.nn.Module):

    def __init__(self, options, input_n, output_n):
        super(ExpandingBlock, self).__init__()

        self.padding_style = options['conv_padding_style']
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if skip_skip:
            self.double_conv = DoubleConv(options, input_n=input_n, output_n=output_n)
        else:
            self.double_conv = DoubleConv(options, input_n=input_n + output_n, output_n=output_n)
        


    def forward(self, x, x_skip):
        x = self.upsample(x)

        # Insure that x and skip H and W dimensions match.
        x = expand_padding(x, x_skip, padding_style=self.padding_style)
        if not skip_skip:
            x = torch.cat([x, x_skip], dim=1)
            # x = torch.cat([x, x], dim=1)
            # x=x

        return self.double_conv(x)
    
    
def expand_padding(x, x_contract, padding_style: str = 'constant'):

    if type(x_contract) == type(x):
        x_contract = x_contract.size()

    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv2d(2, out_channels=64, kernel_size=3, padding=1) 
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
