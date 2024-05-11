import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=2)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out

class NUNet(nn.Module):
    def __init__(self):
        super(NUNet, self).__init__()

        self.activation = F.tanh
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)   
        self.conv_block256_512 = UNetConvBlock(256, 512)  
        self.conv_block512_1024 = UNetConvBlock(512, 1024)  

        self.up_block1024_512 = UNetUpBlock(1024, 512)   
        self.up_block512_256 = UNetUpBlock(512, 256)  
        self.up_block256_128 = UNetUpBlock(256, 128)   
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        block1 = self.conv_block3_16(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block16_32(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block32_64(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block64_128(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block128_256(pool4) 
        pool5 = self.pool4(block5)
         
        block6 = self.conv_block256_512(pool5) 
        pool6 = self.pool4(block6)

        block7 = self.conv_block512_1024(pool6)   

        up1 = self.activation(self.up_block1024_512(block7, block6))  
        up2 = self.activation(self.up_block512_256(up1, block5))   
        up3 = self.activation(self.up_block256_128(up2, block4))  
        up4 = self.activation(self.up_block128_64(up3, block3))

        up5 = self.activation(self.up_block64_32(up4, block2))

        up6 = self.up_block32_16(up5, block1)

        return self.last(up6)
