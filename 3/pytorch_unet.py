import torch
import torch.nn as nn

def double_conv_down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 32, padding=16),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, 32, padding=15),
        nn.ReLU(inplace=True)
    )

def double_conv_up(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 32, padding=15),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, 32, padding=16),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv_down(1, 32)
        self.dconv_down2 = double_conv_down(32, 64)
        self.dconv_down3 = double_conv_down(64, 128)
        self.dconv_down4 = double_conv_down(128, 256)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2)        
        
        self.dconv_up3 = double_conv_up(256 + 128, 128)
        self.dconv_up2 = double_conv_up(128 + 64, 64)
        self.dconv_up1 = double_conv_up(64 + 32, 32)
        
        self.conv_last = nn.Conv1d(32, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)

        x = self.upsample(x)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
