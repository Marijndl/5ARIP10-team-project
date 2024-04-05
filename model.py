import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sperical_coordinates


class DualBranch(nn.Module):
    def __init__(self):
        super(DualBranch, self).__init__()
        self.origin_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),     
        )
        
        self.shape_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            

        )
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, origin, shape):
        origin_feat = self.origin_branch(origin)
        shape_feat = self.shape_branch(shape)

        return self.bn_relu(torch.add(origin_feat, shape_feat))
        


class Downsample(nn.Module):
    def __init__(self, in_channels, out):
        super(Downsample, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=out, kernel_size=1)
        )

        self.bn_relu_add = nn.Sequential(
            nn.BatchNorm1d(out),
            nn.ReLU()
        )

        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=out, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        return self.bn_relu_add(torch.add(self.conv_layers(x), self.conv2(x)))
    
class grey(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(grey, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class orange(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(orange, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding =1, stride =2, output_padding=1)
        
    def forward(self, x, prev_info):
        upsampled_x = self.upconv(x)
        # If upsampled_x is larger, crop it to match prev_info's size
        target_length = prev_info.size()[2]
        current_length = upsampled_x.size()[2]
        if current_length > target_length:
            # Calculate cropping
            crop_left = (current_length - target_length) // 2
            crop_right = current_length - crop_left - target_length
            # Apply cropping
            upsampled_x = upsampled_x[:, :, crop_left:current_length-crop_right]
        return torch.cat((upsampled_x, prev_info), dim=1)


class CARNet(nn.Module):
    def __init__(self):
        super(CARNet, self).__init__()
        self.dual_branch_3d = DualBranch()
        self.dual_branch_2d = DualBranch()
        self.unet_backbone = UNet()

    def forward(self, x3d_origin, x3d_shape, x2d_origin, x2d_shape):
        
        x3d = self.dual_branch_3d(x3d_origin, x3d_shape)
        x2d = self.dual_branch_2d(x2d_origin, x2d_shape)
        
        
        x = torch.cat((x3d, x2d), dim=0)
        
        deformation_field = self.unet_backbone(x)
        
        return deformation_field

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.down1 = grey(64, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)  
        self.down4 = Downsample(256, 512)  
        

        self.bridge = grey(512, 512)
        
        
        self.up1 = orange(512, 256)  
        self.up2 = grey(512, 256)
        self.up3 = orange(256, 128)  
        self.up4 = grey(256, 128)
        self.up5 = orange(128, 64)  
        self.up6 = grey(128, 64)

        
        self.output_layer = nn.Conv1d(64, 2, kernel_size=1)

    def forward(self, x):
        
        conn1 = self.down1(x)
        print(conn1.shape)
        skip_conn2 = self.down2(conn1)
        print(skip_conn2.shape)
        skip_conn3 = self.down3(skip_conn2)
        print(skip_conn3.shape)
        skip_conn4 = self.down4(skip_conn3)
        print(skip_conn4.shape)
        
        x = self.bridge(skip_conn4)
        
        
        x = self.up1(x, skip_conn3)
        x = self.up2(x)
        x = self.up3(x, skip_conn2)
        x = self.up4(x)
        x = self.up5(x, conn1)
        x = self.up6(x)
        
        
        deformation_field = self.output_layer(x)
        return deformation_field
    

    
    
