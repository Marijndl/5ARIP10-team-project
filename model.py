import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1),
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

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out, kernel_size=3, padding=1, stride=2))

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
        self.upconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.upconv(x)


class CARNet(nn.Module):
    def __init__(self):
        super(CARNet, self).__init__()
        self.dual_branch_3d = DualBranch()
        self.dual_branch_2d = DualBranch()
        self.unet_backbone = UNet()

    def forward(self, x3d_origin, x3d_shape, x2d_origin, x2d_shape):

        x3d = self.dual_branch_3d(x3d_origin, x3d_shape)
        x2d = self.dual_branch_2d(x2d_origin, x2d_shape)

        x = torch.cat((x3d, x2d), dim=1)
#         # print(x.shape)

        # Pad the input to the Unet:
        # x = torch.cat((x, torch.zeros((x.shape[0], x.shape[1], 3))), dim=2)
        # x = torch.cat((x, torch.zeros((x.shape[0], x.shape[1], 3)).to(device)), dim=2)
#         # print("Padded "+str(x.shape))

        deformation_field = self.unet_backbone(x)

        return deformation_field

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = grey(128, 64)
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
        # print("Conn" + str(conn1.shape))
        skip_conn2 = self.down2(conn1)
        # print("Conn2" + str(skip_conn2.shape))
        skip_conn3 = self.down3(skip_conn2)
        # print("Conn3" + str(skip_conn3.shape))
        skip_conn4 = self.down4(skip_conn3)
        # print("Conn4" + str(skip_conn4.shape))

        
        x = self.bridge(skip_conn4)

        # print("x " + str(x.shape))
        # # print("skip_conn3 " + str(skip_conn3.shape))
        # # print("skip_conn2 " + str(skip_conn2.shape))
        
        x = self.up1(x)
        x = torch.cat((x, skip_conn3), dim=1)
        # print("x after upconv = " + str(x.shape))
        x = self.up2(x)
        x = self.up3(x)
        x = torch.cat((x, skip_conn2), dim=1)
        # print("x after upconv = " + str(x.shape))
        x = self.up4(x)
        x = self.up5(x)
        x = torch.cat((x, conn1), dim=1)
        # print("x after upconv = " + str(x.shape))
        x = self.up6(x)

        # print("x after up6 = " + str(x.shape))
        deformation_field = self.output_layer(x)
        # print("output " + str(deformation_field.shape))
        return deformation_field
