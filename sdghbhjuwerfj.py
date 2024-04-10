import torch


data_3D = torch.load('D:\\CTA data\\shape_3D.pt')

print(torch.sum(torch.isnan(data_3D)))