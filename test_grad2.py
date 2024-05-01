import torch
from torch import nn

origin_2D = torch.rand((512,3,1), requires_grad=True)
origin_3D = torch.rand((512,3,1), requires_grad=True)

spherical_2D = torch.rand((512,3,350), requires_grad=True)
spherical_3D = torch.rand((512,3,350), requires_grad=True)

deformation = torch.rand((512,2,350), requires_grad=True)

diff = spherical_2D.clone() - spherical_3D.clone()
diff.retain_grad()

loss = torch.sum(torch.mean(torch.sum(torch.abs(diff), dim=1), dim=1))
loss.retain_grad()

loss.backward()

print(spherical_2D.grad)
