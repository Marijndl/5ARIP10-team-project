import torch
from torch import nn

def cal_cart_tensor(origin, spherical):
    r = spherical[:, 0, :].clone()
    theta = spherical[:, 1, :].clone()
    phi = spherical[:, 2, :].clone()

    r.retain_grad()
    theta.retain_grad()
    phi.retain_grad()

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    x.retain_grad()
    y.retain_grad()
    z.retain_grad()

    shape = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1)), dim=1)
    shape.retain_grad()
    full = torch.cat((origin.clone(), shape), dim=2)
    full.retain_grad()

    cartesian = torch.matmul(full, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))))
    cartesian.retain_grad()
    return cartesian

class mPD_loss_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, origin_3D, spherical_3D, origin_2D, spherical_2D, deformation_field):
        # Add deformation to 3D line
        spherical_3D_deformed = spherical_3D.clone()
        spherical_3D_deformed[:, 1:, :] = spherical_3D_deformed[:, 1:, :] + deformation_field

        original_cart = cal_cart_tensor(origin_2D, spherical_2D)
        deformed_cart = cal_cart_tensor(origin_3D, spherical_3D_deformed)

        loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1), dim=1))
        loss.retain_grad()
        return loss


criterion = mPD_loss_2()
origin_2D = torch.rand((512,3,1), requires_grad=True)
origin_3D = torch.rand((512,3,1), requires_grad=True)

spherical_2D = torch.rand((512,3,350), requires_grad=True)
spherical_3D = torch.rand((512,3,350), requires_grad=True)

deformation = torch.rand((512,2,350), requires_grad=True)

spherical_3D_deformed = spherical_3D.clone()
spherical_3D_deformed[:, 1:, :] = spherical_3D_deformed[:, 1:, :] + deformation

r = spherical_3D_deformed[:, 0, :].clone()
theta = spherical_3D_deformed[:, 1, :].clone()
phi = spherical_3D_deformed[:, 2, :].clone()

r.retain_grad()
theta.retain_grad()
phi.retain_grad()

x = r * torch.sin(theta) * torch.cos(phi)
y = r * torch.sin(theta) * torch.sin(phi)
z = r * torch.cos(theta)

x.retain_grad()
y.retain_grad()
z.retain_grad()

shape = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1)), dim=1)
shape.retain_grad()
full = torch.cat((origin_2D.clone(), shape), dim=2)
full.retain_grad()

original_cart = torch.matmul(full, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))))
original_cart.retain_grad()

#___--------------------------------------------------

r2 = spherical_3D_deformed[:, 0, :].clone()
theta2 = spherical_3D_deformed[:, 1, :].clone()
phi2 = spherical_3D_deformed[:, 2, :].clone()

r2.retain_grad()
theta2.retain_grad()
phi2.retain_grad()

x2 = r2 * torch.sin(theta2) * torch.cos(phi2)
y2 = r2 * torch.sin(theta2) * torch.sin(phi2)
z2 = r2 * torch.cos(theta2)

x2.retain_grad()
y2.retain_grad()
z2.retain_grad()

shape2 = torch.cat((x2.unsqueeze(dim=1), y2.unsqueeze(dim=1), z2.unsqueeze(dim=1)), dim=1)
shape2.retain_grad()
full2 = torch.cat((origin_3D.clone(), shape2), dim=2)
full2.retain_grad()

deformed_cart = torch.matmul(full2, torch.triu(torch.ones((x.shape[0], full.shape[2], full.shape[2]))))
deformed_cart.retain_grad()


# loss = criterion(origin_3D, spherical_3D, origin_2D, spherical_2D, deformation)
# loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1), dim=1))
diff = deformed_cart.clone() - original_cart.clone()
diff.retain_grad()

loss = torch.square(diff)
loss.retain_grad()

diff.sum().backward()

print(deformed_cart.grad)



###_________________________


import torch
from torch import nn

origin_2D = torch.rand((512,3,1), requires_grad=True)
origin_3D = torch.rand((512,3,1), requires_grad=True)

spherical_2D = torch.rand((512,3,350), requires_grad=True)
spherical_3D = torch.rand((512,3,350), requires_grad=True)

loss = torch.sum(torch.mean(torch.sqrt(torch.sum(torch.square(spherical_2D - spherical_3D), dim=1), dim=2)), dim=1)

loss = torch.sum(torch.mean(torch.sum(torch.abs(deformed_cart - original_cart), dim=1), dim=1))
# loss = torch.sum(torch.mean(torch.sqrt(torch.sum(torch.square(deformed_cart - original_cart), dim=1)), dim=1), dim=0)
